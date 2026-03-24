import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from methyldl.modelling.classifiers.minirnns.associative_scan import (
    associative_scan_log,
)
from methyldl.modelling.classifiers.minirnns.helpers import g, log_g
from methyldl.modelling.utils import exists


class MinGRUCell(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        use_init_hidden_state: bool = False,
        batch_first: bool = True,
    ):
        """
        A "minimal" GRU-like module. Expects input either:
          - (batch, seq_len, dim) if batch_first == True, or
          - (seq_len, batch, dim) if batch_first == False.
        """
        super().__init__()
        self.to_hidden_and_gate = nn.Linear(input_dim, hidden_dim * 2)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

        # Optional learnable initial state
        self.init_hidden_state = (
            nn.Parameter(torch.randn(hidden_dim), requires_grad=True)
            if use_init_hidden_state
            else None
        )

        self.batch_first = batch_first
        self.hidden_dim = hidden_dim

    def forward(self, x: Tensor, prev_state=None, parallel_scan=True):
        """
        Inputs:
            x: shape = (batch, seq_len, dim) if self.batch_first == True
                      (seq_len, batch, dim) otherwise
            prev_state: optional (prev_hidden, prev_log_hidden),
                        each of shape (batch, 1, dim_inner).

        Returns:
            out:         same shape as x, but last dim is 'dim' (since we do to_out).
            next_state:  (next_hidden, next_log_hidden), each (batch, 1, dim_inner).
        """
        # If not batch_first, transpose to (batch, seq_len, dim)
        if not self.batch_first:
            x = x.transpose(0, 1)  # (seq_len, batch, dim) -> (batch, seq_len, dim)

        batch, seq_len, dim = x.shape
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim=-1)

        # --------- If seq_len == 1, do a trivial step-by-step update --------- #
        if seq_len == 1 or not parallel_scan:
            tilde_h = g(hidden)  # shape = (batch, 1, dim_inner)
            gate_sig = gate.sigmoid()  # shape = (batch, 1, dim_inner)

            if exists(prev_state):
                prev_hidden, _ = prev_state  # each shape = (batch, 1, dim_inner)
                out = (1 - gate_sig) * prev_hidden + gate_sig * tilde_h
            elif exists(self.init_hidden_state):
                init_h = g(self.init_hidden_state).unsqueeze(0)  # (1, dim_inner)
                init_h = init_h.expand(batch, -1)  # (batch, dim_inner)
                init_h = init_h.unsqueeze(1)  # (batch, 1, dim_inner)
                out = (1 - gate_sig) * init_h + gate_sig * tilde_h
            else:
                out = gate_sig * tilde_h

            next_hidden = out[:, -1:]  # (batch, 1, dim_inner)
            next_log_hidden = out[:, -1:].log()  # (batch, 1, dim_inner)

        # ------------- If seq_len > 1, do log-scan approach ------------- #
        else:
            # log-space "coeffs" and "values"
            log_coeffs = -F.softplus(gate)  # log(1 - sigmoid(gate)) in effect
            log_z = -F.softplus(-gate)  # log(sigmoid(gate))
            log_tilde_h = log_g(hidden)  # log of non-linear transform
            log_values = log_z + log_tilde_h  # sum in log-space

            # If we have a previous hidden state or a learnable init, prepend it
            if exists(prev_state) or exists(self.init_hidden_state):
                if exists(prev_state):
                    _, prev_log_hidden = prev_state
                else:
                    # Expand init hidden to entire batch
                    init_h = g(self.init_hidden_state)  # (dim_inner,)
                    init_h = init_h.unsqueeze(0).expand(batch, -1)  # (batch, dim_inner)
                    init_h = init_h.unsqueeze(1)  # (batch, 1, dim_inner)
                    prev_log_hidden = init_h.log()

                # For log_values shape = (batch, seq_len, dim_inner)
                # We prepend one step so cat along time dim = 1
                log_values = torch.cat((prev_log_hidden, log_values), dim=1)
                # Also adjust log_coeffs with a pad
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

            # Perform log-space prefix-scan
            log_out = associative_scan_log(log_coeffs, log_values, return_log=True)
            # We only want the last seq_len steps for the final "output" portion
            out = torch.exp(
                log_out[:, -seq_len:]
            )  # shape = (batch, seq_len, dim_inner)

            next_hidden = out[:, -1:]  # (batch, 1, dim_inner)
            next_log_hidden = log_out[:, -1:]  # (batch, 1, dim_inner)

        # map from dim_inner -> dim
        out = self.to_out(out)

        # If not batch_first, restore shape to (seq_len, batch, dim)
        if not self.batch_first:
            out = out.transpose(0, 1)

        return out, (next_hidden, next_log_hidden)


class MinGRU(nn.Module):
    """
    A stacked mini-GRU with `num_layers`.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        use_init_hidden_state: bool = False,
        batch_first: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first

        # Build a stack of SingleMinGRU
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            # for the first layer, input_dim -> hidden_dim
            # for subsequent layers, hidden_dim -> hidden_dim
            layer_input_dim = input_dim if layer_idx == 0 else hidden_dim

            self.layers.append(
                MinGRUCell(
                    input_dim=layer_input_dim,
                    hidden_dim=hidden_dim,
                    use_init_hidden_state=use_init_hidden_state,
                    batch_first=batch_first,
                )
            )

    def forward(self, x, prev_states=None, parallel_scan=True):
        """
        x: shape = (batch, seq_len, input_dim) or (seq_len, batch, input_dim)
        prev_states: optionally a list of states for each layer:
                     [ (h0_layer0, log_h0_layer0), (h0_layer1, log_h0_layer1), ... ]
        """
        # If no previous states, pass None for each layer
        if prev_states is None:
            prev_states = [None] * self.num_layers
        assert len(prev_states) == self.num_layers

        output = x
        next_states = []
        for layer_idx, (layer, prev_state) in enumerate(zip(self.layers, prev_states)):
            output, next_state = layer.forward(output, prev_state, parallel_scan)
            next_states.append(next_state)

        return output, next_states


class BiMinGRU(nn.Module):
    """
    Bidirectional wrapper around minGRU.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        use_init_hidden_state: bool = False,
        batch_first: bool = True,
        num_layers=1,
    ):
        """
        If batch_first=True, we assume input is (batch, seq_len, dim).
        Otherwise, (seq_len, batch, dim).
        """
        super().__init__()
        self.batch_first = batch_first

        # forward and backward miniGRU
        self.fwd_gru = MinGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            use_init_hidden_state=use_init_hidden_state,
            batch_first=batch_first,
            num_layers=num_layers,
        )
        self.bwd_gru = MinGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            use_init_hidden_state=use_init_hidden_state,
            batch_first=batch_first,
            num_layers=num_layers,
        )

    def forward(
        self, x: Tensor, prev_state_fwd=None, prev_state_bwd=None, parallel_scan=True
    ):
        """
        Inputs:
          x: shape = (batch, seq_len, dim) if self.batch_first=True
                     (seq_len, batch, dim) otherwise.
          prev_state_fwd, prev_state_bwd: each is (hidden, log_hidden).

        Returns:
          out:        shape = same as x, but last dim is 2*dim
                       (concatenation of fwd and bwd).
          next_state: (next_hidden, next_log_hidden),
                      each shape = (batch, 1, 2*dim_inner) always given the code.
        """
        # -- Forward direction --
        out_fwd, next_state_fwd = self.fwd_gru(x, prev_state_fwd, parallel_scan)
        next_state_fwd = next_state_fwd[-1]  # last layer only

        # We need to flip x along the time dimension. That dimension is:
        #   dim=1 if batch_first=True, else dim=0
        seq_dim = 1 if self.batch_first else 0
        x_reversed = torch.flip(x, dims=[seq_dim])

        # -- Backward direction --
        out_bwd_reversed, next_state_bwd = self.bwd_gru(
            x_reversed, prev_state_bwd, parallel_scan
        )  # last layer only
        next_state_bwd = next_state_bwd[-1]  # last layer only

        # Flip the backward output back
        out_bwd = torch.flip(out_bwd_reversed, dims=[seq_dim])

        # Concatenate forward and backward outputs along the last dimension
        # Regardless of batch_first, the "feature" dimension is always -1
        out = torch.cat([out_fwd, out_bwd], dim=-1)

        # Combine hidden states
        # next_state_fwd = (fwd_hidden, fwd_log_hidden) each (batch, 1, dim_inner)
        # next_state_bwd = (bwd_hidden, bwd_log_hidden) each (batch, 1, dim_inner)
        # -> We produce (batch, 1, 2*dim_inner)
        next_hidden_fwd, next_log_hidden_fwd = next_state_fwd
        next_hidden_bwd, next_log_hidden_bwd = next_state_bwd

        next_hidden = torch.cat([next_hidden_fwd, next_hidden_bwd], dim=-1)
        next_log_hidden = torch.cat([next_log_hidden_fwd, next_log_hidden_bwd], dim=-1)
        next_state = (next_hidden, next_log_hidden)

        return out, next_state
