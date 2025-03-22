from typing import Optional, Union, List, Dict, Sequence, Tuple

from transformers import PretrainedConfig,PreTrainedTokenizer

from collections import OrderedDict
from typing import Optional

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn  # Legacy mambav1 file structure
except ImportError:
    try:
        from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn  # mambav2 file structure
    except ImportError:
        RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class RCPSEmbedding(nn.Module):
    """Embedding layer that supports reverse-complement equivariance."""
    def __init__(self, vocab_size: int, d_model: int, complement_map: dict, **factory_kwargs):
        """
        Args:
            vocab_size: Size of vocabulary.
            d_model: Dimensionality of embedding (actual embedding matrix will have 1/2 the output dim).
            complement_map: Dictionary mapping each token id to its complement.
        """
        super().__init__()
        self.register_buffer(
            "complement_map",
            torch.tensor(list(OrderedDict(complement_map).values()), dtype=torch.long)
        )
        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

    @property
    def weight(self):
        """Embedding weights."""
        return self.embedding.weight

    def set_weight(self, value):
        """Set embedding weights."""
        self.embedding.weight = value

    def rc(self, x):
        """Reverse-complement a tensor of input_ids by flipping along length dimension and complementing the ids."""
        return torch.gather(
            self.complement_map.unsqueeze(0).expand(x.shape[0], -1),
            dim=1,
            index=torch.flip(x, dims=[-1])
        )

    def forward(self, input_ids):
        """Reverse-complement equivariant forward pass.

        This embedding module doubles the output dimensionality to support reverse-complement equivariance.

        Args:
            input_ids: Input tensor of shape (batch_size, seq_len)
        Returns:
            Embedding tensor of shape (batch_size, seq_len, d_model * 2)
        """
        fwd_out = self.embedding(input_ids)
        rc_out = torch.flip(self.embedding(self.rc(input_ids)), dims=[-2, -1])

        return torch.cat([fwd_out, rc_out], dim=-1)


class RCPSWrapper(nn.Module):
    """Wrapper to convert arbitrary nn.Module into a reverse-complement equivariant module.

    See ref. "Towards a Better Understanding of Reverse-Complement Equivariance for Deep Learning Models in Regulatory
    Genomics", Zhou et al. (2022), https://proceedings.mlr.press/v165/zhou22a.html for more details.
    """
    def __init__(self, submodule: nn.Module):
        super().__init__()
        self.submodule = submodule

    @staticmethod
    def rc(x):
        """Reverse-complement a tensor by flipping the length (dim=-2) and channel (dim=-1) dimensions."""
        return torch.flip(x, dims=[-2, -1])

    def forward(self, x, **kwargs):
        """Reverse-complement equivariant forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, channels)
        Returns:
            Output tensor of shape (batch_size, seq_len, channels)
        """
        n_channels = x.shape[-1]
        # Run submodule along sequence
        fwd_out = self.submodule(x[..., :n_channels // 2], **kwargs)
        # Run submodule along rc-sequence
        rc_out = self.submodule(self.rc(x[..., n_channels // 2:]), **kwargs)
        # Concatenate along channel dimension (dim=-1)
        return torch.cat([fwd_out, self.rc(rc_out)], dim=-1)


class RCPSAddNormWrapper(RCPSWrapper):
    """RC equivariant AddNorm layer."""
    def __init__(self, submodule: nn.Module):
        super().__init__(submodule)

    def forward(self, x, residual=None, prenorm=False):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, channels)
            residual: Residual tensor of shape (batch_size, seq_len, channels) or None.
            prenorm: Whether to return residual.
        """
        n_channels = x.shape[-1]
        if residual is None:
            residual = x
            x_fwd = self.submodule(x[..., :n_channels // 2].to(dtype=self.submodule.weight.dtype))
            x_rc = self.submodule(self.rc(x[..., n_channels // 2:]).to(dtype=self.submodule.weight.dtype))
            x = torch.cat([x_fwd, self.rc(x_rc)], dim=-1)
        else:
            residual_fwd = x[..., :n_channels // 2] + residual[..., :n_channels // 2]
            x_fwd = self.submodule(residual_fwd.to(dtype=self.submodule.weight.dtype))

            residual_rc = self.rc(x[..., n_channels // 2:]) + self.rc(residual[..., n_channels // 2:])
            x_rc = self.submodule(residual_rc.to(dtype=self.submodule.weight.dtype))

            residual = torch.cat([residual_fwd, self.rc(residual_rc)], dim=-1)
            x = torch.cat([x_fwd, self.rc(x_rc)], dim=-1)

        return x if not prenorm else (x, residual)


class RCPSMambaBlock(nn.Module):
    def __init__(
            self,
            dim,
            mixer_cls,
            norm_cls=nn.LayerNorm,
            fused_add_norm=False,
            residual_in_fp32=False,
            device=None,  # Keep for consistency with original Mamba Block
            dtype=None,  # Keep for consistency with original Mamba Block
    ):
        """RCPS version of simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection.

        Adapted from: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = RCPSWrapper(mixer_cls(dim))
        norm_f = norm_cls(dim)
        self.norm = norm_f if fused_add_norm else RCPSAddNormWrapper(norm_f)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual)).
            inference_params: inference parameters for mixer.
        """
        if not self.fused_add_norm:
            hidden_states, residual = self.norm(hidden_states, residual=residual, prenorm=True)
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn

            hidden_states_fwd, residual_fwd = fused_add_norm_fn(
                hidden_states[..., hidden_states.shape[-1] // 2:],
                self.norm.weight,
                self.norm.bias,
                residual=residual[..., hidden_states.shape[-1] // 2:] if residual is not None else None,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )

            hidden_states_rc, residual_rc = fused_add_norm_fn(
                hidden_states[..., :hidden_states.shape[-1] // 2].flip(dims=[-2, -1]),
                self.norm.weight,
                self.norm.bias,
                residual=residual[..., :hidden_states.shape[-1] // 2].flip(dims=[-2, -1]) if residual is not None else None,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
            hidden_states = torch.cat([hidden_states_fwd, hidden_states_rc.flip(dims=[-2, -1])], dim=-1)
            residual = torch.cat([residual_fwd, residual_rc.flip(dims=[-2, -1])], dim=-1)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """Allocate inference cache for mixer.

        Keep for compatibility with original Mamba Block.
        """
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class RCPSLMHead(nn.Module):
    """LM Head for reverse-complement equivariant inputs, which have dim * 2 relative to standard inputs."""
    def __init__(self, true_dim: int, vocab_size: int, complement_map: dict, **factory_kwargs):
        """
        `true_dim` corresponds to the actual dimensionality of the input were it not reverse-complement
        equivariant, i.e. 0.5 times the actual input dim.
        """
        super().__init__()
        self.register_buffer(
            "complement_map",
            torch.tensor(list(OrderedDict(complement_map).values()), dtype=torch.long)
        )
        self.true_dim = true_dim
        self.lm_head = nn.Linear(true_dim, vocab_size, bias=False, **factory_kwargs)

    @property
    def weight(self):
        """LM head weights."""
        return self.lm_head.weight

    def set_weight(self, value):
        """Set LM head weights."""
        self.lm_head.weight = value

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim), where dim = 2 * true_dim.
        """
        n_channels = x.shape[-1]
        assert n_channels == 2 * self.true_dim, "Input must have 2 * true_dim channels."
        fwd_logits = F.linear(x[..., :n_channels // 2], self.weight, bias=self.lm_head.bias)
        rc_logits = F.linear(
            torch.flip(x[..., n_channels // 2:], dims=[-1]),
            self.weight[self.complement_map, :],
            bias=self.lm_head.bias
        )
        return fwd_logits + rc_logits

class CaduceusConfig(PretrainedConfig):
    """Config that extends the original MambaConfig with params relevant to bi-directionality and RC equivariance."""
    model_type = "caduceus"

    def __init__(
            self,
            # From original MambaConfig
            d_model: int = 2560,
            n_layer: int = 64,
            vocab_size: int = 50277,
            ssm_cfg: Optional[dict] = None,
            rms_norm: bool = True,
            residual_in_fp32: bool = True,
            fused_add_norm: bool = True,
            pad_vocab_size_multiple: int = 8,

            # Not in original MambaConfig, but default arg in create_block in mamba_ssm repo; used in layer norm
            norm_epsilon: float = 1e-5,

            # Used in init_weights
            initializer_cfg: Optional[dict] = None,

            # Caduceus-specific params
            bidirectional: bool = True,
            bidirectional_strategy: Union[str, None] = "add",
            bidirectional_weight_tie: bool = True,
            rcps: bool = False,
            complement_map: Optional[dict] = None,  # used for RCPSEmbedding / RCPSLMHead
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.ssm_cfg = ssm_cfg
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.norm_epsilon = norm_epsilon
        self.initializer_cfg = initializer_cfg
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.bidirectional_weight_tie = bidirectional_weight_tie
        self.rcps = rcps
        self.complement_map = complement_map

class CaduceusTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids"]

    def __init__(self,
                 model_max_length: int,
                 characters: Sequence[str] = ("A", "C", "G", "T", "N"),
                 complement_map=None,
                 bos_token="[BOS]",
                 eos_token="[SEP]",
                 sep_token="[SEP]",
                 cls_token="[CLS]",
                 pad_token="[PAD]",
                 mask_token="[MASK]",
                 unk_token="[UNK]",
                 **kwargs):
        """Character tokenizer for Hugging Face transformers.

        Adapted from https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen-hf/blob/main/tokenization_hyena.py
        Args:
            model_max_length (int): Model maximum sequence length.
            characters (Sequence[str]): List of desired characters. Any character which
                is not included in this list will be replaced by a special token called
                [UNK] with id=6. Following is a list of the special tokens with
                their corresponding ids:
                    "[CLS]": 0
                    "[SEP]": 1
                    "[BOS]": 2
                    "[MASK]": 3
                    "[PAD]": 4
                    "[RESERVED]": 5
                    "[UNK]": 6
                an id (starting at 7) will be assigned to each character.
            complement_map (Optional[Dict[str, str]]): Dictionary with string complements for each character.
        """
        if complement_map is None:
            complement_map = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
        self.characters = characters
        self.model_max_length = model_max_length

        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        add_prefix_space = kwargs.pop("add_prefix_space", False)
        padding_side = kwargs.pop("padding_side", "left")

        self._complement_map = {}
        for k, v in self._vocab_str_to_int.items():
            complement_id = self._vocab_str_to_int[complement_map[k]] if k in complement_map.keys() else v
            self._complement_map[self._vocab_str_to_int[k]] = complement_id

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=add_prefix_space,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    @property
    def complement_map(self) -> Dict[int, int]:
        return self._complement_map

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return list(text.upper())  # Convert all base pairs to uppercase

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)  # Note: this operation has lost info about which base pairs were originally lowercase

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        # cls = [self.cls_token_id]
        result = token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int

    # Fixed vocabulary with no vocab file
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple:
        return ()