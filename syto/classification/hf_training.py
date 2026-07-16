"""Architecture-neutral HuggingFace-Trainer helpers shared across read classifiers.

These were originally written for MethylBERT but depend only on the generic HF
Trainer surface plus a boolean ``signal_mask`` over dataset indices, so any
classifier that trains through ``transformers.Trainer`` (MethylBERT, EpigenBERT)
can reuse them. Dismir trains through a hand-written loop and does NOT use these.
"""

import logging
from dataclasses import replace

import numpy as np
from torch.utils.data import DataLoader, Sampler
from transformers import Trainer, EarlyStoppingCallback

_module_logger = logging.getLogger(__name__)


def apply_early_stopping(training_args, early_stopping_cfg, callbacks, logger=None):
    """Attach a ``transformers.EarlyStoppingCallback`` when requested.

    Parameters
    ----------
    training_args :
        A ``TrainingArguments`` dataclass instance. When early stopping is
        enabled but its prerequisites are unset, a copy with them enforced is
        returned (via ``dataclasses.replace``); the original is left untouched.
    early_stopping_cfg : dict | None
        Parsed ``training.early_stopping`` block. Recognised keys:
        ``enabled`` (default True when the block is present),
        ``early_stopping_patience`` (default 1),
        ``early_stopping_threshold`` (default 0.0).
    callbacks : list | None
        Existing callbacks to preserve.

    Returns
    -------
    tuple
        ``(training_args, callbacks)``. When ``early_stopping_cfg`` is falsy or
        carries ``enabled: false`` both inputs are returned unchanged.
    """
    log = logger or _module_logger
    if not early_stopping_cfg:
        return training_args, callbacks

    cfg = dict(early_stopping_cfg)
    if not cfg.pop("enabled", True):
        return training_args, callbacks

    # EarlyStoppingCallback requires load_best_model_at_end=True and a
    # metric_for_best_model; auto-enforce them (with a warning) when unset.
    updates = {}
    if not getattr(training_args, "load_best_model_at_end", False):
        log.warning(
            "Early stopping enabled: forcing load_best_model_at_end=True "
            "(required by EarlyStoppingCallback)."
        )
        updates["load_best_model_at_end"] = True
    if not getattr(training_args, "metric_for_best_model", None):
        log.warning(
            "Early stopping enabled: defaulting metric_for_best_model='eval_loss' "
            "(required by EarlyStoppingCallback)."
        )
        updates["metric_for_best_model"] = "eval_loss"
    if updates:
        training_args = replace(training_args, **updates)

    callback = EarlyStoppingCallback(
        early_stopping_patience=cfg.get("early_stopping_patience", 1),
        early_stopping_threshold=cfg.get("early_stopping_threshold", 0.0),
    )
    return training_args, list(callbacks or []) + [callback]


def evaluate_train_metrics(trainer, logger=None):
    """Run one evaluation pass over the training set to materialise train metrics.

    HuggingFace ``Trainer`` only runs ``compute_metrics`` on the eval set, so
    accuracy/f1/precision/recall/AP/MCC are never computed for the training
    split. This triggers a single ``evaluate`` over ``trainer.train_dataset``
    with the ``train`` prefix (best model already reloaded when
    ``load_best_model_at_end`` is set), appending a ``train_*`` entry to
    ``trainer.state.log_history`` that ``extract_trainer_metrics`` picks up.

    This is a full extra forward pass over the training data — potentially
    expensive on large datasets. It is a no-op when there is no training set.
    """
    log = logger or _module_logger
    if getattr(trainer, "train_dataset", None) is None:
        return
    log.info("Computing train-set metrics (full evaluation pass over training data)...")

    # This manual evaluate() uses the "train" metric prefix, so the periodic
    # EarlyStoppingCallback would not find its "eval_*" metric and log a
    # misleading "early stopping is disabled" warning. Detach it for the pass.
    handler = getattr(trainer, "callback_handler", None)
    early_stopping_cbs = [
        cb
        for cb in getattr(handler, "callbacks", [])
        if isinstance(cb, EarlyStoppingCallback)
    ]
    for cb in early_stopping_cbs:
        trainer.remove_callback(cb)
    try:
        trainer.evaluate(eval_dataset=trainer.train_dataset, metric_key_prefix="train")
    finally:
        for cb in early_stopping_cbs:
            trainer.add_callback(cb)


def log_fit_completion(trainer, logger=None):
    """Emit an explicit "fit finished successfully" line with the best checkpoint.

    Logged after the training/metrics tables and before MLflow artifacts are
    written, so the captured log records a clean finish plus the best checkpoint
    and its metric value.
    """
    log = logger or _module_logger
    state = getattr(trainer, "state", None)
    best_ckpt = getattr(state, "best_model_checkpoint", None)
    best_metric = getattr(state, "best_metric", None)
    metric_name = getattr(getattr(trainer, "args", None), "metric_for_best_model", None)

    msg = "Fit finished successfully."
    if best_ckpt is not None:
        msg += f" Best checkpoint: {best_ckpt}."
    if best_metric is not None:
        msg += f" Best {metric_name or 'metric'}={best_metric:.6f}."
    log.info(msg)


class BalancedBackgroundBatchSampler(Sampler):
    """
    Yields batches where background reads are capped at bg_ratio of the batch.
    """

    def __init__(
        self, signal_mask, batch_size, bg_ratio=0.3, shuffle=True, drop_last=False
    ):
        self.batch_size = batch_size
        self.bg_ratio = bg_ratio
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.signal_indices = np.where(signal_mask)[0]
        self.bg_indices = np.where(~signal_mask)[0]

        self.n_bg_per_batch = int(batch_size * bg_ratio)
        self.n_signal_per_batch = batch_size - self.n_bg_per_batch

    def __iter__(self):
        if self.shuffle:
            signal = np.random.permutation(self.signal_indices)
            bg = np.random.permutation(self.bg_indices)
        else:
            signal = self.signal_indices.copy()
            bg = self.bg_indices.copy()

        bg_cycle = np.resize(bg, max(len(signal), len(bg) + self.batch_size))
        s_ptr, b_ptr = 0, 0

        while s_ptr + self.n_signal_per_batch <= len(signal):
            batch_signal = signal[s_ptr : s_ptr + self.n_signal_per_batch]
            batch_bg = bg_cycle[b_ptr : b_ptr + self.n_bg_per_batch]
            batch = np.concatenate([batch_signal, batch_bg])
            np.random.shuffle(batch)
            yield batch.tolist()
            s_ptr += self.n_signal_per_batch
            b_ptr += self.n_bg_per_batch

        if not self.drop_last and s_ptr < len(signal):
            remaining = signal[s_ptr:]
            n_bg_rem = int(len(remaining) * self.bg_ratio / (1 - self.bg_ratio))
            batch_bg = bg_cycle[b_ptr : b_ptr + n_bg_rem]
            batch = np.concatenate([remaining, batch_bg])
            np.random.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        n = len(self.signal_indices) // self.n_signal_per_batch
        if not self.drop_last and len(self.signal_indices) % self.n_signal_per_batch:
            n += 1
        return n


class AuxLossLoggingTrainer(Trainer):
    """
    Custom Trainer that optionally logs an additional loss_ce metric if provided
    by the model output (via a ``loss_ce`` attribute). Architecture-neutral: if
    the model output has no ``loss_ce``, this is a silent no-op.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_loss_ce_train = 0.0
        self._custom_loss_ce_train_steps = 0
        self._custom_loss_ce_eval = 0.0
        self._custom_loss_ce_eval_steps = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, **kwargs
        )

        if hasattr(outputs, "loss_ce") and outputs.loss_ce is not None:
            if model.training:
                self._custom_loss_ce_train += outputs.loss_ce.item()
                self._custom_loss_ce_train_steps += 1
            else:
                self._custom_loss_ce_eval += outputs.loss_ce.item()
                self._custom_loss_ce_eval_steps += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, *args, **kwargs) -> None:
        if "loss" in logs and getattr(self, "_custom_loss_ce_train_steps", 0) > 0:
            logs["loss_ce"] = (
                self._custom_loss_ce_train / self._custom_loss_ce_train_steps
            )
            self._custom_loss_ce_train = 0.0
            self._custom_loss_ce_train_steps = 0
        super().log(logs, *args, **kwargs)

    def evaluation_loop(self, *args, **kwargs):
        metric_key_prefix = kwargs.get("metric_key_prefix", "eval")
        if len(args) >= 5:
            metric_key_prefix = args[4]

        self._custom_loss_ce_eval = 0.0
        self._custom_loss_ce_eval_steps = 0

        output = super().evaluation_loop(*args, **kwargs)

        if (
            getattr(self, "_custom_loss_ce_eval_steps", 0) > 0
            and output.metrics is not None
        ):
            output.metrics[f"{metric_key_prefix}_loss_ce"] = (
                self._custom_loss_ce_eval / self._custom_loss_ce_eval_steps
            )
            self._custom_loss_ce_eval = 0.0
            self._custom_loss_ce_eval_steps = 0

        return output


class BalancedTrainer(AuxLossLoggingTrainer):
    """
    HF Trainer that uses BalancedBackgroundBatchSampler for training.
    """

    def __init__(self, *args, signal_mask=None, bg_ratio=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.signal_mask = signal_mask
        self.bg_ratio = bg_ratio

    def get_train_dataloader(self) -> DataLoader:
        if self.signal_mask is None:
            # Fall back to default behavior
            return super().get_train_dataloader()

        batch_sampler = BalancedBackgroundBatchSampler(
            signal_mask=self.signal_mask,
            batch_size=self.args.per_device_train_batch_size,
            bg_ratio=self.bg_ratio,
            shuffle=True,
            drop_last=self.args.dataloader_drop_last,
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


# Back-compat alias for the pre-extraction name.
MethylBertTrainer = AuxLossLoggingTrainer
