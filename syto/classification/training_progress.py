"""
Shared CLI/notebook training-progress reporting for read classifiers.

Provides ``TrainingProgressTracker``, a small tabular progress renderer
(modeled after HuggingFace's ``NotebookTrainingTracker``, but also usable in
a plain terminal via ``tqdm.write``), and ``TableProgressCallback``, a
``TrainerCallback`` that uses it to render evaluation logs from a
HuggingFace ``Trainer`` as a metrics table - replacing the default
``ProgressCallback``/``PrinterCallback``, which dump raw ``logs`` dicts.
"""

import logging

from typing import Dict, Optional

from tqdm import tqdm
from transformers.trainer_callback import (
    PrinterCallback,
    ProgressCallback,
)

try:
    from transformers.utils.import_utils import is_in_notebook
    from transformers.utils.notebook import NotebookTrainingTracker
except ImportError:

    def is_in_notebook():
        return False

    NotebookTrainingTracker = None


def _mirror_to_log_file(message: str) -> None:
    """Mirror a pre-rendered table line into any file-based logging handler.

    The metrics table is rendered to the console via ``tqdm.write`` (stdout) so it
    interleaves cleanly with active progress bars, which bypasses the ``logging``
    framework - so it never reaches a ``FileHandler`` attached via ``--log-file``.
    This writes the same raw line to every ``FileHandler`` on the root logger,
    preserving the table's column alignment, without double-printing to the
    console ``StreamHandler`` (a ``FileHandler`` is a ``StreamHandler`` subclass,
    so the ``isinstance`` check matches file handlers only). A no-op when no file
    handler is configured.
    """
    for handler in logging.getLogger().handlers:
        if not isinstance(handler, logging.FileHandler):
            continue
        handler.acquire()
        try:
            if handler.stream is None:  # delay=True handler not yet opened
                handler.stream = handler._open()
            handler.stream.write(message + handler.terminator)
            handler.flush()
        finally:
            handler.release()


class TrainingProgressTracker:
    """Render one table row per evaluation: train/val loss plus ``val_*`` metrics.

    In a notebook, delegates to HuggingFace's ``NotebookTrainingTracker`` (a
    live-updating HTML table). In a terminal, prints a column-aligned table
    via ``tqdm.write`` so it interleaves cleanly with any active progress
    bars.
    """

    def __init__(self, use_notebook: Optional[bool] = None):
        self.use_notebook = is_in_notebook() if use_notebook is None else use_notebook
        self._tracker = None

    def update(
        self, epoch, epochs, train_loss, val_loss, val_metrics: Dict[str, float]
    ):
        """Print/append a row for one evaluation point.

        Args:
            epoch: Current epoch (or fractional epoch for step-based eval).
            epochs: Total number of epochs (for the "x/y" progress label).
            train_loss: Most recently logged training loss, or ``None``.
            val_loss: Validation loss for this evaluation point.
            val_metrics: Dict of ``val_*``-prefixed scalar metrics.
        """
        columns = ["Epoch", "Training Loss", "Validation Loss"]
        metric_keys = sorted(k for k in val_metrics if k.startswith("val_"))
        metric_display_names = [k[4:].replace("_", " ").title() for k in metric_keys]
        column_names = columns + metric_display_names

        # 1. Initialize tracker / print header on first update
        if self._tracker is None:
            if self.use_notebook and NotebookTrainingTracker is not None:
                self._tracker = NotebookTrainingTracker(
                    num_steps=epochs, column_names=column_names
                )
            else:
                # Terminal/tqdm mode: compute column widths dynamically
                column_widths = [max(len(col) + 2, 10) for col in column_names]
                for idx, col in enumerate(column_names):
                    if col == "Training Loss":
                        column_widths[idx] = max(column_widths[idx], 15)
                    elif col == "Validation Loss":
                        column_widths[idx] = max(column_widths[idx], 17)

                self._tracker = {
                    "column_names": column_names,
                    "column_widths": column_widths,
                }

                header_str = "  ".join(
                    col.ljust(w) for col, w in zip(column_names, column_widths)
                )
                separator_str = "  ".join("-" * w for w in column_widths)
                tqdm.write(header_str)
                tqdm.write(separator_str)
                _mirror_to_log_file(header_str)
                _mirror_to_log_file(separator_str)

        # 2. Format row values
        row_dict = {
            "Epoch": epoch,
            "Training Loss": (f"{train_loss:.6f}" if train_loss is not None else "-"),
            "Validation Loss": f"{val_loss:.6f}" if val_loss is not None else "-",
        }

        metric_key_mapping = {
            col: f"val_{col.lower().replace(' ', '_')}" for col in column_names[3:]
        }
        for display_name in column_names[3:]:
            key = metric_key_mapping[display_name]
            val = val_metrics.get(key, None)
            row_dict[display_name] = f"{val:.6f}" if val is not None else "-"

        # 3. Render the row
        if self.use_notebook and not isinstance(self._tracker, dict):
            self._tracker.write_line(row_dict)
            self._tracker.update(epoch, comment=f"Epoch {epoch}/{epochs}")
            _mirror_to_log_file("  ".join(str(row_dict[col]) for col in column_names))
        else:
            column_widths = self._tracker["column_widths"]
            row_values = [
                f"{epoch}/{epochs}".ljust(column_widths[0]),
                row_dict["Training Loss"].ljust(column_widths[1]),
                row_dict["Validation Loss"].ljust(column_widths[2]),
            ]
            for col_name, w in zip(column_names[3:], column_widths[3:]):
                row_values.append(row_dict[col_name].ljust(w))
            row_str = "  ".join(row_values)
            tqdm.write(row_str)
            _mirror_to_log_file(row_str)


# eval_* keys that report run timing rather than a model metric, plus
# eval_loss itself (already shown as the "Validation Loss" column).
_NON_METRIC_EVAL_KEYS = {
    "loss",
    "runtime",
    "samples_per_second",
    "steps_per_second",
    "steps",
    "jit_compilation_time",
}


class TableProgressCallback(ProgressCallback):
    """Like ``ProgressCallback``, but renders evaluation logs as a metrics table.

    Keeps the same step progress bar as ``ProgressCallback``, but instead of
    writing each raw ``logs`` dict to the console, accumulates the most
    recent training loss and, on evaluation logs (containing ``eval_loss``),
    renders a row via ``TrainingProgressTracker``.
    """

    def __init__(self, max_str_len: int = 100):
        super().__init__(max_str_len=max_str_len)
        self._progress_tracker = TrainingProgressTracker()
        self._last_train_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.is_world_process_zero:
            return

        if "loss" in logs:
            self._last_train_loss = logs["loss"]

        if "eval_loss" in logs:
            val_metrics = {
                f"val_{key[len('eval_'):]}": value
                for key, value in logs.items()
                if key.startswith("eval_")
                and isinstance(value, (int, float))
                and key[len("eval_") :] not in _NON_METRIC_EVAL_KEYS
            }
            self._progress_tracker.update(
                epoch=round(logs.get("epoch", state.epoch), 4),
                epochs=args.num_train_epochs,
                train_loss=self._last_train_loss,
                val_loss=logs["eval_loss"],
                val_metrics=val_metrics,
            )


def use_table_progress_callback(trainer) -> None:
    """Swap a ``Trainer``'s default progress/printer callback for a table-based one."""
    trainer.remove_callback(ProgressCallback)
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(TableProgressCallback())
