"""Fit-deconvolution task wizard: schema and assembler.

Mirrors the live DeconvolutionFittingPipeline (App/deconvolution_pipeline.py) and
App/config/deconvolution/new_deconvolution_template.yaml (HDF5 / pseudobulk-v2).
The ``splits`` section discovers its choices from the pseudobulk HDF5; the
``deconvolvers`` section offers the five known types to fit (no on-disk discovery,
since the models are being created here). Shared deconvolver metadata and split
discovery live in ``deconvolver_common``.
"""

from App.wizard.fields import FieldSpec
from App.wizard.tasks.deconvolver_common import (
    DECONVOLVER_DEFAULT_PARAMS,
    DECONVOLVER_ORDER,
    DEFAULT_SPLITS,
    list_splits as _list_splits,
)

# Splits the fitting pipeline requires when present in the file: 'valid' for the
# feature mask, 'train' for the reference matrix / model fitting.
REQUIRED_SPLITS = ["train", "valid"]


def _splits_section(engine, answers) -> None:
    """Resolve fitting splits from the pseudobulk HDF5, enforcing train+valid."""
    available = _list_splits(answers.get("pseudobulk_h5_path", ""))
    if not available:
        print(
            "  ! Could not read splits from the pseudobulk file; "
            f"falling back to {DEFAULT_SPLITS}."
        )
        answers["splits"] = list(DEFAULT_SPLITS)
        return

    required_present = [s for s in REQUIRED_SPLITS if s in available]
    missing_from_file = [s for s in REQUIRED_SPLITS if s not in available]
    if missing_from_file:
        print(
            f"  ! The pseudobulk file is missing {missing_from_file}, which "
            f"fitting requires. Available: {available}. Fix the inputs before running."
        )

    while True:
        selected = engine.ask_checkbox("Which splits to fit on?", available)
        missing = [s for s in required_present if s not in selected]
        if missing:
            print(
                f"  ✗ Fitting requires {missing}; please include "
                f"{'them' if len(missing) > 1 else 'it'}."
            )
            continue
        break
    answers["splits"] = selected


def _guarantee_columns_section(engine, answers) -> None:
    """Optionally force-select specific prediction-class columns.

    The enable question is wizard-only and never written to the config; when the
    user declines, ``guarantee_columns_selection`` is omitted entirely.
    """
    enabled = engine.ask_one(
        FieldSpec(
            key="_guarantee_columns_enabled",
            label="Force-select specific prediction-class columns?",
            kind="bool",
            default=False,
        ),
        answers,
    )
    if not enabled:
        return

    while True:
        raw = engine.ask_one(
            FieldSpec(
                key="_guarantee_columns_raw",
                label="guarantee_columns_selection (comma-separated column indices)",
                kind="text",
            ),
            answers,
        )
        try:
            cols = [int(tok) for tok in str(raw).split(",") if tok.strip() != ""]
        except ValueError:
            print("  ✗ Enter comma-separated integers, e.g. '38, 5'.")
            continue
        break
    answers["guarantee_columns_selection"] = cols


def _deconvolvers_section(engine, answers) -> None:
    """Pick which deconvolvers to fit and attach default params."""
    selected = engine.ask_checkbox(
        "Which deconvolvers to fit?", list(DECONVOLVER_ORDER)
    )
    entries: list[dict] = []
    for name in selected:
        entry: dict = {"name": name}
        params = DECONVOLVER_DEFAULT_PARAMS.get(name)
        if params is not None:
            entry["params"] = dict(params)
        entries.append(entry)
    answers["deconvolvers"] = entries

    print(
        "  i Per-deconvolver hyperparameters use defaults — "
        "edit the generated YAML to override them."
    )
