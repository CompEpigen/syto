"""Fit-calibration task wizard: schema and assembler.

Mirrors the live CalibratorFittingPipeline (App/calibration_pipeline.py), NOT the
stale example YAML. Two list sections discover their choices from disk: ``splits``
from the pseudobulk HDF5 (``outputs/*`` groups) and ``deconvolvers`` from the saved
model files in ``deconvolvers_dir``.
"""

import os

from App.wizard import validators as v
from App.wizard.config_utils import set_nested as _set_nested
from App.wizard.fields import FieldSpec
from App.wizard.tasks import TASK_REGISTRY

DEFAULT_SPLITS = ["train", "valid", "test"]

# Canonical presentation order for discovered deconvolvers.
DECONVOLVER_ORDER = ["xgb", "swn", "mlp", "nnls", "psls"]

# name -> primary model file written by the deconvolution pipeline.
DECONVOLVER_FILES = {
    "xgb": "xgb_deconvolver.joblib",
    "swn": "swn_best_deconvolver.pt",
    "mlp": "mlp_best_deconvolver.pt",
    "nnls": "nnls_deconvolver.joblib",
    "psls": "psls_deconvolver.joblib",
}

# Default predict-time params attached per deconvolver (others get no params).
DECONVOLVER_DEFAULT_PARAMS = {
    "swn": {"device": "cuda"},
    "mlp": {"device": "cuda"},
    "psls": {"n_workers": 2},
}


def _list_splits(pseudobulk_h5_path: str) -> list[str]:
    """Return the split names present in a pseudobulk HDF5 file (``[]`` on error)."""
    try:
        from syto.data.pseudobulk_hdf5_utils import PseudobulkHDF5Reader

        return list(PseudobulkHDF5Reader(pseudobulk_h5_path).list_splits())
    except Exception:  # pragma: no cover - defensive; any read failure -> fallback
        return []


def _available_deconvolvers(deconvolvers_dir: str) -> list[str]:
    """Return known deconvolver names whose model file exists, in canonical order."""
    present: list[str] = []
    for name in DECONVOLVER_ORDER:
        if os.path.exists(os.path.join(deconvolvers_dir, DECONVOLVER_FILES[name])):
            present.append(name)
    return present


def _splits_section(engine, answers) -> None:
    """Resolve calibration splits from the pseudobulk HDF5, enforcing ``valid``."""
    available = _list_splits(answers.get("pseudobulk_h5_path", ""))
    if not available:
        print(
            "  ! Could not read splits from the pseudobulk file; "
            f"falling back to {DEFAULT_SPLITS}."
        )
        answers["splits"] = list(DEFAULT_SPLITS)
        return

    if "valid" not in available:
        print(
            "  ! The pseudobulk file has no 'valid' split, which calibration "
            f"requires. Available: {available}. Fix the inputs before running."
        )

    while True:
        selected = engine.ask_checkbox("Which splits to calibrate on?", available)
        if "valid" in available and "valid" not in selected:
            print(
                "  ✗ The 'valid' split is required for calibration; "
                "please include it."
            )
            continue
        break
    answers["splits"] = selected


def _deconvolvers_section(engine, answers) -> None:
    """Discover deconvolvers on disk, pick a subset, attach default params."""
    available = _available_deconvolvers(answers.get("deconvolvers_dir", ""))
    if not available:
        available = list(DECONVOLVER_ORDER)
        print(
            "  ! No known deconvolver files found in the directory; "
            f"offering all known types: {available}."
        )

    selected = engine.ask_checkbox("Which deconvolvers to calibrate?", available)
    entries: list[dict] = []
    for name in selected:
        entry: dict = {"name": name}
        params = DECONVOLVER_DEFAULT_PARAMS.get(name)
        if params is not None:
            entry["params"] = dict(params)
        entries.append(entry)
    answers["deconvolvers"] = entries

    print(
        "  i Per-deconvolver 'params' (device/n_workers) and the 'vector_scaling' "
        "CV grid use defaults — edit the generated YAML to override them."
    )
