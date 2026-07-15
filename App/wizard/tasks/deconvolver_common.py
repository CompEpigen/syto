"""Shared deconvolver metadata and pseudobulk-split discovery.

Used by the deconvolution-family task wizards (``fit_deconvolution`` and
``fit_calibration``) so the deconvolver registry and HDF5 split discovery live in
one place instead of being duplicated per task.
"""

# Default splits used when the pseudobulk HDF5 cannot be read.
DEFAULT_SPLITS = ["train", "valid", "test"]

# Canonical presentation order for deconvolvers.
DECONVOLVER_ORDER = ["xgb", "swn", "mlp", "nnls", "psls"]

# name -> primary model file written by the deconvolution pipeline.
DECONVOLVER_FILES = {
    "xgb": "xgb_deconvolver.joblib",
    "swn": "swn_best_deconvolver.pt",
    "mlp": "mlp_best_deconvolver.pt",
    "nnls": "nnls_deconvolver.joblib",
    "psls": "psls_deconvolver.joblib",
}

# Default fit/predict-time params attached per deconvolver (others get none).
DECONVOLVER_DEFAULT_PARAMS = {
    "swn": {"device": "cuda"},
    "mlp": {"device": "cuda"},
    "psls": {"n_workers": 2},
}


def list_splits(pseudobulk_h5_path: str) -> list[str]:
    """Return the split names present in a pseudobulk HDF5 file (``[]`` on error)."""
    try:
        from syto.data.pseudobulk_hdf5_utils import PseudobulkHDF5Reader

        return list(PseudobulkHDF5Reader(pseudobulk_h5_path).list_splits())
    except Exception:  # pragma: no cover - defensive; any read failure -> fallback
        return []
