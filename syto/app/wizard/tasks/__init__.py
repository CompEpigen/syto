"""Task wizard registry."""

TASK_REGISTRY: dict = {}


def _load_tasks() -> None:
    """Import task modules so they self-register in TASK_REGISTRY."""
    from syto.app.wizard.tasks import inference  # noqa: F401
    from syto.app.wizard.tasks import classifier_fit  # noqa: F401
    from syto.app.wizard.tasks import generate_pseudobulk  # noqa: F401
    from syto.app.wizard.tasks import fit_calibration  # noqa: F401
    from syto.app.wizard.tasks import fit_deconvolution  # noqa: F401
    from syto.app.wizard.tasks import deconvolute_pseudobulk  # noqa: F401


_load_tasks()
