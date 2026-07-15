"""Fit-deconvolution task wizard: schema and assembler.

Mirrors the live DeconvolutionFittingPipeline (App/deconvolution_pipeline.py) and
App/config/deconvolution/new_deconvolution_template.yaml (HDF5 / pseudobulk-v2).
The ``splits`` section discovers its choices from the pseudobulk HDF5; the
``deconvolvers`` section offers the five known types to fit (no on-disk discovery,
since the models are being created here). Shared deconvolver metadata and split
discovery live in ``deconvolver_common``.
"""

from App.wizard.tasks.deconvolver_common import (
    DECONVOLVER_DEFAULT_PARAMS,
    DECONVOLVER_ORDER,
)


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
