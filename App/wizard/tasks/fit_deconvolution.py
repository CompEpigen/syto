"""Fit-deconvolution task wizard: schema and assembler.

Mirrors the live DeconvolutionFittingPipeline (App/deconvolution_pipeline.py) and
App/config/deconvolution/new_deconvolution_template.yaml (HDF5 / pseudobulk-v2).
The ``splits`` section discovers its choices from the pseudobulk HDF5; the
``deconvolvers`` section offers the five known types to fit (no on-disk discovery,
since the models are being created here). Shared deconvolver metadata and split
discovery live in ``deconvolver_common``.
"""

from App.wizard import validators as v
from App.wizard.config_utils import set_nested as _set_nested
from App.wizard.fields import FieldSpec
from App.wizard.tasks import TASK_REGISTRY
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


GATE_KEYS = {"feature_selection_mode"}


class FitDeconvolutionWizard:
    task_name = "fit_deconvolution"

    def field_specs(self) -> list[FieldSpec]:
        return [
            FieldSpec(
                key="labels_dict_path",
                label="Labels dict JSON path",
                kind="path",
                default="App/labels_dict.json",
                validate=v.path_exists,
            ),
            FieldSpec(
                key="num_input_labels",
                label="num_input_labels",
                kind="int",
                default=39,
                validate=v.positive_int,
            ),
            FieldSpec(
                key="num_output_labels",
                label="num_output_labels",
                kind="int",
                default=39,
                validate=v.positive_int,
            ),
            FieldSpec(
                key="pseudobulk_h5_path",
                label="Pseudobulk HDF5 path",
                kind="path",
                validate=v.path_exists,
            ),
            FieldSpec(
                key="splits",
                label="Splits",
                kind="list_section",
                handler=_splits_section,
            ),
            FieldSpec(
                key="feature_selection_mode",
                label="Feature selection mode",
                kind="select",
                choices=["cutoff", "top_features"],
                default="cutoff",
            ),
            FieldSpec(
                key="feature_cutoff",
                label="feature_cutoff (max/mean ratio threshold)",
                kind="float",
                default=1.1,
                validate=v.positive_float,
                when=lambda a: a.get("feature_selection_mode") == "cutoff",
            ),
            FieldSpec(
                key="top_features",
                label="top_features (number of features to select)",
                kind="int",
                default=156,
                validate=v.positive_int,
                when=lambda a: a.get("feature_selection_mode") == "top_features",
            ),
            FieldSpec(
                key="guarantee_diagonal_selection",
                label="Always keep the diagonal (cell-type vs. its own class)?",
                kind="bool",
                default=True,
            ),
            FieldSpec(
                key="guarantee_columns_selection",
                label="Guarantee columns selection",
                kind="list_section",
                handler=_guarantee_columns_section,
            ),
            FieldSpec(
                key="generate_feature_selection_plot",
                label="Render the feature-selection heatmap?",
                kind="bool",
                default=False,
                tier="expert",
            ),
            FieldSpec(
                key="deconvolvers",
                label="Deconvolvers",
                kind="list_section",
                handler=_deconvolvers_section,
            ),
            FieldSpec(
                key="output_dir",
                label="Output directory",
                kind="text",
                validate=v.non_empty,
            ),
        ]

    def build_config(self, answers: dict) -> dict:
        """Assemble flat answers into the nested config.

        Drops the wizard-only gate key and any None/blank values so nothing
        unused leaks into the generated config. Section handlers already omit
        their keys entirely when not applicable (e.g. guarantee_columns_selection
        when the user declined).
        """
        config: dict = {}
        for key, value in answers.items():
            if key in GATE_KEYS or value is None:
                continue
            if isinstance(value, str) and value.strip() == "":
                continue
            _set_nested(config, key, value)
        return config


TASK_REGISTRY["fit_deconvolution"] = FitDeconvolutionWizard
