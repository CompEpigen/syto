"""Fit-calibration task wizard: schema and assembler.

Mirrors the live CalibratorFittingPipeline (syto/app/calibration_pipeline.py), NOT the
stale example YAML. Two list sections discover their choices from disk: ``splits``
from the pseudobulk HDF5 (``outputs/*`` groups) and ``deconvolvers`` from the saved
model files in ``deconvolvers_dir``.
"""

import os

from syto.app.wizard import validators as v
from syto.app.wizard.config_utils import set_nested as _set_nested
from syto.app.wizard.fields import FieldSpec
from syto.app.wizard.tasks import TASK_REGISTRY
from syto.app.wizard.tasks.deconvolver_common import (
    DECONVOLVER_DEFAULT_PARAMS,
    DECONVOLVER_FILES,
    DECONVOLVER_ORDER,
    DEFAULT_SPLITS,
    list_splits as _list_splits,
    template_deconvolver_entries,
)


def _available_deconvolvers(deconvolvers_dir: str) -> list[str]:
    """Return known deconvolver names whose model file exists, in canonical order."""
    present: list[str] = []
    for name in DECONVOLVER_ORDER:
        if os.path.exists(os.path.join(deconvolvers_dir, DECONVOLVER_FILES[name])):
            present.append(name)
    return present


def _splits_section(engine, answers) -> None:
    """Resolve calibration splits from the pseudobulk HDF5, enforcing ``valid``."""
    available = _list_splits(answers.get("pseudobulk_path", ""))
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


class FitCalibrationWizard:
    task_name = "fit_calibration"

    def field_specs(self) -> list[FieldSpec]:
        return [
            FieldSpec(
                key="labels_dict_path",
                label="Labels dict JSON path",
                kind="path",
                default="syto/app/labels_dict.json",
                validate=v.path_exists,
            ),
            FieldSpec(
                key="num_output_labels",
                label="num_output_labels",
                kind="int",
                default=39,
                validate=v.positive_int,
            ),
            FieldSpec(
                key="num_input_labels",
                label="num_input_labels",
                kind="int",
                default=39,
                validate=v.positive_int,
                tier="expert",
            ),
            FieldSpec(
                key="pseudobulk_path",
                label="Pseudobulk store path (.h5 file or columnar dir)",
                kind="path",
                validate=v.pseudobulk_store,
            ),
            FieldSpec(
                key="splits",
                label="Splits",
                kind="list_section",
                handler=_splits_section,
            ),
            FieldSpec(
                key="features_mask_path",
                label="Feature-selection mask (.npz) path",
                kind="path",
                validate=v.path_exists,
            ),
            FieldSpec(
                key="deconvolvers_dir",
                label="Deconvolvers directory",
                kind="path",
                validate=v.path_exists,
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

    def template_overrides(self, classifier: str | None = None) -> dict:
        """Template mode: all splits and every deconvolver (nothing on disk yet)."""
        return {
            "splits": list(DEFAULT_SPLITS),
            "deconvolvers": template_deconvolver_entries(),
        }

    def build_config(self, answers: dict) -> dict:
        """Assemble flat answers into the nested config, dropping None/blank."""
        config: dict = {}
        for key, value in answers.items():
            if value is None:
                continue
            if isinstance(value, str) and value.strip() == "":
                continue
            _set_nested(config, key, value)
        return config


TASK_REGISTRY["fit_calibration"] = FitCalibrationWizard
