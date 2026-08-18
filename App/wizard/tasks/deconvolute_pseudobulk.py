"""Deconvolute-pseudobulk task wizard: schema and assembler.

Mirrors the live PseudobulkDeconvolutionPipeline
(App/pseudobulk_deconvolution_pipeline.py) and
App/config/pseudodulk_deconvolution.yaml: read-based baselines are run over a
pre-generated pseudobulk HDF5 file, one parquet of predicted proportions per
model and split.

The ``splits`` section discovers its choices from the pseudobulk HDF5 and is
optional — when the key is omitted the pipeline processes every split present in
the file. The baseline sub-flow is shared with the ``inference`` task and lives
in ``baselines_common``.
"""

from App.wizard import validators as v
from App.wizard.config_utils import set_nested as _set_nested
from App.wizard.fields import FieldSpec
from App.wizard.tasks import TASK_REGISTRY
from App.wizard.tasks.baselines_common import (
    ask_baseline_entries,
    template_baseline_entries,
)
from App.wizard.tasks.deconvolver_common import (
    DEFAULT_SPLITS,
    list_splits as _list_splits,
)


def _splits_section(engine, answers) -> None:
    """Resolve splits from the pseudobulk HDF5; omit the key to process all.

    Unlike the fitting tasks no split is required here, so an unreadable file or
    an empty selection simply leaves ``splits`` out of the config and lets the
    pipeline auto-discover them at run time.
    """
    available = _list_splits(answers.get("pseudobulk_path", ""))
    if not available:
        print(
            "  ! Could not read splits from the pseudobulk file; "
            "omitting 'splits' so the pipeline processes all splits it finds."
        )
        return

    selected = engine.ask_checkbox(
        "Which splits to deconvolute? (select none = all splits in the file)",
        available,
    )
    if not selected:
        print("  i No split selected — the pipeline will process all of them.")
        return
    answers["splits"] = selected


def _baselines_section(engine, answers) -> None:
    """Pick which read-based baselines to run and configure each one."""
    while True:
        entries = ask_baseline_entries(
            engine, answers, "Which baseline deconvolvers to run?"
        )
        if entries:
            break
        print("  ✗ This task runs baselines only; select at least one.")
    answers["baselines"] = entries


class DeconvolutePseudobulkWizard:
    task_name = "deconvolute_pseudobulk"

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
                key="baselines",
                label="Baseline deconvolvers",
                kind="list_section",
                handler=_baselines_section,
            ),
            FieldSpec(
                key="output_dir",
                label="Output directory",
                kind="text",
                validate=v.non_empty,
            ),
            FieldSpec(
                key="n_workers",
                label="n_workers (1 = run in-process, no fork pool)",
                kind="int",
                default=1,
                validate=v.positive_int,
            ),
            FieldSpec(
                key="batch_size",
                label="batch_size (pseudobulks per worker chunk)",
                kind="int",
                default=1,
                validate=v.positive_int,
                tier="expert",
            ),
            FieldSpec(
                key="class_label_column",
                label="class_label_column (read column used as the class label)",
                kind="text",
                default="original_label",
                validate=v.non_empty,
                tier="expert",
            ),
        ]

    def template_overrides(self, classifier: str | None = None) -> dict:
        """Template mode: all splits and every baseline (nothing to discover yet)."""
        return {
            "splits": list(DEFAULT_SPLITS),
            "baselines": template_baseline_entries(),
        }

    def build_config(self, answers: dict) -> dict:
        """Assemble flat answers into the nested config, dropping None/blank.

        Section handlers already omit their keys entirely when not applicable
        (e.g. ``splits`` when all splits should be processed).
        """
        config: dict = {}
        for key, value in answers.items():
            if value is None:
                continue
            if isinstance(value, str) and value.strip() == "":
                continue
            _set_nested(config, key, value)
        return config


TASK_REGISTRY["deconvolute_pseudobulk"] = DeconvolutePseudobulkWizard
