"""Generate-pseudobulk task wizard: schema and assembler.

Mirrors the live PseudoBulkPipeline (syto/app/pseudobulk_pipeline.py) and the working
configs under config/pseudobulk. Two things follow from the input type:

``raw_splits``
    All splits are read from one shared columnar/legacy dataset directory
    (top-level ``data_path`` + ``data_format`` / ``split_column``), then prepared
    against the UXM atlas and run through the classifier. ``split_information``
    only carries each split's target proportions.
``pre_predicted``
    Each split is loaded from its own already-predicted file, so
    ``split_information`` carries a per-split ``data_path`` and the atlas,
    checkpoint and prediction params are all omitted.

``classifier_type`` and ``labeling_scheme`` are kept either way as generation
metadata, and ``classifier_config.num_prediction_classes`` is always written:
PseudoBulkPipeline.__init__ reads it for both input types.
"""

from syto.app.wizard import validators as v
from syto.app.wizard.config_utils import placeholder, set_nested as _set_nested
from syto.app.wizard.fields import FieldSpec
from syto.app.wizard.tasks import TASK_REGISTRY

# Every classifier the pipeline can build; mirrors the _CLF_REGISTRY keys in
# syto.classification.classifiers.lazy_classifier_factory, which is what
# PseudoBulkPipeline._get_classifier dispatches through.
PB_CLASSIFIERS = ["dismir", "methylbert", "cancer_detector", "lookup", "epigenbert2"]
# Neural read classifiers: the only ones whose prediction pass is parameterised
# by sequence length / soft labels / batch size.
NEURAL_TYPES = {"dismir", "methylbert", "epigenbert2"}
# Classifiers loaded on top of a HuggingFace foundation model.
FOUNDATION_TYPES = {"methylbert", "epigenbert2"}
# Classifiers with a GRG-aware classifier head.
GRG_HEAD_TYPES = {"dismir", "methylbert", "epigenbert2"}
INPUT_TYPES = ["raw_splits", "pre_predicted"]
SPLITS = ["train", "valid", "test"]
GATE_KEYS = {"limit_pseudobulks"}


def _ctype(answers):
    return answers.get("classifier_type")


def _raw(answers):
    return answers.get("input_type") == "raw_splits"


def _raw_and(*types):
    types = set(types)
    return lambda a: _raw(a) and _ctype(a) in types


def _split_information_section(engine, answers):
    """Collect per-split inputs, whose shape depends on the input type.

    ``raw_splits`` reads every split out of the shared top-level ``data_path``
    (see PseudoBulkPipeline._load_raw_splits), so only the target proportions are
    per-split; ``pre_predicted`` loads one predicted file per split.
    """
    pre_predicted = not _raw(answers)
    prompt = (
        "Which splits to include (data_path + target proportions)?"
        if pre_predicted
        else "Which splits to include (target proportions)?"
    )
    selected = engine.ask_checkbox(prompt, SPLITS)
    info = {}
    for split in selected:
        entry = {}
        if pre_predicted:
            entry["data_path"] = engine.ask_one(
                FieldSpec(
                    key=f"_{split}_data",
                    label=f"[{split}] data_path (predicted .parquet/.pkl)",
                    kind="path",
                    validate=v.path_exists,
                ),
                answers,
            )
        entry["target_proportions_path"] = engine.ask_one(
            FieldSpec(
                key=f"_{split}_target",
                label=f"[{split}] target_proportions_path (.npz)",
                kind="path",
                validate=v.path_exists,
            ),
            answers,
        )
        info[split] = entry
    answers["split_information"] = info


class GeneratePseudobulkWizard:
    task_name = "generate_pseudobulk"

    def field_specs(self) -> list[FieldSpec]:
        return [
            # ── Input type (asked first) ──
            FieldSpec(
                key="input_type",
                label="Input type",
                kind="select",
                choices=INPUT_TYPES,
                help="raw_splits runs read prep + classifier prediction; "
                "pre_predicted skips straight to pseudobulk generation.",
            ),
            # ── Shared raw dataset (raw_splits only — one dir for all splits) ──
            FieldSpec(
                key="data_path",
                label="Shared dataset directory (columnar or legacy)",
                kind="path",
                validate=v.path_exists,
                when=_raw,
            ),
            FieldSpec(
                key="data_format",
                label="Data format",
                kind="select",
                choices=["auto", "legacy", "columnar"],
                default="auto",
                when=_raw,
            ),
            FieldSpec(
                key="split_column",
                label="Split column name (columnar datasets)",
                kind="text",
                default="split",
                tier="expert",
                when=_raw,
            ),
            FieldSpec(
                key="min_pattern_length",
                label="Min marked CpGs per read (1 = no filtering)",
                kind="int",
                default=1,
                validate=v.positive_int,
                tier="expert",
                when=_raw,
            ),
            # ── Classifier type (always — used as generation metadata) ──
            FieldSpec(
                key="classifier_type",
                label="Classifier type",
                kind="select",
                choices=PB_CLASSIFIERS,
            ),
            # ── Classifier checkpoint + config (raw_splits only) ──
            FieldSpec(
                key="classifier_checkpoint",
                label="Classifier checkpoint path",
                kind="path",
                validate=v.path_exists,
                when=_raw,
            ),
            # num_prediction_classes is how many classes the classifier emits —
            # possibly more than there are cell types (e.g. a background class),
            # which is why it is distinct from the top-level num_labels. The
            # pipeline reads it for *both* input types, so it is always asked.
            FieldSpec(
                key="classifier_config.num_prediction_classes",
                label="classifier num_prediction_classes (classifier output classes)",
                kind="int",
                default=40,
                validate=v.positive_int,
            ),
            FieldSpec(
                key="classifier_config.seq_length",
                label="seq_length",
                kind="int",
                default=150,
                validate=v.positive_int,
                when=_raw_and(*NEURAL_TYPES),
            ),
            FieldSpec(
                key="classifier_config.soft_labels",
                label="soft_labels?",
                kind="bool",
                default=True,
                when=_raw_and(*NEURAL_TYPES),
            ),
            FieldSpec(
                key="classifier_config.batch_size",
                label="prediction batch_size",
                kind="int",
                default=2200,
                validate=v.positive_int,
                when=_raw_and(*NEURAL_TYPES, "cancer_detector"),
            ),
            FieldSpec(
                key="classifier_config.classifier_head_implementation",
                label="classifier head",
                kind="select",
                choices=["grg_attention_based", "simple"],
                default="grg_attention_based",
                when=_raw_and(*GRG_HEAD_TYPES),
            ),
            FieldSpec(
                key="classifier_config.num_grg_labels",
                label="num_grg_labels",
                kind="int",
                default=39,
                validate=v.positive_int,
                when=_raw_and(*GRG_HEAD_TYPES),
            ),
            FieldSpec(
                key="classifier_config.grg_label_column",
                label="classifier grg_label_column",
                kind="text",
                default="dmr_ctype_label",
                when=_raw_and(*GRG_HEAD_TYPES),
            ),
            FieldSpec(
                key="classifier_config.foundation_model",
                label="foundation_model path",
                kind="path",
                validate=v.path_exists,
                when=_raw_and(*FOUNDATION_TYPES),
            ),
            FieldSpec(
                key="classifier_config.col_label",
                label="[lookup] col_label (label column the lookup was fit on)",
                kind="text",
                default="hard_label_with_background",
                when=_raw_and("lookup"),
                help="Recorded for provenance — prediction uses the column stored "
                "in the checkpoint.",
            ),
            FieldSpec(
                key="classifier_config.dismir_flavor",
                label="dismir_flavor",
                kind="select",
                choices=["lstm", "mingru"],
                default="lstm",
                when=_raw_and("dismir"),
            ),
            # ── Atlas (raw_splits only — used for read preparation) ──
            FieldSpec(
                key="atlas_path",
                label="UXM atlas TSV path",
                kind="path",
                validate=v.path_exists,
                when=_raw,
            ),
            FieldSpec(
                key="atlas_name",
                label="Atlas name",
                kind="text",
                validate=v.non_empty,
                when=_raw,
            ),
            FieldSpec(
                key="trim_reads_to_grg_regions",
                label="Trim reads to their GRG regions during preparation?",
                kind="bool",
                default=False,
                tier="expert",
                when=_raw,
            ),
            FieldSpec(
                key="generate_predictions_only",
                label="Stop after writing predicted splits (skip generation)?",
                kind="bool",
                default=False,
                tier="expert",
                when=_raw,
            ),
            # ── Labels & reference (always) ──
            FieldSpec(
                key="labels_dict_path",
                label="Labels dict JSON path",
                kind="path",
                default="syto/app/labels_dict.json",
                validate=v.path_exists,
            ),
            FieldSpec(
                key="num_labels",
                label="Number of labels",
                kind="int",
                default=39,
                validate=v.positive_int,
            ),
            FieldSpec(
                key="labeling_scheme",
                label="Labeling scheme (metadata)",
                kind="select",
                choices=["soft_labels", "hard_labels"],
                default="soft_labels",
            ),
            # ── Per-split inputs ──
            FieldSpec(
                key="split_information",
                label="Split information",
                kind="list_section",
                handler=_split_information_section,
            ),
            # ── Sampling parameters (always) ──
            FieldSpec(
                key="n_reads_to_sample",
                label="n_reads_to_sample (per pseudobulk)",
                kind="int",
                default=475000,
                validate=v.positive_int,
            ),
            FieldSpec(
                key="class_label_column",
                label="class_label_column",
                kind="text",
                default="original_label",
            ),
            FieldSpec(
                key="grg_label_column",
                label="grg_label_column",
                kind="text",
                default="dmr_ctype_label",
            ),
            FieldSpec(
                key="grg_sampling_method",
                label="grg_sampling_method",
                kind="select",
                choices=["uniform_multinomial"],
                default="uniform_multinomial",
            ),
            FieldSpec(
                key="substitution_method",
                label="substitution_method",
                kind="select",
                choices=["uniform_number", "prior_blending", "prior_imputation"],
                default="uniform_number",
            ),
            # ── Parallelization & checkpointing (always) ──
            FieldSpec(
                key="n_workers",
                label="n_workers (Dask)",
                kind="int",
                default=5,
                validate=v.positive_int,
            ),
            FieldSpec(
                key="batch_size",
                label="batch_size (pseudobulks per batch)",
                kind="int",
                default=100,
                validate=v.positive_int,
            ),
            # ── Output ──
            FieldSpec(
                key="output_dir",
                label="Output directory",
                kind="text",
                default="tmp/",
                validate=v.non_empty,
            ),
            # ── Optional: cap number of pseudobulks ──
            FieldSpec(
                key="limit_pseudobulks",
                label="Limit number of pseudobulks to sample?",
                kind="bool",
                default=False,
            ),
            FieldSpec(
                key="n_pseudobulks_to_sample",
                label="n_pseudobulks_to_sample",
                kind="int",
                default=2000,
                validate=v.positive_int,
                when=lambda a: a.get("limit_pseudobulks"),
            ),
        ]

    def template_overrides(self, classifier: str | None = None) -> dict:
        """Template mode: every split stubbed out, classifier pinned when known.

        A template materialises the first ``input_type`` choice, so the per-split
        entries must carry exactly what that branch reads — a ``data_path`` only
        when the splits are pre-predicted.
        """
        per_split_keys = ["target_proportions_path"]
        if INPUT_TYPES[0] == "pre_predicted":
            per_split_keys.insert(0, "data_path")

        overrides: dict = {
            "split_information": {
                split: {key: placeholder(f"{split}_{key}") for key in per_split_keys}
                for split in SPLITS
            }
        }
        if classifier in PB_CLASSIFIERS:
            overrides["classifier_type"] = classifier
        elif classifier is not None:
            print(
                f"  ! generate_pseudobulk does not support '{classifier}'; "
                f"its template keeps classifier_type from {PB_CLASSIFIERS}."
            )
        return overrides

    def build_config(self, answers: dict) -> dict:
        """Assemble flat answers into the nested config.

        The engine already omits fields not applicable to the chosen input_type
        / classifier_type. Here we drop None/blank values and the wizard-only
        gate key so nothing unused leaks into the generated config.
        """
        config: dict = {}
        for key, value in answers.items():
            if key in GATE_KEYS or value is None:
                continue
            if isinstance(value, str) and value.strip() == "":
                continue
            _set_nested(config, key, value)
        return config


TASK_REGISTRY["generate_pseudobulk"] = GeneratePseudobulkWizard
