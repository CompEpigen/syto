"""Generate-pseudobulk task wizard: schema and assembler.

Mirrors App/config/pseudobulk_v2_test.yaml (the HDF5-based PseudoBulkPipeline).
The classifier checkpoint/config and the UXM atlas are only used when reads must
be prepared and predicted (input_type == "raw_splits"); for "pre_predicted"
input those are omitted. classifier_type and labeling_scheme are always kept as
generation metadata.
"""

from App.wizard import validators as v
from App.wizard.config_utils import set_nested as _set_nested
from App.wizard.fields import FieldSpec
from App.wizard.tasks import TASK_REGISTRY

PB_CLASSIFIERS = ["dismir", "methylbert", "cancer_detector"]
GRG_HEAD_TYPES = {"dismir", "methylbert"}
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
    """Collect per-split data_path + target_proportions_path into a nested dict."""
    selected = engine.ask_checkbox(
        "Which splits to include (data_path + target proportions)?", SPLITS
    )
    info = {}
    for split in selected:
        data_path = engine.ask_one(
            FieldSpec(key=f"_{split}_data", label=f"[{split}] data_path",
                      kind="path", validate=v.path_exists),
            answers,
        )
        target = engine.ask_one(
            FieldSpec(key=f"_{split}_target",
                      label=f"[{split}] target_proportions_path (.npz)",
                      kind="path", validate=v.path_exists),
            answers,
        )
        info[split] = {"data_path": data_path, "target_proportions_path": target}
    answers["split_information"] = info


class GeneratePseudobulkWizard:
    task_name = "generate_pseudobulk"

    def field_specs(self) -> list[FieldSpec]:
        return [
            # ── Input type (asked first) ──
            FieldSpec(key="input_type", label="Input type", kind="select",
                      choices=["raw_splits", "pre_predicted"],
                      help="raw_splits runs read prep + classifier prediction; "
                           "pre_predicted skips straight to pseudobulk generation."),

            # ── Classifier type (always — used as generation metadata) ──
            FieldSpec(key="classifier_type", label="Classifier type",
                      kind="select", choices=PB_CLASSIFIERS),

            # ── Classifier checkpoint + config (raw_splits only) ──
            FieldSpec(key="classifier_checkpoint", label="Classifier checkpoint path",
                      kind="path", validate=v.path_exists, when=_raw),
            FieldSpec(key="classifier_config.seq_length", label="seq_length",
                      kind="int", default=150, validate=v.positive_int, when=_raw),
            FieldSpec(key="classifier_config.soft_labels", label="soft_labels?",
                      kind="bool", default=True, when=_raw),
            FieldSpec(key="classifier_config.num_labels", label="classifier num_labels",
                      kind="int", default=39, validate=v.positive_int, when=_raw),
            FieldSpec(key="classifier_config.batch_size", label="prediction batch_size",
                      kind="int", default=2200, validate=v.positive_int, when=_raw),
            FieldSpec(key="classifier_config.classifier_head_implementation",
                      label="classifier head", kind="select",
                      choices=["grg_attention_based", "simple"],
                      default="grg_attention_based",
                      when=_raw_and("dismir", "methylbert")),
            FieldSpec(key="classifier_config.num_grg_labels", label="num_grg_labels",
                      kind="int", default=39, validate=v.positive_int,
                      when=_raw_and("dismir", "methylbert")),
            FieldSpec(key="classifier_config.grg_label_column",
                      label="classifier grg_label_column", kind="text",
                      default="dmr_ctype_label",
                      when=_raw_and("dismir", "methylbert")),
            FieldSpec(key="classifier_config.foundation_model",
                      label="foundation_model path", kind="path",
                      validate=v.path_exists, when=_raw_and("methylbert")),
            FieldSpec(key="classifier_config.dismir_flavor", label="dismir_flavor",
                      kind="select", choices=["lstm", "mingru"], default="lstm",
                      when=_raw_and("dismir")),

            # ── Atlas (raw_splits only — used for read preparation) ──
            FieldSpec(key="atlas_path", label="UXM atlas TSV path",
                      kind="path", validate=v.path_exists, when=_raw),
            FieldSpec(key="atlas_name", label="Atlas name",
                      kind="text", validate=v.non_empty, when=_raw),

            # ── Labels & reference (always) ──
            FieldSpec(key="labels_dict_path", label="Labels dict JSON path",
                      kind="path", default="App/labels_dict.json",
                      validate=v.path_exists),
            FieldSpec(key="num_labels", label="Number of labels",
                      kind="int", default=39, validate=v.positive_int),
            FieldSpec(key="labeling_scheme", label="Labeling scheme (metadata)",
                      kind="select", choices=["soft_labels", "hard_labels"],
                      default="soft_labels"),

            # ── Per-split inputs ──
            FieldSpec(key="split_information", label="Split information",
                      kind="list_section", handler=_split_information_section),

            # ── Sampling parameters (always) ──
            FieldSpec(key="n_reads_to_sample", label="n_reads_to_sample (per pseudobulk)",
                      kind="int", default=475000, validate=v.positive_int),
            FieldSpec(key="class_label_column", label="class_label_column",
                      kind="text", default="original_label"),
            FieldSpec(key="grg_label_column", label="grg_label_column",
                      kind="text", default="dmr_ctype_label"),
            FieldSpec(key="grg_sampling_method", label="grg_sampling_method",
                      kind="select", choices=["uniform_multinomial"],
                      default="uniform_multinomial"),
            FieldSpec(key="substitution_method", label="substitution_method",
                      kind="select",
                      choices=["uniform_number", "prior_blending", "prior_imputation"],
                      default="uniform_number"),

            # ── Parallelization & checkpointing (always) ──
            FieldSpec(key="n_workers", label="n_workers (Dask)",
                      kind="int", default=5, validate=v.positive_int),
            FieldSpec(key="batch_size", label="batch_size (pseudobulks per batch)",
                      kind="int", default=100, validate=v.positive_int),

            # ── Output ──
            FieldSpec(key="output_dir", label="Output directory",
                      kind="text", default="tmp/", validate=v.non_empty),

            # ── Optional: cap number of pseudobulks ──
            FieldSpec(key="limit_pseudobulks",
                      label="Limit number of pseudobulks to sample?",
                      kind="bool", default=False),
            FieldSpec(key="n_pseudobulks_to_sample", label="n_pseudobulks_to_sample",
                      kind="int", default=2000, validate=v.positive_int,
                      when=lambda a: a.get("limit_pseudobulks")),
        ]

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
