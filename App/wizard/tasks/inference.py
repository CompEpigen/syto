"""Inference task wizard: field schema, deconvolution sub-flows, and assembler."""

from App.wizard import validators as v
from App.wizard.config_utils import set_nested as _set_nested
from App.wizard.fields import FieldSpec
from App.wizard.tasks import TASK_REGISTRY
from App.wizard.tasks.baselines_common import (
    BASELINES,  # noqa: F401 - re-exported for callers/tests of this task module
    ask_baseline_entries,
)

CLASSIFIER_TYPES = ["dismir", "methylbert", "cancer_detector", "lookup", "epigenbert2"]
TRANSFORMER_ARCHS = {"methylbert", "epigenbert2"}
# Architectures that use a read-level classifier head and a soft/hard labeling
# scheme. cancer_detector and lookup use neither.
HEAD_LABEL_ARCHS = {"dismir", "methylbert", "epigenbert2"}
SYTO_DECONVOLVERS = ["xgboost", "mlp", "swn", "nnls", "psls"]
# Friendly checkbox label -> the method-dict fields the pipeline dispatches on
# (see App/inference.py: ``name``/``flavor`` resolution). ls-based methods carry
# their variant in ``flavor``; the others are matched by ``name`` directly.
SYTO_METHOD_DISPATCH = {
    "xgboost": {"name": "xgboost"},
    "mlp": {"name": "3Layer_MLP"},
    "swn": {"name": "Shallow_Wide_Network"},
    "nnls": {"name": "ls", "flavor": "nnls"},
    "psls": {"name": "ls", "flavor": "psls"},
}
LABELING_SCHEMES = ["Soft Labels", "Hard Labels"]
DEFAULT_LABELS_DICT = "App/labels_dict.json"


def _ctype(answers):
    return answers.get("classifier.classifier_type")


def _run_syto(answers):
    return bool(answers.get("run_syto"))


def _input_is_bam(answers):
    return answers.get("input.type") == "bam"


def _deconv_methods_section(engine, answers):
    """Build deconvolution.syto.methods as a list of dicts."""
    selected = engine.ask_checkbox(
        "Select syto feature-based deconvolvers to enable:", SYTO_DECONVOLVERS
    )
    methods = []
    for label in selected:
        ckpt = engine.ask_one(
            FieldSpec(
                key=f"_{label}_ckpt",
                label=f"[{label}] checkpoint_path",
                kind="path",
                validate=v.path_exists,
            ),
            answers,
        )
        cal_dir = engine.ask_one(
            FieldSpec(
                key=f"_{label}_cal",
                label=f"[{label}] calibrators_dir",
                kind="path",
                validate=v.path_exists,
            ),
            answers,
        )
        use_cal = engine.ask_one(
            FieldSpec(
                key=f"_{label}_use",
                label=f"[{label}] use calibration?",
                kind="bool",
                default=True,
            ),
            answers,
        )
        entry = dict(SYTO_METHOD_DISPATCH[label])  # name (+ flavor for ls)
        entry.update(
            {
                "enabled": True,
                "use_callibration": use_cal,
                "checkpoint_path": ckpt,
                "calibrators_dir": cal_dir,
            }
        )
        methods.append(entry)
    answers["deconvolution.syto.methods"] = methods


def _deconv_baselines_section(engine, answers):
    """Build deconvolution.baselines as a list of dicts."""
    answers["deconvolution.baselines"] = ask_baseline_entries(
        engine, answers, "Select read-based baseline deconvolvers to enable:"
    )


class InferenceWizard:
    task_name = "inference"

    def field_specs(self) -> list[FieldSpec]:
        return [
            # ── Syto classification gate (asked first) ──
            FieldSpec(
                key="run_syto",
                label="Run syto classification-based deconvolution?",
                kind="bool",
                default=True,
                help="If no, the classifier and syto blocks are skipped; "
                "only baseline deconvolvers run.",
            ),
            # ── Classifier (only when syto is enabled) ──
            FieldSpec(
                key="classifier.classifier_type",
                label="Classifier type",
                kind="select",
                choices=CLASSIFIER_TYPES,
                when=_run_syto,
            ),
            FieldSpec(
                key="classifier.dismir_flavor",
                label="Dismir flavor",
                kind="select",
                choices=["lstm", "mingru"],
                default="lstm",
                when=lambda a: _run_syto(a) and _ctype(a) == "dismir",
            ),
            FieldSpec(
                key="classifier.foundation_model",
                label="Foundation model path",
                kind="path",
                validate=v.path_exists,
                when=lambda a: _run_syto(a) and _ctype(a) in TRANSFORMER_ARCHS,
            ),
            FieldSpec(
                key="classifier.classifier_head_implementation",
                label="Classifier head",
                kind="select",
                choices=["grg_attention_based", "simple"],
                default="grg_attention_based",
                when=lambda a: _run_syto(a) and _ctype(a) in HEAD_LABEL_ARCHS,
            ),
            FieldSpec(
                key="classifier.labeling_scheme",
                label="Classifier labeling scheme",
                kind="select",
                choices=LABELING_SCHEMES,
                default="Soft Labels",
                when=lambda a: _run_syto(a) and _ctype(a) in HEAD_LABEL_ARCHS,
            ),
            FieldSpec(
                key="checkpoint_path",
                label="Checkpoint path",
                kind="path",
                validate=v.path_exists,
                when=_run_syto,
            ),
            FieldSpec(
                key="features_mask_path",
                label="Features mask (.npz) path",
                kind="path",
                validate=v.path_exists,
                when=_run_syto,
            ),
            # ── Labels / shared (always) ──
            FieldSpec(
                key="labels_dict_path",
                label="Labels dict JSON path",
                kind="path",
                default=DEFAULT_LABELS_DICT,
                validate=v.path_exists,
            ),
            FieldSpec(
                key="num_labels",
                label="Number of labels",
                kind="int",
                default=39,
                validate=v.positive_int,
            ),
            # ── Syto atlas (only when syto is enabled) ──
            FieldSpec(
                key="deconvolution.syto.atlas_path",
                label="Syto atlas TSV path",
                kind="path",
                validate=v.path_exists,
                when=_run_syto,
            ),
            FieldSpec(
                key="deconvolution.syto.atlas_name",
                label="Syto atlas name",
                kind="text",
                when=_run_syto,
            ),
            # ── Input (always) ──
            FieldSpec(
                key="input.type",
                label="Input type",
                kind="select",
                choices=["bam", "parsed_reads", "predicted_reads"],
            ),
            FieldSpec(
                key="input.data_path",
                label="Input data path",
                kind="path",
                validate=v.path_exists,
            ),
            FieldSpec(
                key="input.reference_path",
                label="Reference FASTA path",
                kind="path",
                validate=v.path_exists,
                when=lambda a: a.get("input.type") == "bam",
            ),
            FieldSpec(
                key="input.data_type",
                label="Data type",
                kind="select",
                choices=["wgbs", "ont"],
                default="wgbs",
                when=lambda a: a.get("input.type") == "bam",
            ),
            FieldSpec(
                key="input.chromosomes",
                label="Chromosomes ('all' or comma-separated)",
                kind="text",
                default="all",
            ),
            # ── Missing-label handling (always) ──
            FieldSpec(
                key="fill_in_missing_labels",
                label="Fill in missing labels?",
                kind="bool",
                default=True,
            ),
            FieldSpec(
                key="missing_label_strategy",
                label="Missing label strategy",
                kind="select",
                choices=["prior_blending", "zeroes", "prior_imputation"],
                default="prior_blending",
                when=lambda a: a.get("fill_in_missing_labels"),
            ),
            FieldSpec(
                key="pseudobulk_h5_path",
                label="Pseudobulk HDF5 path",
                kind="path",
                validate=v.path_exists,
                when=lambda a: a.get("fill_in_missing_labels")
                and a.get("missing_label_strategy")
                in {"prior_blending", "prior_imputation"},
            ),
            # ── Deconvolution (list-sections) ──
            FieldSpec(
                key="deconvolution.syto.methods",
                label="Syto deconvolvers",
                kind="list_section",
                default=[],
                handler=_deconv_methods_section,
                when=_run_syto,
            ),
            FieldSpec(
                key="deconvolution.baselines",
                label="Baseline deconvolvers",
                kind="list_section",
                default=[],
                handler=_deconv_baselines_section,
            ),
            # ── Output (always) ──
            FieldSpec(
                key="output_dir",
                label="Output directory",
                kind="text",
                validate=v.non_empty,
            ),
            FieldSpec(
                key="output.save_processed_reads",
                label="Save processed reads?",
                kind="bool",
                default=False,
            ),
            FieldSpec(
                key="output.save_predictions",
                label="Save predictions?",
                kind="bool",
                default=True,
            ),
            FieldSpec(
                key="output.save_deconvolution",
                label="Save deconvolution?",
                kind="bool",
                default=True,
            ),
            # ── Expert: BAM processing (only relevant for bam input) ──
            FieldSpec(
                key="bam_processing.n_jobs",
                label="bam n_jobs",
                kind="int",
                default=2,
                tier="expert",
                validate=v.positive_int,
                when=_input_is_bam,
            ),
            FieldSpec(
                key="bam_processing.min_mapq",
                label="bam min_mapq",
                kind="int",
                default=10,
                tier="expert",
                when=_input_is_bam,
            ),
            FieldSpec(
                key="bam_processing.require_flags",
                label="bam require_flags",
                kind="int",
                default=3,
                tier="expert",
                when=_input_is_bam,
            ),
            FieldSpec(
                key="bam_processing.exclude_flags",
                label="bam exclude_flags",
                kind="int",
                default=1796,
                tier="expert",
                when=_input_is_bam,
            ),
            FieldSpec(
                key="bam_processing.min_cpgs",
                label="bam min_cpgs",
                kind="int",
                default=4,
                tier="expert",
                validate=v.positive_int,
                when=_input_is_bam,
            ),
            FieldSpec(
                key="bam_processing.merge_pairs",
                label="bam merge_pairs",
                kind="bool",
                default=True,
                tier="expert",
                when=_input_is_bam,
            ),
            FieldSpec(
                key="bam_processing.ont_methyl_tr",
                label="bam ont_methyl_tr",
                kind="int",
                default=180,
                tier="expert",
                when=_input_is_bam,
            ),
            # ── Expert: classifier prediction params (only when syto runs) ──
            FieldSpec(
                key="max_sequence_length",
                label="max_sequence_length",
                kind="int",
                default=150,
                tier="expert",
                validate=v.positive_int,
                when=_run_syto,
            ),
            FieldSpec(
                key="prediction_batch_size",
                label="prediction_batch_size",
                kind="int",
                default=2200,
                tier="expert",
                validate=v.positive_int,
                when=_run_syto,
            ),
        ]

    def build_config(self, answers: dict) -> dict:
        run_syto = bool(answers.get("run_syto"))
        config: dict = {}

        # Keys assembled explicitly below rather than copied verbatim.
        skip_keys = {
            "run_syto",
            "input.chromosomes",
            "classifier.labeling_scheme",
            "deconvolution.syto.methods",
            "deconvolution.baselines",
        }

        # The wizard engine omits fields that are not applicable to this config
        # (their `when` predicate was False), so the answers dict already
        # excludes irrelevant keys. A None value marks a not-applicable field
        # supplied explicitly (e.g. in tests); skip those too so nothing unused
        # leaks into the generated config.
        for key, value in answers.items():
            if key in skip_keys or value is None:
                continue
            _set_nested(config, key, value)

        # Labeling scheme -> soft_labels bool (only when it was asked).
        scheme = answers.get("classifier.labeling_scheme")
        if scheme is not None:
            _set_nested(config, "classifier.soft_labels", scheme == "Soft Labels")

        # chromosomes: keep "all", else split to list
        chrom = answers.get("input.chromosomes", "all")
        if isinstance(chrom, str) and chrom.strip() != "all":
            chrom = [c.strip() for c in chrom.split(",") if c.strip()]
        _set_nested(config, "input.chromosomes", chrom)

        # Deconvolution block
        config.setdefault("deconvolution", {})
        if run_syto:
            config["deconvolution"].setdefault("syto", {})
            config["deconvolution"]["syto"]["methods"] = answers.get(
                "deconvolution.syto.methods", []
            )
        config["deconvolution"]["baselines"] = answers.get(
            "deconvolution.baselines", []
        )
        return config


TASK_REGISTRY["inference"] = InferenceWizard
