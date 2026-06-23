"""Inference task wizard: field schema, deconvolution sub-flows, and assembler."""

from App.wizard import validators as v
from App.wizard.fields import FieldSpec
from App.wizard.tasks import TASK_REGISTRY

CLASSIFIER_TYPES = ["dismir", "methylbert", "cancer_detector", "lookup", "epigenbert2"]
TRANSFORMER_ARCHS = {"methylbert", "epigenbert2"}
SYTO_DECONVOLVERS = ["xgboost", "3Layer_MLP", "Shallow_Wide_Network", "nnls", "psls"]
BASELINES = ["uxm", "celfieish", "celfie"]


def _ctype(answers):
    return answers.get("classifier.classifier_type")


def _deconv_methods_section(engine, answers):
    """Build deconvolution.methods as a list of dicts."""
    selected = engine.ask_checkbox(
        "Select syto feature-based deconvolvers to enable:", SYTO_DECONVOLVERS
    )
    methods = []
    for name in selected:
        ckpt = engine.ask_one(
            FieldSpec(key=f"_{name}_ckpt", label=f"[{name}] checkpoint_path",
                      kind="path", validate=v.path_exists),
            answers,
        )
        cal_dir = engine.ask_one(
            FieldSpec(key=f"_{name}_cal", label=f"[{name}] calibrators_dir",
                      kind="path", validate=v.path_exists),
            answers,
        )
        use_cal = engine.ask_one(
            FieldSpec(key=f"_{name}_use", label=f"[{name}] use calibration?",
                      kind="bool", default=True),
            answers,
        )
        methods.append({
            "name": name, "enabled": True, "use_callibration": use_cal,
            "checkpoint_path": ckpt, "calibrators_dir": cal_dir,
        })
    answers["deconvolution.methods"] = methods


def _deconv_baselines_section(engine, answers):
    """Build deconvolution.baselines as a list of dicts."""
    selected = engine.ask_checkbox(
        "Select read-based baseline deconvolvers to enable:", BASELINES
    )
    baselines = []
    for model in selected:
        atlas = engine.ask_one(
            FieldSpec(key=f"_{model}_atlas", label=f"[{model}] atlas_path",
                      kind="path", validate=v.path_exists),
            answers,
        )
        entry = {"model": model, "enabled": True, "atlas_path": atlas}
        if model == "uxm":
            ignore = engine.ask_one(
                FieldSpec(key="_uxm_ignore",
                          label="[uxm] ignore_cells (comma-separated)",
                          kind="text", default="Megakaryocytes"),
                answers,
            )
            entry["ignore_cells"] = [c.strip() for c in ignore.split(",") if c.strip()]
        else:  # celfieish / celfie
            entry["reference_genome"] = engine.ask_one(
                FieldSpec(key=f"_{model}_ref", label=f"[{model}] reference_genome",
                          kind="select", choices=["hg38", "hg19"], default="hg38"),
                answers,
            )
            entry["num_iterations"] = engine.ask_one(
                FieldSpec(key=f"_{model}_iter", label=f"[{model}] num_iterations",
                          kind="int", default=400, validate=v.positive_int),
                answers,
            )
            entry["convergence_criteria"] = engine.ask_one(
                FieldSpec(key=f"_{model}_conv",
                          label=f"[{model}] convergence_criteria",
                          kind="float", default=0.001, validate=v.positive_float),
                answers,
            )
            if model == "celfie":
                entry["random_restarts"] = engine.ask_one(
                    FieldSpec(key="_celfie_rr", label="[celfie] random_restarts",
                              kind="int", default=1, validate=v.positive_int),
                    answers,
                )
        baselines.append(entry)
    answers["deconvolution.baselines"] = baselines


class InferenceWizard:
    task_name = "inference"

    def field_specs(self) -> list[FieldSpec]:
        return [
            # ── Classifier (core) ──
            FieldSpec(key="classifier.classifier_type", label="Classifier type",
                      kind="select", choices=CLASSIFIER_TYPES),
            FieldSpec(key="classifier.dismir_flavor", label="Dismir flavor",
                      kind="select", choices=["lstm", "mingru"], default="lstm",
                      when=lambda a: _ctype(a) == "dismir"),
            FieldSpec(key="classifier.foundation_model", label="Foundation model path",
                      kind="path", validate=v.path_exists,
                      when=lambda a: _ctype(a) in TRANSFORMER_ARCHS),
            FieldSpec(key="classifier.classifier_head_implementation",
                      label="Classifier head", kind="select",
                      choices=["grg_attention_based", "simple"],
                      default="grg_attention_based",
                      when=lambda a: _ctype(a) in {"methylbert", "dismir"}),
            FieldSpec(key="classifier.soft_labels", label="Soft labels?",
                      kind="bool", default=True),
            FieldSpec(key="checkpoint_path", label="Checkpoint path",
                      kind="path", validate=v.path_exists),
            FieldSpec(key="labels_dict_path", label="Labels dict JSON path",
                      kind="path", validate=v.path_exists),
            FieldSpec(key="num_labels", label="Number of labels",
                      kind="int", default=39, validate=v.positive_int),
            # ── Atlas (core) ──
            FieldSpec(key="atlas_path", label="Atlas TSV path",
                      kind="path", validate=v.path_exists),
            FieldSpec(key="atlas_name", label="Atlas name", kind="text"),
            # ── Input (core) ──
            FieldSpec(key="input.type", label="Input type",
                      kind="select", choices=["bam", "parsed_reads"]),
            FieldSpec(key="input.bam_path", label="BAM path", kind="path",
                      validate=v.path_exists,
                      when=lambda a: a.get("input.type") == "bam"),
            FieldSpec(key="input.reference_path", label="Reference FASTA path",
                      kind="path", validate=v.path_exists,
                      when=lambda a: a.get("input.type") == "bam"),
            FieldSpec(key="input.data_type", label="Data type",
                      kind="select", choices=["wgbs", "ont"], default="wgbs",
                      when=lambda a: a.get("input.type") == "bam"),
            FieldSpec(key="input.parsed_reads_path", label="Parsed reads path",
                      kind="path", validate=v.path_exists,
                      when=lambda a: a.get("input.type") == "parsed_reads"),
            FieldSpec(key="input.chromosomes",
                      label="Chromosomes ('all' or comma-separated)",
                      kind="text", default="all"),
            # ── Missing-label handling (core) ──
            FieldSpec(key="fill_in_missing_labels",
                      label="Fill in missing labels?", kind="bool", default=True),
            FieldSpec(key="missing_label_strategy", label="Missing label strategy",
                      kind="select",
                      choices=["prior_blending", "zeroes", "prior_imputation"],
                      default="prior_blending",
                      when=lambda a: a.get("fill_in_missing_labels")),
            FieldSpec(key="pseudobulk_h5_path", label="Pseudobulk HDF5 path",
                      kind="path", validate=v.path_exists,
                      when=lambda a: a.get("fill_in_missing_labels")
                      and a.get("missing_label_strategy")
                      in {"prior_blending", "prior_imputation"}),
            # ── Deconvolution (core, list-sections) ──
            FieldSpec(key="deconvolution.methods", label="Syto deconvolvers",
                      kind="list_section", handler=_deconv_methods_section),
            FieldSpec(key="deconvolution.baselines", label="Baseline deconvolvers",
                      kind="list_section", handler=_deconv_baselines_section),
            # ── Output (core) ──
            FieldSpec(key="output_dir", label="Output directory",
                      kind="text", validate=v.non_empty),
            FieldSpec(key="output.save_processed_reads",
                      label="Save processed reads?", kind="bool", default=False),
            FieldSpec(key="output.save_predictions", label="Save predictions?",
                      kind="bool", default=True),
            FieldSpec(key="output.save_deconvolution", label="Save deconvolution?",
                      kind="bool", default=True),
            # ── Expert ──
            FieldSpec(key="bam_processing.n_jobs", label="bam n_jobs",
                      kind="int", default=2, tier="expert", validate=v.positive_int),
            FieldSpec(key="bam_processing.min_mapq", label="bam min_mapq",
                      kind="int", default=10, tier="expert"),
            FieldSpec(key="bam_processing.require_flags", label="bam require_flags",
                      kind="int", default=3, tier="expert"),
            FieldSpec(key="bam_processing.exclude_flags", label="bam exclude_flags",
                      kind="int", default=1796, tier="expert"),
            FieldSpec(key="bam_processing.min_cpgs", label="bam min_cpgs",
                      kind="int", default=4, tier="expert", validate=v.positive_int),
            FieldSpec(key="bam_processing.merge_pairs", label="bam merge_pairs",
                      kind="bool", default=True, tier="expert"),
            FieldSpec(key="bam_processing.ont_methyl_tr", label="bam ont_methyl_tr",
                      kind="int", default=180, tier="expert"),
            FieldSpec(key="max_sequence_length", label="max_sequence_length",
                      kind="int", default=150, tier="expert", validate=v.positive_int),
            FieldSpec(key="prediction_batch_size", label="prediction_batch_size",
                      kind="int", default=2200, tier="expert",
                      validate=v.positive_int),
            FieldSpec(key="features_mask_path",
                      label="features_mask_path (optional, blank to skip)",
                      kind="path", default="", tier="expert",
                      validate=v.path_exists_or_blank),
            FieldSpec(key="cell_type_match_dict_path",
                      label="cell_type_match_dict_path (optional, blank to skip)",
                      kind="path", default="", tier="expert",
                      validate=v.path_exists_or_blank),
        ]

    def build_config(self, answers: dict) -> dict:
        config: dict = {}
        list_keys = {"deconvolution.methods", "deconvolution.baselines"}
        special = {"input.chromosomes"} | list_keys
        optional_blank = {"features_mask_path", "cell_type_match_dict_path"}

        for key, value in answers.items():
            if key in special:
                continue
            if key in optional_blank and (value is None or str(value).strip() == ""):
                value = None
            _set_nested(config, key, value)

        # chromosomes: keep "all", else split to list
        chrom = answers.get("input.chromosomes", "all")
        if isinstance(chrom, str) and chrom.strip() != "all":
            chrom = [c.strip() for c in chrom.split(",") if c.strip()]
        _set_nested(config, "input.chromosomes", chrom)

        # deconvolution lists
        config.setdefault("deconvolution", {})
        config["deconvolution"]["methods"] = answers.get("deconvolution.methods", [])
        config["deconvolution"]["baselines"] = answers.get(
            "deconvolution.baselines", []
        )
        return config


def _set_nested(config: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


TASK_REGISTRY["inference"] = InferenceWizard
