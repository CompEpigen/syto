"""Classifier-fit task wizard: per-architecture field schema and assembler.

Mirrors the five working configs under App/config/classifiers_fitting. Each
architecture exposes a different set of model and training fields; the schema
gates every field with a `when` predicate keyed on the chosen architecture, so
the generated config never carries attributes irrelevant to that architecture.
"""

from App.wizard import validators as v
from App.wizard.config_utils import set_nested as _set_nested
from App.wizard.fields import FieldSpec
from App.wizard.tasks import TASK_REGISTRY

ARCHITECTURES = ["dismir", "methylbert", "cancer_detector", "lookup", "epigenbert2"]
TRANSFORMER_ARCHS = {"methylbert", "epigenbert2"}
HF_ARCHS = {"methylbert", "epigenbert2"}  # HuggingFace Trainer based
NEURAL_ARCHS = {"dismir", "methylbert", "epigenbert2"}  # num_labels/head/soft_labels
GRG_LABEL_ARCHS = {"dismir", "methylbert"}  # num_grg_labels


def _arch(answers):
    return answers.get("model.architecture")


def _is(*archs):
    """Build a `when` predicate that is True for the given architectures."""
    archs = set(archs)
    return lambda a: _arch(a) in archs


def _in(archs):
    return lambda a: _arch(a) in archs


class ClassifierFitWizard:
    task_name = "classifier_fit"

    def field_specs(self) -> list[FieldSpec]:
        return [
            # ── Architecture (asked first) ──
            FieldSpec(
                key="model.architecture",
                label="Classifier architecture",
                kind="select",
                choices=ARCHITECTURES,
            ),
            # ── Model: neural / transformer architectures ──
            FieldSpec(
                key="model.flavor",
                label="Dismir flavor",
                kind="select",
                choices=["lstm", "mingru"],
                default="lstm",
                when=_is("dismir"),
            ),
            FieldSpec(
                key="model.foundation_model",
                label="Foundation model path",
                kind="path",
                validate=v.path_exists,
                when=_in(TRANSFORMER_ARCHS),
            ),
            FieldSpec(
                key="model.num_labels",
                label="num_labels",
                kind="int",
                default=39,
                validate=v.positive_int,
                when=_in(NEURAL_ARCHS),
            ),
            FieldSpec(
                key="model.num_grg_labels",
                label="num_grg_labels",
                kind="int",
                default=39,
                validate=v.positive_int,
                when=_in(GRG_LABEL_ARCHS),
            ),
            FieldSpec(
                key="model.classifier_head_implementation",
                label="Classifier head",
                kind="select",
                choices=["grg_attention_based", "simple"],
                default="grg_attention_based",
                when=_in(NEURAL_ARCHS),
            ),
            FieldSpec(
                key="model.grg_label_column",
                label="GRG label column",
                kind="text",
                default="dmr_ctype_label",
                when=_is("dismir"),
            ),
            FieldSpec(
                key="model.soft_labels",
                label="Use soft labels?",
                kind="bool",
                default=True,
                when=_in(NEURAL_ARCHS),
            ),
            FieldSpec(
                key="model.use_cpg_methylation",
                label="Use CpG methylation?",
                kind="bool",
                default=True,
                when=_is("epigenbert2"),
            ),
            FieldSpec(
                key="model.use_m6a_methylation",
                label="Use m6A methylation?",
                kind="bool",
                default=False,
                when=_is("epigenbert2"),
            ),
            # ── Model: lookup ──
            FieldSpec(
                key="model.lookup_config.num_classes",
                label="lookup num_classes",
                kind="int",
                default=39,
                validate=v.positive_int,
                when=_is("lookup"),
            ),
            FieldSpec(
                key="model.lookup_config.label_col",
                label="lookup label_col",
                kind="text",
                default="original_label",
                when=_is("lookup"),
            ),
            FieldSpec(
                key="model.lookup_config.label_mode",
                label="lookup label_mode",
                kind="select",
                choices=["soft", "hard"],
                default="soft",
                when=_is("lookup"),
            ),
            FieldSpec(
                key="model.lookup_config.min_reads",
                label="lookup min_reads",
                kind="int",
                default=0,
                validate=v.non_negative_int,
                when=_is("lookup"),
            ),
            FieldSpec(
                key="model.lookup_config.max_distance",
                label="lookup max_distance",
                kind="int",
                default=0,
                validate=v.non_negative_int,
                when=_is("lookup"),
            ),
            # ── Shared data / run config (always) ──
            FieldSpec(
                key="data_path", label="Data path", kind="path", validate=v.path_exists
            ),
            FieldSpec(
                key="datasets",
                label="Datasets ('all' or a dataset name)",
                kind="text",
                default="all",
                validate=v.non_empty,
            ),
            FieldSpec(
                key="label_column",
                label="Training label column (columnar datasets)",
                kind="text",
                validate=v.non_empty,
                when=_in(NEURAL_ARCHS),
            ),
            FieldSpec(
                key="data_format",
                label="Data format",
                kind="select",
                choices=["auto", "legacy", "columnar"],
                default="auto",
            ),
            FieldSpec(
                key="split_column",
                label="Split column name (columnar datasets)",
                kind="text",
                default="split",
                tier="expert",
            ),
            FieldSpec(
                key="min_pattern_length",
                label="Min marked CpGs per read (1 = no filtering)",
                kind="int",
                default=1,
                validate=v.positive_int,
                tier="expert",
            ),
            FieldSpec(
                key="max_sequence_length",
                label="max_sequence_length",
                kind="int",
                default=150,
                validate=v.positive_int,
            ),
            FieldSpec(
                key="prediction_batch_size",
                label="prediction_batch_size",
                kind="int",
                default=128,
                validate=v.positive_int,
            ),
            FieldSpec(
                key="output.output_dir",
                label="Output directory",
                kind="text",
                validate=v.non_empty,
            ),
            # ── MLflow (experiment/uri only when enabled) ──
            FieldSpec(
                key="mlflow.enabled",
                label="Enable MLflow tracking?",
                kind="bool",
                default=True,
            ),
            FieldSpec(
                key="mlflow.experiment_name",
                label="MLflow experiment name",
                kind="text",
                validate=v.non_empty,
                when=lambda a: a.get("mlflow.enabled"),
            ),
            FieldSpec(
                key="mlflow.tracking_uri",
                label="MLflow tracking URI (blank for local)",
                kind="text",
                default="",
                when=lambda a: a.get("mlflow.enabled"),
            ),
            # ── Training: cancer_detector ──
            FieldSpec(
                key="training.col_label",
                label="col_label",
                kind="text",
                default="label",
                when=_is("cancer_detector"),
            ),
            FieldSpec(
                key="training.col_marker_label",
                label="col_marker_label",
                kind="text",
                default="dmr_ctype_label",
                when=_is("cancer_detector"),
            ),
            FieldSpec(
                key="training.eps_beta_fit",
                label="eps_beta_fit",
                kind="float",
                default=0.01,
                validate=v.positive_float,
                when=_is("cancer_detector"),
            ),
            FieldSpec(
                key="training.class_prior_type",
                label="class_prior_type",
                kind="select",
                choices=["train_freq", "uniform"],
                default="train_freq",
                when=_is("cancer_detector"),
            ),
            # ── Training: dismir ──
            FieldSpec(
                key="training.epochs",
                label="epochs",
                kind="int",
                default=100,
                validate=v.positive_int,
                when=_is("dismir"),
            ),
            FieldSpec(
                key="training.batch_size",
                label="batch_size",
                kind="int",
                default=128,
                validate=v.positive_int,
                when=_is("dismir"),
            ),
            FieldSpec(
                key="training.patience",
                label="patience",
                kind="int",
                default=30,
                validate=v.positive_int,
                when=_is("dismir"),
            ),
            FieldSpec(
                key="training.optimizer_type",
                label="optimizer_type",
                kind="text",
                default="ADAM",
                when=_is("dismir"),
            ),
            FieldSpec(
                key="training.lr",
                label="learning rate",
                kind="float",
                default=0.001,
                validate=v.positive_float,
                when=_is("dismir"),
            ),
            FieldSpec(
                key="training.weight_decay",
                label="weight_decay",
                kind="float",
                default=1.0e-6,
                when=_is("dismir"),
            ),
            FieldSpec(
                key="training.variable_length",
                label="variable_length?",
                kind="bool",
                default=False,
                when=_is("dismir"),
            ),
            FieldSpec(
                key="training.verbose",
                label="verbose",
                kind="int",
                default=1,
                validate=v.non_negative_int,
                when=_is("dismir"),
            ),
            # ── Training: lookup + HuggingFace (train-set metric computation) ──
            # For HF archs this triggers a full evaluation pass over the training
            # set after fit to log train_* metrics (potentially expensive).
            FieldSpec(
                key="training.compute_train_metrics",
                label="compute_train_metrics?",
                kind="bool",
                default=True,
                when=_in({"lookup"} | HF_ARCHS),
            ),
            # ── Training: HuggingFace (methylbert / epigenbert2) ──
            FieldSpec(
                key="training.grg_label_column",
                label="GRG label column",
                kind="text",
                default="dmr_ctype_label",
                when=_in(HF_ARCHS),
            ),
            # early stopping (transformers.EarlyStoppingCallback)
            FieldSpec(
                key="training.early_stopping.enabled",
                label="Enable early stopping?",
                kind="bool",
                default=False,
                when=_in(HF_ARCHS),
            ),
            FieldSpec(
                key="training.early_stopping.early_stopping_patience",
                label="early_stopping_patience",
                kind="int",
                default=5,
                validate=v.positive_int,
                when=lambda a: _arch(a) in HF_ARCHS
                and bool(a.get("training.early_stopping.enabled")),
            ),
            FieldSpec(
                key="training.early_stopping.early_stopping_threshold",
                label="early_stopping_threshold",
                kind="float",
                default=0.0,
                validate=v.non_negative_float,
                when=lambda a: _arch(a) in HF_ARCHS
                and bool(a.get("training.early_stopping.enabled")),
            ),
            # core training_args
            FieldSpec(
                key="training.training_args.learning_rate",
                label="learning_rate",
                kind="float",
                default=0.0001,
                validate=v.positive_float,
                when=_in(HF_ARCHS),
            ),
            FieldSpec(
                key="training.training_args.per_device_train_batch_size",
                label="per_device_train_batch_size",
                kind="int",
                default=512,
                validate=v.positive_int,
                when=_in(HF_ARCHS),
            ),
            FieldSpec(
                key="training.training_args.per_device_eval_batch_size",
                label="per_device_eval_batch_size",
                kind="int",
                default=2200,
                validate=v.positive_int,
                when=_in(HF_ARCHS),
            ),
            FieldSpec(
                key="training.training_args.num_train_epochs",
                label="num_train_epochs",
                kind="int",
                default=1,
                validate=v.positive_int,
                when=_in(HF_ARCHS),
            ),
            FieldSpec(
                key="training.training_args.weight_decay",
                label="weight_decay",
                kind="float",
                default=0.01,
                when=_in(HF_ARCHS),
            ),
            FieldSpec(
                key="training.training_args.warmup_steps",
                label="warmup_steps",
                kind="int",
                default=100,
                validate=v.non_negative_int,
                when=_in(HF_ARCHS),
            ),
            FieldSpec(
                key="training.training_args.eval_strategy",
                label="eval_strategy",
                kind="select",
                choices=["steps", "epoch", "no"],
                default="steps",
                when=_in(HF_ARCHS),
            ),
            # expert training_args
            FieldSpec(
                key="training.training_args.gradient_accumulation_steps",
                label="gradient_accumulation_steps",
                kind="int",
                default=1,
                tier="expert",
                validate=v.positive_int,
                when=_in(HF_ARCHS),
            ),
            FieldSpec(
                key="training.training_args.fp16",
                label="fp16?",
                kind="bool",
                default=False,
                tier="expert",
                when=_in(HF_ARCHS),
            ),
            FieldSpec(
                key="training.training_args.save_steps",
                label="save_steps",
                kind="int",
                default=200,
                tier="expert",
                validate=v.positive_int,
                when=_in(HF_ARCHS),
            ),
            FieldSpec(
                key="training.training_args.eval_steps",
                label="eval_steps",
                kind="int",
                default=200,
                tier="expert",
                validate=v.positive_int,
                when=_in(HF_ARCHS),
            ),
            FieldSpec(
                key="training.training_args.logging_steps",
                label="logging_steps",
                kind="int",
                default=200,
                tier="expert",
                validate=v.positive_int,
                when=_in(HF_ARCHS),
            ),
            FieldSpec(
                key="training.training_args.max_grad_norm",
                label="max_grad_norm",
                kind="float",
                default=1.0,
                tier="expert",
                when=_in(HF_ARCHS),
            ),
            FieldSpec(
                key="training.training_args.adam_beta1",
                label="adam_beta1",
                kind="float",
                default=0.9,
                tier="expert",
                when=_in(HF_ARCHS),
            ),
            FieldSpec(
                key="training.training_args.adam_beta2",
                label="adam_beta2",
                kind="float",
                default=0.98,
                tier="expert",
                when=_in(HF_ARCHS),
            ),
            FieldSpec(
                key="training.training_args.adam_epsilon",
                label="adam_epsilon",
                kind="float",
                default=0.000001,
                tier="expert",
                when=_in(HF_ARCHS),
            ),
            FieldSpec(
                key="training.training_args.gradient_checkpointing",
                label="gradient_checkpointing?",
                kind="bool",
                default=False,
                tier="expert",
                when=_in(HF_ARCHS),
            ),
            FieldSpec(
                key="training.training_args.seed",
                label="seed",
                kind="int",
                default=42,
                tier="expert",
                validate=v.non_negative_int,
                when=_in(HF_ARCHS),
            ),
        ]

    def build_config(self, answers: dict) -> dict:
        """Assemble the flat dot-notation answers into the nested config.

        Fields not applicable to the chosen architecture were already omitted by
        the engine (their `when` was False). Here we additionally drop any
        None/blank values so nothing unused leaks into the generated config.
        """
        config: dict = {}
        for key, value in answers.items():
            if value is None:
                continue
            if isinstance(value, str) and value.strip() == "":
                continue
            _set_nested(config, key, value)
        return config


TASK_REGISTRY["classifier_fit"] = ClassifierFitWizard
