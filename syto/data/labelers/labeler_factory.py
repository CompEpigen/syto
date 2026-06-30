"""Config-driven construction and application of labelers.

A ``labelers`` config is a dict keyed by the destination column name; each
value is a dict with a ``type`` key selecting the labeler and the remaining
keys as that labeler's parameters. The same type may appear under multiple
keys with different parameters.

The ``_LABELER_REGISTRY`` / ``labeler_factory`` pair mirrors
``syto/classification/classifiers/lazy_classifier_factory.py``: labeler classes
are referenced by lazily-imported ``"module:Class"`` strings, validated at
import time.
"""

import importlib
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from syto.data.labelers.abstract_labeler import AbstractLabeler
from syto.data.omics_signatures_handlers.binary_cpg_signature import (
    BinaryCpGSignatureHandler,
)

_LABELER_REGISTRY = {
    "data_driven_soft": (
        "syto.data.labelers.data_driven_soft_labeler:DataDrivenSoftLabeler"
    ),
    "archetype": "syto.data.labelers.archetype_labeler:ArchetypeLabeler",
    "hard_with_background": (
        "syto.data.labelers.hard_with_background_labeler:HardWithBackgroundLabeler"
    ),
    "hard_binary": "syto.data.labelers.hard_binary_labeler:HardBinaryLabeler",
    "smoothed_hard_with_background": (
        "syto.data.labelers.smoothed_hard_with_background:"
        "SmoothedHardWithBackgroundLabeler"
    ),
}


# Validate module paths at import time (no heavy deps triggered)
for _key, _entry in _LABELER_REGISTRY.items():
    _module_path = _entry.split(":", maxsplit=1)[0]
    if importlib.util.find_spec(_module_path) is None:
        warnings.warn(
            f"Registry entry '{_key}': module '{_module_path}' not found.",
            ImportWarning,
            stacklevel=2,
        )


def labeler_factory(name: str, **kwargs) -> AbstractLabeler:
    """Construct a labeler instance by its registry name.

    Args:
        name: Identifier for the labeler type (e.g. "data_driven_soft").
        **kwargs: Constructor arguments for the labeler class.

    Returns:
        An instance of a subclass of AbstractLabeler.
    """
    if name not in _LABELER_REGISTRY:
        raise ValueError(
            f"Unknown labeler '{name}'. Valid types: {sorted(_LABELER_REGISTRY)}"
        )
    module_path, cls_name = _LABELER_REGISTRY[name].split(":")
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(**kwargs)


@dataclass
class LabelerContext:
    """Shared resources passed to every labeler builder."""

    num_classes: int
    signature_handler: BinaryCpGSignatureHandler
    signature_column: str = "signature"
    global_prior: Optional[np.ndarray] = None
    class_label_column: str = "original_label"
    grg_class_label_column: str = "dmr_ctype_label"


def _build_data_driven_soft(params, ctx):
    labeler = labeler_factory(
        "data_driven_soft",
        distance_name=params.get("distance_name", "jaccard"),
        signature_handler=ctx.signature_handler,
    )
    compute_kwargs = dict(
        perform_pooling=params.get("perform_pooling", True),
        min_reads=params.get("min_reads", 30),
        max_distance=params.get("max_distance", 0.5),
        num_classes=ctx.num_classes,
        keep_intermediate_values=False,
        precomputed_signature_column=ctx.signature_column,
        add_superset_counts=params.get("add_superset_counts", False),
    )
    return labeler, compute_kwargs, "soft_label"


def _build_archetype(params, ctx):
    labeler = labeler_factory(
        "archetype",
        signature_handler=ctx.signature_handler,
        eps=params.get("eps", 1e-10),
    )
    prior_type = params.get("ctype_prior_type", "uniform")
    predefined = params.get("predefined_prior")
    if prior_type == "inv_global_freq":
        if ctx.global_prior is None:
            raise ValueError(
                "archetype labeler requested 'inv_global_freq' but no "
                "global_prior was provided in the context."
            )
        prior_type = "predefined"
        predefined = ctx.global_prior
    compute_kwargs = dict(
        num_classes=ctx.num_classes,
        ctype_prior_type=prior_type,
        predefined_prior=predefined,
        keep_likelihoods=False,
        precomputed_signature_column=ctx.signature_column,
    )
    return labeler, compute_kwargs, "soft_label"


def _build_hard_with_background(params, ctx):
    labeler = labeler_factory("hard_with_background")
    compute_kwargs = dict(
        num_original_classes=ctx.num_classes,
        class_label_column=params.get("class_label_column", ctx.class_label_column),
        grg_class_label_column=params.get(
            "grg_class_label_column", ctx.grg_class_label_column
        ),
    )
    return labeler, compute_kwargs, "label"


def _build_hard_binary(params, ctx):
    labeler = labeler_factory("hard_binary")
    compute_kwargs = dict(
        class_label_column=params.get("class_label_column", ctx.class_label_column),
        grg_class_label_column=params.get(
            "grg_class_label_column", ctx.grg_class_label_column
        ),
    )
    return labeler, compute_kwargs, "label"


def _build_smoothed_hard_with_background(params, ctx):
    labeler = labeler_factory("smoothed_hard_with_background")
    compute_kwargs = dict(
        num_original_classes=ctx.num_classes,
        epsilon=params.get("epsilon", 0.1),
        class_label_column=params.get("class_label_column", ctx.class_label_column),
        grg_class_label_column=params.get(
            "grg_class_label_column", ctx.grg_class_label_column
        ),
    )
    return labeler, compute_kwargs, "smoothed_label"


_BUILDERS: dict[str, Callable] = {
    "data_driven_soft": _build_data_driven_soft,
    "archetype": _build_archetype,
    "hard_with_background": _build_hard_with_background,
    "hard_binary": _build_hard_binary,
    "smoothed_hard_with_background": _build_smoothed_hard_with_background,
}

# Types that consume the binary-CpG signature (so we compute it once up front).
SIGNATURE_LABELER_TYPES = {"data_driven_soft", "archetype"}


def build_labeler(
    labeler_type: str, params: dict, context: LabelerContext
) -> tuple[AbstractLabeler, dict, str]:
    """Construct a labeler and its compute kwargs from a config entry."""
    if labeler_type not in _BUILDERS:
        raise ValueError(
            f"Unknown labeler type {labeler_type!r}. Valid types: {sorted(_BUILDERS)}"
        )
    return _BUILDERS[labeler_type](params, context)


def apply_labelers(
    df: pd.DataFrame, labelers_config: dict, context: LabelerContext
) -> pd.DataFrame:
    """Run every configured labeler, writing each output to its config key.

    Each labeler's native output column is renamed to a unique temporary name
    immediately after it runs, then to the destination key at the end, so two
    labelers sharing a native output column (e.g. two soft labelers both
    producing ``soft_label``) never collide.
    """
    if not labelers_config:
        raise ValueError("'labelers' config is empty; at least one entry is required.")

    # Fail fast on unknown types before any expensive work.
    for entry in labelers_config.values():
        labeler_type = entry.get("type")
        if labeler_type not in _BUILDERS:
            raise ValueError(
                f"Unknown labeler type {labeler_type!r}. "
                f"Valid types: {sorted(_BUILDERS)}"
            )

    df = df.copy()

    # Compute the shared binary-CpG signature once if any labeler needs it.
    needs_signature = any(
        entry["type"] in SIGNATURE_LABELER_TYPES for entry in labelers_config.values()
    )
    if needs_signature:
        tqdm.pandas(desc="Extracting signatures")
        df[context.signature_column] = df.progress_apply(
            context.signature_handler.extract_signature, axis=1
        )

    temp_renames: dict[str, str] = {}
    for i, (dest_column, entry) in enumerate(labelers_config.items()):
        params = {k: v for k, v in entry.items() if k != "type"}
        labeler, compute_kwargs, native_col = build_labeler(
            entry["type"], params, context
        )
        df = labeler.compute_labels(df, **compute_kwargs)
        temp = f"__labeler_output_{i}"
        df = df.rename(columns={native_col: temp})
        temp_renames[temp] = dest_column

    df = df.rename(columns=temp_renames)
    return df
