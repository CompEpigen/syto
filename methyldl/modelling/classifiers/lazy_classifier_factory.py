"""
Defines a factory function to load read classifiers from checkpoints,
using a registry of known classifier types.
"""

import importlib
import warnings

from methyldl.modelling.classifiers.abstract_read_classifier import (
    AbstractReadClassifier,
)

_CLF_REGISTRY = {
    "methylbert": "methyldl.modelling.classifiers.methylbert:MethylBert",
    "dismir": "methyldl.modelling.classifiers.dismir:Dismir",
    "lookup": "methyldl.modelling.classifiers.lookup:LookupClassifier",
    "cancer_detector": "methyldl.modelling.classifiers.cancer_detector:CancerDetectorClassifier",
}


# Validate module paths at import time (no heavy deps triggered)
for _key, _entry in _CLF_REGISTRY.items():
    _module_path = _entry.split(":", maxsplit=1)[0]
    if importlib.util.find_spec(_module_path) is None:
        warnings.warn(
            f"Registry entry '{_key}': module '{_module_path}' not found.",
            ImportWarning,
            stacklevel=2,
        )


def read_classifier_factory(name: str, path: str, **kwargs) -> AbstractReadClassifier:
    """
    Factory function to load a read classifier from a checkpoint.

    Args:
        name: Identifier for the classifier type (e.g. "methylbert").
        path: Path to the saved classifier checkpoint.
        **kwargs: Additional arguments to pass to the classifier's load method.

    Returns:
        An instance of a subclass of AbstractReadClassifier loaded from the checkpoint.
    """
    if name not in _CLF_REGISTRY:
        raise ValueError(f"Unknown classifier '{name}'")
    module_path, cls_name = _CLF_REGISTRY[name].split(":")

    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls.load(path, **kwargs)
