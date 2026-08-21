"""Single source of truth for the torch device Syto runs on.

Importing this module also teaches torch how to load a GPU-saved checkpoint on
a machine that has no GPU, which is why it is imported for its side effect by
every module that deserializes torch state.
"""

import logging

import torch

#: The device every model is loaded onto and every tensor is built on.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_CPU_NOTICE = "Inference on CPU is slow. Consider using a GPU-enabled machine."


def _never_tag(_storage):
    """Leave save-time tagging alone; we only override the load direction."""
    return None


def _restore_cuda_on_cpu(storage, location):
    """Land a CUDA-tagged storage on the CPU when there is no CUDA.

    A checkpoint saved on a GPU carries a ``cuda:N`` tag, and ``torch.load``
    honours it unless the caller passes ``map_location``.  Anything unpickled
    outside ``torch.load`` -- a joblib-saved calibrator, say -- has no such
    caller, so on a CPU-only machine it fails hard instead of falling back.
    Running ahead of torch's own CUDA deserializer turns that failure into a
    CPU load; where CUDA does exist this returns None and the normal path
    takes over.
    """
    if location.startswith("cuda") and not torch.cuda.is_available():
        return torch.serialization.default_restore_location(storage, "cpu")
    return None


# Priority 15 puts this between torch's CPU (10) and CUDA (20) handlers.
torch.serialization.register_package(15, _never_tag, _restore_cuda_on_cpu)


def warn_if_cpu(logger: logging.Logger) -> None:
    """Note that a transformer classifier is about to run without a GPU.

    MethylBERT and DNABERT-2 are not blocked on CPU, only slow, so this is a
    warning.
    """
    if DEVICE == "cpu":
        logger.warning(_CPU_NOTICE)
