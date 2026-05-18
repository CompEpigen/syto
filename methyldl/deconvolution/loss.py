# methyldl/deconvolution/losses.py

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ArrayLike = Union[torch.Tensor, np.ndarray]


SUPPORTED_LOSSES = {}


def register_loss(*names):
    def deco(cls):
        for n in names:
            SUPPORTED_LOSSES[n] = cls
        SUPPORTED_LOSSES[cls.__name__] = cls
        return cls

    return deco


class DeconvolutionLoss(ABC):
    """
    Base class for losses usable in PyTorch and Numpy pipelines

    Subclasses implement two methods:
      - `torch_loss(pred, target)` returning a scalar torch.Tensor (with grad).
      - `numpy_loss(pred, target)` returning a Python float.

    The `__call__` dispatches based on input type, so training loops can stay
    type-agnostic.
    """

    @abstractmethod
    def torch_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def numpy_loss(self, pred: np.ndarray, target: np.ndarray) -> float: ...

    def __call__(self, pred: ArrayLike, target: ArrayLike):
        if isinstance(pred, torch.Tensor):
            return self.torch_loss(pred, target)
        return self.numpy_loss(pred, target)

    @property
    def name(self) -> str:
        return self.__class__.__name__


@register_loss("combined_mse_kl")
class CombinedMSEKLLoss(DeconvolutionLoss):
    """
    MSE + KL(target || pred)
    """

    def __init__(
        self, mse_weight: float = 1.0, kl_weight: float = 0.5, eps: float = 1e-8
    ):
        self.mse_weight = mse_weight
        self.kl_weight = kl_weight
        self.eps = eps

    def torch_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        kl = (
            (
                target
                * (target.clamp(min=self.eps).log() - pred.clamp(min=self.eps).log())
            )
            .sum(dim=-1)
            .mean()
        )
        return self.mse_weight * mse + self.kl_weight * kl

    def numpy_loss(self, pred: np.ndarray, target: np.ndarray) -> float:
        assert pred.shape == target.shape
        mse = ((pred - target) ** 2).mean()
        t = np.clip(target, self.eps, None)
        p = np.clip(pred, self.eps, None)
        kl = (t * (np.log(t) - np.log(p))).sum(axis=-1).mean()
        return float(self.mse_weight * mse + self.kl_weight * kl)


def build_loss_from_config(**kwargs):
    criterion = kwargs.get("loss", "combined_mse_kl")
    if criterion in SUPPORTED_LOSSES.keys():
        if criterion == "combined_mse_kl":
            loss_weights = kwargs.get(
                "loss_weights", {"mse_weight": 1.0, "kl_weight": 0.5}
            )
            return SUPPORTED_LOSSES[criterion](
                mse_weight=loss_weights["mse_weight"],
                kl_weight=loss_weights["kl_weight"],
            )
        else:
            return SUPPORTED_LOSSES[criterion]()
    else:
        return ValueError(
            f"Wrong loss is specified. Must be one of {SUPPORTED_LOSSES.keys()}"
        )
