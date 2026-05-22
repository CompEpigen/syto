# https://github.com/lucidrains/minGRU-pytorch/blob/main/minGRU_pytorch/minGRU.py
import torch
import torch.nn.functional as F


def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())


def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))
