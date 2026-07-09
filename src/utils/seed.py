"""
Reproducibility utilities for PapsAI XNet.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True, benchmark: bool = False) -> None:
    """
    Set random seeds for reproducible experiments.

    Parameters
    ----------
    seed : int
        Random seed value.

    deterministic : bool
        Whether to enforce deterministic CUDA operations.

    benchmark : bool
        Whether to allow cuDNN benchmarking.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = benchmark


def get_device(preferred_device: str = "cuda") -> torch.device:
    """
    Return CUDA device if available, otherwise CPU.
    """

    if preferred_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")
