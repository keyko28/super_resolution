import torch


import numpy as np
import torch

class UnNormalize:
    """
    class to denormilze normalized input at the end
    """
    def __init__(self, mean: np.ndaarray, std: np.ndaarray):

        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor):
        """
        Args:
            tensor - Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor - Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)

        return tensor