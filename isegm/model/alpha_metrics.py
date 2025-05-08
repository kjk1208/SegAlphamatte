# isegm/modelalpha_metrics.py

import torch
import torch.nn.functional as F
import math

class AlphaMAE:
    def __call__(self, pred, target):
        """
        Mean Absolute Error
        :param pred: [B, 1, H, W]
        :param target: [B, 1, H, W]
        """
        return torch.mean(torch.abs(pred - target)).item()

    @property
    def name(self):
        return "AlphaMAE"


class AlphaMSE:
    def __call__(self, pred, target):
        """
        Mean Squared Error
        :param pred: [B, 1, H, W]
        :param target: [B, 1, H, W]
        """
        return torch.mean((pred - target) ** 2).item()

    @property
    def name(self):
        return "AlphaMSE"


class AlphaPSNR:
    def __call__(self, pred, target):
        """
        PSNR (Peak Signal to Noise Ratio)
        :param pred: [B, 1, H, W]
        :param target: [B, 1, H, W]
        """
        mse = torch.mean((pred - target) ** 2)
        if mse.item() == 0:
            return float('inf')
        return 20 * math.log10(1.0) - 10 * math.log10(mse.item())  # assuming pixel range is [0, 1]

    @property
    def name(self):
        return "AlphaPSNR"


class AlphaGradientError:
    def __call__(self, pred, target):
        """
        Gradient error (based on x and y directional differences)
        :param pred: [B, 1, H, W]
        :param target: [B, 1, H, W]
        """
        pred_dx = F.pad(pred[:, :, :, 1:] - pred[:, :, :, :-1], (0, 1))
        pred_dy = F.pad(pred[:, :, 1:, :] - pred[:, :, :-1, :], (0, 0, 0, 1))

        target_dx = F.pad(target[:, :, :, 1:] - target[:, :, :, :-1], (0, 1))
        target_dy = F.pad(target[:, :, 1:, :] - target[:, :, :-1, :], (0, 0, 0, 1))

        grad_diff = (pred_dx - target_dx).abs() + (pred_dy - target_dy).abs()
        return grad_diff.mean().item()

    @property
    def name(self):
        return "AlphaGradientError"
