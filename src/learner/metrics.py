"""
Module Description:
 
This script ....
 
 
Author:
 
Ahmed Telili
 
Date Created:
 
29 01 2024
 
"""

import torch
import math
import torch.nn.functional as F

class WSPSNRMetric:
    """Weighted-Spherical Peak Signal-to-Noise Ratio (WS-PSNR) Metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the metric."""
        self.total_wspnsr = 0.0
        self.count = 0

    def update(self, pred, target):
        """Update the metric with new predictions and targets."""
        wspnsr_value = self.ws_psnr(pred, target)
        self.total_wspnsr += wspnsr_value
        self.count += 1

    def get_result(self):
        """Return the average WS-PSNR."""
        return self.total_wspnsr / self.count if self.count != 0 else float('inf')

    @staticmethod
    def genERPWeights(height, device):
        """
        Generate ERP weights for each row in the image using PyTorch.
        """
        y_coords = torch.arange(height, dtype=torch.float32, device=device) - height / 2 + 0.5
        w_map = torch.cos(y_coords * math.pi / height)
        return w_map

    @staticmethod
    def compute_map_ws(height, width, device):
        """
        Compute the ERP weighting map for the entire image.
        """
        w_map_row = WSPSNRMetric.genERPWeights(height, device)
        w_map = w_map_row.view(1, height, 1).expand(-1, -1, width)
        return w_map

    @staticmethod
    def ws_psnr(image1, image2):
        """
        Calculate WS-PSNR between two batches of images.
        """
        batch_size, channels, height, width = image1.shape
        ws_psnr_values = torch.zeros(batch_size, device=image1.device)

        for i in range(batch_size):
            ws_map = WSPSNRMetric.compute_map_ws(height, width, image1.device)
            ws_map = ws_map.expand(channels, -1, -1)

            ws_mse = torch.sum((image1[i] - image2[i]) ** 2 * ws_map) / torch.sum(ws_map)
            if ws_mse == 0:
                ws_psnr_values[i] = float('inf')
            else:
                ws_psnr_values[i] = 20 * torch.log10(255.0 / torch.sqrt(ws_mse))

        return torch.mean(ws_psnr_values)


class WSSSIMMetric:
    """Weighted-Spherical Structural Similarity Index Measure (WS-SSIM) Metric."""

    def __init__(self, window_size=11, sigma=1.5):
        self.window_size = window_size
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the metric."""
        self.total_wsssim = 0.0
        self.count = 0

    def update(self, pred, target):
        """Update the metric with new predictions and targets."""
        wsssim_value = self.ws_ssim(pred, target, self.window_size, self.sigma)
        self.total_wsssim += wsssim_value
        self.count += 1

    def get_result(self):
        """Return the average WS-SSIM."""
        return self.total_wsssim / self.count if self.count != 0 else float('inf')

    @staticmethod
    def gaussian_kernel(size, sigma):
        """Generate a Gaussian kernel for SSIM calculation."""
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2

        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()

        return g.view(1, -1) * g.view(-1, 1)

    @staticmethod
    def ssim(img1, img2, window_size=11, sigma=1.5):
        """Calculate SSIM for a pair of images."""
        c1 = (0.01 * 255)**2
        c2 = (0.03 * 255)**2

        window = gaussian_kernel(window_size, sigma).to(img1.device)
        window = window.repeat(1, 1, 1, 1)

        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=1)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=1)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=1) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=1) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        return ssim_map.mean()

    @staticmethod
    def ws_ssim(pred, target, window_size, sigma):
        """Compute WS-SSIM between two images."""
        # Basic SSIM calculation
        basic_ssim = WSSSIMMetric.ssim(pred, target, window_size, sigma)

        # Apply weighting for WS-SSIM
        height, width = pred.shape[2:]
        ws_map = WSPSNRMetric.compute_map_ws(height, width).to(pred.device)
        ws_ssim = torch.sum(basic_ssim * ws_map) / torch.sum(ws_map)

        return ws_ssim

