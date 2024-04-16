"""
Module Description:
 
This script describes metrics, WS-PSNR and WS-SSIM
 
 
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




