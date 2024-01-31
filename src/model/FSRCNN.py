"""
Module Description:
 
This script defines th FSRCNN model
 
Original Author:
    Lorna
    Original Source: https://github.com/Lornatang/FSRCNN-PyTorch

 
"""
from math import sqrt
import torch
from torch import nn

class FSRCNN(nn.Module):
    """
    Args:
        upscale_factor (int): Image magnification factor. Default is 4.
    """

    def __init__(self, upscale_factor: int = 4) -> None:
        super(FSRCNN, self).__init__()
        # Feature extraction layer.
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, 56, (5, 5), (1, 1), (2, 2)),
            nn.PReLU(56)
        )

        # Shrinking layer.
        self.shrink = nn.Sequential(
            nn.Conv2d(56, 12, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(12)
        )

        # Mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12)
        )

        # Expanding layer.
        self.expand = nn.Sequential(
            nn.Conv2d(12, 56, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(56)
        )

        # Deconvolution layer.
        self.deconv = nn.ConvTranspose2d(
            56, 3, (9, 9),
            (upscale_factor, upscale_factor),
            (4, 4),
            (upscale_factor - 1, upscale_factor - 1)
        )
        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.feature_extraction(x)
        out = self.shrink(out)
        out = self.map(out)
        out = self.expand(out)
        out = self.deconv(out)

        return out

    # Initialize weights
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(
                    m.weight.data,
                    mean=0.0,
                    std=sqrt(2 / (m.out_channels * m.weight.data[0][0].numel()))
                )
                nn.init.zeros_(m.bias.data)

        nn.init.normal_(self.deconv.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.deconv.bias.data)


if __name__ == '__main__':
    height = 64
    width = 64
    model = FSRCNN()

    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)


    x = torch.randn((16, 3, height, width))
    x = model(x)
    print(x.shape)
