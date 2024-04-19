from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_padding(kernel_size: int, dilation: int) -> int:
    """Padding mode = `same`"""

    if isinstance(kernel_size, int):
        return (kernel_size - 1) // 2 * dilation
    elif isinstance(kernel_size, tuple) and len(kernel_size) == 2:
        return (kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation
    else:
        raise ValueError("Kernel size must be an integer or a tuple of two integers.")

class OutConv(nn.Module):
    """Ouput Convolutional Block"""
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return self.conv(x)  

class Conv(nn.Module):
    """Convolutional Block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Optional[int] = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        dilation: int = 1,
        bias: bool = False,
        act: bool = True,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = calculate_padding(kernel_size, dilation)
            self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class DoubleConv(nn.Module):
    """There are double Convolutional Block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        kernel_size: Optional[int] = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        act: bool = True,
    ) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            act=act,
        )
        self.conv2 = Conv(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            act=act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Down(nn.Module):
    """Downscaling with MaxPool then apply Double Convolutional Block"""

    def __init__(self, in_channels: int, out_channels: int, factor: int = 2) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=factor)
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x

class Up(nn.Module):
    """Upscaling then apply Double Convolutional Block"""

    def __init__(self, in_channels: int, out_channels: int, factor: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2, stride=factor
        )
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:        
        x1 = self.up(x1)

        # solution to padding issues
        # this issues from if you change image and mask sizes
        ############# starts #################
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        ############# ends #################
        
        x_ = torch.cat([x2, x1], dim=1)
        return self.conv(x_)

class UNet(nn.Module):
    """UNet Segmentation Model"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_conv = DoubleConv(in_channels, out_channels=64)

        # Downscaling 
        self.down1 = Down(in_channels=64, out_channels=128, factor=2)  # P/2
        self.down2 = Down(in_channels=128, out_channels=256, factor=2)  # P/4
        self.down3 = Down(in_channels=256, out_channels=512, factor=2)  # P/8
        self.down4 = Down(in_channels=512, out_channels=1024, factor=2)  # P/16

        # Upscaling
        self.up1 = Up(in_channels=1024, out_channels=512, factor=2)
        self.up2 = Up(in_channels=512, out_channels=256, factor=2)
        self.up3 = Up(in_channels=256, out_channels=128, factor=2)
        self.up4 = Up(in_channels=128, out_channels=64, factor=2)
      
        self.output_conv = (OutConv(in_channels=64, out_channels=out_channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.input_conv(x)

        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        x = self.output_conv(x)

        return x