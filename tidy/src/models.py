# U-Net architecture, powerful CNN specially designed for Semantic Segmentation tasks i.e,. road lane detection.

import torch
import torch.nn as nn


class DoubleConv(nn.Module):  # This class implements the basic block used throughout the U-Net.
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), # 1st block convolutional.
            nn.BatchNorm2d(out_ch),  # 1st block normalization.
            nn.ReLU(inplace=True),  # 1st block activation function.

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),  # 2nd block -- repeat of 1st convolutional.
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):  # Builds the full network, 
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)  # Encoder -- capture context by reducing spatial dimension and increasing the number of feature channels.
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 4, base * 8)  # Bottleneck -- Connecting encoder and decoder part of the network. Applies one final doubleconv block.

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)  # Decoder -- precisely localize the features by increasing the spatial dimensions and reducing the no.of feature channels.
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)  # Output layer -- The final layer is standard 1 x 1 convolution. It maps the final feature maps to the desired no.of output channels. 

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)
