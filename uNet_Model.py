import os
import glob
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import matplotlib.pyplot as plt # type: ignore

from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


print(f"PyTorch version: {torch.__version__}")


#UNet Model
class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):

        return self.block(x)


#encoder block of UNet
class DownBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.MaxPool2d(
                kernel_size=2
            ),

            DoubleConv(
                in_channels,
                out_channels
            )
        )

    def forward(self, x):

        return self.block(x)


#decoder block of UNet
class UpBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(
            in_channels // 2 + skip_channels,
            out_channels
        )

    def forward(
        self,
        x,
        skip
    ):

        x = self.up(x)

        # Handle possible size differences
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        x = F.pad(
            x,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2
            ]
        )

        x = torch.cat(
            [skip, x],
            dim=1
        )

        return self.conv(x)



#complete U-Net
class UNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        num_classes=11
    ):

        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(
            in_channels,
            64
        )

        self.enc2 = DownBlock(
            64,
            128
        )

        self.enc3 = DownBlock(
            128,
            256
        )

        self.enc4 = DownBlock(
            256,
            512
        )

        # Bottleneck
        self.bottleneck = DownBlock(
            512,
            1024
        )

        # Decoder
        self.dec4 = UpBlock(
            1024,
            512,
            512
        )

        self.dec3 = UpBlock(
            512,
            256,
            256
        )

        self.dec2 = UpBlock(
            256,
            128,
            128
        )

        self.dec1 = UpBlock(
            128,
            64,
            64
        )

        # Output
        self.output = nn.Conv2d(
            64,
            num_classes,
            kernel_size=1
        )



    def forward(self, x, return_features=False):

    # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

    # Bottleneck
        b = self.bottleneck(e4)

    # Decoder
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        logits = self.output(d1)

        #if return_features is True, return the logits and the bottleneck features
        if return_features: 
            return logits, b

        return logits
