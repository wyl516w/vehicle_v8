import torch
import ultralytics
from torch import nn
import torch.nn.functional as F
import math

class Conv2convweight(nn.Module):
    r"This class is used to convert the output of a convolutional kernel to the weight of another convolutional kernel"
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(Conv2convweight, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.convoutchannels = out_channels // (
            in_channels // (in_channels // math.gcd(in_channels, out_channels))
        )
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.convoutchannels,
            kernel_size=kernel_size,
            padding=0,
        )

    def forward(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        x = x.reshape(-1, 1, x.shape[2], x.shape[3])
        x = self.conv(x)
        x = x.reshape(
            batch_size, self.in_channels, self.convoutchannels, x.shape[2], x.shape[3]
        )
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batch_size, self.out_channels, -1, x.shape[3], x.shape[4])
        x = x.mean(dim=2, keepdim=False)
        return x


class Conv2convbias(nn.Module):
    r"This class is used to convert the convolutional kernel weight to the bias"
    def __init__(self, kernel_size: int = 3):
        super(Conv2convbias, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=kernel_size, padding=0
        )

    def forward(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        x = x.reshape(-1, 1, x.shape[2], x.shape[3])
        x = self.conv(x)
        x = x.reshape(batch_size, -1)
        return x


class FeatureConv(nn.Module):
    r"This class is used to extract features from the input image"
    def __init__(self):
        super(FeatureConv, self).__init__()  # 3x32x32
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1
        )  # 16x32x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x16x16
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=0
        )  # 32x14x14
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x7x7
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=0
        )  # 64x5x5

    def forward(self, x):
        x = self.conv(x)
        return x


class Supernet(nn.Module):
    def __init__(self, basenet: nn.Module, featurenet: nn.Module = FeatureConv()):
        super(Supernet, self).__init__()
        self.basenet = basenet
        self.featurenet = featurenet
        self.modellist = nn.ModuleList()
        for modelname, modelparams in self.basenet.parameters().items():
            if modelname == "conv2convweight":
                self.modellist.append(Conv2convweight(**modelparams))
            elif modelname == "conv2convbias":
                self.modellist.append(Conv2convbias(**modelparams))
            else:
                raise ValueError(f"Unknown model name: {modelname}")
            
from ultralytics.nn.modules import C2f