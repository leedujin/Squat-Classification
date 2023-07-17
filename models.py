import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
    def __init__(self, in_channels):
        super(CNN, self).__init__()
        
        self.layer1 = nn.Conv2d(in_channels=in_channels, out_channels = 64, kernel_size = 5)
        self.layer2 = nn.Conv2d(in_channels=64, out_channels= 64, kernel_size = 5)
        self.layer3 = nn.Conv2d(in_channels=64, out_channels= 128, kernel_size = 5)
        self.pool1 = nn.AdaptiveMaxPool2d(128)
        self.layer4 = nn.Conv2d(in_channels=128, out_channels = 256, kernel_size = 3)
        self.layer5 = nn.Conv2d(in_channels=256, out_channels= 256, kernel_size = 3)
        self.layer6 = nn.Conv2d(in_channels=256, out_channels= 512, kernel_size = 3)
        self.pool2 = nn.AdaptiveMaxPool2d(512)

        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.pool1,
            self.layer4,
            self.layer5,
            self.layer6,
        )
        self.pool4 = nn.AdaptiveAvgPool2d(1)
        self.fcn = nn.Linear(512, 7)

    def forward(self, x):
        N, T, V, C = x.size()
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.pool4(x)
        x = x.view(N, -1)
        return self.fcn(x)
    

class Conv3D(nn.Module):
    def __init__(self, in_channels):
        super(CNN, self).__init__()
        
        self.layer1 = nn.Conv2d(in_channels=in_channels, out_channels = 64, kernel_size = 5)
        self.layer2 = nn.Conv2d(in_channels=64, out_channels= 64, kernel_size = 5)
        self.layer3 = nn.Conv2d(in_channels=64, out_channels= 128, kernel_size = 5)
        self.pool1 = nn.AdaptiveMaxPool2d(128)
        self.layer4 = nn.Conv2d(in_channels=128, out_channels = 256, kernel_size = 3)
        self.layer5 = nn.Conv2d(in_channels=256, out_channels= 256, kernel_size = 3)
        self.layer6 = nn.Conv2d(in_channels=256, out_channels= 512, kernel_size = 3)
        self.pool2 = nn.AdaptiveMaxPool2d(512)

        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.pool1,
            self.layer4,
            self.layer5,
            self.layer6,
        )
        self.pool4 = nn.AdaptiveAvgPool2d(1)
        self.fcn = nn.Linear(512, 7)

    def forward(self, x):
        N, T, V, C = x.size()
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.pool4(x)
        x = x.view(N, -1)
        return self.fcn(x)

