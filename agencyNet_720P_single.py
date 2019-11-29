import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import time
import math

drop_rate = 0.5
width = 1280
height = 720

class Basic_layer(nn.Module):
    """docstring for Basic_layer"""
    def __init__(self, in_channels, out_channels):
        super(Basic_layer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 5, padding = 2),
            nn.Dropout(p = drop_rate),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
            )

    def forward(self, input_x):
        x = self.layer(input_x)

        return x
        

class CNN_Net(nn.Module):
    """docstring for CNN_Net"""
    def __init__(self, in_channels, out_channels):
        super(CNN_Net, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Conv_layer = nn.Sequential(
            Basic_layer(in_channels, 16),
            Basic_layer(16, 32),
            Basic_layer(32, 64)
            )
        self.Linear_layer = nn.Sequential(
            nn.Linear(64 * 160 * 90, 1)
            )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def forward(self, input_x):
        x = self.Conv_layer(input_x)
        x = x.view(-1, 64 * 160 * 90)
        x = self.Linear_layer(x)

        return x


