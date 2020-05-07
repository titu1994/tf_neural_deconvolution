import torch
import torch.nn as nn
import torch.nn.functional as F

from pt_deconv import FastDeconv1D, FastDeconv2D, ChannelDeconv1D, ChannelDeconv2D


class SimpleNet1D(nn.Module):

    def __init__(self, num_classes, num_channels=64, groups=1, channel_deconv_loc="pre", blocks=64):
        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_channels
        self.channel_deconv_loc = channel_deconv_loc

        self.conv1 = FastDeconv1D(3, num_channels, kernel_size=3, stride=2,
                                  groups=1, padding=1,
                                  n_iter=5, momentum=0.1, block=blocks)

        self.conv2 = FastDeconv1D(num_channels, num_channels, kernel_size=3, stride=2,
                                  groups=groups, padding=1,
                                  n_iter=5, momentum=0.1, block=blocks)

        self.final_conv = ChannelDeconv1D(block=num_channels, momentum=0.1)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Linear(num_channels, num_classes)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)

        if self.channel_deconv_loc == 'pre':
            x = self.final_conv(x)
            x = self.gap(x)
        else:
            x = self.gap(x)
            x = self.final_conv(x)

        x = x.view(-1, self.num_channels)
        x = self.clf(x)
        return x


class SimpleNet2D(nn.Module):

    def __init__(self, num_classes, num_channels=64, groups=1, channel_deconv_loc="pre", blocks=64):
        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_channels
        self.channel_deconv_loc = channel_deconv_loc

        self.conv1 = FastDeconv2D(3, num_channels, kernel_size=(3, 3), stride=(2, 2),
                                  groups=1, padding=1,
                                  n_iter=5, momentum=0.9, block=blocks)

        self.conv2 = FastDeconv2D(num_channels, num_channels, kernel_size=(3, 3), stride=(2, 2),
                                  groups=groups, padding=1,
                                  n_iter=5, momentum=0.9, block=blocks)

        self.final_conv = ChannelDeconv2D(block=num_channels, momentum=0.1)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.clf = nn.Linear(num_channels, num_classes)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)

        if self.channel_deconv_loc == 'pre':
            x = self.final_conv(x)
            x = self.gap(x)
        else:
            x = self.gap(x)
            x = self.final_conv(x)

        x = x.view(-1, self.num_channels)
        x = self.clf(x)
        return x
