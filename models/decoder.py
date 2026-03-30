import torch
import torch.nn.functional as F
from torch import nn


def upsample(in_channels, out_channels, upscale, kernel_size=3):
    layers = []

    # Middle channels
    mid_channels = 32

    #First conv layer to reduce number of channels
    diff_conv1x1 = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=1, bias=False)
    nn.init.kaiming_normal_(diff_conv1x1.weight.data, nonlinearity='relu')
    layers.append(diff_conv1x1)

    #ReLU
    diff_relu = nn.ReLU()
    layers.append(diff_relu)

    #Upsampling to original size
    up      = nn.Upsample(scale_factor=upscale, mode='bilinear')
    layers.append(up)

    #Classification layer
    conv1x1 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
    nn.init.kaiming_normal_(conv1x1.weight.data, nonlinearity='relu')
    layers.append(conv1x1)

    return nn.Sequential(*layers)


class _PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes):
        super(_PSPModule, self).__init__()

        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(bin_sizes)), out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                       align_corners=False) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class Decoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(Decoder, self).__init__()
        self.psp = _PSPModule(2048, bin_sizes=[1, 2, 3, 6])
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x):
        x = self.psp(x)
        x = self.upsample(x)
        return x


class ReDecoder(nn.Module):
    def __init__(self, input_channels, output_channels=3):
        super(ReDecoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, output_channels, kernel_size=3, stride=1, padding=1)

        self.conv_res1 = nn.Conv2d(input_channels, 64, kernel_size=1, stride=1, padding=0)
        self.conv_res2 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.conv_res3 = nn.Conv2d(32, output_channels, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU(True)

    def forward(self, x):
        residual1 = self.conv_res1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = x + residual1

        residual2 = self.conv_res2(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x + residual2

        residual3 = self.conv_res3(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x + residual3
        return x
