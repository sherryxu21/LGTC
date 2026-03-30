from models.backbones.resnet_backbone import ResNetBackbone
import torch
import torch.nn as nn
import os


pretrained_url = {
    "resnet50": "models/backbones/pretrained/3x3resnet50-imagenet.pth"
}


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        return self.sigmoid(avg_out) * x


class FrequencyAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = ChannelAttention(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # 转置为 [B, H, W, C]
        x = self.norm(x)
        xf = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')  #[B, H, W//2 + 1, C]
        xf_real = torch.real(xf)
        xf_imag = torch.imag(xf)
        xf_real = self.attn(xf_real)
        xf_imag = self.attn(xf_imag)
        xf_concat = torch.stack((xf_real, xf_imag), dim=-1)
        xf_reconstructed = torch.view_as_complex(xf_concat)
        x = torch.fft.irfft2(xf_reconstructed, dim=(-2, -1), norm='ortho')
        return x


class GFTBlock(nn.Module):
    def __init__(self, channels):
        super(GFTBlock, self).__init__()
        self.silu = nn.SiLU()
        self.fa = FrequencyAttention(channels // 2)
        self.conv = nn.Conv2d(channels // 2, channels, kernel_size=1, padding=0, bias=False)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False)
        )
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        x1, x2 = torch.split(self.norm1(x), c // 2, dim=1)
        x = x + self.conv(self.silu(x1) * self.fa(x2))
        x = x + self.feed_forward(self.norm2(x))
        return x


class LGIEncoder(nn.Module):
    def __init__(self, pretrained):
        super(LGIEncoder, self).__init__()
        if pretrained and not os.path.isfile(pretrained_url["resnet50"]):
            print("Downloading pretrained resnet (source : https://github.com/donnyyou/torchcv)")
            os.system('sh models/backbones/get_resnet50_pretrained_model.sh')
        self.model = ResNetBackbone(backbone='deepbase_resnet50_dilated8', pretrained=pretrained)

        # 四个阶段，每个阶段由下采样的卷积层和多个SFBlock组成
        self.stage1_gftblocks = self._make_gftblocks(128, num_blocks=1)
        self.stage1_downsample = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.stage2_gftblocks = self._make_gftblocks(256, num_blocks=1)
        self.stage2_downsample = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.stage3_gftblocks = self._make_gftblocks(512, num_blocks=1)
        self.stage3_downsample = nn.Conv2d(512, 1024, kernel_size=3, padding=1)

        self.stage4_gftblocks = self._make_gftblocks(1024, num_blocks=1)
        self.stage4_downsample = nn.Conv2d(1024, 2048, kernel_size=3, padding=1)

    def _make_gftblocks(self, in_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(GFTBlock(in_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        c0 = self.model.prefix(x)
        c0 = self.model.maxpool(c0)  # [8,128,64,64]

        c1 = self.model.layer1(c0)  # [8,256,64,64]
        f1 = self.stage1_downsample(self.stage1_gftblocks(c0))
        c1 = c1 + f1

        c2 = self.model.layer2(c1)  # [8,512,32,32]
        f2 = self.stage2_downsample(self.stage2_gftblocks(c1))
        c2 = c2 + f2

        c3 = self.model.layer3(c2)  # [8,1024,32,32]
        f3 = self.stage3_downsample(self.stage3_gftblocks(c2))
        c3 = c3 + f3

        c4 = self.model.layer4(c3)  # [8,2048,32,32]
        f4 = self.stage4_downsample(self.stage4_gftblocks(c3))
        c4 = c4 + f4
        return c1, c2, c3, c4
