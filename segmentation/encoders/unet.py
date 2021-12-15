import loguru
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation.encoders._base import EncoderMixin


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2))
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, out_channels=(3, 64, 128, 256, 512, 1024), n_class=1):
        super(UNet, self).__init__()
        self.inc = DoubleConv(out_channels[0], out_channels[1])  # 64x512x512
        self.down1 = Down(out_channels[1], out_channels[2])      # 128x256x256
        self.down2 = Down(out_channels[2], out_channels[3])      # 256x128x128
        self.down3 = Down(out_channels[3], out_channels[4])      # 512x64x64
        self.down4 = Down(out_channels[4], out_channels[5])      # 1024x32x32
        self.up1 = Up(out_channels[5], out_channels[4])          # 512x64x64
        self.up2 = Up(out_channels[4], out_channels[3])          # 256x128x128
        self.up3 = Up(out_channels[3], out_channels[2])          # 128x256x256
        self.up4 = Up(out_channels[2], out_channels[1])          # 64x512x512
        self.outc = OutConv(out_channels[1], n_class)            # 1x512x512

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNetEncoder(UNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

    def get_stages(self):
        return [
            nn.Identity(),             # input
            self.inc,                  # h1
            self.down1,                # h2
            self.down2,                # h3
            self.down3,                # h4
            self.down4,                # h5
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)


from loguru import logger

if __name__ == "__main__":
    input = torch.rand(4, 3, 512, 512)
    model = UNet()
    output = model(input)
    logger.info(output.shape)
