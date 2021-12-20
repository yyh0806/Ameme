import torch.nn as nn

from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck

from ._base import EncoderMixin

from loguru import logger


class ResNetEncoder(ResNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        del self.fc
        del self.avgpool

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
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

    # -------- resnet50 --------
    # Encoder:
    #   name: ResNetEncoder
    #   params:
    #     depth: 5
    #     out_channels: &out_channels
    #       (3, 64, 256, 512, 1024, 2048)
    #     block: Bottleneck
    #     layers: [3, 4, 6, 3]
    # --------------------------

    # -------- resnet101 -------
    # Encoder:
    #   name: ResNetEncoder
    #   params:
    #     depth: 5
    #     out_channels: &out_channels
    #       (3, 64, 256, 512, 1024, 2048)
    #     block: Bottleneck
    #     layers: [3, 4, 23, 3]
    # --------------------------

    # -------- resnet152 -------
    # Encoder:
    #   name: ResNetEncoder
    #   params:
    #     depth: 5
    #     out_channels: &out_channels
    #       (3, 64, 256, 512, 1024, 2048)
    #     block: Bottleneck
    #     layers: [3, 8, 36, 3]
    # --------------------------