import torch.nn as nn
from torchvision.models.vgg import VGG
from torchvision.models.vgg import make_layers

from ._base import EncoderMixin

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGEncoder(VGG, EncoderMixin):
    def __init__(self, out_channels, config, batch_norm=False, depth=5, **kwargs):
        super().__init__(make_layers(config, batch_norm=batch_norm), **kwargs)
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3
        del self.classifier

    def get_stages(self):
        stages = []
        stage_modules = []
        for module in self.features:
            if isinstance(module, nn.MaxPool2d):
                stages.append(nn.Sequential(*stage_modules))
                stage_modules = []
            stage_modules.append(module)
        stages.append(nn.Sequential(*stage_modules))
        return stages

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        keys = list(state_dict.keys())
        for k in keys:
            if k.startswith("classifier"):
                state_dict.pop(k, None)
        super().load_state_dict(state_dict, **kwargs)

    # -------- vgg11 --------
    # Encoder:
    #   name: VGGEncoder
    #   params:
    #     out_channels: &out_channels
    #       (64, 128, 256, 512, 512, 512)
    #     config: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    #     batch_norm: False
    # --------------------------
