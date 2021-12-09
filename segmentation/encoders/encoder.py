from torchvision.models.resnet import BasicBlock, Bottleneck
from segmentation.encoders.resnet import ResNetEncoder
from segmentation.encoders.unet import UNetEncoder


def Encoder(name, in_channels=3, weights=None, output_stride=32, **params):
    p = {}
    for k, v in dict(params["params"]).items():
        p[k] = eval(v) if isinstance(v, str) else v
    encoder = eval(name)(**p)
    encoder.set_in_channels(in_channels, pretrained=weights is not None)
    if output_stride != 32:
        encoder.make_dilated(output_stride)
    return encoder
