from segmentation.decoders.unet import UNetDecoder
from segmentation.decoders.fpn import FPNDecoder
from segmentation.decoders.unetplusplus import UNetPlusPlusDecoder


def Decoder(name, **params):
    p = {}
    for k, v in dict(params["params"]).items():
        p[k] = eval(v) if isinstance(v, str) else v
    decoder = eval(name)(**p)
    return decoder
