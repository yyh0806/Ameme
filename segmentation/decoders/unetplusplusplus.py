import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from ..base import modules as md

from loguru import logger

# TODO


class CBR(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1):
        super(CBR, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.padding = padding

        self.cbr = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                      kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.cbr.cuda()(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        od_block = collections.OrderedDict()
        n_blocks = 5
        for i in range(1, n_blocks):
            scale = 2 ** i
            od_block['up_{0}'.format(scale)] = nn.Upsample(scale_factor=scale)
            od_block['down_{0}'.format(scale)] = nn.MaxPool2d(scale, scale, ceil_mode=True)
        od_block['cbr'] = CBR(in_channels=in_channels, out_channels=out_channels)
        self.module_dict = nn.ModuleDict(od_block)

    def forward(self, x, choice, scale):
        if choice in ['down', 'up']:
            x = self.module_dict["{0}_{1}".format(choice, scale)](x)
        x = self.module_dict['cbr'](x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, scales: list[tuple], cat_channels: list, encoder_channels):
        super(DecoderBlock, self).__init__()
        assert len(scales) == len(encoder_channels)
        self.scales = scales                        # [('down', 8), ('up', 4)]
        self.encoder_channels = encoder_channels    # (64, 128, 256, 512, 1024)
        od_block = collections.OrderedDict()
        upper_channel = 320
        cat_channel = 64
        for i in range(len(encoder_channels)):
            in_ch = encoder_channels[i]
            if i in cat_channels:
                in_ch = upper_channel
            od_block['block_{0}'.format(i)] = BasicBlock(in_channels=in_ch, out_channels=cat_channel)

        od_block['out_cbr'] = CBR(in_channels=upper_channel, out_channels=upper_channel)
        self.module_dict = nn.ModuleDict(od_block)

    def forward(self, x1, x2, x3, x4, x5):
        decode_list = []
        for idx, x in enumerate([x1, x2, x3, x4, x5]):
            feature = self.module_dict['block_{0}'.format(idx)](x, self.scales[idx][0], self.scales[idx][1])
            decode_list.append(feature)
        feature_cat = torch.cat(decode_list, 1)
        y = self.module_dict['out_cbr'](feature_cat)
        return y


class UNetPlusPlusPlusDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )
        encoder_channels = encoder_channels[1:]  # (64, 128, 256, 512, 1024)
        self.cat_channels = 64
        self.upper_channels = self.cat_channels * 5
        self.decoderBlock4 = DecoderBlock([('down', 8), ('down', 4), ('down', 2), ('cbr', 0), ('up', 2)], [], encoder_channels)
        self.decoderBlock3 = DecoderBlock([('down', 4), ('down', 2), ('cbr', 0), ('up', 2), ('up', 4)], [3], encoder_channels)
        self.decoderBlock2 = DecoderBlock([('down', 2), ('cbr', 0), ('up', 2), ('up', 4), ('up', 8)], [2, 3], encoder_channels)
        self.decoderBlock1 = DecoderBlock([('cbr', 0), ('up', 2), ('up', 4), ('up', 8), ('up', 16)], [1, 2, 3], encoder_channels)

    def forward(self, *features):
        h1, h2, h3, h4, f5 = features[-5:]
        f4 = self.decoderBlock4(h1, h2, h3, h4, f5)
        f3 = self.decoderBlock3(h1, h2, h3, f4, f5)
        f2 = self.decoderBlock2(h1, h2, f3, f4, f5)
        f1 = self.decoderBlock1(h1, f2, f3, f4, f5)
        return f1


# class UNetPPPPDecoder(nn.Module):
#     def __init__(
#             self,
#             encoder_channels,
#             decoder_channels,
#             n_blocks=5
#     ):
#         super().__init__()
#         self.encoder_channels = encoder_channels[1:]
#         self.decoder_channels = decoder_channels
#         self.cat_channels = 64
#         if n_blocks != len(decoder_channels):
#             raise ValueError(
#                 "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
#                     n_blocks, len(decoder_channels)
#                 )
#             )
#
#     def forward(self, *features):
#         h1, h2, h3, h4, f5 = features[-5:]
#         h1_1 = BasicBlock(in_channels=h1.shape[1], out_channels=self.cat_channels)(h1, 'down', 8)
#         h1_2 = BasicBlock(in_channels=h2.shape[1], out_channels=self.cat_channels)(h2, 'down', 4)
#         h1_3 = BasicBlock(in_channels=h3.shape[1], out_channels=self.cat_channels)(h3, 'down', 2)
#         h1_4 = BasicBlock(in_channels=h4.shape[1], out_channels=self.cat_channels)(h4, 'cbr', 0)
#         f1 = torch.cat([h1_1, h1_2, h1_3, h1_4], dim=1)
#         h2_1 = BasicBlock(in_channels=h1.shape[1], out_channels=self.cat_channels)(h1, 'down', 4)
#         h2_2 = BasicBlock(in_channels=h2.shape[1], out_channels=self.cat_channels)(h2, 'down', 2)
#         h2_3 = BasicBlock(in_channels=h3.shape[1], out_channels=self.cat_channels)(h3, 'cbr', 0)
#         f2 = torch.cat([h2_1, h2_2, h2_3], dim=1)
#         h3_1 = BasicBlock(in_channels=h1.shape[1], out_channels=self.cat_channels)(h1, 'down', 2)
#         h3_2 = BasicBlock(in_channels=h2.shape[1], out_channels=self.cat_channels)(h2, 'cbr', 0)
#         f3 = torch.cat([h3_1, h3_2], dim=1)
#         f4 = BasicBlock(in_channels=h1.shape[1], out_channels=self.cat_channels)(h1, 'cbr', 0)
#
#         u1 = nn.Upsample(scale_factor=8)(f1)
#         u2 = nn.Upsample(scale_factor=4)(f2)
#         u3 = nn.Upsample(scale_factor=2)(f3)
#
#         fea = torch.cat([u1, u2, u3, f4], dim=1)
#
#         return fea