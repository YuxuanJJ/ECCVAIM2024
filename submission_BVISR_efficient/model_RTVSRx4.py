# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math

import torch
from torch import nn
import torch.nn.functional as F




class ResidualConvBlock(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualConvBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)


class UpsampleBlock(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(UpsampleBlock, self).__init__(*m)


class Generator(nn.Module):
    def __init__(self, upscale_factor=4) -> None:
        super(Generator, self).__init__()
        self.img_range = 1.

        num_c = 24
        num_b = 3
        res_scale = 2

        # First layer
        self.PixelunShuffle = nn.PixelUnshuffle(2)
        self.conv_first = nn.Conv2d(3 * 4, num_c, (3, 3), (1, 1), (1, 1))

        # Residual blocks
        body = []
        for _ in range(num_b):
            body.append(ResidualConvBlock(num_c, res_scale))
        self.body = nn.Sequential(*body)

        # Second layer
        self.conv_after_body = nn.Conv2d(num_c, num_c, (3, 3), (1, 1), (1, 1))

        # Upsampling layers

        self.upsample1 = UpsampleBlock(scale=2, num_feat=num_c)

        self.upsample2 = nn.Sequential(
            UpsampleOneStep(scale=4, num_feat=num_c, num_out_ch=1),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frame = x
        input = x
        x = self.PixelunShuffle(input)  # ds by 2
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x
        x = self.upsample1(res)
        YUV_n = F.interpolate(input, scale_factor=4, mode='nearest', recompute_scale_factor=True)
        x = self.upsample2(x) + YUV_n[:, 0:1, :, :]

        YUV = F.interpolate(input, scale_factor=4, mode='bicubic', recompute_scale_factor=True)
        UV = YUV[:, 1:3, :, :]

        return torch.cat([x, UV], dim=1)


    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
