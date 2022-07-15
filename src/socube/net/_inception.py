# MIT License
#
# Copyright (c) 2022 Zhang.H.N
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn

from ._base import ModuleBase

__all__ = [
    "InceptionBlock2D", "InceptionBlock2DLayer"
]


class InceptionBlock2D(ModuleBase):
    """
    Inception block with multiple conv kernels

    Parameters
    ----------
    in_channels: integer scalar value
        input data's channel count
    out_channels: integer scalar value
        output data's channel count,
        it must be a multiple of `num`
    nums: integer scalar value
        Number of convolution kernels

    Examples
    ----------
    >>> iblock = InceptionBlock2D(48, 96, 3)
    """
    def __init__(self, in_channels: int, out_channels: int, nums: int = 3):
        super(InceptionBlock2D, self).__init__(in_channels, out_channels)
        assert isinstance(nums, int) and nums > 0
        assert out_channels % nums == 0, f"Out channels should be multiple of {nums}"
        out_channels //= nums
        self._inceptions = nn.ModuleList()
        for padding in range(nums):
            self._inceptions.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,
                              out_channels,
                              2 * padding + 1,
                              padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([inception(x) for inception in self._inceptions],
                         dim=1)


class InceptionBlock2DLayer(ModuleBase):
    """
    Inception block with multiple layers.

    Parameters
    ----------
    in_channels: integer scalar value
        input data's channel count
    out_channels: integer scalar value
        output data's channel count
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(InceptionBlock2DLayer, self).__init__(in_channels, out_channels)
        self.__extracts = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
            InceptionBlock2D(48, 96),
            nn.MaxPool2d(3, 2, padding=1),
            nn.Dropout(0.1),
            InceptionBlock2D(96, out_channels),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.__extracts(x)
