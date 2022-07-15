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

from socube.net import (NetBase, InceptionBlock2DLayer)

__all__ = ["SoCubeNet"]


class SoCubeNet(NetBase):
    """
    Neural network model constructed for doublet detection task. previous name
    is Conv2DClassifyNet.

    Parameters
    ----------
    in_channels: integer scalar value
        input data's channel count
    out_channels: integer scalar value
        output data's channel count
    freeze: boolean value
        Whether to freeze the feature extraction layer, default is `False`
    binary: boolean value
        Whether the output probability is binary or multicategorical, default is `True` for binary

    Examples
    ----------
    >>>  SoCubeNet(10, 2)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 freeze: bool = False,
                 binary: bool = True,
                 **kwargs):
        super(SoCubeNet, self).__init__(in_channels, out_channels, **kwargs)
        self._feature = InceptionBlock2DLayer(in_channels, 192)
        self._binary = binary and (out_channels == 2)
        if self._binary:
            out_channels = 1
        if freeze:
            self.freeze(self._feature)

        self._classify = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, out_channels))
        if self._binary:
            self._classify.add_module(str(len(self._classify)), nn.Flatten(0))
            self._classify.add_module(str(len(self._classify)), nn.Sigmoid())
        else:
            self._classify.add_module(str(len(self._classify)), nn.Softmax(1))

    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        hidden = self._feature.forward(x1)
        return self._classify(hidden)

    def criterion(self, yPredict: torch.Tensor,
                  yTrue: torch.Tensor) -> torch.Tensor:
        if self._binary:
            predictLoss = nn.BCELoss()(yPredict, yTrue.float())
        else:
            predictLoss = nn.CrossEntropyLoss()(yPredict, yTrue)
        return predictLoss
