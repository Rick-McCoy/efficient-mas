# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""LSTM layers module."""

from torch import Tensor, nn

from encodec.typing import ConvolutionalTensor, LSTMTensor


class LSTM(nn.LSTM):
    def __call__(self, x: LSTMTensor) -> tuple[LSTMTensor, tuple[Tensor, Tensor]]:
        return super().__call__(x)


class SLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """

    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = LSTM(dimension, dimension, num_layers)

    def __call__(self, x: ConvolutionalTensor) -> ConvolutionalTensor:
        return super().__call__(x)

    def forward(self, x: ConvolutionalTensor) -> ConvolutionalTensor:
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y
