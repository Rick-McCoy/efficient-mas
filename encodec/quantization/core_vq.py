# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This implementation is inspired from
# https://github.com/lucidrains/vector-quantize-pytorch
# which is released under MIT License. Hereafter, the original license:
# MIT License
#
# Copyright (c) 2020 Phil Wang
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

"""Core vector quantization implementation."""

from typing import cast

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and
            use the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at
            initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any
            codes that have an exponential moving average cluster size less than the
            specified threshold with randomly selected vector from the current batch.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        init_fn = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer("embed", embed)
        self.embed: Tensor

    def preprocess(self, x: Tensor):
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x: Tensor) -> Tensor:
        dist = torch.cdist(x.unsqueeze(0), self.embed.unsqueeze(0))
        embed_ind = dist.argmin(dim=-1).squeeze(0)
        return embed_ind

    def dequantize(self, embed_ind: Tensor):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x: Tensor):
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = embed_ind.view(shape[:-1])
        return embed_ind

    def decode(self, embed_ind: Tensor):
        quantize = self.dequantize(embed_ind)
        return quantize


class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    Currently supports only euclidean distance.
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified
            dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any
            codes that have an exponential moving average cluster size less than the
            specified threshold with randomly selected vector from the current batch.
        commitment_weight (float): Weight for commitment loss.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: int | None = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 1.0,
    ):
        super().__init__()
        _codebook_dim = codebook_dim or dim

        requires_projection = _codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity()
        )

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self._codebook = EuclideanCodebook(
            dim=_codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x: Tensor):
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind: Tensor) -> Tensor:
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)

        quantize, embed_ind = self._codebook(x)

        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, *, num_quantizers: int, **kwargs):
        super().__init__()
        self.layers = cast(
            list[VectorQuantization],
            nn.ModuleList(VectorQuantization(**kwargs) for _ in range(num_quantizers)),
        )

    def forward(self, x: Tensor, n_q: int | None = None):
        quantized_out = 0.0
        residual = x

        all_indices = []

        n_q = len(self.layers) if n_q is None else n_q

        for i, layer in enumerate(self.layers):
            quantized, indices = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            if i + 1 == n_q:
                out_indices = torch.stack(all_indices)
                return quantized_out, out_indices

        raise ValueError("n_q must be less than or equal to the number of quantizers")

    def encode(self, x: Tensor, n_q: int | None = None) -> Tensor:
        residual = x
        all_indices = []
        n_q = len(self.layers) if n_q is None else n_q
        for i, layer in enumerate(self.layers):
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
            if i + 1 == n_q:
                out_indices = torch.stack(all_indices)
                return out_indices

        raise ValueError("n_q must be less than or equal to the number of quantizers")

    def decode(self, q_indices: Tensor) -> Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, layer in enumerate(self.layers):
            quantized = layer.decode(q_indices[i])
            quantized_out = quantized_out + quantized
            if i + 1 == q_indices.shape[0]:
                return quantized_out

        raise ValueError("n_q must be less than or equal to the number of quantizers")
