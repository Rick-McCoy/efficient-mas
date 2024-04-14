# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EnCodec model implementation."""

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from encodec.modules.seanet import SEANetDecoder, SEANetEncoder
from encodec.quantization.vq import ResidualVectorQuantizer
from encodec.utils import _check_checksum, _get_checkpoint_url, _linear_overlap_add
from util.typing import AudioChannelTensor, AudioCodeTensor, AudioSegmentTensor

ROOT_URL = "https://dl.fbaipublicfiles.com/encodec/v0/"


class EncodecModel(nn.Module):
    """EnCodec model operating on the raw waveform.
    Args:
        target_bandwidths (list of float): Target bandwidths.
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        normalize (bool): Whether to apply audio normalization.
        segment (float or None): segment duration in sec. when doing overlap-add.
        overlap (float): overlap between segment, given as a fraction of the segment
                         duration.
        name (str): name of the model, used as metadata when compressing audio.
    """

    def __init__(
        self,
        encoder: SEANetEncoder,
        decoder: SEANetDecoder,
        quantizer: ResidualVectorQuantizer,
        target_bandwidths: list[float],
        sample_rate: int,
        channels: int,
        normalize: bool = False,
        segment: float | None = None,
        overlap: float = 0.01,
        name: str = "unset",
    ):
        super().__init__()
        self.bandwidth = 0.0
        self.target_bandwidths = target_bandwidths
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.sample_rate = sample_rate
        self.channels = channels
        self.normalize = normalize
        self.segment = segment
        self.overlap = overlap
        self.frame_rate = math.ceil(self.sample_rate / np.prod(self.encoder.ratios))
        self.name = name
        self.bits_per_codebook = int(math.log2(self.quantizer.bins))
        assert (
            2**self.bits_per_codebook == self.quantizer.bins
        ), "quantizer bins must be a power of 2."

    @property
    def segment_length(self) -> int | None:
        if self.segment is None:
            return None
        return int(self.segment * self.sample_rate)

    @property
    def segment_stride(self) -> int | None:
        segment_length = self.segment_length
        if segment_length is None:
            return None
        return max(1, int((1 - self.overlap) * segment_length))

    def encode(self, x: AudioChannelTensor):
        """Given a tensor `x`, returns a list of frames containing
        the discrete encoded codes for `x`, along with rescaling factors
        for each segment, when `self.normalize` is True.

        Each frames is a tuple `(codebook, scale)`, with `codebook` of
        shape `[B, K, T]`, with `K` the number of codebooks.
        """
        _, channels, length = x.shape
        assert channels > 0 and channels <= 2
        segment_length = self.segment_length
        if segment_length is None:
            segment_length = length
            stride = length
        else:
            stride = self.segment_stride
            assert stride is not None

        encoded_frames = [
            self._encode_frame(x[:, :, offset : offset + segment_length])
            for offset in range(0, length, stride)
        ]
        return encoded_frames

    def _encode_frame(
        self, x: AudioSegmentTensor
    ) -> tuple[AudioCodeTensor, Tensor | None]:
        length = x.shape[-1]
        duration = length / self.sample_rate
        segment = self.segment
        # assert segment is None or duration <= 1e-5 + segment
        if segment is not None:
            assert duration <= 1e-5 + segment

        if self.normalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None

        emb = self.encoder(x)
        codes = self.quantizer.encode(emb, self.frame_rate, self.bandwidth)
        codes = codes.transpose(0, 1)
        # codes is [B, K, T], with T frames, K nb of codebooks.
        return codes, scale

    def decode(
        self, encoded_frames: list[tuple[AudioCodeTensor, Tensor | None]]
    ) -> AudioChannelTensor:
        """Decode the given frames into a waveform.
        Note that the output might be a bit bigger than the input. In that case,
        any extra steps at the end can be trimmed.
        """
        segment_length = self.segment_length
        if segment_length is None:
            assert len(encoded_frames) == 1
            return self._decode_frame(encoded_frames[0])

        frames = [self._decode_frame(frame) for frame in encoded_frames]
        segment_stride = self.segment_stride
        return _linear_overlap_add(
            frames, segment_stride if segment_stride is not None else 1
        )

    def _decode_frame(
        self, encoded_frame: tuple[AudioCodeTensor, Tensor | None]
    ) -> AudioSegmentTensor:
        codes, scale = encoded_frame
        codes = codes.transpose(0, 1)
        emb = self.quantizer.decode(codes)
        out = self.decoder(emb)
        if scale is not None:
            out = out * scale.view(-1, 1, 1)
        return out

    def forward(self, x: AudioChannelTensor) -> AudioChannelTensor:
        frames = self.encode(x)
        return self.decode(frames)[:, :, : x.shape[-1]]

    def set_target_bandwidth(self, bandwidth: float):
        if bandwidth not in self.target_bandwidths:
            raise ValueError(
                f"This model doesn't support the bandwidth {bandwidth}. "
                f"Select one of {self.target_bandwidths}."
            )
        self.bandwidth = bandwidth

    @staticmethod
    def _get_model(
        target_bandwidths: list[float],
        sample_rate: int = 24_000,
        channels: int = 1,
        causal: bool = True,
        model_norm: str = "weight_norm",
        audio_normalize: bool = False,
        segment: float | None = None,
        name: str = "unset",
    ):
        encoder = SEANetEncoder(channels=channels, norm=model_norm, causal=causal)
        decoder = SEANetDecoder(channels=channels, norm=model_norm, causal=causal)
        n_q = int(
            1000
            * target_bandwidths[-1]
            // (math.ceil(sample_rate / encoder.hop_length) * 10)
        )
        quantizer = ResidualVectorQuantizer(
            dimension=encoder.dimension,
            n_q=n_q,
            bins=1024,
        )
        model = EncodecModel(
            encoder,
            decoder,
            quantizer,
            target_bandwidths,
            sample_rate,
            channels,
            normalize=audio_normalize,
            segment=segment,
            name=name,
        )
        return model

    @staticmethod
    def _get_pretrained(checkpoint_name: str, repository: Path | None = None):
        if repository is not None:
            if not repository.is_dir():
                raise ValueError(f"{repository} must exist and be a directory.")
            file = repository / checkpoint_name
            checksum = file.stem.split("-")[1]
            _check_checksum(file, checksum)
            return torch.load(file)
        else:
            url = _get_checkpoint_url(ROOT_URL, checkpoint_name)
            return torch.hub.load_state_dict_from_url(
                url, map_location="cpu", check_hash=True
            )

    @staticmethod
    def encodec_model_24khz(pretrained: bool = True, repository: Path | None = None):
        """Return the pretrained causal 24khz model."""
        if repository:
            assert pretrained
        target_bandwidths = [1.5, 3.0, 6.0, 12.0, 24.0]
        checkpoint_name = "encodec_24khz-d7cc33bc.th"
        sample_rate = 24_000
        channels = 1
        model = EncodecModel._get_model(
            target_bandwidths,
            sample_rate,
            channels,
            causal=True,
            model_norm="weight_norm",
            audio_normalize=False,
            name="encodec_24khz" if pretrained else "unset",
        )
        if pretrained:
            state_dict = EncodecModel._get_pretrained(checkpoint_name, repository)
            model.load_state_dict(fix_state_dict(state_dict))
        model.eval()
        return model

    @staticmethod
    def encodec_model_48khz(pretrained: bool = True, repository: Path | None = None):
        """Return the pretrained 48khz model."""
        if repository:
            assert pretrained
        target_bandwidths = [3.0, 6.0, 12.0, 24.0]
        checkpoint_name = "encodec_48khz-7e698e3e.th"
        sample_rate = 48_000
        channels = 2
        model = EncodecModel._get_model(
            target_bandwidths,
            sample_rate,
            channels,
            causal=False,
            model_norm="time_group_norm",
            audio_normalize=True,
            segment=1.0,
            name="encodec_48khz" if pretrained else "unset",
        )
        if pretrained:
            state_dict = EncodecModel._get_pretrained(checkpoint_name, repository)
            model.load_state_dict(fix_state_dict(state_dict))
        model.eval()
        return model


def fix_state_dict(state_dict: dict[str, Any]):
    for key in list(state_dict.keys()):
        if key.endswith("weight_g"):
            prefix = key[:-8]
            weight_g = state_dict[key]
            weight_v = state_dict[prefix + "weight_v"]

            state_dict[prefix + "weight"] = (
                weight_g
                * weight_v
                / torch.linalg.vector_norm(weight_v, dim=(1, 2), keepdim=True)
            )
            del state_dict[key]
            del state_dict[prefix + "weight_v"]
        if key.endswith("inited"):
            del state_dict[key]
        if key.endswith("cluster_size"):
            del state_dict[key]
        if key.endswith("embed_avg"):
            del state_dict[key]

    return state_dict
