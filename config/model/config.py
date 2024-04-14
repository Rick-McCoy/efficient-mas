from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class AlignerBaseConfig:
    dim: int = MISSING
    max_audio_len: int = 2048
    max_phoneme_len: int = 512


@dataclass
class AlignerLargeConfig(AlignerBaseConfig):
    dim: int = 512


@dataclass
class AlignerMediumConfig(AlignerBaseConfig):
    dim: int = 256


@dataclass
class AlignerSmallConfig(AlignerBaseConfig):
    dim: int = 128
