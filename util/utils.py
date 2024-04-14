import numpy as np
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from torch import Tensor

from util.mask import mask_from_lengths
from util.typing import (
    AudioEncTensor,
    LengthTensor,
)

plt.switch_backend("agg")


def plot_with_cmap(mels: list[np.ndarray], sharex: bool = True):
    fig, axes = plt.subplots(len(mels), 1, figsize=(20, 4 * len(mels)), sharex=sharex)

    if len(mels) == 1:
        axes = [axes]

    for i, mel in enumerate(mels):
        im = axes[i].imshow(mel, aspect="auto", origin="lower", interpolation="none")
        fig.colorbar(im, ax=axes[i])

    fig.canvas.draw()
    plt.close(fig)

    return np.array(
        fig.canvas.buffer_rgba()  # pyright: ignore [reportAttributeAccessIssue]
    )


def normalize_audio(
    audio_enc: AudioEncTensor, audio_lens: LengthTensor
) -> tuple[AudioEncTensor, Tensor, Tensor]:
    """
    Normalize audio encodings to have zero mean and unit variance.
    Each audio encoding is 2-dimensional and has zero padding to the right.
    Return normalized audio encodings, mean, and standard deviation.

    Args:
        audio_enc: Audio encodings. Shape: (batch, time, channel).
        audio_lens: Lengths of audio encodings. Shape: (batch,).

    Returns:
        audio_enc: Normalized audio encodings. Shape: (batch, time, channel).
        audio_mean: Mean of audio encodings. Shape: (batch,).
        audio_std: Standard deviation of audio encodings. Shape: (batch,).
    """
    audio_mask = mask_from_lengths(audio_lens, audio_enc.shape[1])
    # audio_mean = (audio_enc.mean(dim=2) * audio_mask).sum(dim=1) / audio_lens
    # audio_sq_mean = ((audio_enc**2).mean(dim=2) * audio_mask).sum(dim=1) / audio_lens
    # nelem = audio_lens * audio_enc.shape[2]
    # bessel_correction = nelem / (nelem - 1)
    # audio_std = torch.sqrt((audio_sq_mean - audio_mean**2) * bessel_correction)
    batch_size = audio_enc.shape[0]
    audio_mean = torch.full((batch_size,), -1.430645).to(audio_enc.device)
    audio_std = torch.full((batch_size,), 2.1208718).to(audio_enc.device)
    audio_mean = rearrange(audio_mean, "b -> b () ()")
    audio_std = rearrange(audio_std, "b -> b () ()")
    normalized_audio_enc = (
        (audio_enc - audio_mean)
        / (audio_std + 1e-5)
        * rearrange(audio_mask, "b l -> b l ()")
    )
    return normalized_audio_enc, audio_mean, audio_std
