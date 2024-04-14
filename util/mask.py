import torch
from einops import rearrange, repeat

from util.typing import AudioMaskTensor, LengthTensor


@torch.no_grad()
def mask_from_lengths(lengths: LengthTensor, max_len: int) -> AudioMaskTensor:
    seq = torch.arange(max_len, device=lengths.device)
    seq = repeat(seq, "l -> b l", b=lengths.shape[0])

    mask = seq < rearrange(lengths, "b -> b 1")
    return mask
