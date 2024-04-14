import torch
from einops import rearrange

from util.mask import mask_from_lengths
from util.typing import AlignProbTensor, LengthTensor


@torch.no_grad()
def log_beta_binomial(
    audio_lens: LengthTensor,
    phoneme_lens: LengthTensor,
    max_audio_len: int,
    max_phoneme_len: int,
) -> AlignProbTensor:
    audio_mask = rearrange(
        mask_from_lengths(audio_lens, max_audio_len), "b m -> b m ()"
    )
    phoneme_mask = rearrange(
        mask_from_lengths(phoneme_lens, max_phoneme_len), "b p -> b () p"
    )
    audio_lens = rearrange(audio_lens, "b -> b () ()").double()
    phoneme_lens = rearrange(phoneme_lens, "b -> b () ()").double()
    const = (
        torch.lgamma(phoneme_lens + 1)
        + torch.lgamma(audio_lens + 1)
        - torch.lgamma(audio_lens + phoneme_lens + 1)
    )
    p_range = rearrange(
        torch.arange(max_phoneme_len, device=phoneme_lens.device), "p -> () () p"
    )
    i_range = torch.lgamma(p_range + 1) + torch.lgamma((phoneme_lens + 1 - p_range))
    m_range = rearrange(
        torch.arange(max_audio_len, device=audio_lens.device), "m -> () m ()"
    )
    j_range = torch.lgamma(m_range + 1) + torch.lgamma((audio_lens - m_range))
    full_range = torch.lgamma(p_range + m_range + 1) + torch.lgamma(
        (audio_lens + phoneme_lens - p_range - m_range)
    )
    result = const + full_range - i_range - j_range
    result.masked_fill_(~audio_mask, torch.finfo(torch.float16).min)
    result.masked_fill_(~phoneme_mask, torch.finfo(torch.float16).min)
    return result.float()
