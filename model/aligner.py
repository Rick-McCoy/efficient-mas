import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from util.ipa import VOCAB_SIZE, VOCAB_TO_ID
from util.mask import mask_from_lengths
from util.prior import log_beta_binomial
from util.typing import (
    AlignMaskTensor,
    AlignProbTensor,
    AudioEncTensor,
    LengthTensor,
    PhonemeTensor,
)


@torch.compiler.disable
def for_loop(log_p: AlignProbTensor):
    for i in range(1, log_p.shape[1]):
        log_p[:, i, 1:] += torch.maximum(log_p[:, i - 1, :-1], log_p[:, i - 1, 1:])


def batched_mas_log(
    align_prob: AlignProbTensor, mask: AlignMaskTensor
) -> AlignMaskTensor:
    _, mel, phone = align_prob.shape
    device = align_prob.device
    mel_size = 1
    while mel_size < mel:
        mel_size *= 2
    log_p = F.pad(align_prob, (0, 0, 0, mel_size - mel))
    pad_mask = F.pad(mask, (0, 0, 0, mel_size - mel))
    log_p.masked_fill_(~pad_mask, -torch.inf)

    direction = torch.zeros_like(log_p).to(torch.long)

    log_p[:, 0, 1:] = -torch.inf
    log_p[:, 1:, 0] = torch.cumsum(log_p[:, 1:, 0], dim=1)
    # for i in range(1, mel):
    #     log_p[:, i, 1:] += torch.maximum(log_p[:, i - 1, :-1], log_p[:, i - 1, 1:])
    for_loop(log_p=log_p)

    direction[:, 1:, 1:] = log_p[:, :-1, :-1] >= log_p[:, :-1, 1:]
    direction.masked_fill_(~pad_mask, 0)

    memoized_index = torch.arange(phone, device=device) - direction
    memoized_index[:, :, 0] = 0

    mel_range = torch.arange(mel_size, device=device)
    index = repeat(mask[:, 0].sum(dim=1) - 1, "b -> b m", m=mel_size)

    power = 1
    while power < mel_size:
        binary_mask = ~(mel_range & power).bool()
        gather = torch.gather(memoized_index[:, -power:], 2, index[:, -power:, None])
        select = repeat(gather, "b m () -> b (i m)", i=mel_size // power)
        index = torch.where(binary_mask, select, index)
        memoized_index[:, power:] = torch.gather(
            memoized_index[:, :-power], 2, memoized_index[:, power:]
        )
        power *= 2

    path = torch.zeros_like(align_prob).bool()
    path = path.scatter(2, rearrange(index[:, :mel], "b m -> b m ()"), 1)
    return path * mask


def mas_approx(
    align_prob: AlignProbTensor, audio_lens: LengthTensor, phoneme_lens: LengthTensor
) -> AlignMaskTensor:
    _, mel, phone = align_prob.shape
    device = align_prob.device
    mel_size = 1
    while mel_size < mel:
        mel_size *= 2

    log_p = F.pad(align_prob, (0, 0, 0, mel_size - mel))
    audio_mask = rearrange(mask_from_lengths(audio_lens, mel_size), "b m -> b m ()")
    phoneme_mask = rearrange(mask_from_lengths(phoneme_lens, phone), "b p -> b () p")
    log_p.masked_fill_(~audio_mask, -torch.inf)
    log_p.masked_fill_(~phoneme_mask, -torch.inf)

    direction = torch.zeros_like(log_p).long()

    log_p[:, 0, 1:] = -torch.inf

    log_p_clone = log_p.clone()
    for _ in range(1, 64):
        log_p[:, 1:, 1:] = log_p_clone[:, 1:, 1:] + torch.maximum(
            log_p[:, :-1, :-1], log_p[:, :-1, 1:]
        )

    direction[:, 1:, 1:] = log_p[:, :-1, :-1] >= log_p[:, :-1, 1:]
    direction.masked_fill_(~audio_mask, 0)
    direction.masked_fill_(~phoneme_mask, 0)

    memoized_index = torch.arange(phone, device=device) - direction
    memoized_index[:, :, 0] = 0

    mel_range = torch.arange(mel_size, device=device)
    index = repeat(phoneme_lens - 1, "b -> b m", m=mel_size)

    power = 1
    while power < mel_size:
        gather = torch.gather(memoized_index[:, -power:], 2, index[:, -power:, None])
        select = repeat(gather, "b m () -> b (i m)", i=mel_size // power)
        index = torch.where((mel_range & power).bool(), index, select)
        memoized_index[:, power:] = torch.gather(
            memoized_index[:, :-power], 2, memoized_index[:, power:]
        )
        power *= 2

    path = torch.zeros_like(log_p).bool()
    path = path.scatter(2, rearrange(index, "b m -> b m ()"), 1)
    path.masked_fill_(~audio_mask, 0)
    path.masked_fill_(~phoneme_mask, 0)
    return path[:, :mel]


class Aligner(nn.Module):
    def __init__(self, dim_query: int, dim_key: int, dim_hidden: int):
        super().__init__()

        self.key_emb = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=dim_key,
            padding_idx=VOCAB_TO_ID["<pad>"],
        )

        self.key_layers = nn.Sequential(
            nn.Conv1d(dim_key, dim_hidden, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv1d(dim_hidden, dim_hidden, kernel_size=1, padding=0, bias=True),
        )

        self.query_layers = nn.Sequential(
            nn.Conv1d(dim_query, dim_hidden, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv1d(dim_hidden, dim_hidden * 2, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv1d(dim_hidden * 2, dim_hidden, kernel_size=1, padding=0, bias=True),
        )

    def __call__(
        self,
        query: AudioEncTensor,
        query_lens: LengthTensor,
        key: PhonemeTensor,
        key_lens: LengthTensor,
    ) -> AlignProbTensor:
        return super().__call__(
            query=query, query_lens=query_lens, key=key, key_lens=key_lens
        )

    def forward(
        self,
        query: AudioEncTensor,
        query_lens: LengthTensor,
        key: PhonemeTensor,
        key_lens: LengthTensor,
    ) -> AlignProbTensor:
        key = self.key_emb(key)

        query = rearrange(query, "b q d -> b d q")
        key = rearrange(key, "b k d -> b d k")

        query = self.query_layers(query)
        key = self.key_layers(key)

        query = rearrange(query, "b d q -> b q d")
        key = rearrange(key, "b d k -> b k d")

        attn_soft = -torch.cdist(query, key)

        log_prior = log_beta_binomial(
            query_lens,
            key_lens,
            max_audio_len=attn_soft.shape[1],
            max_phoneme_len=attn_soft.shape[2],
        )
        attn_prior = attn_soft + log_prior
        attn_logprob = attn_prior.log_softmax(dim=-1)

        return attn_logprob
