import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from util.typing import AlignMaskTensor, AlignProbTensor, LengthTensor, LossBatchTensor


def attention_bin_loss(
    alignment_hard: AlignMaskTensor, alignment_logprob: AlignProbTensor
) -> LossBatchTensor:
    masked_log = torch.where(alignment_hard, alignment_logprob, 0)
    loss = -masked_log.sum(dim=(1, 2)) / alignment_hard.sum(dim=(1, 2))
    return loss


class AttentionCTCLoss(nn.Module):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.blank_logprob = blank_logprob
        self.ctc_loss = nn.CTCLoss(zero_infinity=True, reduction="none")

    def __call__(
        self,
        attn_logprob: AlignProbTensor,
        key_lens: LengthTensor,
        query_lens: LengthTensor,
    ) -> LossBatchTensor:
        return super().__call__(attn_logprob, key_lens, query_lens)

    def forward(
        self,
        attn_logprob: AlignProbTensor,
        key_lens: LengthTensor,
        query_lens: LengthTensor,
    ) -> LossBatchTensor:
        batch, _, max_key_len = attn_logprob.shape

        # Reorder input to [query_len, batch_size, key_len]
        attn_logprob = rearrange(attn_logprob, "b t c -> t b c")

        # Add blank label
        attn_logprob = F.pad(attn_logprob, (1, 0), value=self.blank_logprob)

        # Convert to log probabilities
        # Note: Mask out probs beyond key_len
        key_inds = torch.arange(max_key_len + 1, device=attn_logprob.device)
        attn_logprob.masked_fill_(
            rearrange(key_inds, "c -> () () c") > rearrange(key_lens, "b -> () b ()"),
            torch.finfo(attn_logprob.dtype).min,
        )
        attn_logprob = attn_logprob.log_softmax(dim=-1)

        # Target sequences
        target_seqs = repeat(key_inds[1:], "c -> b c", b=batch)

        # Evaluate CTC loss
        loss = self.ctc_loss(
            attn_logprob, target_seqs, input_lengths=query_lens, target_lengths=key_lens
        )

        # Average loss over batch
        loss = loss / key_lens

        return loss
