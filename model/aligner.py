import torch
from einops import rearrange
from torch import nn

from util.prior import log_beta_binom


class Aligner(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, hidden_dim: int):
        super().__init__()
        self.query_layers = nn.Sequential(
            nn.Conv1d(query_dim, hidden_dim * 2, kernel_size=1, padding=1),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=1, padding=1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.key_layers = nn.Sequential(
            nn.Conv1d(key_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, query, key, query_lens, key_lens):
        query = rearrange(query, "b n c -> b c n")
        key = rearrange(key, "b n c -> b c n")

        query = self.query_layers(query)
        key = self.key_layers(key)

        dist = -torch.cdist(query, key)
        prior = log_beta_binom(query_lens, key_lens)

        log_prob = (dist + prior).log_softmax(dim=-1)

        return log_prob
