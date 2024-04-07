import torch
from einops import rearrange
from torch import Tensor


def log_beta_binom(query_lens: Tensor, key_lens: Tensor):
    """
    Compute the log probability of the beta-binomial distribution.

    With a single query_len `q` and key_len `k`, the log-probability at position (i, j)
    is given by (0-indexed):
    ```
        log_p(i, j) = log(BetaBinom(j | k, i + 1, q - i))
                    = lgamma(k + 1) - lgamma(j + 1) - lgamma(k - j + 1)
                    + lgamma(i + j + 1) + lgamma(q + k - i - j) - lgamma(q + k + 1)
                    + lgamma(q + 1) - lgamma(i + 1) - lgamma(q - i)
    ```

    Args:
        query_lens: The lengths of the query sequences. Shape: `(B,)`.
        key_lens: The lengths of the key sequences. Shape: `(B,)`.

    Returns:
        The log probabilities of the beta-binomial distribution. Shape: `(B, max(query_lens), max(key_lens))`.
    """

    max_query_len = query_lens.max().item()
    max_key_len = key_lens.max().item()

    query_lens = rearrange(query_lens.double(), "b -> b () ()")
    key_lens = rearrange(key_lens.double(), "b -> b () ()")

    constant = (
        torch.lgamma(query_lens + 1)
        + torch.lgamma(key_lens + 1)
        - torch.lgamma(query_lens + key_lens + 1)
    )

    i_range = rearrange(
        torch.arange(max_query_len, device=query_lens.device), "q -> () q ()"
    )
    q_range = torch.lgamma(i_range + 1) + torch.lgamma(query_lens - i_range)

    j_range = rearrange(
        torch.arange(max_key_len, device=key_lens.device), "k -> () () k"
    )
    k_range = torch.lgamma(j_range + 1) + torch.lgamma(key_lens - j_range + 1)

    full_range = torch.lgamma(i_range + j_range + 1) + torch.lgamma(
        query_lens + key_lens - i_range - j_range
    )

    log_p = constant - q_range - k_range + full_range
    log_p.masked_fill_(query_lens <= i_range, float("-inf"))
    log_p.masked_fill_(key_lens <= j_range, float("-inf"))

    return log_p
