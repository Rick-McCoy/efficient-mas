from torch import nn


class Aligner(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, hidden_dim: int):
        super().__init__()
        self.query_layers = nn.Sequential(
            nn.Conv1d(query_dim, hidden_dim * 2, kernel_size=5, padding=2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.key_layers = nn.Sequential(
            nn.Conv1d(key_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, query, key):
        query = self.query_layers(query)
        key = self.key_layers(key)
        return query, key
