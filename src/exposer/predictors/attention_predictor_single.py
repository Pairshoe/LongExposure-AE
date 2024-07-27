import torch
import torch.nn as nn


class AttentionPredictor(nn.Module):
    def __init__(self, hidden_dim: int, r: int):
        super(AttentionPredictor, self).__init__()
        self.r = r
        self.approx_query = nn.Linear(hidden_dim, r)
        self.approx_key = nn.Linear(hidden_dim, r)

    def forward(self, x: torch.Tensor):
        q = self.approx_query(x)
        k = self.approx_key(x)
        approx_score = nn.functional.sigmoid(q @ k.mT)
        return approx_score
