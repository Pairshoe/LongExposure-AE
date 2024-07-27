import torch
import torch.nn as nn


class AttentionPredictor(nn.Module):
    def __init__(self, hidden_dim, num_heads, r=4):
        super(AttentionPredictor, self).__init__()
        self.r = r  # head_dim
        self.num_heads = num_heads

        # each head has its own query and key
        self.approx_query = nn.Linear(hidden_dim, r * num_heads)
        self.approx_key = nn.Linear(hidden_dim, r * num_heads)

    def forward(self, x):
        # down sample x from [batch_size, seq_len, hidden_dim] to [batch_size, \sqrt{seq_len}, hidden_dim]
        batch_size, seq_len, _ = x.size()
        seq_len = (seq_len + int(seq_len ** 0.5) - 1) // int(seq_len ** 0.5)
        seq_idx = torch.linspace(0, x.size(1) - 1, seq_len).long()
        x = x[:, seq_idx, :]

        q = self.approx_query(x)
        k = self.approx_key(x)

        # split query and key into num_heads
        q = q.view(batch_size, seq_len, self.num_heads, self.r)
        k = k.view(batch_size, seq_len, self.num_heads, self.r)

        # compute the approximate score: batch_size, num_heads, new_seq_len, new_seq_len
        approx_score = torch.einsum('bqhr,bkhr->bhqk', q, k)
        approx_score = torch.sigmoid(approx_score)
        
        return approx_score
