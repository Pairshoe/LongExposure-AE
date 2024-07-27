import torch.nn as nn


class MLPPredictor(nn.Module):
    def __init__(self, embed_dim, hidden_dim, blk_size):
        super(MLPPredictor, self).__init__()
        assert hidden_dim % blk_size == 0, 'hidden_dim must be divisible by blk_size'
        self.predictor = nn.Linear(embed_dim, hidden_dim // blk_size)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.predictor(x))
