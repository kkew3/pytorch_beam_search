import torch
from torch import nn
from torch.nn import functional as F


# Dummy model: simple 2-layer language model for tests.
class DummyLM(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        x = self.embed(input_ids)  # (batch, seq_len, hidden)
        x = F.gelu(self.l1(x))  # (batch, seq_len, hidden)
        x = self.l2(x)  # (batch, seq_len, vocab_size)
        return x.mean(1)  # (batch, vocab_size)
