import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean


class InterScaleBlock(nn.Module):
    """
    Interleaved multiscale message passing:
      0 -> 1 (bottom-up), 1 -> 0 (top-down),
      1 -> 2 (bottom-up), 2 -> 1 (top-down)
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout

        self.up01 = nn.Linear(hidden_dim, hidden_dim)
        self.down10 = nn.Linear(hidden_dim, hidden_dim)

        self.up12 = nn.Linear(hidden_dim, hidden_dim)
        self.down21 = nn.Linear(hidden_dim, hidden_dim)

        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, h0, h1, h2, assign0, assign1):
        # 0 -> 1
        msg01 = scatter_mean(h0, assign0, dim=0, dim_size=h1.size(0))
        msg01 = self.up01(msg01)
        h1 = self.norm1(h1 + F.dropout(msg01, p=self.dropout, training=self.training))

        # 1 -> 0
        msg10 = self.down10(h1[assign0])
        h0 = self.norm0(h0 + F.dropout(msg10, p=self.dropout, training=self.training))

        # 1 -> 2
        msg12 = scatter_mean(h1, assign1, dim=0, dim_size=h2.size(0))
        msg12 = self.up12(msg12)
        h2 = self.norm2(h2 + F.dropout(msg12, p=self.dropout, training=self.training))

        # 2 -> 1
        msg21 = self.down21(h2[assign1])
        h1 = self.norm1(h1 + F.dropout(msg21, p=self.dropout, training=self.training))

        return h0, h1, h2
