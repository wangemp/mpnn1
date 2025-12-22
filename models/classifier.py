import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from .gnn_encoder import GraphEncoder
from .im_mpnn import InterScaleBlock


class IMMPNNWebshellClassifier(nn.Module):
    def __init__(
        self,
        in_dim0: int,
        in_dim1: int,
        in_dim2: int,
        hidden_dim: int = 128,
        gnn_layers: int = 2,
        inter_steps: int = 2,
        dropout: float = 0.2,
        num_classes: int = 2
    ):
        super().__init__()
        self.inter_steps = inter_steps

        # ① 输入编码器：raw_dim -> hidden_dim（只用一次）
        self.enc0_in = GraphEncoder(in_dim0, hidden_dim, num_layers=gnn_layers, dropout=dropout)
        self.enc1_in = GraphEncoder(in_dim1, hidden_dim, num_layers=gnn_layers, dropout=dropout)
        self.enc2_in = GraphEncoder(in_dim2, hidden_dim, num_layers=gnn_layers, dropout=dropout)

        # ② 局部传播编码器：hidden_dim -> hidden_dim（循环里反复用）
        self.enc0 = GraphEncoder(hidden_dim, hidden_dim, num_layers=gnn_layers, dropout=dropout)
        self.enc1 = GraphEncoder(hidden_dim, hidden_dim, num_layers=gnn_layers, dropout=dropout)
        self.enc2 = GraphEncoder(hidden_dim, hidden_dim, num_layers=gnn_layers, dropout=dropout)

        self.inter = InterScaleBlock(hidden_dim, dropout=dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        x0, e0, b0 = data.x0, data.edge_index0, data.x0_batch
        x1, e1, b1 = data.x1, data.edge_index1, data.x1_batch
        x2, e2, b2 = data.x2, data.edge_index2, data.x2_batch

        # ✅ assign0/assign1 不用改，照常用（它们是跨尺度映射）
        assign0, assign1 = data.assign0, data.assign1

        # ① 第一次：raw -> hidden
        h0 = self.enc0_in(x0, e0)
        h1 = self.enc1_in(x1, e1)
        h2 = self.enc2_in(x2, e2)

        # ② 循环：跨尺度交互 + hidden 空间内的局部传播
        for _ in range(self.inter_steps):
            h0, h1, h2 = self.inter(h0, h1, h2, assign0, assign1)
            h0 = self.enc0(h0, e0)
            h1 = self.enc1(h1, e1)
            h2 = self.enc2(h2, e2)

        g0 = global_mean_pool(h0, b0)
        g1 = global_mean_pool(h1, b1)
        g2 = global_mean_pool(h2, b2)

        g = torch.cat([g0, g1, g2], dim=-1)
        return self.head(g)
