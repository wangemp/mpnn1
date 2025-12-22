import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphEncoder(nn.Module):
    """
    Single-scale GNN encoder: GraphSAGE + LayerNorm + residual projection.
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i+1]))
            self.norms.append(nn.LayerNorm(dims[i+1]))

        self.res_proj = None
        if in_dim != hidden_dim:
            self.res_proj = nn.Linear(in_dim, hidden_dim)

    def forward(self, x, edge_index):
        base = self.res_proj(x) if self.res_proj is not None else x
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h, inplace=True)
            h = F.dropout(h, p=self.dropout, training=self.training)
        if h.shape == base.shape:
            h = h + base
        return h
