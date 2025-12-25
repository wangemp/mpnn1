import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import scatter

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

        # 输入投影 (CodeBERT 773 -> 128)
        self.input_proj = nn.Linear(in_dim0, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.act = nn.ReLU()

        # GNN 编码器
        self.enc0_in = GraphEncoder(hidden_dim, hidden_dim, num_layers=gnn_layers, dropout=dropout)
        self.enc1_in = GraphEncoder(hidden_dim, hidden_dim, num_layers=gnn_layers, dropout=dropout)
        self.enc2_in = GraphEncoder(hidden_dim, hidden_dim, num_layers=gnn_layers, dropout=dropout)

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
        # 1. 解包数据
        # 兼容处理：新版 preprocess 生成的是 x, edge_index
        if hasattr(data, 'x') and not hasattr(data, 'x0'):
            x0 = data.x
            e0 = data.edge_index
            # 如果 batch 属性缺失（如 batch_size=1），手动生成全0 batch
            b0 = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x0.size(0), dtype=torch.long, device=x0.device)
            assign0_local = data.assign_index if hasattr(data, 'assign_index') else None
        else:
            x0, e0, b0 = data.x0, data.edge_index0, data.x0_batch
            assign0_local = data.assign0

        device = x0.device

        # 2. [核心修复] 动态推断 Scale 1 结构 (稳健模式)
        # 我们不再依赖 data.num_scale1_nodes，而是直接从 assign0_local 推断
        if assign0_local is not None:
            # assign0_local 是局部的函数ID (0..k)。
            # 我们通过 scatter_max 找出每个图最大的函数ID，然后 +1 得到数量
            # nums: [batch_size]，表示每个图有多少个函数
            max_func_ids = scatter(assign0_local, b0, dim=0, reduce='max')
            # 如果某个图没有节点/函数，max_func_ids 可能是初始值 0，这里假设至少有1个函数(<global>)
            nums = max_func_ids + 1
            
            # 2.1 构建 offsets 用于生成 assign0 (全局)
            offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long), torch.cumsum(nums, dim=0)[:-1]])
            assign0 = assign0_local + offsets[b0]
            
            # 2.2 构建 assign1 (全局函数 -> 图ID)
            # graph_ids: [0, 1, 2...]
            graph_ids = torch.arange(nums.size(0), device=device, dtype=torch.long)
            # 重复 graph_id: [0, 0, 1, 1, 1...]
            assign1 = torch.repeat_interleave(graph_ids, nums)
            
            total_funcs = nums.sum().item()
        else:
            # Fallback: 无函数层级信息
            assign0 = None
            assign1 = None
            total_funcs = 0

        # 3. 特征投影 (773 -> 128)
        x0 = self.input_proj(x0)
        x0 = self.act(x0)
        x0 = self.dropout_layer(x0)

        # 4. 初始化特征
        x1 = torch.zeros(total_funcs, 128, device=device)
        e1 = torch.empty(2, 0, dtype=torch.long, device=device)
        
        batch_size = data.num_graphs if hasattr(data, 'num_graphs') else (b0.max().item() + 1)
        x2 = torch.zeros(batch_size, 128, device=device)
        e2 = torch.empty(2, 0, dtype=torch.long, device=device)

        # 5. 初始编码
        h0 = self.enc0_in(x0, e0)
        h1 = self.enc1_in(x1, e1)
        h2 = self.enc2_in(x2, e2)

        # 6. 交互与传播
        for _ in range(self.inter_steps):
            if assign0 is not None and assign1 is not None:
                # 只有当映射存在且有效时才交互
                h0, h1, h2 = self.inter(h0, h1, h2, assign0, assign1)
            
            h0 = self.enc0(h0, e0)
            h1 = self.enc1(h1, e1)
            h2 = self.enc2(h2, e2)

        # 7. 读出 (Readout)
        g0 = global_mean_pool(h0, b0)
        
        # Scale 1 -> Global (直接利用 assign1 聚合)
        if assign1 is not None and h1.size(0) > 0:
            # 确保 assign1 没有越界
            if assign1.max() < batch_size:
                g1 = scatter(h1, assign1, dim=0, dim_size=batch_size, reduce='mean')
            else:
                # 理论上不会发生，作为保险
                g1 = torch.zeros_like(g0)
        else:
            g1 = torch.zeros_like(g0)

        g2 = h2 

        g = torch.cat([g0, g1, g2], dim=-1)
        return self.head(g)