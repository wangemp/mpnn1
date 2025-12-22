import torch
from torch_geometric.data import Data
from typing import List, Dict

# 引用 behavior_extractor 里定义的结构
from .behavior_extractor import BehaviorNode

class MultiScaleGraphBuilder:
    def __init__(self, behavior_types: List[str]):
        self.behavior_types = behavior_types

    def build_graph(self, nodes: List[BehaviorNode], label: int) -> Data:
        """
        将行为节点列表转换为 PyG 的 Data 对象
        """
        if not nodes:
            # 如果没有节点，返回空或者抛出异常（会在上层被 catch）
            raise ValueError("Empty node list")

        # =======================================================
        # 1. 构建节点特征矩阵 (X)
        # =======================================================
        # nodes[i].feat 现在是一个 773 维的列表 (768 CodeBERT + 5 Stat)
        # 我们直接将其转换为 FloatTensor
        feats = [n.feat for n in nodes]
        x = torch.tensor(feats, dtype=torch.float)

        num_nodes = len(nodes)

        # =======================================================
        # 2. 构建 Scale 0 边 (Edge Index) - 序列边
        # =======================================================
        # 简单的近邻连接：Node[i] <-> Node[i+1]
        # 代表代码的执行流顺序
        src_list = []
        dst_list = []

        for i in range(num_nodes - 1):
            # 正向边 i -> i+1
            src_list.append(i)
            dst_list.append(i + 1)
            # 反向边 i+1 -> i (无向图通常有利于信息传递)
            src_list.append(i + 1)
            dst_list.append(i)
        
        # 如果只有一个节点，加自环，防止 GNN 报错
        if num_nodes == 1:
            src_list.append(0)
            dst_list.append(0)

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        # =======================================================
        # 3. 构建 Scale 1 分配矩阵 (Assignment Matrix for Pooling)
        # =======================================================
        # 目的是将 Scale 0 的节点聚合到 Scale 1 (函数级)
        # 我们需要知道每个节点属于哪个函数
        
        # 1. 找出所有唯一的函数名
        unique_funcs = list(set(n.func_name for n in nodes))
        # 排序以保证确定性
        unique_funcs.sort() 
        func_to_id = {name: i for i, name in enumerate(unique_funcs)}
        
        # 2. 生成分配索引 (assign_index)
        # 这通常用于 scatter 操作: scatter_mean(x, assign_index, dim=0)
        assign_list = []
        for n in nodes:
            func_id = func_to_id[n.func_name]
            assign_list.append(func_id)
            
        # assign_index 形状: [num_nodes]
        # assign_index[i] = k 表示第 i 个节点属于第 k 个函数聚类
        assign_index = torch.tensor(assign_list, dtype=torch.long)

        # =======================================================
        # 4. 组装 Data 对象
        # =======================================================
        data = Data(
            x=x, 
            edge_index=edge_index, 
            y=torch.tensor([label], dtype=torch.long),
            
            # 额外信息存入 Data 对象
            assign_index=assign_index,   # 用于 Scale 0 -> Scale 1 的池化
            num_scale1_nodes=len(unique_funcs), # Scale 1 有多少个节点
            
            # 调试信息 (可选)
            num_nodes=num_nodes
        )
        
        return data