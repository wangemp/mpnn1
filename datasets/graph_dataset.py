import torch
from torch.utils.data import Dataset
from typing import List
from torch_geometric.data import Data


class GraphPTDataset(Dataset):
    def __init__(self, pt_path: str):
        super().__init__()
        self.data_list: List[Data] = torch.load(pt_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
