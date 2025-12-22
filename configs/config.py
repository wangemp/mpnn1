from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class Config:
    #Paths
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_raw_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"
    ))
    data_processed_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed"
    ))
    data_cache_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache"
    ))
    ckpt_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints"
    ))
    log_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
    ))

    # [新增] CodeBERT 本地保存路径 (项目根目录/pretrained_models/codebert-base)
    codebert_local_path: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        "pretrained_models", 
        "codebert-base"
    ))

    # [新增] CodeBERT 远程模型名称 (用于下载)
    codebert_remote_name: str = "microsoft/codebert-base"

    # Data split
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Supported file extensions
    exts: List[str] = field(default_factory=lambda: [".php", ".asp", ".aspx", ".txt"])

    # Preprocess behavior types
    behavior_types: List[str] = field(default_factory=lambda: [
        "INPUT", 
        "DECODE", 
        "STRING_OP", 
        "EXECUTE", 
        "FILE_OP", 
        "NET_OP", 
        "CRYPTO", 
        "CONTROL", 
        "OTHER",
        "OBFUSCATION" 
    ])

    # Training
    device: str = "cuda"
    num_workers: int = 0
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    grad_clip: float = 2.0

    # Model
    hidden_dim: int = 128
    gnn_layers: int = 2
    inter_steps: int = 2
    dropout: float = 0.2
    num_classes: int = 2

    def ensure_dirs(self):
        os.makedirs(self.data_processed_dir, exist_ok=True)
        os.makedirs(self.data_cache_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)