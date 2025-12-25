import os
import torch
import random
from typing import List, Dict
from tqdm import tqdm

from configs import Config
from .parser import parse_file
from .graph_builder import MultiScaleGraphBuilder
from .behavior_extractor import BehaviorExtractor

def build_and_save_dataset(raw_dir=None, processed_dir=None, behavior_types=None, exts=None, **kwargs):
    
    # 1. 初始化配置
    cfg = Config()
    cfg.ensure_dirs()
    
    # 2. 参数覆盖
    if raw_dir: cfg.data_raw_dir = raw_dir
    if processed_dir: cfg.data_processed_dir = processed_dir
    if behavior_types: cfg.behavior_types = behavior_types
    if exts: cfg.exts = exts
    
    # 3. 检测设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Preprocess using device: {device}")
    
    # 4. 初始化特征提取器
    extractor = BehaviorExtractor(
        behavior_types=cfg.behavior_types,
        local_path=cfg.codebert_local_path,
        remote_name=cfg.codebert_remote_name,
        device=device
    )
    
    builder = MultiScaleGraphBuilder(cfg.behavior_types)
    
    # 5. 收集文件
    print(f"Collecting files from {cfg.data_raw_dir}...")
    raw_files = []
    if os.path.exists(cfg.data_raw_dir):
        for root, dirs, files in os.walk(cfg.data_raw_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in cfg.exts:
                    raw_files.append(os.path.join(root, f))
    else:
        print(f"[Error] Raw directory not found: {cfg.data_raw_dir}")
        return {}

    if not raw_files:
        print(f"[Warning] No files found in {cfg.data_raw_dir} with extensions {cfg.exts}")
        return {}

    random.shuffle(raw_files)
    
    # 6. 数据集划分
    n_total = len(raw_files)
    n_train = int(n_total * cfg.train_ratio)
    n_val = int(n_total * cfg.val_ratio)
    
    train_files = raw_files[:n_train]
    val_files = raw_files[n_train:n_train+n_val]
    test_files = raw_files[n_train+n_val:]
    
    print(f"Total: {n_total}, Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    # 7. 处理函数
    def _process_split(files: List[str], split_name: str):
        data_list = []
        print(f"Processing {split_name} data...")
        
        for file_path in tqdm(files):
            try:
                # 读取源码
                with open(file_path, "r", errors="ignore", encoding="utf-8") as f:
                    source_code = f.read()
                
                # 简单标签逻辑
                label = 1 if "webshell" in file_path.lower() else 0
                
                # 解析 AST
                lang, calls = parse_file(file_path, source_code)
                
                # 提取特征
                nodes = extractor.build_nodes(lang, calls)
                
                # 构建图
                data = builder.build_graph(nodes, label)
                
                # ============================================================
                # [核心修复] 必须把 lang 存进去，否则统计时全是 'other'
                # ============================================================
                data.lang = lang       # <--- 加上这一行！
                data.file_path = file_path
                
                data_list.append(data)
                
            except Exception as e:
                # print(f"Error processing {file_path}: {e}")
                pass
        
        out_path = os.path.join(cfg.data_processed_dir, f"{split_name}.pt")
        torch.save(data_list, out_path)
        print(f"Saved {len(data_list)} graphs to {out_path}")

    # 8. 执行处理
    if len(train_files) > 0: _process_split(train_files, "train")
    if len(val_files) > 0: _process_split(val_files, "val")
    if len(test_files) > 0: _process_split(test_files, "test")
    
    return {
        "train": len(train_files),
        "val": len(val_files),
        "test": len(test_files)
    }