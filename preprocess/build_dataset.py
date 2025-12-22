import os
import torch
import random
from typing import List
from tqdm import tqdm

from configs import Config
from .parser import parse_file
from .graph_builder import MultiScaleGraphBuilder
from .behavior_extractor import BehaviorExtractor

# =================================================================
# [修正] 增加参数接收，解决 TypeError 报错
# =================================================================
def build_and_save_dataset(raw_dir=None, processed_dir=None, behavior_types=None, exts=None, **kwargs):
    
    # 1. 初始化基础配置 (主要是为了拿到 CodeBERT 的路径)
    cfg = Config()
    cfg.ensure_dirs()
    
    # 2. [关键] 参数覆盖：如果 scripts/preprocess.py 传了参数进来，就用传进来的
    # 这样既兼容了旧的调用方式，又保留了新加的 CodeBERT 配置
    if raw_dir: 
        cfg.data_raw_dir = raw_dir
    if processed_dir: 
        cfg.data_processed_dir = processed_dir
    if behavior_types: 
        cfg.behavior_types = behavior_types
    if exts: 
        cfg.exts = exts
    
    # 3. 检测设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Preprocess using device: {device}")
    
    # 4. 初始化特征提取器
    # 注意：codebert_local_path 和 codebert_remote_name 来自 Config
    # 因为 scripts/preprocess.py 不会传这两个新参数，必须从 cfg 里读
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
    for root, dirs, files in os.walk(cfg.data_raw_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in cfg.exts:
                raw_files.append(os.path.join(root, f))

    if not raw_files:
        print(f"[Warning] No files found in {cfg.data_raw_dir} with extensions {cfg.exts}")
        return {} # 返回空字典防止报错

    random.shuffle(raw_files)
    
    # 6. 分割数据集
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
                
                # 标签逻辑
                label = 1 if "webshell" in file_path.lower() else 0
                
                # 解析 AST
                lang, calls = parse_file(file_path, source_code)
                
                # 提取特征 (CodeBERT)
                nodes = extractor.build_nodes(lang, calls)
                
                # 构建图
                data = builder.build_graph(nodes, label)
                
                # 保存路径方便调试
                data.file_path = file_path
                
                data_list.append(data)
                
            except Exception as e:
                # print(f"Error processing {file_path}: {e}")
                pass
        
        out_path = os.path.join(cfg.data_processed_dir, f"{split_name}.pt")
        torch.save(data_list, out_path)
        print(f"Saved {len(data_list)} graphs to {out_path}")

    # 8. 执行
    if len(train_files) > 0: _process_split(train_files, "train")
    if len(val_files) > 0: _process_split(val_files, "val")
    if len(test_files) > 0: _process_split(test_files, "test")
    
    # 返回一些元数据，防止 caller 报错
    return {
        "train": len(train_files),
        "val": len(val_files),
        "test": len(test_files)
    }

if __name__ == "__main__":
    build_and_save_dataset()