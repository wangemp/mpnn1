import os
import torch
from torch_geometric.loader import DataLoader

from configs import Config
from utils.logger import get_logger
from utils.seed import set_seed
from models.classifier import IMMPNNWebshellClassifier
from trainers.trainer import Trainer
from trainers.evaluator import evaluate

def main():
    cfg = Config()
    cfg.ensure_dirs()
    set_seed(cfg.seed)

    # 日志初始化
    log_path = os.path.join(cfg.log_dir, "train.log")
    if os.path.exists(log_path): os.remove(log_path) # 清理旧日志
    logger = get_logger(log_path) 

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 1. 加载数据
    # 注意：新的 preprocess.py 生成的是 .pt 文件，里面是 Data 对象的 list
    logger.info("Loading datasets...")
    train_ds = torch.load(os.path.join(cfg.data_processed_dir, "train.pt"))
    val_ds = torch.load(os.path.join(cfg.data_processed_dir, "val.pt"))
    test_ds = torch.load(os.path.join(cfg.data_processed_dir, "test.pt"))

    # 2. DataLoader
    # follow_batch 设为 None，因为新的 Data 可能没有复杂的 x0_batch 结构，
    # PyG 的 DataLoader 会自动处理 .batch 属性
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # 3. 动态获取维度 (关键步骤)
    if len(train_ds) > 0:
        sample = train_ds[0]
        # 兼容性处理：CodeBERT 预处理生成的是 .x，旧代码生成的是 .x0
        if hasattr(sample, 'x0'):
            in_dim0 = sample.x0.size(-1)
        elif hasattr(sample, 'x'):
            in_dim0 = sample.x.size(-1)
        else:
            in_dim0 = 773 # 默认fallback
            
        # Scale 1 和 Scale 2 如果没有特征，给个默认值 128 (后面模型里会初始化为0)
        in_dim1 = sample.x1.size(-1) if hasattr(sample, 'x1') else 128
        in_dim2 = sample.x2.size(-1) if hasattr(sample, 'x2') else 128
        
        logger.info(f"Detected dims: in_dim0={in_dim0}, in_dim1={in_dim1}, in_dim2={in_dim2}")
    else:
        in_dim0, in_dim1, in_dim2 = 773, 128, 128

    # 4. 初始化模型
    model = IMMPNNWebshellClassifier(
        in_dim0=in_dim0,
        in_dim1=in_dim1,
        in_dim2=in_dim2,
        hidden_dim=cfg.hidden_dim,
        gnn_layers=cfg.gnn_layers,
        inter_steps=cfg.inter_steps,
        dropout=cfg.dropout,
        num_classes=cfg.num_classes
    ).to(device)

    # 5. 初始化 Trainer
    # [修正] 这里严格匹配 trainers/trainer.py 的 __init__
    # 不要传 optimizer, criterion
    trainer = Trainer(
        model=model,
        device=device,
        logger=logger,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip
    )

    # 6. 开始训练
    logger.info("Start training...")
    state = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.epochs,
        ckpt_dir=cfg.ckpt_dir,
        ckpt_name="im_mpnn_best.pt"
    )

    # 7. 测试
    if state.best_path:
        logger.info(f"Loading best checkpoint: {state.best_path}")
        model.load_state_dict(torch.load(state.best_path, map_location=device))

    test_loss, test_metrics = evaluate(model, test_loader, device)
    logger.info(
        f"[TEST] loss={test_loss:.4f} "
        f"acc={test_metrics['acc']:.4f} f1={test_metrics['f1']:.4f} "
        f"prec={test_metrics['precision']:.4f} rec={test_metrics['recall']:.4f}"
    )

if __name__ == "__main__":
    main()
    