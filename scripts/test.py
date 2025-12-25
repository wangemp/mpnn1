import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from configs import Config
from models.classifier import IMMPNNWebshellClassifier
from utils.logger import get_logger
from utils.seed import set_seed

def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs
            
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    return avg_loss, acc, f1, p, r

def main():
    cfg = Config()
    set_seed(cfg.seed)
    
    logger = get_logger(os.path.join(cfg.log_dir, "test.log"))
    logger.info(f"Start testing on device: {cfg.device}")

    # 1. Load Data
    test_path = os.path.join(cfg.data_processed_dir, "test.pt")
    if not os.path.exists(test_path):
        logger.error("Test data not found.")
        return
        
    test_dataset = torch.load(test_path)
    logger.info(f"Test size: {len(test_dataset)}")
    
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    # ============================================================
    # [核心修改] 动态获取维度
    # ============================================================
    if len(test_dataset) > 0:
        in_dim = test_dataset[0].x.shape[1]
    else:
        in_dim = 773 

    # 2. Build Model
    model = IMMPNNWebshellClassifier(
        in_channels=in_dim,  # <--- 使用动态获取的维度
        hidden_dim=cfg.hidden_dim,
        num_classes=cfg.num_classes,
        gnn_layers=cfg.gnn_layers,
        inter_steps=cfg.inter_steps,
        dropout=cfg.dropout
    ).to(cfg.device)

    # 3. Load Checkpoint
    ckpt_path = os.path.join(cfg.ckpt_dir, "im_mpnn_best.pt")
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found at {ckpt_path}")
        return
        
    logger.info(f"Loading checkpoint from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))

    # 4. Evaluate
    criterion = nn.CrossEntropyLoss()
    loss, acc, f1, prec, rec = evaluate(model, test_loader, cfg.device, criterion)
    
    logger.info(f"[TEST] loss={loss:.4f} acc={acc:.4f} f1={f1:.4f} prec={prec:.4f} rec={rec:.4f}")

if __name__ == "__main__":
    main()