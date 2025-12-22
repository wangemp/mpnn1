import os
import torch
from torch_geometric.loader import DataLoader

from configs import Config
from utils import get_logger, set_seed
from datasets import GraphPTDataset
from models import IMMPNNWebshellClassifier
from trainers.evaluator import evaluate


def main():
    cfg = Config()
    cfg.ensure_dirs()
    set_seed(cfg.seed)

    logger = get_logger("test", os.path.join(cfg.log_dir, "test.log"))
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 1) Load test dataset
    test_ds = GraphPTDataset(os.path.join(cfg.data_processed_dir, "test.pt"))

    # 2) IMPORTANT: follow_batch makes PyG auto-generate x0_batch/x1_batch/x2_batch
    #    which your classifier forward expects.
    test_loader = DataLoader(
        test_ds,
        batch_size=max(32, cfg.batch_size),
        shuffle=False,
        follow_batch=["x0", "x1", "x2"]
    )

    # 3) Build model with inferred input dims from a sample
    sample = test_ds[0]
    model = IMMPNNWebshellClassifier(
        in_dim0=sample.x0.size(-1),
        in_dim1=sample.x1.size(-1),
        in_dim2=sample.x2.size(-1),
        hidden_dim=cfg.hidden_dim,
        gnn_layers=cfg.gnn_layers,
        inter_steps=cfg.inter_steps,
        dropout=cfg.dropout,
        num_classes=cfg.num_classes
    ).to(device)

    # 4) Load checkpoint
    ckpt_path = os.path.join(cfg.ckpt_dir, "im_mpnn_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # 5) Evaluate
    loss, metrics = evaluate(model, test_loader, device)
    logger.info(
        f"[TEST] loss={loss:.4f} "
        f"acc={metrics['acc']:.4f} f1={metrics['f1']:.4f} "
        f"prec={metrics['precision']:.4f} rec={metrics['recall']:.4f}"
    )


if __name__ == "__main__":
    main()
