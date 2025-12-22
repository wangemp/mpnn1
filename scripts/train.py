import os
import torch
from torch_geometric.loader import DataLoader

from configs import Config
from utils import get_logger, set_seed
from datasets import GraphPTDataset
from models import IMMPNNWebshellClassifier
from trainers import Trainer
from trainers.evaluator import evaluate


def main():
    cfg = Config()
    cfg.ensure_dirs()
    set_seed(cfg.seed)

    log_path = os.path.join(cfg.log_dir, "train.log")
    logger = get_logger("train", log_path)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_ds = GraphPTDataset(os.path.join(cfg.data_processed_dir, "train.pt"))
    val_ds = GraphPTDataset(os.path.join(cfg.data_processed_dir, "val.pt"))
    test_ds = GraphPTDataset(os.path.join(cfg.data_processed_dir, "test.pt"))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, follow_batch=["x0", "x1", "x2"])
    val_loader = DataLoader(val_ds, batch_size=max(32, cfg.batch_size), shuffle=False, follow_batch=["x0", "x1", "x2"])
    test_loader = DataLoader(test_ds, batch_size=max(32, cfg.batch_size), shuffle=False, follow_batch=["x0", "x1", "x2"])


    # infer input dims from one sample
    sample = train_ds[0]
    in_dim0 = sample.x0.size(-1)
    in_dim1 = sample.x1.size(-1)
    in_dim2 = sample.x2.size(-1)

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

    trainer = Trainer(
        model=model,
        device=device,
        logger=logger,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip
    )

    state = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.epochs,
        ckpt_dir=cfg.ckpt_dir,
        ckpt_name="im_mpnn_best.pt"
    )

    # load best and test
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
