import os
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

from trainers.evaluator import evaluate


@dataclass
class TrainState:
    best_f1: float = -1.0
    best_path: Optional[str] = None


class Trainer:
    def __init__(self, model, device, logger, lr=1e-3, weight_decay=1e-4, grad_clip=2.0):
        self.model = model
        self.device = device
        self.logger = logger
        self.grad_clip = grad_clip
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def fit(self, train_loader, val_loader, epochs: int, ckpt_dir: str, ckpt_name: str = "best.pt") -> TrainState:
        os.makedirs(ckpt_dir, exist_ok=True)
        state = TrainState()

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_n = 0

            for data in train_loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()

                logits = self.model(data)
                y = data.y.view(-1).long()
                loss = F.cross_entropy(logits, y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                total_loss += loss.item() * y.size(0)
                total_n += y.size(0)

            train_loss = total_loss / max(1, total_n)
            val_loss, val_metrics = evaluate(self.model, val_loader, self.device)

            self.logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"acc={val_metrics['acc']:.4f} f1={val_metrics['f1']:.4f} "
                f"prec={val_metrics['precision']:.4f} rec={val_metrics['recall']:.4f}"
            )

            if val_metrics["f1"] > state.best_f1:
                state.best_f1 = val_metrics["f1"]
                best_path = os.path.join(ckpt_dir, ckpt_name)
                torch.save(self.model.state_dict(), best_path)
                state.best_path = best_path
                self.logger.info(f"Saved best model to: {best_path} (best_f1={state.best_f1:.4f})")

        return state
