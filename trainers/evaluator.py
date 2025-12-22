import torch
import torch.nn.functional as F
from typing import Dict, Tuple, List

from utils.metrics import compute_metrics


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_n = 0

    ys: List[int] = []
    preds: List[int] = []

    for data in loader:
        data = data.to(device)
        logits = model(data)
        y = data.y.view(-1).long()

        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * y.size(0)
        total_n += y.size(0)

        probs = torch.softmax(logits, dim=1)[:, 1]  # 取出属于 Webshell 类别的概率
        threshold = 0.3  # 🔥 关键：降低阈值！从 0.5 降到 0.3 甚至 0.2
        pred = (probs > threshold).long().detach().cpu().tolist()
        ys.extend(y.detach().cpu().tolist())
        preds.extend(pred)

    avg_loss = total_loss / max(1, total_n)
    metrics = compute_metrics(ys, preds)
    return avg_loss, metrics
