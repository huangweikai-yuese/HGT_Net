from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hgtnet.config import load_config, merge_overrides
from hgtnet.data.ospi import make_ospi_loaders
from hgtnet.models.ospi_transformer import FeatureInteractionTransformer
from hgtnet.training.early_stopping import EarlyStopping
from hgtnet.training.losses import BinaryFocalLoss
from hgtnet.training.metrics import binary_metrics
from hgtnet.utils import ensure_dir, get_device, save_json, set_seed


def predict(model: torch.nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/ospi_transformer.yaml"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/ospi_transformer"))
    args = parser.parse_args()

    cfg = merge_overrides(load_config(args.config), epochs=args.epochs)
    set_seed(int(cfg["seed"]))
    output_dir = ensure_dir(args.output_dir)
    device = get_device()

    data = make_ospi_loaders(
        ROOT / cfg["data_dir"],
        batch_size=int(cfg["batch_size"]),
        seed=int(cfg["seed"]),
        val_size=float(cfg["val_size"]),
        test_size=float(cfg["test_size"]),
    )
    model = FeatureInteractionTransformer(
        input_dim=data.input_dim,
        hidden_dim=int(cfg["hidden_dim"]),
        num_heads=int(cfg["num_heads"]),
        num_layers=int(cfg["num_layers"]),
        dropout=float(cfg["dropout"]),
    ).to(device)
    criterion = BinaryFocalLoss(gamma=float(cfg["focal_gamma"]))
    optimizer = AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(cfg["epochs"])))
    stopper = EarlyStopping(patience=int(cfg["early_stopping_patience"]), mode="max")

    history = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        losses = []
        for x, y in data.train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        y_val, p_val = predict(model, data.val_loader, device)
        val_metrics = binary_metrics(y_val, p_val)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(row)
        if stopper.step(val_metrics["pr_auc"], model):
            break

    stopper.restore(model)
    y_val, p_val = predict(model, data.val_loader, device)
    threshold = binary_metrics(y_val, p_val)["threshold"]
    y_test, p_test = predict(model, data.test_loader, device)
    metrics = binary_metrics(y_test, p_test, threshold=threshold)
    save_json({"config": cfg, "metrics": metrics, "history": history, "features": data.feature_names}, output_dir / "results.json")
    torch.save(model.state_dict(), output_dir / "model.pt")
    print({"test": metrics, "output_dir": str(output_dir)})


if __name__ == "__main__":
    main()

