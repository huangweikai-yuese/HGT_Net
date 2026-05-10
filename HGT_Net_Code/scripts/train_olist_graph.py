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
from hgtnet.data.olist_graph import build_olist_heterodata
from hgtnet.models.olist_hgt import OlistHGT
from hgtnet.training.early_stopping import EarlyStopping
from hgtnet.training.losses import BinaryFocalLoss
from hgtnet.training.metrics import binary_metrics
from hgtnet.utils import ensure_dir, get_device, save_json, set_seed


def eval_split(model, data, mask_name: str):
    model.eval()
    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict)
        mask = data["customer"][mask_name]
        y = data["customer"].y[mask].detach().cpu().numpy()
        p = torch.sigmoid(logits[mask]).detach().cpu().numpy()
    return y, p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/olist_hgt.yaml"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/olist_hgt"))
    args = parser.parse_args()
    cfg = merge_overrides(load_config(args.config), epochs=args.epochs)
    set_seed(int(cfg["seed"]))
    out = ensure_dir(args.output_dir)
    device = get_device()

    data, input_dims, feature_cols = build_olist_heterodata(
        ROOT / cfg["data_dir"],
        seed=int(cfg["seed"]),
        observation_months=int(cfg["observation_months"]),
        prediction_months=int(cfg["prediction_months"]),
        val_size=float(cfg["val_size"]),
        label_strategy=str(cfg.get("label_strategy", "temporal_future")),
        feature_set=str(cfg.get("feature_set", "all")),
    )
    data = data.to(device)
    model = OlistHGT(
        metadata=data.metadata(),
        input_dims=input_dims,
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
        optimizer.zero_grad(set_to_none=True)
        logits = model(data.x_dict, data.edge_index_dict)
        mask = data["customer"].train_mask
        loss = criterion(logits[mask], data["customer"].y[mask])
        loss.backward()
        optimizer.step()
        scheduler.step()
        y_val, p_val = eval_split(model, data, "val_mask")
        val_metrics = binary_metrics(y_val, p_val)
        row = {"epoch": epoch, "loss": float(loss.detach().cpu()), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(row)
        if stopper.step(val_metrics["pr_auc"], model):
            break

    stopper.restore(model)
    y_val, p_val = eval_split(model, data, "val_mask")
    threshold = binary_metrics(y_val, p_val)["threshold"]
    y_test, p_test = eval_split(model, data, "test_mask")
    metrics = binary_metrics(y_test, p_test, threshold=threshold)
    save_json({"config": cfg, "metrics": metrics, "history": history, "features": feature_cols}, out / "results.json")
    torch.save(model.state_dict(), out / "model.pt")
    print({"test": metrics, "output_dir": str(out)})


if __name__ == "__main__":
    main()
