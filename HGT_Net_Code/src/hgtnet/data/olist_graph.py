from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from hgtnet.data.olist import OLIST_TOP5_FEATURES, build_customer_features


def build_olist_heterodata(
    data_dir: str | Path,
    seed: int,
    observation_months: int = 18,
    prediction_months: int = 6,
    val_size: float = 0.2,
    label_strategy: str = "temporal_future",
    feature_set: str = "all",
):
    try:
        from torch_geometric.data import HeteroData
    except ImportError as exc:
        raise ImportError("Olist graph construction requires torch-geometric.") from exc

    root = Path(data_dir)
    customer_df = build_customer_features(root, observation_months, prediction_months, label_strategy=label_strategy)
    if feature_set == "all":
        feature_cols = [c for c in customer_df.columns if c not in {"customer_unique_id", "label"}]
    elif feature_set == "olist_top5":
        feature_cols = OLIST_TOP5_FEATURES
    else:
        raise ValueError(f"Unknown Olist feature_set: {feature_set}")
    scaler = StandardScaler()
    customer_x = scaler.fit_transform(customer_df[feature_cols].to_numpy(dtype="float32")).astype("float32")
    y = customer_df["label"].to_numpy(dtype=int)

    orders = pd.read_csv(root / "olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp"])
    items = pd.read_csv(root / "olist_order_items_dataset.csv")
    products = pd.read_csv(root / "olist_products_dataset.csv")
    sellers = pd.read_csv(root / "olist_sellers_dataset.csv")
    customers = pd.read_csv(root / "olist_customers_dataset.csv")
    orders = orders.merge(customers[["customer_id", "customer_unique_id"]], on="customer_id", how="left")
    start = orders["order_purchase_timestamp"].min()
    cutoff = start + pd.DateOffset(months=observation_months)
    if label_strategy == "temporal_future":
        orders = orders[(orders["order_status"] == "delivered") & (orders["order_purchase_timestamp"] <= cutoff)]
    elif label_strategy == "first_order_repeat":
        orders = orders[orders["order_status"] == "delivered"].sort_values("order_purchase_timestamp")
        first_ids = orders.groupby("customer_unique_id", as_index=False).head(1)["order_id"]
        orders = orders[orders["order_id"].isin(first_ids)]
    elif label_strategy == "all_history_repeat":
        orders = orders[orders["order_status"] == "delivered"]
    else:
        raise ValueError(f"Unknown Olist label_strategy: {label_strategy}")

    customer_ids = customer_df["customer_unique_id"].tolist()
    order_ids = orders["order_id"].drop_duplicates().tolist()
    items = items[items["order_id"].isin(order_ids)].copy()
    product_ids = pd.Index(items["product_id"].dropna().unique()).intersection(products["product_id"]).tolist()
    seller_ids = pd.Index(items["seller_id"].dropna().unique()).intersection(sellers["seller_id"]).tolist()

    maps = {
        "customer": {v: i for i, v in enumerate(customer_ids)},
        "order": {v: i for i, v in enumerate(order_ids)},
        "product": {v: i for i, v in enumerate(product_ids)},
        "seller": {v: i for i, v in enumerate(seller_ids)},
    }

    data = HeteroData()
    data["customer"].x = torch.from_numpy(customer_x)
    data["customer"].y = torch.from_numpy(y.astype("float32"))
    data["order"].x = torch.ones((len(order_ids), 1), dtype=torch.float32)
    data["product"].x = torch.ones((len(product_ids), 1), dtype=torch.float32)
    data["seller"].x = torch.ones((len(seller_ids), 1), dtype=torch.float32)

    co = orders[orders["customer_unique_id"].isin(maps["customer"]) & orders["order_id"].isin(maps["order"])]
    co_src = [maps["customer"][v] for v in co["customer_unique_id"]]
    co_dst = [maps["order"][v] for v in co["order_id"]]
    data["customer", "places", "order"].edge_index = torch.tensor([co_src, co_dst], dtype=torch.long)
    data["order", "rev_places", "customer"].edge_index = torch.tensor([co_dst, co_src], dtype=torch.long)

    op = items[items["order_id"].isin(maps["order"]) & items["product_id"].isin(maps["product"])]
    op_src = [maps["order"][v] for v in op["order_id"]]
    op_dst = [maps["product"][v] for v in op["product_id"]]
    data["order", "contains", "product"].edge_index = torch.tensor([op_src, op_dst], dtype=torch.long)
    data["product", "rev_contains", "order"].edge_index = torch.tensor([op_dst, op_src], dtype=torch.long)

    ps = items[items["product_id"].isin(maps["product"]) & items["seller_id"].isin(maps["seller"])]
    ps_src = [maps["product"][v] for v in ps["product_id"]]
    ps_dst = [maps["seller"][v] for v in ps["seller_id"]]
    data["product", "sold_by", "seller"].edge_index = torch.tensor([ps_src, ps_dst], dtype=torch.long)
    data["seller", "rev_sold_by", "product"].edge_index = torch.tensor([ps_dst, ps_src], dtype=torch.long)

    idx = np.arange(len(customer_ids))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_size, random_state=seed, stratify=y[train_idx])
    for name, split_idx in {"train_mask": train_idx, "val_mask": val_idx, "test_mask": test_idx}.items():
        mask = torch.zeros(len(customer_ids), dtype=torch.bool)
        mask[torch.from_numpy(split_idx)] = True
        data["customer"][name] = mask

    return data, {"customer": customer_x.shape[1], "order": 1, "product": 1, "seller": 1}, feature_cols
