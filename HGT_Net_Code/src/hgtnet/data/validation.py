from __future__ import annotations

from pathlib import Path

import pandas as pd

OLIST_FILES = [
    "olist_customers_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_geolocation_dataset.csv",
    "product_category_name_translation.csv",
]

OSPI_FILE = "online_shoppers_intention.csv"


def validate_olist(data_dir: str | Path) -> dict[str, object]:
    data_dir = Path(data_dir)
    missing = [name for name in OLIST_FILES if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Olist files in {data_dir}: {missing}")
    orders = pd.read_csv(data_dir / "olist_orders_dataset.csv")
    required_cols = {"order_id", "customer_id", "order_status", "order_purchase_timestamp"}
    absent = required_cols - set(orders.columns)
    if absent:
        raise ValueError(f"Olist orders file missing columns: {sorted(absent)}")
    return {
        "dataset": "olist",
        "data_dir": str(data_dir),
        "files": len(OLIST_FILES),
        "orders": int(len(orders)),
        "delivered_orders": int((orders["order_status"] == "delivered").sum()),
    }


def validate_ospi(data_dir: str | Path) -> dict[str, object]:
    data_dir = Path(data_dir)
    path = data_dir / OSPI_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing OSPI file: {path}")
    df = pd.read_csv(path)
    if "Revenue" not in df.columns:
        raise ValueError("OSPI file must contain a Revenue target column")
    return {
        "dataset": "ospi",
        "data_dir": str(data_dir),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "positive": int(df["Revenue"].astype(bool).sum()),
    }

