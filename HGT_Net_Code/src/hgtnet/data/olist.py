from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class OlistTabularData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    customer_ids: np.ndarray
    train_customer_ids: np.ndarray
    val_customer_ids: np.ndarray
    test_customer_ids: np.ndarray


OLIST_TOP5_FEATURES = ["retraso_dias", "tiempo_entrega", "total_pago", "total_flete", "tiempo_estimado"]


def _read_olist(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(data_dir)
    return {
        "customers": pd.read_csv(root / "olist_customers_dataset.csv"),
        "orders": pd.read_csv(root / "olist_orders_dataset.csv", parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]),
        "items": pd.read_csv(root / "olist_order_items_dataset.csv"),
        "payments": pd.read_csv(root / "olist_order_payments_dataset.csv"),
        "reviews": pd.read_csv(root / "olist_order_reviews_dataset.csv"),
    }


def build_customer_features(
    data_dir: str | Path,
    observation_months: int = 18,
    prediction_months: int = 6,
    label_strategy: str = "temporal_future",
) -> pd.DataFrame:
    data = _read_olist(data_dir)
    orders = data["orders"]
    delivered = orders[orders["order_status"] == "delivered"].copy()
    delivered = delivered.dropna(subset=["order_purchase_timestamp", "order_delivered_customer_date"])
    start = delivered["order_purchase_timestamp"].min()
    cutoff = start + pd.DateOffset(months=observation_months)
    future_end = cutoff + pd.DateOffset(months=prediction_months)

    customers = data["customers"][["customer_id", "customer_unique_id"]]
    delivered = delivered.merge(customers, on="customer_id", how="left")
    if label_strategy == "temporal_future":
        obs_orders = delivered[delivered["order_purchase_timestamp"] <= cutoff].copy()
        future_orders = delivered[
            (delivered["order_purchase_timestamp"] > cutoff)
            & (delivered["order_purchase_timestamp"] <= future_end)
        ].copy()
        repeat_customers = set(future_orders["customer_unique_id"].dropna())
    elif label_strategy == "first_order_repeat":
        delivered = delivered.sort_values("order_purchase_timestamp")
        counts = delivered.groupby("customer_unique_id")["order_id"].nunique()
        first_ids = delivered.groupby("customer_unique_id", as_index=False).head(1)["order_id"]
        obs_orders = delivered[delivered["order_id"].isin(first_ids)].copy()
        repeat_customers = set(counts[counts > 1].index)
    elif label_strategy == "all_history_repeat":
        counts = delivered.groupby("customer_unique_id")["order_id"].nunique()
        obs_orders = delivered.copy()
        repeat_customers = set(counts[counts > 1].index)
    else:
        raise ValueError(f"Unknown Olist label_strategy: {label_strategy}")

    payments = data["payments"].groupby("order_id", as_index=False).agg(
        total_pago=("payment_value", "sum"),
        installments=("payment_installments", "sum"),
    )
    items = data["items"].groupby("order_id", as_index=False).agg(
        total_flete=("freight_value", "sum"),
        item_count=("order_item_id", "count"),
        product_count=("product_id", "nunique"),
        seller_count=("seller_id", "nunique"),
    )
    reviews = data["reviews"].groupby("order_id", as_index=False).agg(review_score=("review_score", "mean"))

    obs = obs_orders.merge(payments, on="order_id", how="left").merge(items, on="order_id", how="left").merge(reviews, on="order_id", how="left")
    obs["tiempo_entrega"] = (obs["order_delivered_customer_date"] - obs["order_purchase_timestamp"]).dt.days
    obs["tiempo_estimado"] = (obs["order_estimated_delivery_date"] - obs["order_purchase_timestamp"]).dt.days
    obs["retraso_dias"] = (obs["order_delivered_customer_date"] - obs["order_estimated_delivery_date"]).dt.days.clip(lower=0)

    agg = obs.groupby("customer_unique_id", as_index=False).agg(
        historical_order_count=("order_id", "nunique"),
        total_pago=("total_pago", "sum"),
        avg_pago=("total_pago", "mean"),
        total_flete=("total_flete", "sum"),
        avg_flete=("total_flete", "mean"),
        item_count=("item_count", "sum"),
        product_count=("product_count", "sum"),
        seller_count=("seller_count", "sum"),
        installments=("installments", "sum"),
        review_score=("review_score", "mean"),
        tiempo_entrega=("tiempo_entrega", "mean"),
        tiempo_estimado=("tiempo_estimado", "mean"),
        retraso_dias=("retraso_dias", "mean"),
        first_order=("order_purchase_timestamp", "min"),
        last_order=("order_purchase_timestamp", "max"),
    )
    agg["recency_days"] = (cutoff - agg["last_order"]).dt.days
    agg["customer_lifetime_days"] = (agg["last_order"] - agg["first_order"]).dt.days
    agg["label"] = agg["customer_unique_id"].isin(repeat_customers).astype(int)
    agg = agg.drop(columns=["first_order", "last_order"]).fillna(0)
    return agg


def make_olist_tabular(
    data_dir: str | Path,
    seed: int,
    observation_months: int = 18,
    prediction_months: int = 6,
    val_size: float = 0.2,
    label_strategy: str = "temporal_future",
    feature_set: str = "all",
) -> OlistTabularData:
    df = build_customer_features(data_dir, observation_months, prediction_months, label_strategy=label_strategy)
    if feature_set == "all":
        feature_names = [c for c in df.columns if c not in {"customer_unique_id", "label"}]
    elif feature_set == "olist_top5":
        feature_names = OLIST_TOP5_FEATURES
    else:
        raise ValueError(f"Unknown Olist feature_set: {feature_set}")
    y = df["label"].to_numpy(dtype=int)
    ids = df["customer_unique_id"].to_numpy()
    x = df[feature_names].to_numpy(dtype="float32")

    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=seed, stratify=y)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_size, random_state=seed, stratify=y[train_idx])

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_idx]).astype("float32")
    x_val = scaler.transform(x[val_idx]).astype("float32")
    x_test = scaler.transform(x[test_idx]).astype("float32")

    return OlistTabularData(
        x_train=x_train,
        y_train=y[train_idx],
        x_val=x_val,
        y_val=y[val_idx],
        x_test=x_test,
        y_test=y[test_idx],
        feature_names=feature_names,
        customer_ids=ids,
        train_customer_ids=ids[train_idx],
        val_customer_ids=ids[val_idx],
        test_customer_ids=ids[test_idx],
    )
