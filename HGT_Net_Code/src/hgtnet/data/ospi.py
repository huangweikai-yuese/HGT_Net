from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class OspiData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    input_dim: int
    feature_names: list[str]
    y_test: np.ndarray


NUMERIC_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]


def load_ospi_dataframe(data_dir: str | Path) -> pd.DataFrame:
    path = Path(data_dir) / "online_shoppers_intention.csv"
    df = pd.read_csv(path)
    if "Revenue" not in df.columns:
        raise ValueError(f"Missing Revenue column in {path}")
    return df


def make_ospi_loaders(
    data_dir: str | Path,
    batch_size: int,
    seed: int,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> OspiData:
    df = load_ospi_dataframe(data_dir)
    y = df["Revenue"].astype(bool).astype(int).to_numpy()
    x = df.drop(columns=["Revenue"])
    categorical = [c for c in x.columns if c not in NUMERIC_COLUMNS]
    numeric = [c for c in NUMERIC_COLUMNS if c in x.columns]

    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    val_fraction = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=val_fraction, random_state=seed, stratify=y_trainval
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scale", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    x_train_np = preprocessor.fit_transform(x_train).astype("float32")
    x_val_np = preprocessor.transform(x_val).astype("float32")
    x_test_np = preprocessor.transform(x_test).astype("float32")
    feature_names = list(preprocessor.get_feature_names_out())

    def loader(features: np.ndarray, labels: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels.astype("float32")))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return OspiData(
        train_loader=loader(x_train_np, y_train, True),
        val_loader=loader(x_val_np, y_val, False),
        test_loader=loader(x_test_np, y_test, False),
        input_dim=x_train_np.shape[1],
        feature_names=feature_names,
        y_test=y_test,
    )

