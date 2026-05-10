from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hgtnet.data.olist import make_olist_tabular
from hgtnet.data.ospi import make_ospi_loaders
from hgtnet.training.metrics import binary_metrics
from hgtnet.utils import ensure_dir, save_json, set_seed


def optional_models(seed: int):
    models = {}
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03, eval_metric="logloss", random_state=seed)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(n_estimators=400, learning_rate=0.03, random_state=seed, verbose=-1)
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier
        models["catboost"] = CatBoostClassifier(iterations=400, learning_rate=0.03, depth=6, verbose=False, random_seed=seed)
    except ImportError:
        pass
    return models


def ospi_arrays(seed: int):
    data = make_ospi_loaders(ROOT / "data/raw/ospi", batch_size=100000, seed=seed)
    def unpack(loader):
        x, y = next(iter(loader))
        return x.numpy(), y.numpy().astype(int)
    return (*unpack(data.train_loader), *unpack(data.val_loader), *unpack(data.test_loader))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ospi", "olist"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    out = ensure_dir(args.output_dir)

    if args.dataset == "ospi":
        x_train, y_train, x_val, y_val, x_test, y_test = ospi_arrays(args.seed)
    else:
        data = make_olist_tabular(
            ROOT / "data/raw/olist",
            seed=args.seed,
            label_strategy="all_history_repeat",
            feature_set="olist_top5",
        )
        x_train, y_train, x_val, y_val, x_test, y_test = data.x_train, data.y_train, data.x_val, data.y_val, data.x_test, data.y_test

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=args.seed, n_jobs=-1),
        "naive_bayes": GaussianNB(),
        "knn": KNeighborsClassifier(n_neighbors=25),
        **optional_models(args.seed),
    }
    report = {}
    for name, model in models.items():
        print(f"training {name}")
        model.fit(x_train, y_train)
        val_score = model.predict_proba(x_val)[:, 1]
        threshold = binary_metrics(y_val, val_score)["threshold"]
        test_score = model.predict_proba(x_test)[:, 1]
        report[name] = binary_metrics(y_test, test_score, threshold=threshold)
        print(name, report[name])
    save_json(report, out / "baseline_results.json")


if __name__ == "__main__":
    main()
