# HGT-Net

Official project code for **HGT-Net: A Unified Prediction Framework for E-Commerce Customer Retention and Session Purchase Intention via Heterogeneous Graph Attention and Feature-Interaction Transformers**.

The repository contains two pipelines:

- **Customer retention / repeat-purchase prediction** using heterogeneous graph modeling over customers, orders, products, and sellers.
- **Session purchase-intention prediction** using a feature-interaction Transformer over session-level tabular features.

Dataset links are listed in [data/README.md](data/README.md).

## Quick Checks

```bash
python scripts/train_ospi.py --config configs/ospi_transformer.yaml --epochs 3 --output-dir outputs/ospi_smoke
```

## Main Experiments

```bash
python scripts/train_ospi.py --config configs/ospi_transformer.yaml --output-dir outputs/ospi_transformer
python scripts/train_baselines.py --dataset ospi --output-dir outputs/ospi_baselines
python scripts/train_baselines.py --dataset olist --output-dir outputs/olist_baselines
python scripts/train_olist_graph.py --config configs/olist_hgt.yaml --output-dir outputs/olist_hgt
```
## Metrics

All training scripts report ROC-AUC, PR-AUC, F1, recall, MCC, Brier score, and the selected classification threshold.

## Configuration

The checked-in YAML files contain the selected training settings and a `search_space` section with the candidate values used during model selection. The active scalar values are the final settings used by the training scripts.

The graph configuration uses five delivery and payment features: `retraso_dias`, `tiempo_entrega`, `total_pago`, `total_flete`, and `tiempo_estimado`.
