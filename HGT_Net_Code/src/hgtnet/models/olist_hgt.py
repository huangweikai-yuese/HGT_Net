from __future__ import annotations

import torch
from torch import nn


class OlistHGT(nn.Module):
    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        input_dims: dict[str, int],
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        try:
            from torch_geometric.nn import HGTConv
        except ImportError as exc:
            raise ImportError("OlistHGT requires torch-geometric.") from exc

        self.projections = nn.ModuleDict({
            node_type: nn.Linear(dim, hidden_dim) for node_type, dim in input_dims.items()
        })
        self.convs = nn.ModuleList([
            HGTConv(hidden_dim, hidden_dim, metadata, heads=num_heads) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleDict({node_type: nn.LayerNorm(hidden_dim) for node_type in input_dims})
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict) -> torch.Tensor:
        x_dict = {
            node_type: self.dropout(torch.relu(self.projections[node_type](x)))
            for node_type, x in x_dict.items()
        }
        for conv in self.convs:
            updated = conv(x_dict, edge_index_dict)
            x_dict = {
                node_type: self.norms[node_type](self.dropout(updated[node_type]) + x_dict[node_type])
                for node_type in x_dict
            }
        return self.classifier(x_dict["customer"]).squeeze(-1)
