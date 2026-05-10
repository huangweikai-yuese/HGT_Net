from __future__ import annotations

import torch
from torch import nn


class FeatureInteractionTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.value_projection = nn.Linear(1, hidden_dim)
        self.feature_embedding = nn.Parameter(torch.empty(input_dim, hidden_dim))
        nn.init.normal_(self.feature_embedding, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.value_projection(x.unsqueeze(-1)) + self.feature_embedding.unsqueeze(0)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.head(pooled).squeeze(-1)

