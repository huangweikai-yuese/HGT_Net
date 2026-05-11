from __future__ import annotations

import copy
from dataclasses import dataclass

import torch


@dataclass
class EarlyStopping:
    patience: int
    mode: str = "max"

    def __post_init__(self) -> None:
        self.best_score: float | None = None
        self.best_state: dict[str, torch.Tensor] | None = None
        self.bad_epochs = 0

    def step(self, score: float, model: torch.nn.Module) -> bool:
        improved = self.best_score is None
        if self.best_score is not None:
            improved = score > self.best_score if self.mode == "max" else score < self.best_score
        if improved:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model: torch.nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

