from __future__ import annotations

import torch
import torch.nn as nn


class CTRMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DeepFM(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 16):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.feature_embeddings = nn.Parameter(torch.randn(input_dim, embedding_dim) * 0.01)
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.output = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_term = self.linear(x)

        xv = torch.matmul(x, self.feature_embeddings)
        xv_square = xv * xv
        x_square = x * x
        v_square = self.feature_embeddings * self.feature_embeddings
        second_order = 0.5 * torch.sum(xv_square - torch.matmul(x_square, v_square), dim=1, keepdim=True)

        deep_term = self.deep(x)
        return self.output(linear_term + second_order + deep_term)


class TwoTowerRecall(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, embedding_dim: int = 64):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def encode_user(self, x: torch.Tensor) -> torch.Tensor:
        return self.user_tower(x)

    def encode_item(self, x: torch.Tensor) -> torch.Tensor:
        return self.item_tower(x)

    def forward(self, user_x: torch.Tensor, item_x: torch.Tensor) -> torch.Tensor:
        user_emb = self.encode_user(user_x)
        item_emb = self.encode_item(item_x)
        user_emb = torch.nn.functional.normalize(user_emb, dim=1)
        item_emb = torch.nn.functional.normalize(item_emb, dim=1)
        return torch.sum(user_emb * item_emb, dim=1, keepdim=True)
