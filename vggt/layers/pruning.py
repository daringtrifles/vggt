import torch
from torch import nn, Tensor


class TokenPruner(nn.Module):
    """
    A small MLP that predicts a scalar importance score per token.

    Inputs: x with shape [B, N, C]
    Outputs: scores with shape [B, N]
    """

    def __init__(self, dim: int, hidden_ratio: float = 0.25, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = max(1, int(dim * hidden_ratio))
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, N, C] -> scores: [B, N]
        scores = self.mlp(x).squeeze(-1)
        return scores


