"""ST-GCN (Spatial-Temporal Graph Convolutional Network) gesture classifier.

Primary architecture (Yan, Xiong, Lin, 2018) applied to hand landmarks.

Input:  (N, T=40, V=42, C=3)
Output: (N, n_classes) logits

Spatial dimension is the 42 hand joints; the two hands (21 landmarks each)
share an identical intra-hand skeleton graph, so the adjacency is a 42x42
block-diagonal matrix. Temporal convolution runs across the T frames.
"""

from __future__ import annotations

import torch
from torch import nn

NUM_LANDMARKS = 21

# MediaPipe hand skeleton: landmark 0 is the wrist; 4 finger chains + thumb.
_HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
]


def hand_adjacency(v: int = NUM_LANDMARKS) -> torch.Tensor:
    """Return a (v, v) normalized adjacency for a single hand skeleton."""
    adj = torch.zeros(v, v)
    for i, j in _HAND_EDGES:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    adj = adj + torch.eye(v)
    deg = adj.sum(dim=1).clamp_min(1.0)
    d_inv_sqrt = deg.pow(-0.5)
    return d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]


def build_adjacency(v: int = 2 * NUM_LANDMARKS) -> torch.Tensor:
    """Build a (v, v) block-diagonal adjacency for both hands."""
    block = hand_adjacency(NUM_LANDMARKS)
    adj = torch.zeros(v, v)
    adj[:NUM_LANDMARKS, :NUM_LANDMARKS] = block
    adj[NUM_LANDMARKS:, NUM_LANDMARKS:] = block
    return adj


class STGCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, adj: torch.Tensor, stride: int = 1) -> None:
        super().__init__()
        self.register_buffer("adj", adj, persistent=False)
        self.gcn = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.tconv = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=(9, 1),
            padding=(4, 0),
            stride=(stride, 1),
        )
        self.tbn = nn.BatchNorm2d(out_ch)
        self.residual = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        N, C, T, V = x.shape
        res = self.residual(x)
        # spatial graph conv: aggregate neighbor features
        x_flat = x.permute(0, 2, 3, 1).reshape(N * T, V, C)  # (N*T, V, C)
        agg = torch.einsum("nvw, nwc -> nvc", self.adj.unsqueeze(0), x_flat)
        x = agg.reshape(N, T, V, C).permute(0, 3, 1, 2)  # (N, C, T, V)
        x = self.relu(self.bn(self.gcn(x)))
        x = self.relu(self.tbn(self.tconv(x)))
        return x + res


class STGCN(nn.Module):
    def __init__(
        self,
        n_classes: int,
        T: int = 40,
        V: int = 42,
        C: int = 3,
        hidden: list[int] | None = None,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.T = T
        self.V = V
        self.C = C
        adj = build_adjacency(V)
        channels = hidden or [64, 128, 128]

        layers: list[nn.Module] = [STGCNBlock(C, channels[0], adj, stride=1)]
        for i in range(1, len(channels)):
            stride = 2 if channels[i] != channels[i - 1] else 1
            layers.append(STGCNBlock(channels[i - 1], channels[i], adj, stride=stride))
        self.blocks = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(channels[-1], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, V, C) -> (N, C, T, V)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.blocks(x)
        x = x.mean(dim=(2, 3))  # global avg pool over T and V
        x = self.dropout(x)
        return self.head(x)
