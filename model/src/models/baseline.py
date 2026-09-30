"""Conv1D + BiLSTM baseline gesture classifier.

Input:  (N, T, V=42, C=3)
Output: (N, n_classes) logits

The (T, V, C) tensor is flattened to (N, C*V, T) so a 1D convolution runs
over the time axis (capturing local temporal patterns), followed by a
bidirectional LSTM over time, temporal pooling, and a linear head.
"""

from __future__ import annotations

import torch
from torch import nn


class Conv1D_BiLSTM(nn.Module):
    def __init__(
        self,
        n_classes: int,
        T: int = 40,
        V: int = 42,
        C: int = 3,
        conv_channels: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.T = T
        self.V = V
        self.C = C
        in_channels = C * V  # 126

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
        )

        self.lstm = nn.LSTM(
            conv_channels,
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, V, C) -> (N, C*V, T)
        x = x.reshape(x.shape[0], self.T, self.V * self.C).transpose(1, 2)
        x = self.conv(x)  # (N, conv_channels, T)
        x = x.transpose(1, 2)  # (N, T, conv_channels)
        x, _ = self.lstm(x)  # (N, T, 2*lstm_hidden)
        x = x.mean(dim=1)  # temporal mean pooling
        x = self.dropout(x)
        return self.head(x)
