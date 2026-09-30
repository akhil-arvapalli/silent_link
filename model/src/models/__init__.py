"""Model architectures for gesture classification.

Input contract (Phase 1 target):
    (N, T=40, V=42, C=3)  ->  logits over gloss classes

- baseline.py: Conv1D + BiLSTM baseline
- stgcn.py:    ST-GCN (spatial-temporal graph conv) primary
"""

from .baseline import Conv1D_BiLSTM
from .stgcn import STGCN

__all__ = ["Conv1D_BiLSTM", "STGCN"]
