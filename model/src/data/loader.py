"""PyTorch dataset + stratified train/val split for landmark sequences.

Wraps data/processed sequences (see dataset.py) into a torch Dataset with
optional on-the-fly augmentation, and provides a stratified split helper so
every gloss appears in both train and val.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .dataset import discover_sequences, load_sequence


class LandmarkDataset(Dataset):
    """Holds (sequence_id, gloss) records; loads + augments lazily."""

    def __init__(self, records: list[tuple[np.ndarray, int]], augment_fn=None) -> None:
        """records: list of (sequence, gloss_index) tuples."""
        self.records = records
        self.augment_fn = augment_fn

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        seq, label = self.records[idx]
        if self.augment_fn is not None:
            seq = self.augment_fn(seq)
        return torch.from_numpy(np.asarray(seq, dtype=np.float32)), label


def load_dataset(root, glosses: list[str], augment_fn=None) -> LandmarkDataset:
    """Load all sequences for the given glosses into an in-memory dataset.

    Returns (dataset, gloss_to_index) where gloss_to_index maps gloss name
    -> integer label.
    """
    gloss_to_index = {g: i for i, g in enumerate(glosses)}
    records: list[tuple[np.ndarray, int]] = []
    found = discover_sequences(root)
    for gloss in glosses:
        for path in found.get(gloss, []):
            seq = load_sequence(path)
            records.append((seq, gloss_to_index[gloss]))
    return LandmarkDataset(records, augment_fn=augment_fn)


def stratified_split(
    dataset: LandmarkDataset,
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """Split a dataset stratified by gloss so both splits see every gloss.

    Per-gloss indices are shuffled and the last val_frac are held out.
    """
    from collections import defaultdict

    rng = np.random.default_rng(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.records):
        by_label[label].append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for label, idxs in by_label.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n_val = max(1, round(len(idxs) * val_frac))
        n_val = min(n_val, len(idxs) - 1) if len(idxs) > 1 else 0
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def make_loaders(
    dataset: LandmarkDataset,
    train_subset: Subset,
    val_subset: Subset,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
