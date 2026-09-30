"""Training loop with early stopping, validation split, and model checkpointing.

Follows AGENTS.md guardrails: proper validation split + early stopping
(NOT the old epochs=2000 / test_size=0.05 anti-pattern).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from .baseline import Conv1D_BiLSTM
from .stgcn import STGCN


def build_model(model_name: str, n_classes: int, device: torch.device, **kwargs) -> nn.Module:
    if model_name == "baseline":
        model = Conv1D_BiLSTM(n_classes=n_classes, **kwargs)
    elif model_name == "stgcn":
        model = STGCN(n_classes=n_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model '{model_name}' (choose 'baseline' or 'stgcn')")
    return model.to(device)


def _epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train() if optimizer is not None else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.shape[0]
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.shape[0]
    return total_loss / total, correct / total


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15,
    out_dir: Path | None = None,
) -> dict:
    """Train with early stopping on validation accuracy.

    Returns metrics dict including best epoch and best val accuracy, and
    saves best model weights + a metrics.json if out_dir is given.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = 0.0
    best_state = None
    best_epoch = -1
    wait = 0
    history: list[dict] = []

    for epoch in range(epochs):
        train_loss, train_acc = _epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = _epoch(model, val_loader, criterion, None, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    metrics = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val,
        "epochs_run": len(history),
        "history": history,
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, out_dir / "best_model.pt")
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics
