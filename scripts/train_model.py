#!/usr/bin/env python
"""Train a gesture classification model (baseline or ST-GCN) with early stopping.

Reads data/processed sequences, splits train/val stratified per gloss, trains
with early stopping, saves best weights, and exports ONNX.

Usage:
    python scripts/train_model.py --model stgcn
    python scripts/train_model.py --model baseline --epochs 60 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "model" / "src"))

from data.augment import augment  # noqa: E402
from data.loader import load_dataset, make_loaders, stratified_split  # noqa: E402
from models.export import export_onnx  # noqa: E402
from models.training import build_model, train_model  # noqa: E402


def load_glosses(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return sorted(data["glosses"].keys())


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ISL gesture classifier")
    parser.add_argument("--model", choices=["baseline", "stgcn"], default="stgcn")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument("--glosses-json", type=Path, default=REPO_ROOT / "model" / "configs" / "glosses.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "model" / "runs" / "default")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--no-augment", action="store_true", help="Disable train-time augmentation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    glosses = load_glosses(args.glosses_json)
    n_classes = len(glosses)
    augment_fn = None if args.no_augment else augment

    dataset = load_dataset(args.data, glosses, augment_fn=augment_fn)
    if len(dataset) == 0:
        print(f"No sequences found in {args.data}. Run scripts/capture_data.py or scripts/make_synthetic_data.py first.")
        return 1
    print(f"loaded {len(dataset)} sequences across {n_classes} glosses")

    train_subset, val_subset = stratified_split(dataset, val_frac=args.val_frac)
    train_loader, val_loader = make_loaders(
        dataset, train_subset, val_subset, batch_size=args.batch_size
    )
    print(f"train: {len(train_subset)}  val: {len(val_subset)}")

    model = build_model(args.model, n_classes, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model}: {n_params:,} params")

    metrics = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        out_dir=args.out,
    )
    print(
        f"best_epoch={metrics['best_epoch']}  best_val_acc={metrics['best_val_acc']:.4f}  "
        f"epochs_run={metrics['epochs_run']}"
    )

    onnx_path = export_onnx(
        args.model,
        n_classes,
        args.out / "best_model.pt",
        args.out / "model.onnx",
    )
    print(f"exported ONNX: {onnx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
