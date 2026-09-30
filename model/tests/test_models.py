import numpy as np
import pytest
import torch

from data.loader import LandmarkDataset, stratified_split
from models.baseline import Conv1D_BiLSTM
from models.export import export_onnx
from models.stgcn import STGCN
from models.training import build_model, train_model

T, V, C = 40, 42, 3
N_CLASSES = 11


def _rand_input(batch=4, T=T, V=V, C=C):
    return torch.randn(batch, T, V, C)


@pytest.mark.parametrize("model_cls", [Conv1D_BiLSTM, STGCN])
def test_forward_shape(model_cls):
    model = model_cls(n_classes=N_CLASSES)
    out = model(_rand_input())
    assert out.shape == (4, N_CLASSES)


def test_stgcn_graph_edges_produce_expected_degree():
    from models.stgcn import build_adjacency, hand_adjacency

    adj = build_adjacency(V)
    assert adj.shape == (V, V)
    # two hands => two block-diagonal subgraphs, no cross-hand edges
    assert adj[:21, 21:].sum() == 0
    assert adj[21:, :21].sum() == 0
    hand = hand_adjacency(21)
    # wrist (node 0) connects to itself + 5 fingers => 6 nonzero neighbors
    assert int((hand[0] > 0).sum()) == 6


def test_build_model_unknown_raises():
    with pytest.raises(ValueError):
        build_model("nope", N_CLASSES, torch.device("cpu"))


def test_models_backpropagate():
    for name in ("baseline", "stgcn"):
        model = build_model(name, N_CLASSES, torch.device("cpu"))
        logits = model(_rand_input())
        loss = logits.sum()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, f"{name} produced no gradients"


def _synthetic_dataset(n_per_gloss=20, n_glosses=4):
    glosses = [f"g{i}" for i in range(n_glosses)]
    records = []
    rng = np.random.default_rng(0)
    for gi, gloss in enumerate(glosses):
        for seq_id in range(n_per_gloss):
            t = np.linspace(0, 2 * np.pi, T)[:, None, None]
            phase = rng.uniform(0, 2 * np.pi, size=(V, C))
            base = np.sin((0.5 + 0.3 * gi) * t + phase) * 0.5
            seq = (base + rng.normal(0, 0.02, size=(T, V, C))).astype(np.float32)
            records.append((seq, gi))
    return glosses, LandmarkDataset(records)


def test_stratified_split_all_glosses_present():
    glosses, dataset = _synthetic_dataset()
    train_sub, val_sub = stratified_split(dataset, val_frac=0.2, seed=7)
    train_labels = {dataset.records[i][1] for i in train_sub.indices}
    val_labels = {dataset.records[i][1] for i in val_sub.indices}
    assert train_labels == set(range(len(glosses)))
    assert val_labels == set(range(len(glosses)))
    assert len(train_sub) > 0 and len(val_sub) > 0


def test_dataset_label_mapping_is_0_to_n_minus_1():
    _, dataset = _synthetic_dataset(n_per_gloss=5, n_glosses=3)
    assert {label for _, label in dataset.records} == {0, 1, 2}


def test_train_model_saves_checkpoint_and_improves(tmp_path):
    glosses, dataset = _synthetic_dataset(n_per_gloss=30, n_glosses=6)
    train_sub, val_sub = stratified_split(dataset, val_frac=0.25, seed=1)
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_sub, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=16)

    model = build_model("stgcn", 6, torch.device("cpu"))
    metrics = train_model(
        model,
        train_loader,
        val_loader,
        torch.device("cpu"),
        epochs=20,
        patience=5,
        out_dir=tmp_path,
    )
    assert (tmp_path / "best_model.pt").exists()
    assert (tmp_path / "metrics.json").exists()
    assert metrics["best_val_acc"] > 0.5
    assert metrics["best_epoch"] >= 0


def test_export_onnx_runs_onnxruntime(tmp_path):
    import onnxruntime as ort

    model = build_model("baseline", 4, torch.device("cpu"))
    torch.save(model.state_dict(), tmp_path / "best_model.pt")

    onnx_path = export_onnx("baseline", 4, tmp_path / "best_model.pt", tmp_path / "model.onnx")
    assert onnx_path.exists()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.randn(1, T, V, C).astype(np.float32)
    out = sess.run(None, {"landmarks": x})[0]
    assert out.shape == (1, 4)
