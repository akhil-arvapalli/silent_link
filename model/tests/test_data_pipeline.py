import numpy as np
import pytest

from data.augment import augment, mirror_hands, rotate_sequence
from data.dataset import SEQUENCE_LENGTH, build_dataset, discover_sequences, save_sequence
from data.normalize import pack_hands, sequence_to_input


def _fake_sequence(T=SEQUENCE_LENGTH, joints=42):
    rng = np.random.default_rng(0)
    return rng.normal(size=(T, joints, 3)).astype(np.float32)


def test_sequence_to_input_shape():
    seq = _fake_sequence()
    out = sequence_to_input(seq)
    assert out.shape == (SEQUENCE_LENGTH, 42, 3)
    assert out.dtype == np.float32


def test_pack_hands_placement():
    left = np.ones((21, 3), dtype=np.float32)
    right = np.full((21, 3), 2.0, dtype=np.float32)
    frame = pack_hands(left, right)
    assert frame.shape == (42, 3)
    assert frame[0, 0] == 1.0
    assert frame[21, 0] == 2.0


def test_missing_hand_zero():
    frame = pack_hands(None, None)
    assert frame.shape == (42, 3)
    assert not frame.any()


def test_normalize_scale_invariance():
    seq = _fake_sequence()
    a = sequence_to_input(seq)
    b = sequence_to_input(seq * 2.0)
    np.testing.assert_allclose(a, b, atol=1e-3)


def test_augment_preserves_shape():
    seq = _fake_sequence()
    out = augment(seq)
    assert out.shape == seq.shape


def test_mirror_swaps_hands():
    seq = _fake_sequence()
    mirrored = mirror_hands(seq)
    np.testing.assert_allclose(mirrored[:, :21], seq[:, 21:])
    np.testing.assert_allclose(mirrored[:, 21:], seq[:, :21])


def test_rotate_preserves_shape():
    seq = _fake_sequence()
    out = rotate_sequence(seq)
    assert out.shape == seq.shape


def test_sequence_length_enforced(tmp_path):
    with pytest.raises(ValueError):
        save_sequence(_fake_sequence(T=10), "hello", 0, tmp_path)


def test_save_discover_build_roundtrip(tmp_path):
    save_sequence(_fake_sequence(), "hello", 0, tmp_path)
    save_sequence(_fake_sequence(), "hello", 1, tmp_path)
    save_sequence(_fake_sequence(), "thanks", 0, tmp_path)

    disc = discover_sequences(tmp_path)
    assert set(disc.keys()) == {"hello", "thanks"}
    assert len(disc["hello"]) == 2

    X, labels = build_dataset(tmp_path)
    assert X.shape[0] == 3
    assert X.shape[1:] == (SEQUENCE_LENGTH, 42, 3)
    assert labels.count("hello") == 2
