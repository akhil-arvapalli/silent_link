from silentlink.config import gloss_labels, load_glosses


def test_load_glosses_is_isl() -> None:
    glosses = load_glosses()
    assert glosses["language"] == "ISL"
    assert "hello" in glosses["glosses"]


def test_gloss_labels_are_sorted() -> None:
    labels = gloss_labels()
    assert labels == sorted(labels)
    assert "hello" in labels
