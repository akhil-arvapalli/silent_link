"""Config loader for Silent Link.

Reads canonical ISL gloss vocabulary from model/configs/glosses.json.
Single source of truth — never hardcode gloss lists in code.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GLOSSES_PATH = REPO_ROOT / "model" / "configs" / "glosses.json"


def load_glosses(path: Path = DEFAULT_GLOSSES_PATH) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def gloss_labels(path: Path = DEFAULT_GLOSSES_PATH) -> list[str]:
    return sorted(load_glosses(path)["glosses"].keys())
