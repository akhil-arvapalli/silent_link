"""Rule-based sentence -> gloss normalization (Phase 2).

Maps free-form English input text onto the canonical ISL gloss vocabulary
defined in model/configs/glosses.json, using each gloss's "words" list.

This is the first, deterministic stage of Text->Sign. It handles:
  - lowercasing + basic punctuation stripping
  - longest-phrase-first matching against the gloss vocabulary
  - collapsing duplicate adjacent glosses

A later seq2seq stage (Phase 3 stretch) may replace this, but the interface
(GlossNormalizer.normalize(text) -> list[str]) stays stable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_PUNCT = re.compile(r"[^\w\s']")


def load_gloss_specs(path: Path) -> dict[str, dict[str, object]]:
    """Load the canonical gloss specs, keyed by gloss name."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data["glosses"])


class GlossNormalizer:
    """Match input text against the canonical gloss vocabulary."""

    def __init__(self, gloss_specs: dict[str, dict[str, object]]) -> None:
        # Longest word-phrase first so multi-word phrases win over their parts.
        self._by_phrase: dict[str, str] = {}
        for gloss, spec in gloss_specs.items():
            words = spec["words"]
            if isinstance(words, str):
                words = [words]
            for phrase in words:
                key = _canonical(phrase)
                if key:
                    self._by_phrase[key] = gloss
        self._phrases = sorted(self._by_phrase, key=len, reverse=True)

    def normalize(self, text: str) -> list[str]:
        """Return the canonical gloss sequence for a sentence (empty if none)."""
        tokens = _canonical(text).split()
        out: list[str] = []
        i = 0
        while i < len(tokens):
            matched = False
            for phrase in self._phrases:
                n = len(phrase.split())
                window = " ".join(tokens[i : i + n])
                if window == phrase:
                    gloss = self._by_phrase[phrase]
                    if not out or out[-1] != gloss:
                        out.append(gloss)
                    i += n
                    matched = True
                    break
            if not matched:
                i += 1
        return out


def _canonical(text: str) -> str:
    return _PUNCT.sub(" ", text).strip().lower()
