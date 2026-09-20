"""Merge and deduplicate triplets from all chunks into a final graph."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models import Triplet

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Lowercase + strip for key comparison."""
    return text.lower().strip()


def merge(triplets: list[Triplet]) -> list[Triplet]:
    """Deduplicate *triplets* by ``(subject, canonical_relation, object)`` after normalization.

    Normalization is case-insensitive and whitespace-stripped.
    When the same fact appears in multiple chunks (due to overlap or repetition),
    only the first occurrence is kept.

    Parameters
    ----------
    triplets:
        All canonicalized triplets from all chunks, in chunk order.

    Returns
    -------
    list[Triplet]
        Deduplicated triplets preserving original order of first occurrence.
    """
    seen: set[tuple[str, str, str]] = set()
    unique: list[Triplet] = []

    for t in triplets:
        effective_relation = t.canonical_relation or t.relation
        key = (
            _normalize(t.subject),
            _normalize(effective_relation),
            _normalize(t.object_entity),
        )
        if key not in seen:
            seen.add(key)
            unique.append(t)

    removed = len(triplets) - len(unique)
    if removed:
        logger.info("Merger removed %d duplicate triplet(s).", removed)

    return unique

