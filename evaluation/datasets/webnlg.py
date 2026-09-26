"""WebNLG dataset loader."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from evaluation.datasets.base import BaseDatasetLoader
from evaluation.models import BenchmarkSample, GroundTruthTriplet

logger = logging.getLogger(__name__)

_DEFAULT_SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "webnlg_sample.json"


def _unslug(text: str) -> str:
    """Convert slugged DBpedia names like 'Alan_Shepard' to 'Alan Shepard'."""
    return text.replace("_", " ").strip()


class WebNLGDatasetLoader(BaseDatasetLoader):
    """Loads WebNLG entries (JSON format).

    Parameters
    ----------
    file_path:
        Path to a WebNLG JSON file. If omitted, uses the built-in sample file.
    """

    def __init__(self, file_path: Optional[str | Path] = None) -> None:
        self.file_path = Path(file_path) if file_path else _DEFAULT_SAMPLE_PATH

    def load(self, limit: Optional[int] = None) -> list[BenchmarkSample]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"WebNLG dataset file not found: {self.file_path}")

        logger.info("Loading WebNLG dataset from %s", self.file_path)
        data = json.loads(self.file_path.read_text(encoding="utf-8"))

        samples: list[BenchmarkSample] = []
        for item in data:
            gt_triplets: list[GroundTruthTriplet] = []
            for t in item.get("triplets", []):
                gt_triplets.append(
                    GroundTruthTriplet(
                        subject=_unslug(t.get("subject", "")),
                        relation=t.get("relation", "").strip(),
                        object_entity=_unslug(t.get("object", "")),
                    )
                )

            samples.append(
                BenchmarkSample(
                    id=item.get("id", f"sample_{len(samples)}"),
                    text=item.get("text", "").strip(),
                    ground_truth=gt_triplets,
                    metadata={"category": item.get("category", "General")},
                )
            )
            if limit and len(samples) >= limit:
                break

        logger.info("Loaded %d WebNLG document(s).", len(samples))
        return samples

