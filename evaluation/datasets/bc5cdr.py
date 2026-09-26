"""BC5CDR (BioCreative V Chemical Disease Relation) dataset loader."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Optional

from evaluation.datasets.base import BaseDatasetLoader
from evaluation.models import BenchmarkSample, GroundTruthTriplet

logger = logging.getLogger(__name__)

_DEFAULT_SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "bc5cdr_sample.pubtator"


# ---------------------------------------------------------------------------
# Dataset loader class
# ---------------------------------------------------------------------------

class BC5CDRDatasetLoader(BaseDatasetLoader):
    """Parses BC5CDR dataset files in PubTator format.

    Each document block consists of:
    - ``PMID|t|Title text``
    - ``PMID|a|Abstract text``
    - Mentions: ``PMID \\t start \\t end \\t text \\t type \\t MeSH_ID``
    - Relations: ``PMID \\t CID \\t Chem_MeSH \\t Dis_MeSH``

    Parameters
    ----------
    file_path:
        Path to a PubTator file. If omitted, uses the built-in sample file.
    """

    def __init__(self, file_path: Optional[str | Path] = None) -> None:
        self.file_path = Path(file_path) if file_path else _DEFAULT_SAMPLE_PATH

    # -------------------------------------------------------------------
    # Public interface: load()
    # -------------------------------------------------------------------

    def load(self, limit: Optional[int] = None) -> list[BenchmarkSample]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"BC5CDR dataset file not found: {self.file_path}")

        logger.info("Loading BC5CDR dataset from %s", self.file_path)
        content = self.file_path.read_text(encoding="utf-8")

        # Split the file into per-document blocks (blank-line separated)
        blocks = content.strip().split("\n\n")

        samples: list[BenchmarkSample] = []
        for block in blocks:
            if not block.strip():
                continue
            sample = self._parse_pubtator_block(block.strip())
            if sample:
                samples.append(sample)
                if limit and len(samples) >= limit:
                    break

        logger.info("Loaded %d BC5CDR document(s).", len(samples))
        return samples

    # -------------------------------------------------------------------
    # Internal: PubTator block parser
    # -------------------------------------------------------------------

    def _parse_pubtator_block(self, block: str) -> Optional[BenchmarkSample]:
        lines = block.splitlines()
        pmid = ""
        title = ""
        abstract = ""

        # MeSH ID -> canonical mention name (first mention seen or most frequent)
        mesh_to_names: dict[str, list[str]] = {}
        # CID relations: list of (chem_mesh, dis_mesh)
        cid_pairs: list[tuple[str, str]] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if "|t|" in line:
                parts = line.split("|t|", 1)
                pmid = parts[0].strip()
                title = parts[1].strip()
            elif "|a|" in line:
                parts = line.split("|a|", 1)
                abstract = parts[1].strip()
            elif "\t" in line:
                fields = line.split("\t")
                if len(fields) >= 6:
                    # Entity mention line: pmid, start, end, mention, type, mesh_id
                    _, _, _, mention, entity_type, mesh_id = fields[:6]
                    mesh_id = mesh_id.strip()
                    mention = mention.strip()
                    if mesh_id and mesh_id != "-1":
                        mesh_to_names.setdefault(mesh_id, []).append(mention)
                elif len(fields) >= 4 and fields[1] == "CID":
                    # Relation line: pmid, "CID", chem_id, dis_id
                    _, _, chem_id, dis_id = fields[:4]
                    cid_pairs.append((chem_id.strip(), dis_id.strip()))

        if not pmid or (not title and not abstract):
            return None

        full_text = f"{title}\n{abstract}".strip() if title and abstract else (title or abstract)

        # Build ground truth triplets by resolving MeSH IDs to mention names
        ground_truth: list[GroundTruthTriplet] = []
        for chem_id, dis_id in cid_pairs:
            # Pick canonical mention name (shortest or most common)
            chem_names = mesh_to_names.get(chem_id, [chem_id])
            dis_names = mesh_to_names.get(dis_id, [dis_id])
            chem_name = chem_names[0] if chem_names else chem_id
            dis_name = dis_names[0] if dis_names else dis_id

            ground_truth.append(
                GroundTruthTriplet(
                    subject=chem_name,
                    relation="CID",
                    object_entity=dis_name,
                    subject_id=chem_id,
                    object_id=dis_id,
                )
            )

        return BenchmarkSample(
            id=f"PMID:{pmid}",
            text=full_text,
            ground_truth=ground_truth,
            metadata={
                "pmid": pmid,
                "title": title,
                "mesh_to_names": {k: list(dict.fromkeys(v)) for k, v in mesh_to_names.items()},
                "cid_count": len(ground_truth),
            },
        )

