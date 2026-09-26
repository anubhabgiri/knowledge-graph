"""Data models for benchmark evaluation."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ground-truth and sample input models
# ---------------------------------------------------------------------------

class GroundTruthTriplet(BaseModel):
    """A reference knowledge graph triplet from a ground truth dataset."""

    subject: str = Field(description="Subject entity text or concept name")
    relation: str = Field(description="Predicate or relation type (e.g. 'CID' or 'birthPlace')")
    object_entity: str = Field(description="Object entity text or value")
    subject_id: Optional[str] = Field(default=None, description="Ontology concept ID if available (e.g. MeSH ID)")
    object_id: Optional[str] = Field(default=None, description="Ontology concept ID if available (e.g. MeSH ID)")


class BenchmarkSample(BaseModel):
    """A single evaluation benchmark instance (e.g., one PubMed abstract or one WebNLG entry)."""

    id: str = Field(description="Sample identifier (e.g., PMID or WebNLG ID)")
    text: str = Field(description="Full input text provided to the extraction pipeline")
    ground_truth: list[GroundTruthTriplet] = Field(
        default_factory=list,
        description="List of reference triplets known to be true in this passage",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Domain, category, entity mentions, or other sample annotations",
    )


# ---------------------------------------------------------------------------
# Triplet match (alignment between a predicted and a reference triplet)
# ---------------------------------------------------------------------------

class TripletMatch(BaseModel):
    """The alignment result between one predicted triplet and one ground truth triplet."""

    predicted_subject: str
    predicted_relation: str
    predicted_object: str
    predicted_canonical: Optional[str] = None

    gt_subject: str
    gt_relation: str
    gt_object: str

    entity_similarity: float = Field(description="Similarity score between subjects and objects [0.0, 1.0]")
    relation_similarity: float = Field(description="Semantic or exact similarity score between relations [0.0, 1.0]")
    overall_score: float = Field(description="Weighted combined match score [0.0, 1.0]")
    is_strict_match: bool = Field(description="True if exact match on all components")
    is_soft_match: bool = Field(description="True if overall_score >= matching threshold")


# ---------------------------------------------------------------------------
# Per-sample and aggregate metric models
# ---------------------------------------------------------------------------

class SampleMetrics(BaseModel):
    """Evaluation metrics for a single sample."""

    sample_id: str
    num_ground_truth: int
    num_predicted: int
    num_strict_matches: int
    num_soft_matches: int

    strict_precision: float
    strict_recall: float
    strict_f1: float

    soft_precision: float
    soft_recall: float
    soft_f1: float

    matches: list[TripletMatch] = Field(default_factory=list)
    unmatched_predicted: list[dict[str, Any]] = Field(default_factory=list)
    unmatched_ground_truth: list[GroundTruthTriplet] = Field(default_factory=list)


class BenchmarkReport(BaseModel):
    """Aggregate benchmark evaluation report across all samples."""

    dataset_name: str
    domain: str
    timestamp: str
    sample_count: int

    # Micro metrics (pooled across all instances)
    micro_strict_precision: float
    micro_strict_recall: float
    micro_strict_f1: float

    micro_soft_precision: float
    micro_soft_recall: float
    micro_soft_f1: float

    # Macro metrics (average of per-sample scores)
    macro_strict_precision: float
    macro_strict_recall: float
    macro_strict_f1: float

    macro_soft_precision: float
    macro_soft_recall: float
    macro_soft_f1: float

    # Pipeline stats
    total_ground_truth_triplets: int
    total_predicted_triplets: int
    total_strict_matches: int
    total_soft_matches: int

    canonicalization_compression_ratio: float = Field(
        description="1 - (canonical relations / raw relations), measuring schema compactness"
    )

    per_sample_results: list[SampleMetrics] = Field(default_factory=list)

