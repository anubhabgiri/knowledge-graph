"""Unit tests for the evaluation suite (loaders, matcher, metrics)."""
import pytest
from evaluation.datasets.bc5cdr import BC5CDRDatasetLoader
from evaluation.datasets.webnlg import WebNLGDatasetLoader
from evaluation.matcher import (
    compute_string_similarity,
    compute_relation_similarity,
    match_triplets,
)
from evaluation.models import GroundTruthTriplet


def test_bc5cdr_loader():
    loader = BC5CDRDatasetLoader()
    samples = loader.load(limit=2)
    assert len(samples) == 2
    first = samples[0]
    assert first.id.startswith("PMID:")
    assert len(first.ground_truth) > 0
    assert first.ground_truth[0].relation == "CID"
    assert first.ground_truth[0].subject_id is not None
    assert first.ground_truth[0].object_id is not None


def test_webnlg_loader():
    loader = WebNLGDatasetLoader()
    samples = loader.load()
    assert len(samples) >= 2
    first = samples[0]
    assert first.id == "webnlg_001"
    assert len(first.ground_truth) == 3


def test_string_similarity():
    # Exact
    assert compute_string_similarity("Cisplatin", "cisplatin") == 1.0
    # Substring containment
    sim = compute_string_similarity("lithium", "lithium chloride")
    assert sim > 0.70
    # Fuzzy overlap
    sim2 = compute_string_similarity("acute renal failure", "renal failure")
    assert sim2 > 0.70


def test_relation_similarity_biomedical():
    # Causal
    assert compute_relation_similarity("inducesCondition", "CID", domain="biomedical") >= 0.90
    assert compute_relation_similarity("causes", "CID", domain="biomedical") >= 0.90
    # Therapeutic / Non-causal
    assert compute_relation_similarity("treatsCondition", "CID", domain="biomedical") <= 0.10
    assert compute_relation_similarity("administeredTo", "CID", domain="biomedical") <= 0.10


def test_hungarian_bipartite_matching():
    preds = [
        {"subject": "Lithium", "relation": "inducesCondition", "object": "nephrogenic diabetes insipidus"},
        {"subject": "Lithium", "relation": "treatsCondition", "object": "bipolar disorder"},
    ]
    gt = [
        GroundTruthTriplet(subject="lithium", relation="CID", object_entity="nephrogenic diabetes insipidus"),
    ]

    matches, unmatched_preds, unmatched_gt = match_triplets(preds, gt, domain="biomedical", threshold=0.65)
    assert len(matches) == 1
    assert matches[0].is_soft_match is True
    assert matches[0].predicted_relation == "inducesCondition"
    assert len(unmatched_preds) == 1
    assert unmatched_preds[0]["relation"] == "treatsCondition"
    assert len(unmatched_gt) == 0

