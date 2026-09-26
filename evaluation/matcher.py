"""Matcher algorithms for aligning predicted triplets with reference ground-truth triplets."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import re
from difflib import SequenceMatcher
from typing import Any, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluation.models import GroundTruthTriplet, TripletMatch


# ---------------------------------------------------------------------------
# Text normalization utilities
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Normalize text for lexical matching: lowercase, strip punctuation and whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _split_camel_case(s: str) -> str:
    """Split camelCase string into space-separated lowercase words."""
    if not s:
        return ""
    s2 = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s2).lower().strip()


# ---------------------------------------------------------------------------
# Entity similarity
# ---------------------------------------------------------------------------

def compute_string_similarity(a: str, b: str) -> float:
    """Compute string similarity based on token overlap, substring containment, and edit distance."""
    norm_a = _normalize_text(a)
    norm_b = _normalize_text(b)

    if not norm_a or not norm_b:
        return 0.0
    if norm_a == norm_b:
        return 1.0

    # Substring containment bonus (e.g. "lithium" in "lithium chloride", "renal failure" in "acute renal failure")
    if norm_a in norm_b or norm_b in norm_a:
        len_ratio = min(len(norm_a), len(norm_b)) / max(len(norm_a), len(norm_b))
        return 0.70 + 0.30 * len_ratio

    # Token-level Jaccard overlap
    tokens_a = set(norm_a.split())
    tokens_b = set(norm_b.split())
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    jaccard = len(intersection) / len(union) if union else 0.0

    # Character-level SequenceMatcher ratio
    seq_ratio = SequenceMatcher(None, norm_a, norm_b).ratio()

    return max(jaccard, seq_ratio)


def compute_entity_pair_similarity(pred_s: str, pred_o: str, gt_s: str, gt_o: str) -> float:
    """Compute aggregate similarity between predicted (subject, object) and reference (subject, object)."""
    # Direct orientation: subject -> subject, object -> object
    sim_direct_s = compute_string_similarity(pred_s, gt_s)
    sim_direct_o = compute_string_similarity(pred_o, gt_o)
    direct_score = 0.5 * (sim_direct_s + sim_direct_o)

    # Inverted orientation: subject -> object, object -> subject (in case of flipped passive predicates)
    sim_inv_s = compute_string_similarity(pred_s, gt_o)
    sim_inv_o = compute_string_similarity(pred_o, gt_s)
    inv_score = 0.5 * (sim_inv_s + sim_inv_o) * 0.75  # minor penalty for flipped direction

    return max(direct_score, inv_score)


# ---------------------------------------------------------------------------
# Relation similarity — domain-aware keyword sets and scoring
# ---------------------------------------------------------------------------

# Causal predicates indicative of Chemical-Induced Disease (CID)
_CID_CAUSAL_KEYWORDS = {
    "induce", "induces", "induced", "inducescondition", "cause", "causes", "caused",
    "causescondition", "toxicity", "toxicto", "toxicityobserved", "toxic", "sideeffect",
    "sideeffectof", "leadsto", "produces", "associatedwith", "triggered", "trigger",
}

# Therapeutic or non-causal relations that must NOT be mapped to CID
_CID_NON_CAUSAL_KEYWORDS = {
    "treat", "treats", "treatscondition", "treatedwith", "cure", "cures", "prevent",
    "prevents", "alleviate", "alleviates", "inhibit", "inhibits", "inhibitedprocess",
    "administer", "administered", "administeredto", "administration", "studiedin",
    "evaluate", "evaluatedin", "measure",
}


def compute_relation_similarity(
    pred_rel: str,
    gt_rel: str,
    domain: str = "general",
) -> float:
    """Compute semantic or lexical similarity between predicted relation and ground-truth relation."""
    norm_pred = _normalize_text(pred_rel)
    norm_gt = _normalize_text(gt_rel)

    if norm_pred == norm_gt:
        return 1.0

    # In biomedical / BC5CDR benchmarks, the ground-truth relation is typically "CID"
    if norm_gt == "cid" or domain == "biomedical":
        tokens = set(_split_camel_case(pred_rel).split())
        if tokens & _CID_CAUSAL_KEYWORDS:
            return 0.95
        if tokens & _CID_NON_CAUSAL_KEYWORDS:
            return 0.05
        # Partial match on words like "effect", "damage", "injury"
        if any(w in norm_pred for w in ["damage", "injury", "lesion", "syndrome", "hazard"]):
            return 0.85
        return 0.20

    # In general domain (e.g. WebNLG)
    pred_words = _split_camel_case(pred_rel)
    gt_words = _split_camel_case(gt_rel)
    return compute_string_similarity(pred_words, gt_words)


# ---------------------------------------------------------------------------
# Hungarian bipartite matching (main entry point)
# ---------------------------------------------------------------------------

def match_triplets(
    predicted_triplets: list[Any],
    ground_truth_triplets: list[GroundTruthTriplet],
    domain: str = "general",
    threshold: float = 0.65,
    entity_weight: float = 0.60,
) -> tuple[list[TripletMatch], list[dict[str, Any]], list[GroundTruthTriplet]]:
    """Match predicted triplets with ground truth using the Hungarian algorithm (Maximum Weight Bipartite Matching).

    Parameters
    ----------
    predicted_triplets:
        List of predicted Triplet objects (or dicts).
    ground_truth_triplets:
        List of GroundTruthTriplet objects.
    domain:
        Domain preset ("general" or "biomedical").
    threshold:
        Overall score threshold [0, 1] required to count as a soft match.
    entity_weight:
        Weight assigned to entity similarity vs relation similarity (default 0.6 / 0.4).

    Returns
    -------
    matches:
        List of aligned pairs (TripletMatch).
    unmatched_predicted:
        List of predictions that could not be paired above threshold.
    unmatched_gt:
        List of ground-truth triplets that had no matching prediction.
    """
    if not predicted_triplets or not ground_truth_triplets:
        unmatched_pred = [
            t.model_dump() if hasattr(t, "model_dump") else dict(t)
            for t in predicted_triplets
        ]
        return [], unmatched_pred, list(ground_truth_triplets)

    n_pred = len(predicted_triplets)
    n_gt = len(ground_truth_triplets)

    # Build pairwise score matrices
    cost_matrix = np.zeros((n_pred, n_gt), dtype=float)
    entity_matrix = np.zeros((n_pred, n_gt), dtype=float)
    relation_matrix = np.zeros((n_pred, n_gt), dtype=float)

    for i, pred in enumerate(predicted_triplets):
        p_sub = getattr(pred, "subject", "") or pred.get("subject", "")
        p_obj = getattr(pred, "object_entity", "") or getattr(pred, "object", "") or pred.get("object", "")
        p_rel = getattr(pred, "canonical_relation", None) or getattr(pred, "relation", "") or pred.get("relation", "")

        for j, gt in enumerate(ground_truth_triplets):
            e_sim = compute_entity_pair_similarity(p_sub, p_obj, gt.subject, gt.object_entity)
            r_sim = compute_relation_similarity(p_rel, gt.relation, domain=domain)
            overall = entity_weight * e_sim + (1.0 - entity_weight) * r_sim

            cost_matrix[i, j] = overall
            entity_matrix[i, j] = e_sim
            relation_matrix[i, j] = r_sim

    # Solve Maximum Weight Matching: scipy solves min-cost, so pass -cost_matrix
    pred_indices, gt_indices = linear_sum_assignment(-cost_matrix)

    # Collect matched pairs that exceed the threshold
    matches: list[TripletMatch] = []
    matched_preds: set[int] = set()
    matched_gts: set[int] = set()

    for p_idx, g_idx in zip(pred_indices, gt_indices):
        score = float(cost_matrix[p_idx, g_idx])
        e_sim = float(entity_matrix[p_idx, g_idx])
        r_sim = float(relation_matrix[p_idx, g_idx])

        pred = predicted_triplets[p_idx]
        gt = ground_truth_triplets[g_idx]

        p_sub = getattr(pred, "subject", "") or pred.get("subject", "")
        p_obj = getattr(pred, "object_entity", "") or getattr(pred, "object", "") or pred.get("object", "")
        p_rel = getattr(pred, "relation", "") or pred.get("relation", "")
        p_can = getattr(pred, "canonical_relation", None) or pred.get("canonical_relation")

        is_strict = bool(
            _normalize_text(p_sub) == _normalize_text(gt.subject)
            and _normalize_text(p_obj) == _normalize_text(gt.object_entity)
            and (
                _normalize_text(p_rel) == _normalize_text(gt.relation)
                or (bool(p_can) and _normalize_text(p_can) == _normalize_text(gt.relation))
            )
        )

        is_soft = score >= threshold

        if is_soft or is_strict:
            matched_preds.add(p_idx)
            matched_gts.add(g_idx)
            matches.append(
                TripletMatch(
                    predicted_subject=p_sub,
                    predicted_relation=p_rel,
                    predicted_object=p_obj,
                    predicted_canonical=p_can,
                    gt_subject=gt.subject,
                    gt_relation=gt.relation,
                    gt_object=gt.object_entity,
                    entity_similarity=round(e_sim, 4),
                    relation_similarity=round(r_sim, 4),
                    overall_score=round(score, 4),
                    is_strict_match=is_strict,
                    is_soft_match=is_soft,
                )
            )

    # Collect unmatched predictions and ground-truth triplets
    unmatched_pred = [
        predicted_triplets[i].model_dump() if hasattr(predicted_triplets[i], "model_dump") else dict(predicted_triplets[i])
        for i in range(n_pred)
        if i not in matched_preds
    ]
    unmatched_gt = [ground_truth_triplets[j] for j in range(n_gt) if j not in matched_gts]

    return matches, unmatched_pred, unmatched_gt
