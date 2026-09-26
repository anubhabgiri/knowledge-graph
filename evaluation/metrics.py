"""Metric computation and report generation for knowledge graph evaluation."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from datetime import datetime, timezone
from typing import Any

from evaluation.models import BenchmarkReport, SampleMetrics, TripletMatch


# ---------------------------------------------------------------------------
# Helper: precision / recall / F1 calculation
# ---------------------------------------------------------------------------

def calc_prf(tp: int, pred_count: int, gt_count: int) -> tuple[float, float, float]:
    """Calculate precision, recall, and F1 safely avoiding division by zero."""
    precision = tp / pred_count if pred_count > 0 else 0.0
    recall = tp / gt_count if gt_count > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


# ---------------------------------------------------------------------------
# Per-sample metric computation
# ---------------------------------------------------------------------------

def compute_sample_metrics(
    sample_id: str,
    num_ground_truth: int,
    num_predicted: int,
    matches: list[TripletMatch],
    unmatched_predicted: list[dict[str, Any]],
    unmatched_gt: list[Any],
) -> SampleMetrics:
    """Compute strict and soft metrics for a single sample."""
    strict_tp = sum(1 for m in matches if m.is_strict_match)
    soft_tp = sum(1 for m in matches if m.is_soft_match)

    strict_p, strict_r, strict_f1 = calc_prf(strict_tp, num_predicted, num_ground_truth)
    soft_p, soft_r, soft_f1 = calc_prf(soft_tp, num_predicted, num_ground_truth)

    return SampleMetrics(
        sample_id=sample_id,
        num_ground_truth=num_ground_truth,
        num_predicted=num_predicted,
        num_strict_matches=strict_tp,
        num_soft_matches=soft_tp,
        strict_precision=strict_p,
        strict_recall=strict_r,
        strict_f1=strict_f1,
        soft_precision=soft_p,
        soft_recall=soft_r,
        soft_f1=soft_f1,
        matches=matches,
        unmatched_predicted=unmatched_predicted,
        unmatched_ground_truth=unmatched_gt,
    )


# ---------------------------------------------------------------------------
# Dataset-level aggregation (micro + macro)
# ---------------------------------------------------------------------------

def aggregate_report(
    dataset_name: str,
    domain: str,
    sample_metrics_list: list[SampleMetrics],
    raw_relation_count: int = 0,
    canonical_relation_count: int = 0,
) -> BenchmarkReport:
    """Aggregate individual sample metrics into a dataset-level BenchmarkReport."""
    n_samples = len(sample_metrics_list)
    if n_samples == 0:
        return BenchmarkReport(
            dataset_name=dataset_name,
            domain=domain,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            sample_count=0,
            micro_strict_precision=0.0,
            micro_strict_recall=0.0,
            micro_strict_f1=0.0,
            micro_soft_precision=0.0,
            micro_soft_recall=0.0,
            micro_soft_f1=0.0,
            macro_strict_precision=0.0,
            macro_strict_recall=0.0,
            macro_strict_f1=0.0,
            macro_soft_precision=0.0,
            macro_soft_recall=0.0,
            macro_soft_f1=0.0,
            total_ground_truth_triplets=0,
            total_predicted_triplets=0,
            total_strict_matches=0,
            total_soft_matches=0,
            canonicalization_compression_ratio=0.0,
            per_sample_results=[],
        )

    # Micro totals
    total_gt = sum(s.num_ground_truth for s in sample_metrics_list)
    total_pred = sum(s.num_predicted for s in sample_metrics_list)
    total_strict_tp = sum(s.num_strict_matches for s in sample_metrics_list)
    total_soft_tp = sum(s.num_soft_matches for s in sample_metrics_list)

    micro_strict_p, micro_strict_r, micro_strict_f1 = calc_prf(total_strict_tp, total_pred, total_gt)
    micro_soft_p, micro_soft_r, micro_soft_f1 = calc_prf(total_soft_tp, total_pred, total_gt)

    # Macro averages
    macro_strict_p = round(sum(s.strict_precision for s in sample_metrics_list) / n_samples, 4)
    macro_strict_r = round(sum(s.strict_recall for s in sample_metrics_list) / n_samples, 4)
    macro_strict_f1 = round(sum(s.strict_f1 for s in sample_metrics_list) / n_samples, 4)

    macro_soft_p = round(sum(s.soft_precision for s in sample_metrics_list) / n_samples, 4)
    macro_soft_r = round(sum(s.soft_recall for s in sample_metrics_list) / n_samples, 4)
    macro_soft_f1 = round(sum(s.soft_f1 for s in sample_metrics_list) / n_samples, 4)

    # Canonicalization compression ratio
    if raw_relation_count > 0:
        compression = round(1.0 - (canonical_relation_count / raw_relation_count), 4)
    else:
        compression = 0.0

    return BenchmarkReport(
        dataset_name=dataset_name,
        domain=domain,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        sample_count=n_samples,
        micro_strict_precision=micro_strict_p,
        micro_strict_recall=micro_strict_r,
        micro_strict_f1=micro_strict_f1,
        micro_soft_precision=micro_soft_p,
        micro_soft_recall=micro_soft_r,
        micro_soft_f1=micro_soft_f1,
        macro_strict_precision=macro_strict_p,
        macro_strict_recall=macro_strict_r,
        macro_strict_f1=macro_strict_f1,
        macro_soft_precision=macro_soft_p,
        macro_soft_recall=macro_soft_r,
        macro_soft_f1=macro_soft_f1,
        total_ground_truth_triplets=total_gt,
        total_predicted_triplets=total_pred,
        total_strict_matches=total_strict_tp,
        total_soft_matches=total_soft_tp,
        canonicalization_compression_ratio=compression,
        per_sample_results=sample_metrics_list,
    )


# ---------------------------------------------------------------------------
# Report formatting (ASCII table output)
# ---------------------------------------------------------------------------

def format_report_summary(report: BenchmarkReport) -> str:
    """Format evaluation results into a clean ASCII table."""
    lines = [
        "=" * 72,
        f"  BENCHMARK EVALUATION REPORT: {report.dataset_name.upper()} (Domain: {report.domain})",
        "=" * 72,
        f"  Samples Evaluated: {report.sample_count}",
        f"  Reference Ground-Truth Triplets: {report.total_ground_truth_triplets}",
        f"  Total Extracted Triplets:        {report.total_predicted_triplets}",
        f"  Canonicalization Compression:    {report.canonicalization_compression_ratio:.1%}",
        "-" * 72,
        f"  {'Metric':<24} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12}",
        "-" * 72,
        f"  {'Micro Strict Match':<24} | {report.micro_strict_precision:<12.4f} | {report.micro_strict_recall:<12.4f} | {report.micro_strict_f1:<12.4f}",
        f"  {'Micro Soft Match':<24}   | {report.micro_soft_precision:<12.4f} | {report.micro_soft_recall:<12.4f} | {report.micro_soft_f1:<12.4f}",
        f"  {'Macro Strict Match':<24} | {report.macro_strict_precision:<12.4f} | {report.macro_strict_recall:<12.4f} | {report.macro_strict_f1:<12.4f}",
        f"  {'Macro Soft Match':<24}   | {report.macro_soft_precision:<12.4f} | {report.macro_soft_recall:<12.4f} | {report.macro_soft_f1:<12.4f}",
        "=" * 72,
    ]
    return "\n".join(lines)

