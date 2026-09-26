"""Benchmark runner orchestrating pipeline execution and metric calculation."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Imports and project-root path setup
# ---------------------------------------------------------------------------

import logging
import sys
from pathlib import Path
from typing import Optional

# Ensure creation/without-schema is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "creation" / "without-schema"))

from pipeline import run as run_pipeline

from evaluation.datasets.base import BaseDatasetLoader
from evaluation.datasets.bc5cdr import BC5CDRDatasetLoader
from evaluation.datasets.webnlg import WebNLGDatasetLoader
from evaluation.matcher import match_triplets
from evaluation.metrics import aggregate_report, compute_sample_metrics, format_report_summary
from evaluation.models import BenchmarkReport, SampleMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loader factory
# ---------------------------------------------------------------------------

def get_dataset_loader(dataset_name: str, file_path: Optional[str | Path] = None) -> BaseDatasetLoader:
    """Return the appropriate dataset loader for a given benchmark name."""
    name = dataset_name.lower().strip()
    if name in {"bc5cdr", "cdr", "biocreative"}:
        return BC5CDRDatasetLoader(file_path=file_path)
    elif name in {"webnlg"}:
        return WebNLGDatasetLoader(file_path=file_path)
    else:
        raise ValueError(f"Unknown dataset name '{dataset_name}'. Supported: 'bc5cdr', 'webnlg'")


# ---------------------------------------------------------------------------
# Benchmark orchestrator
# ---------------------------------------------------------------------------

def run_benchmark(
    dataset_name: str = "bc5cdr",
    data_path: Optional[str | Path] = None,
    limit: Optional[int] = 3,
    domain: str = "biomedical",
    threshold: float = 0.65,
    provider: str = "gemini",
    model: Optional[str] = None,
    output_report_path: Optional[str | Path] = None,
) -> BenchmarkReport:
    """Run an end-to-end benchmark on the given dataset.

    Parameters
    ----------
    dataset_name:
        "bc5cdr" or "webnlg".
    data_path:
        Optional custom path to a dataset file.
    limit:
        Maximum number of samples to evaluate (default: 3).
    domain:
        Pipeline domain preset ("biomedical" or "general").
    threshold:
        Soft-match similarity threshold (default: 0.65).
    provider:
        LLM provider ("gemini", "openai", "ollama").
    model:
        Model identifier.
    output_report_path:
        Optional path to write the JSON evaluation report.

    Returns
    -------
    BenchmarkReport
        Comprehensive evaluation report with micro/macro metrics and sample breakdowns.
    """
    # --- Stage 1: Load dataset samples ---
    loader = get_dataset_loader(dataset_name, file_path=data_path)
    samples = loader.load(limit=limit)

    if not samples:
        logger.warning("No benchmark samples loaded!")
        empty_report = aggregate_report(dataset_name, domain, [])
        return empty_report

    logger.info(
        "Starting benchmark evaluation: dataset=%s | samples=%d | domain=%s | provider=%s | model=%s",
        dataset_name, len(samples), domain, provider, model or "default",
    )

    sample_metrics_list: list[SampleMetrics] = []
    total_raw_relations = 0
    total_canonical_relations = 0

    tmp_dir = _PROJECT_ROOT / "output" / "eval_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(samples, 1):
        logger.info("[%d/%d] Evaluating sample '%s' (%d chars, %d GT triplets) …",
                    idx, len(samples), sample.id, len(sample.text), len(sample.ground_truth))

        # --- Stage 2: Run knowledge graph extraction pipeline ---
        sample_output_path = tmp_dir / f"{sample.id.replace(':', '_')}.json"
        graph_output = run_pipeline(
            text=sample.text,
            output_path=str(sample_output_path),
            domain=domain,
            provider=provider,
            model=model,
        )

        total_raw_relations += graph_output.metadata.get("raw_triplet_count", len(graph_output.triplets))
        total_canonical_relations += len(graph_output.relation_definitions)

        # --- Stage 3: Match predicted triplets with reference ground-truth triplets ---
        matches, unmatched_pred, unmatched_gt = match_triplets(
            predicted_triplets=graph_output.triplets,
            ground_truth_triplets=sample.ground_truth,
            domain=domain,
            threshold=threshold,
        )

        # --- Stage 4: Compute per-sample metrics ---
        metrics = compute_sample_metrics(
            sample_id=sample.id,
            num_ground_truth=len(sample.ground_truth),
            num_predicted=len(graph_output.triplets),
            matches=matches,
            unmatched_predicted=unmatched_pred,
            unmatched_gt=unmatched_gt,
        )
        sample_metrics_list.append(metrics)

        logger.info(
            "   → Pred: %d | GT: %d | Strict TP: %d | Soft TP: %d | Soft F1: %.3f",
            metrics.num_predicted,
            metrics.num_ground_truth,
            metrics.num_strict_matches,
            metrics.num_soft_matches,
            metrics.soft_f1,
        )

    # --- Stage 5: Aggregate results into final report ---
    report = aggregate_report(
        dataset_name=dataset_name,
        domain=domain,
        sample_metrics_list=sample_metrics_list,
        raw_relation_count=total_raw_relations,
        canonical_relation_count=total_canonical_relations,
    )

    # --- Stage 6: Print summary table ---
    summary_text = format_report_summary(report)
    print("\n" + summary_text + "\n")

    # --- Stage 7: Save detailed JSON report ---
    if output_report_path:
        out_path = Path(output_report_path)
    else:
        out_path = _PROJECT_ROOT / "output" / f"eval_{dataset_name}_{domain}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Detailed benchmark report written to '%s'", out_path)

    return report

