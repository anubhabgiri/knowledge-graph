"""CLI entry point for running knowledge graph extraction benchmarks."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Imports, path setup, and logging configuration
# ---------------------------------------------------------------------------

import logging
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "creation" / "without-schema"))

# Load .env
load_dotenv(_PROJECT_ROOT / ".env")

from evaluation.runner import run_benchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)


# ---------------------------------------------------------------------------
# CLI options and command definition
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--dataset", "-d",
    default="bc5cdr",
    show_default=True,
    type=click.Choice(["bc5cdr", "webnlg"], case_sensitive=False),
    help="Target benchmark dataset to evaluate.",
)
@click.option(
    "--data-path",
    default=None,
    show_default=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Custom path to dataset file (uses built-in sample if not provided).",
)
@click.option(
    "--limit", "-n",
    default=3,
    show_default=True,
    type=int,
    help="Number of samples to evaluate (use smaller limit to conserve LLM tokens).",
)
@click.option(
    "--domain",
    default=None,
    show_default=True,
    type=click.Choice(["biomedical", "general"], case_sensitive=False),
    help="Pipeline domain preset ('biomedical' for BC5CDR, 'general' for WebNLG). Defaults automatically.",
)
@click.option(
    "--threshold", "-t",
    default=0.65,
    show_default=True,
    type=float,
    help="Similarity score threshold required to declare a soft triplet match.",
)
@click.option(
    "--provider", "-p",
    default="gemini",
    show_default=True,
    type=click.Choice(["gemini", "openai", "ollama"], case_sensitive=False),
    help="LLM provider for extraction and canonicalization.",
)
@click.option(
    "--model", "-m",
    default=None,
    show_default=True,
    help="LLM model identifier (e.g. 'qwen2.5-coder:7b', 'gemini-2.5-flash').",
)
@click.option(
    "--output", "-o",
    default=None,
    show_default=True,
    type=click.Path(),
    help="Output JSON file path for detailed evaluation metrics and matches.",
)
def main(
    dataset: str,
    data_path: str | None,
    limit: int,
    domain: str | None,
    threshold: float,
    provider: str,
    model: str | None,
    output: str | None,
) -> None:
    """Benchmark knowledge graph extraction pipelines against standard evaluation datasets.

    Supports BC5CDR (Biomedical Chemical-Disease Relations) and WebNLG (General Domain RDF).
    Computes strict and soft Precision, Recall, F1, and schema compression.
    """
    # ---------------------------------------------------------------------------
    # Main function body
    # ---------------------------------------------------------------------------

    # Auto-select domain preset if omitted
    if not domain:
        domain = "biomedical" if dataset.lower() == "bc5cdr" else "general"

    click.echo(
        f"\nStarting Evaluation | Dataset: {dataset} | Domain: {domain} | Limit: {limit} | Provider: {provider}\n",
        err=True,
    )

    run_benchmark(
        dataset_name=dataset,
        data_path=data_path,
        limit=limit,
        domain=domain,
        threshold=threshold,
        provider=provider,
        model=model,
        output_report_path=output,
    )


if __name__ == "__main__":
    main()

