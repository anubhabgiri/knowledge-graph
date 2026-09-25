"""CLI entry point for the without-schema KG pipeline.

Usage
-----
    python creation/without-schema/main.py --input data.txt --output graph.json

Or, from the project root with .env loaded automatically:

    python creation/without-schema/main.py \\
        --input  path/to/document.txt \\
        --output path/to/output.json  \\
        --chunk-size 512              \\
        --chunk-overlap 64            \\
        --top-k 5                     \\
        --model gemini-2.5-flash
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow sibling-module imports regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

import click
from dotenv import load_dotenv

from pipeline import run

# Load .env from the project root (two levels up from this file)
load_dotenv(Path(__file__).parent.parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)


@click.command()
@click.option(
    "--input", "-i", "input_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
    help="Path to the input text file.",
)
@click.option(
    "--output", "-o", "output_path",
    default="output.json",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Destination path for the output JSON file.",
)
@click.option(
    "--chunk-size",
    default=512,
    show_default=True,
    type=int,
    help="Maximum tokens per chunk.",
)
@click.option(
    "--chunk-overlap",
    default=64,
    show_default=True,
    type=int,
    help="Overlap tokens between consecutive chunks.",
)
@click.option(
    "--top-k",
    default=5,
    show_default=True,
    type=int,
    help="Number of canonical-relation candidates surfaced per LLM call.",
)
@click.option(
    "--provider",
    default="gemini",
    show_default=True,
    type=click.Choice(["gemini", "openai", "ollama"], case_sensitive=False),
    help="LLM provider to use for all pipeline stages.",
)
@click.option(
    "--model",
    default=None,
    show_default=True,
    help=(
        "Model identifier for the chosen provider.  "
        "Defaults: gemini→gemini-2.5-flash, openai→gpt-4o-mini, ollama→llama3.2"
    ),
)
def main(
    input_path: Path,
    output_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    provider: str,
    model: str | None,
) -> None:
    """Extract a knowledge graph from a large text file — no predefined schema required.

    The pipeline runs three LLM-powered steps:

    \b
      1. Open triplet extraction  →  [Subject, Relation, Object]
      2. Relation definition      →  natural-language definitions (batched)
      3. Canonicalization         →  merge similar relations via vector similarity

    Output is written as a structured JSON file containing triplets, relation
    definitions, and run metadata.
    """
    text = input_path.read_text(encoding="utf-8")
    click.echo(
        f"Input : {input_path}  ({len(text):,} characters)",
        err=True,
    )

    result = run(
        text=text,
        output_path=str(output_path),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k_choices=top_k,
        provider=provider,
        model=model,
    )

    click.echo(
        f"Done  : {len(result.triplets)} unique triplet(s) → {output_path}",
        err=True,
    )


if __name__ == "__main__":
    main()

