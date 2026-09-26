"""Pipeline orchestrator — wires all stages and writes the output JSON."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from canonicalizer import Canonicalizer
from chunker import TextChunker
from definer import RelationDefiner
from extractor import TripletExtractor
from llm_manager import LLMManager
from merger import merge
from models import GraphOutput, Triplet

logger = logging.getLogger(__name__)


def run(
    text: str,
    output_path: str = "output.json",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    top_k_choices: int = 5,
    min_similarity: float = 0.45,
    domain: str = "general",
    provider: str = "gemini",
    model: str | None = None,
    max_retries: int = 2,
) -> GraphOutput:
    """Run the full schema-free knowledge graph extraction pipeline.

    The pipeline follows the three steps from the notebook:

    1. **Extraction** — each chunk → raw ``[Subject, Relation, Object]`` triplets.
    2. **Definition** — all unique relations → natural-language definitions (batched).
    3. **Canonicalization** — definitions bootstrapped into a schema; similar
       relations merged via embedding similarity + LLM choice.

    Then triplets are merged/deduplicated and written to ``output_path`` as JSON.

    Parameters
    ----------
    text:
        Raw input text.
    output_path:
        Destination file for the JSON output (created if it does not exist).
    chunk_size:
        Maximum tokens per chunk.
    chunk_overlap:
        Overlap tokens between consecutive chunks.
    top_k_choices:
        Number of schema candidates to surface per canonicalization LLM call.
    min_similarity:
        Minimum cosine similarity threshold to consider candidate relations for merging.
    domain:
        Domain configuration preset: ``"general"`` or ``"biomedical"``.
    provider:
        LLM provider to use: ``"gemini"``, ``"openai"``, or ``"ollama"``.
    model:
        Model identifier for the chosen provider.  When ``None`` the provider's
        default model is used (see :class:`llm_manager.LLMManager`).
    max_retries:
        Number of automatic retries on transient API errors (not used by Ollama).

    Returns
    -------
    GraphOutput
        The final structured knowledge graph (also written to *output_path*).
    """
    # Build a single shared LLM instance for all pipeline stages
    llm = LLMManager.build(provider=provider, model=model, max_retries=max_retries)
    model_label = f"{provider}/{model or 'default'}"

    logger.info(
        "Pipeline start | domain=%s provider=%s  model=%s  chunk_size=%d  overlap=%d  top_k=%d  min_sim=%.2f",
        domain, provider, model or "default", chunk_size, chunk_overlap, top_k_choices, min_similarity,
    )

    # ------------------------------------------------------------------
    # Stage 1: Chunking
    # ------------------------------------------------------------------
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.split(text)
    logger.info("Input split into %d chunk(s).", len(chunks))

    # ------------------------------------------------------------------
    # Stage 2: Triplet extraction  (Step 1 from notebook)
    # ------------------------------------------------------------------
    extractor = TripletExtractor(llm=llm, domain=domain)
    all_triplets: list[Triplet] = []

    for i, chunk in enumerate(chunks):
        logger.info("[%d/%d] Extracting triplets …", i + 1, len(chunks))
        chunk_triplets = extractor.extract(chunk, chunk_index=i)
        logger.info("        → %d triplet(s)", len(chunk_triplets))
        all_triplets.extend(chunk_triplets)

    if not all_triplets:
        logger.warning("No triplets extracted — writing empty graph.")
        output = _empty_output(model_label, len(chunks), domain=domain)
        _write_json(output, output_path)
        return output

    logger.info("Total raw triplets: %d across all chunks.", len(all_triplets))

    # ------------------------------------------------------------------
    # Stage 3: Relation definition  (Step 2 from notebook)
    # ------------------------------------------------------------------
    unique_count = len({t.relation for t in all_triplets})
    logger.info("Defining %d unique relation(s) …", unique_count)
    definer = RelationDefiner(llm=llm, domain=domain)
    relation_definitions = definer.define(all_triplets)

    # ------------------------------------------------------------------
    # Stage 4: Schema bootstrap + Canonicalization  (Step 3 from notebook)
    # ------------------------------------------------------------------
    canonicalizer = Canonicalizer(
        llm=llm,
        top_k=top_k_choices,
        min_similarity=min_similarity,
        domain=domain,
    )
    canonicalizer.bootstrap_schema(relation_definitions)

    canonical_mapping = canonicalizer.canonicalize_all(relation_definitions, all_triplets)

    # Apply mapping to every triplet
    canonical_triplets = [
        t.model_copy(
            update={"canonical_relation": canonical_mapping.get(t.relation, t.relation)}
        )
        for t in all_triplets
    ]

    # ------------------------------------------------------------------
    # Stage 5: Merge & deduplicate
    # ------------------------------------------------------------------
    final_triplets = merge(canonical_triplets)
    logger.info(
        "Final graph: %d unique triplet(s)  (from %d raw).",
        len(final_triplets), len(all_triplets),
    )

    # ------------------------------------------------------------------
    # Stage 6: Build output + write JSON
    # ------------------------------------------------------------------
    # Rebuild definitions keyed on canonical relation names
    canonical_definitions: dict[str, str] = {}
    for raw_rel, defn in relation_definitions.items():
        canonical = canonical_mapping.get(raw_rel, raw_rel)
        # Keep the first definition seen for a given canonical relation
        canonical_definitions.setdefault(canonical, defn)

    output = GraphOutput(
        triplets=final_triplets,
        relation_definitions=canonical_definitions,
        metadata={
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "domain": domain,
            "provider": provider,
            "model": model_label,
            "chunk_count": len(chunks),
            "raw_triplet_count": len(all_triplets),
            "unique_relation_count_before_canonicalization": unique_count,
            "unique_relation_count_after_canonicalization": len(canonical_definitions),
            "input_character_count": len(text),
        },
    )
    _write_json(output, output_path)
    logger.info("Output written to '%s'.", output_path)

    return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_output(model: str, chunk_count: int, domain: str = "general") -> GraphOutput:
    return GraphOutput(
        triplets=[],
        relation_definitions={},
        metadata={
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "domain": domain,
            "model": model,
            "chunk_count": chunk_count,
            "raw_triplet_count": 0,
        },
    )


def _write_json(output: GraphOutput, path: str) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(output.model_dump_json(indent=2, by_alias=True), encoding="utf-8")

