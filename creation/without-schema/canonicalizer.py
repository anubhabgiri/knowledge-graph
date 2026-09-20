"""Step 3 — Canonicalize extracted relations using embedding similarity + LLM.

Design
------
Instead of making one LLM call per triplet (which would be N calls for N triplets),
this module canonicalizes per **unique relation** — typically M << N calls — then
applies the resulting mapping to all triplets in O(N).

Schema bootstrapping (Option A)
--------------------------------
The canonical schema is seeded from the full set of relation definitions produced
by Step 2 (``bootstrap_schema``).  The embeddings of those definitions are then
used to find the top-K most semantically similar schema entries for any new
relation being canonicalized.  Because all extracted relations *are* the initial
schema, canonicalization here acts as a **merging / normalisation step**: two
relations that mean the same thing (e.g. ``selectedByNASA`` and ``recruitedBy``)
will be collapsed into whichever the LLM deems the better canonical form.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer

from models import CanonicalizationDecision, Triplet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert in ontology and knowledge graph construction. "
    "Select the most semantically appropriate canonical relation from a ranked list of candidates."
)

_USER_PROMPT_TEMPLATE = """\
A relation was extracted from text. Your task is to decide whether it should be merged \
into one of the provided canonical relations, or kept as-is.

Extracted relation: "{relation}"
Definition:         {definition}

Example triplet using this relation:
  [{subject}] --[{relation}]--> [{object_entity}]

Candidate canonical relations (ranked by semantic similarity, most similar first):
{choices}

Choose the candidate whose meaning best matches "{relation}" in the context above.
If no candidate is a good semantic match, respond with the original relation name "{relation}".
"""


class Canonicalizer:
    """Maps raw extracted relations to canonical forms.

    Parameters
    ----------
    top_k:
        Number of nearest-neighbour candidates to surface from the schema for
        each LLM canonicalization call.
    embedder_model:
        ``sentence-transformers`` model name for computing relation embeddings.
    llm_model:
        Gemini model identifier for the LLM decision step.
    max_retries:
        Automatic retries on transient API errors.
    """

    def __init__(
        self,
        top_k: int = 5,
        embedder_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gemini-2.5-flash",
        max_retries: int = 2,
    ) -> None:
        self.top_k = top_k
        self._embedder = SentenceTransformer(embedder_model)

        llm = ChatGoogleGenerativeAI(model=llm_model, max_retries=max_retries)
        self._chain = llm.with_structured_output(CanonicalizationDecision)

        # Schema state (populated by bootstrap_schema)
        self._schema: dict[str, str] = {}
        self._schema_items: list[tuple[str, str]] = []
        self._schema_embeddings: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def bootstrap_schema(self, relation_definitions: dict[str, str]) -> None:
        """Seed the canonical schema from Step 2's relation definitions.

        This is the Option A "self-bootstrapping" approach: the relations
        extracted from the text themselves form the initial canonical vocabulary.
        Canonicalization then merges near-duplicate relations into one.
        """
        self._schema = dict(relation_definitions)
        self._recompute_embeddings()
        logger.info(
            "Schema bootstrapped with %d canonical relation(s).", len(self._schema)
        )

    def _recompute_embeddings(self) -> None:
        self._schema_items = list(self._schema.items())
        if not self._schema_items:
            self._schema_embeddings = None
            return
        texts = [f"{rel}: {defn}" for rel, defn in self._schema_items]
        self._schema_embeddings = self._embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

    # ------------------------------------------------------------------
    # Canonicalization
    # ------------------------------------------------------------------

    def canonicalize_all(
        self,
        relation_definitions: dict[str, str],
        triplets: list[Triplet],
    ) -> dict[str, str]:
        """Return a ``{raw_relation: canonical_relation}`` mapping for all unique relations.

        One LLM call is made per unique relation (not per triplet), using an
        example triplet for context.

        Parameters
        ----------
        relation_definitions:
            Output of :class:`definer.RelationDefiner.define`.
        triplets:
            All extracted triplets (used to find one example per relation).
        """
        # Build a lookup: one example triplet per relation for prompt context
        examples: dict[str, Triplet] = {}
        for t in triplets:
            if t.relation not in examples:
                examples[t.relation] = t

        mapping: dict[str, str] = {}
        unique_relations = list(relation_definitions.keys())
        logger.info("Canonicalizing %d unique relation(s) …", len(unique_relations))

        for relation in unique_relations:
            definition = relation_definitions.get(relation, "")
            example = examples.get(relation)
            canonical = self._canonicalize_one(
                relation=relation,
                definition=definition,
                subject=example.subject if example else "",
                object_entity=example.object_entity if example else "",
            )
            mapping[relation] = canonical
            if canonical != relation:
                logger.debug("  '%s'  →  '%s'", relation, canonical)

        return mapping

    def _canonicalize_one(
        self,
        relation: str,
        definition: str,
        subject: str = "",
        object_entity: str = "",
    ) -> str:
        """Run one LLM canonicalization call for a single relation. Returns canonical name."""
        candidates = self._retrieve_top_k(relation, definition)
        if not candidates:
            return relation

        # Build numbered choice list
        choices_lines = [
            f"  {i + 1}. '{rel}': {defn}" for i, (rel, defn) in enumerate(candidates)
        ]
        choices_str = "\n".join(choices_lines)

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=_USER_PROMPT_TEMPLATE.format(
                    relation=relation,
                    definition=definition or "(no definition available)",
                    subject=subject,
                    object_entity=object_entity,
                    choices=choices_str,
                )
            ),
        ]
        try:
            decision: CanonicalizationDecision = self._chain.invoke(messages)  # type: ignore[assignment]
            return decision.canonical_relation.strip() or relation
        except Exception:
            logger.exception(
                "Canonicalization failed for '%s' — keeping original.", relation
            )
            return relation

    def _retrieve_top_k(
        self, relation: str, definition: str
    ) -> list[tuple[str, str]]:
        """Return the top-K (relation, definition) pairs by cosine similarity."""
        if self._schema_embeddings is None or not self._schema_items:
            return []
        query_text = f"{relation}: {definition}"
        query_emb = self._embedder.encode(
            query_text, normalize_embeddings=True, show_progress_bar=False
        )
        # Dot product == cosine similarity for L2-normalized vectors
        scores: np.ndarray = self._schema_embeddings @ query_emb
        k = min(self.top_k, len(self._schema_items))
        top_indices = np.argsort(scores)[::-1][:k]
        return [self._schema_items[int(i)] for i in top_indices]

