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
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer

from models import CanonicalizationDecision, Triplet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain Prompts
# ---------------------------------------------------------------------------

_PROMPTS = {
    "general": {
        "system": (
            "You are an expert in ontology engineering and knowledge graph construction. "
            "Your objective is to maintain high semantic precision and prevent schema degradation. "
            "Only merge relations that are genuine, unambiguous semantic equivalents."
        ),
        "user_template": """\
A relation was extracted from text. Your task is to decide whether it should be merged \
into one of the candidate canonical relations, or kept as-is.

Extracted relation: "{relation}"
Definition:         {definition}

Example triplet using this relation:
  [{subject}] --[{relation}]--> [{object_entity}]

Candidate canonical relations (ranked by semantic similarity):
{choices}

CRITICAL RULES TO AVOID OVER-MERGING:
1. Strict Semantic Equivalence: Merge ONLY if the candidate relation conveys the EXACT same meaning and can replace "{relation}" without loss of specificity (e.g. "bornIn" <-> "birthPlace", "ceoOf" <-> "chiefExecutiveOfficerOf").
2. Polarity & Antonym Guard: NEVER merge relations with opposing meanings or effects (e.g. NEVER merge "parentOf" with "childOf", "foundedBy" with "acquiredBy").
3. Causal Strength Guard: NEVER merge direct causation with indirect association (e.g. "caused" vs "associatedWith").
4. Directionality Guard: NEVER merge inverse relations (e.g. "employerOf" vs "employedBy").
5. Specificity Guard: NEVER merge a specific relation into an overly broad, vague relation (e.g. do NOT merge "graduatedFrom" into "associatedWith" or "relatedTo").
6. Conservative Default: If NONE of the candidates is an exact semantic match, or if you are in ANY doubt, respond with the original relation name "{relation}".
""",
    },
    "biomedical": {
        "system": (
            "You are an expert in biomedical ontology engineering, pharmacology, and clinical knowledge graphs. "
            "Your objective is to maintain strict medical accuracy and prevent schema degradation. "
            "Only merge relations that are genuine, unambiguous medical equivalents."
        ),
        "user_template": """\
A biomedical relation was extracted from literature. Your task is to decide whether it should be merged \
into one of the candidate canonical relations, or kept as-is.

Extracted relation: "{relation}"
Definition:         {definition}

Example triplet using this relation:
  [{subject}] --[{relation}]--> [{object_entity}]

Candidate canonical relations (ranked by semantic similarity):
{choices}

CRITICAL RULES TO AVOID OVER-MERGING IN BIOMEDICAL CONTEXTS:
1. Strict Semantic Equivalence: Merge ONLY if the candidate relation conveys the EXACT same clinical/pharmacological meaning (e.g. "causesCondition" <-> "inducesDisease", "administeredTo" <-> "givenTo").
2. Polarity & Antonym Guard: NEVER merge opposing clinical effects (e.g. NEVER merge "treats" or "prevents" with "causes" or "induces"; NEVER merge "inhibits" with "activates"; NEVER merge "increasesRisk" with "decreasesRisk").
3. Causality vs Indication Guard: NEVER merge drug-induced toxicity (CID) with therapeutic indication or off-label use.
4. Causal Strength Guard: NEVER merge direct causal etiology ("inducesCondition") with mere observational correlation ("associatedWith", "coOccursWith", "studiedIn").
5. Directionality Guard: NEVER merge inverse relationships (e.g. "administeredTo" vs "receivedBy", "metabolizedBy" vs "metabolizes").
6. Specificity Guard: NEVER collapse fine-grained clinical relations into broad catch-alls (e.g. do NOT merge "diagnosedWith" or "treatedWith" into "affects" or "relatedTo").
7. Conservative Default: If NONE of the candidates is an exact semantic match, or if you are in ANY doubt, respond with the original relation name "{relation}".
""",
    },
}

# Project root is three levels up from this file (project/creation/without-schema/canonicalizer.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_embedder_path(model_name: str) -> str:
    """Return a local filesystem path if the model was pre-downloaded, else the HF model name.

    Pre-download once with:  python creation/without-schema/download_models.py
    Model is saved to:       <project_root>/models/<model_name>/
    """
    local_path = _PROJECT_ROOT / "models" / model_name
    if local_path.exists():
        logger.info("Loading embedder from local path: %s", local_path)
        return str(local_path)
    logger.info(
        "Local model not found at '%s' — loading '%s' from HuggingFace (run "
        "download_models.py to cache it locally).",
        local_path,
        model_name,
    )
    return model_name


class Canonicalizer:
    """Maps raw extracted relations to canonical forms.

    Parameters
    ----------
    llm:
        A pre-built LangChain ``BaseChatModel`` instance.  Use
        :class:`llm_manager.LLMManager` to construct one for any supported
        provider (Gemini, OpenAI, Ollama).
    top_k:
        Number of nearest-neighbour candidates to surface from the schema for
        each LLM canonicalization call.
    embedder_model:
        ``sentence-transformers`` model name for computing relation embeddings.
        If a local copy exists under ``<project_root>/models/<embedder_model>/``
        (placed there by ``download_models.py``), it is loaded from disk with no
        network requests.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        top_k: int = 5,
        min_similarity: float = 0.45,
        embedder_model: str = "all-MiniLM-L6-v2",
        domain: str = "general",
    ) -> None:
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.domain = domain.lower() if domain else "general"
        resolved = _resolve_embedder_path(embedder_model)
        self._embedder = SentenceTransformer(resolved, local_files_only=resolved != embedder_model)

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
        # If there are no candidate alternatives other than the relation itself, skip LLM
        alternatives = [c for c in candidates if c[0] != relation]
        if not alternatives:
            logger.debug("No close alternative candidates for '%s' — keeping original.", relation)
            return relation

        # Build numbered choice list including candidate relation names, scores, and definitions
        choices_lines = [
            f"  {i + 1}. '{rel}' (similarity: {score:.2f}): {defn}"
            for i, (rel, defn, score) in enumerate(candidates)
        ]
        choices_str = "\n".join(choices_lines)

        prompt_config = _PROMPTS.get(self.domain, _PROMPTS["general"])
        messages = [
            SystemMessage(content=prompt_config["system"]),
            HumanMessage(
                content=prompt_config["user_template"].format(
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
            canonical = decision.canonical_relation.strip() or relation
            valid_choices = {c[0] for c in candidates} | {relation}
            if canonical not in valid_choices:
                logger.warning(
                    "Canonicalizer produced '%s' not in valid candidates for '%s' — keeping original.",
                    canonical,
                    relation,
                )
                return relation
            return canonical
        except Exception:
            logger.exception(
                "Canonicalization failed for '%s' — keeping original.", relation
            )
            return relation

    def _retrieve_top_k(
        self, relation: str, definition: str
    ) -> list[tuple[str, str, float]]:
        """Return the top-K (relation, definition, score) tuples by cosine similarity >= min_similarity."""
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

        results: list[tuple[str, str, float]] = []
        for i in top_indices:
            score = float(scores[int(i)])
            if score >= self.min_similarity:
                rel, defn = self._schema_items[int(i)]
                results.append((rel, defn, score))
        return results

