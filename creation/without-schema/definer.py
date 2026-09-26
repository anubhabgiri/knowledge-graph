"""Step 2 — Generate natural-language definitions for all unique extracted relations."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from models import RelationDefinitionList, Triplet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain Prompts
# ---------------------------------------------------------------------------

_PROMPTS = {
    "general": {
        "system": (
            "You are an expert in ontology, semantics, and knowledge graph construction. "
            "Write concise, precise definitions for semantic relation predicates.\n\n"
            "DEFINITION GUIDELINES:\n"
            "- Explicitly state directionality: clearly describe what the subject entity is or does relative to the object entity.\n"
            "- Avoid vague definitions like 'relates to' or 'connects X and Y'."
        ),
        "user_template": """\
Below are relational triplets extracted from a document. Write a concise one-sentence definition \
for each unique relation listed, capturing its general semantic meaning and directionality — not just this specific instance.

Triplet examples (for context):
{examples}

Relations to define (write exactly one definition per relation):
{relations}

Return a definition for EVERY relation in the list above. \
If a relation name is ambiguous, use the triplet examples for context.
""",
    },
    "biomedical": {
        "system": (
            "You are an expert in biomedical semantics, pharmacology, and clinical ontology engineering. "
            "Write concise, rigorous definitions for biomedical relation predicates.\n\n"
            "CRITICAL BIOMEDICAL DEFINITION GUIDELINES:\n"
            "- Explicitly clarify causality vs indication: specify whether the relation denotes causing/inducing an adverse effect, "
            "acting as a therapeutic treatment/cure, biological modulation (inhibition/activation), or general clinical observation.\n"
            "- State directionality clearly (e.g., chemical -> target, drug -> disease).\n"
            "- Ensure definitions clearly discriminate between opposing clinical outcomes (e.g. therapeutic efficacy vs adverse toxicity)."
        ),
        "user_template": """\
Below are relational triplets extracted from biomedical literature. Write a concise one-sentence definition \
for each unique relation listed, explicitly capturing its clinical/biological nature, directionality, and causal polarity.

Triplet examples (for context):
{examples}

Relations to define (write exactly one definition per relation):
{relations}

Return a definition for EVERY relation in the list above. \
If a relation name is ambiguous, use the triplet examples for context.
""",
    },
}


def _build_example_lines(triplets: list[Triplet], unique_relations: list[str]) -> str:
    """One representative triplet per relation, formatted as a readable line."""
    seen: dict[str, str] = {}
    for t in triplets:
        if t.relation in unique_relations and t.relation not in seen:
            seen[t.relation] = (
                f"  [{t.subject}] --[{t.relation}]--> [{t.object_entity}]"
            )
    return "\n".join(seen.values()) or "  (no examples available)"


class RelationDefiner:
    """Batch-generates definitions for all unique relations in a single LLM call.

    Parameters
    ----------
    llm:
        A pre-built LangChain ``BaseChatModel`` instance.
    domain:
        Domain configuration preset: ``"general"`` or ``"biomedical"``.
    """

    def __init__(self, llm: BaseChatModel, domain: str = "general") -> None:
        self._chain = llm.with_structured_output(RelationDefinitionList)
        self.domain = domain.lower() if domain else "general"

    def define(self, triplets: list[Triplet]) -> dict[str, str]:
        """Return ``{relation_name: definition}`` for every unique relation in *triplets*.

        Any relation the LLM omits receives a placeholder definition so
        downstream stages are never blocked by a missing entry.
        """
        unique_relations = sorted({t.relation for t in triplets})
        if not unique_relations:
            return {}

        examples = _build_example_lines(triplets, unique_relations)
        relations_list = "\n".join(f"- {r}" for r in unique_relations)
        prompt_config = _PROMPTS.get(self.domain, _PROMPTS["general"])
        messages = [
            SystemMessage(content=prompt_config["system"]),
            HumanMessage(
                content=prompt_config["user_template"].format(
                    examples=examples,
                    relations=relations_list,
                )
            ),
        ]

        try:
            result: RelationDefinitionList = self._chain.invoke(messages)  # type: ignore[assignment]
            definitions = {item.relation_name: item.definition for item in result.definitions}
        except Exception:
            logger.exception("Relation definition generation failed — using empty definitions.")
            definitions = {}

        # Fallback: fill any relation the LLM omitted
        for rel in unique_relations:
            if rel not in definitions:
                logger.warning("LLM omitted definition for '%s' — using placeholder.", rel)
                definitions[rel] = f"Relation '{rel}' between the subject and object entities."

        logger.info("Defined %d relation(s).", len(definitions))
        return definitions

