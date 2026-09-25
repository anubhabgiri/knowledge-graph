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
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert in ontology, semantics, and knowledge graph construction. "
    "Write concise, precise definitions for semantic relation predicates."
)

_USER_PROMPT_TEMPLATE = """\
Below are relational triplets extracted from a document. Write a concise one-sentence definition \
for each unique relation listed, capturing its general semantic meaning — not just this specific instance.

Triplet examples (for context):
{examples}

Relations to define (write exactly one definition per relation):
{relations}

Return a definition for EVERY relation in the list above. \
If a relation name is ambiguous, use the triplet examples for context.
"""


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

    Making one batched call rather than one-per-relation keeps latency and
    cost low even for documents that yield dozens of unique relations.

    Parameters
    ----------
    llm:
        A pre-built LangChain ``BaseChatModel`` instance.  Use
        :class:`llm_manager.LLMManager` to construct one for any supported
        provider (Gemini, OpenAI, Ollama).
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self._chain = llm.with_structured_output(RelationDefinitionList)

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
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=_USER_PROMPT_TEMPLATE.format(
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

