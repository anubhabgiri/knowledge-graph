"""Step 1 — Open Information Extraction: raw [Subject, Relation, Object] triplets per chunk."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow running this file directly or importing it from the pipeline
sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from models import RawTripletList, Triplet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert in ontology, semantic relations, and knowledge graph construction. "
    "Extract precise, atomic relational triplets from text. "
    "Use concise camelCase predicate names (e.g. bornIn, ceoOf, locatedIn, foundedBy)."
)

_USER_PROMPT_TEMPLATE = """\
Extract ALL relational triplets from the text below.
Each triplet captures one factual relationship between two entities or an entity and a value.

--- Examples ---

Text: "The 17068.8 millimeter long ALCO RS-3 has a diesel-electric transmission."
Triplets:
  subject="ALCO RS-3"  relation="length"     object="17068.8 millimetres"
  subject="ALCO RS-3"  relation="powerType"  object="diesel-electric transmission"

Text: "Alan Shepard was born on Nov 18, 1923 and selected by NASA in 1959. He was a member of the Apollo 14 crew."
Triplets:
  subject="Alan Shepard"  relation="birthDate"    object="1923-11-18"
  subject="Alan Shepard"  relation="selectedBy"   object="NASA"
  subject="Alan Shepard"  relation="selectedYear" object="1959"
  subject="Alan Shepard"  relation="crewOf"       object="Apollo 14"

--- End Examples ---

Now extract all triplets from:
Text: \"\"\"{chunk}\"\"\"
"""


class TripletExtractor:
    """Extracts raw [Subject, Relation, Object] triplets from a text chunk.

    Uses LangChain's ``with_structured_output`` to get guaranteed JSON that
    conforms to :class:`models.RawTripletList`.

    Parameters
    ----------
    model:
        Gemini model identifier.
    max_retries:
        Number of automatic retries on transient API errors.
    """

    def __init__(self, model: str = "gemini-2.5-flash", max_retries: int = 2) -> None:
        llm = ChatGoogleGenerativeAI(model=model, max_retries=max_retries)
        self._chain = llm.with_structured_output(RawTripletList)

    def extract(self, chunk: str, chunk_index: int) -> list[Triplet]:
        """Return a list of :class:`Triplet` extracted from *chunk*.

        Falls back to an empty list on any LLM or parsing error so the
        pipeline can continue processing remaining chunks.
        """
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=_USER_PROMPT_TEMPLATE.format(chunk=chunk)),
        ]
        try:
            result: RawTripletList = self._chain.invoke(messages)  # type: ignore[assignment]
        except Exception:
            logger.exception("Triplet extraction failed for chunk %d — skipping.", chunk_index)
            return []

        triplets = []
        for raw in result.triplets:
            triplets.append(
                Triplet(
                    subject=raw.subject.strip(),
                    relation=raw.relation.strip(),
                    object=raw.object_entity.strip(),
                    source_chunk=chunk_index,
                )
            )
        logger.debug(
            "Chunk %d: extracted %d triplet(s).", chunk_index, len(triplets)
        )
        return triplets

