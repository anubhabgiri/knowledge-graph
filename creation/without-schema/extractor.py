"""Step 1 — Open Information Extraction: raw [Subject, Relation, Object] triplets per chunk."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow running this file directly or importing it from the pipeline
sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from models import RawTripletList, Triplet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain Prompts
# ---------------------------------------------------------------------------

_PROMPTS = {
    "general": {
        "system": (
            "You are an expert in ontology, semantic relations, and knowledge graph construction. "
            "Extract precise, atomic relational triplets from general text.\n\n"
            "CRITICAL EXTRACTION GUIDELINES:\n"
            "1. Coreference & Anaphora Resolution:\n"
            "   - NEVER use pronouns (e.g., 'he', 'she', 'it', 'they', 'this', 'these') as subjects or objects.\n"
            "   - Always resolve pronouns and definite descriptions (e.g., 'the executive', 'the company', 'this city') "
            "to the explicit, canonical entity name identified in the text.\n"
            "2. Document-Level & Cross-Sentence Reasoning:\n"
            "   - Read the entire passage as a coherent document. Synthesize relationships that span across sentences.\n"
            "   - If an entity is introduced in one sentence and its actions, attributes, roles, or relationships are "
            "described in subsequent sentences, extract the full cross-sentence relation.\n"
            "3. Entity Self-Containment:\n"
            "   - Both subject and object must be atomic, fully qualified entities or literal values that make complete sense independently.\n"
            "4. Directionality & Semantic Precision:\n"
            "   - Preserve strict directionality: [Subject/Agent] -> [relation] -> [Object/Target/Value].\n"
            "   - Use concise camelCase predicate names (e.g., bornIn, ceoOf, locatedIn, foundedBy, acquiredCompany)."
        ),
        "user_template": """\
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

Text: "Satya Nadella joined Microsoft in 1992. The executive was appointed CEO in 2014, succeeding Steve Ballmer. Under his leadership, the company acquired LinkedIn."
Triplets:
  subject="Satya Nadella"  relation="joinedCompany"  object="Microsoft"
  subject="Satya Nadella"  relation="joinedYear"     object="1992"
  subject="Satya Nadella"  relation="appointedRole"  object="CEO"
  subject="Satya Nadella"  relation="appointedYear"  object="2014"
  subject="Satya Nadella"  relation="succeeded"      object="Steve Ballmer"
  subject="Microsoft"      relation="acquired"       object="LinkedIn"

--- End Examples ---

Now extract all triplets from:
Text: \"\"\"{chunk}\"\"\"
""",
    },
    "biomedical": {
        "system": (
            "You are an expert in biomedical ontology, clinical knowledge graphs, and pharmacovigilance. "
            "Extract precise, atomic relational triplets from biomedical and clinical literature.\n\n"
            "CRITICAL EXTRACTION GUIDELINES FOR BIOMEDICAL TEXT:\n"
            "1. Coreference & Anaphora Resolution:\n"
            "   - NEVER use pronouns (e.g., 'it', 'they', 'this', 'these') or generic references ('the drug', 'the compound', 'the patient', 'the disease') as entities.\n"
            "   - Always resolve pronouns and generic references to their specific chemical, drug, gene, or disease name.\n"
            "2. Document-Level & Cross-Sentence Reasoning:\n"
            "   - Connect findings across the entire abstract. For example, if a drug is administered in the methods sentence "
            "and an adverse pathology or clinical outcome is reported sentences later, extract the direct causal or associative relation.\n"
            "3. Directionality & Causality vs. Indication:\n"
            "   - Differentiate Chemical-Induced Disease (causation/adverse effect/toxicity) from Therapeutic Indication (treatment/prevention):\n"
            "     * Induction / Toxicity: [Chemical] -> [inducesCondition / causesCondition / toxicTo] -> [Disease / Symptom]\n"
            "     * Treatment / Prevention: [Chemical] -> [treatsCondition / preventsCondition] -> [Disease]\n"
            "     * Biological mechanism: [Chemical] -> [inhibitsProcess / activatesProcess / metabolizes] -> [Target / Process]\n"
            "   - Ensure the subject is the chemical/agent and the object is the condition/outcome where causation is established.\n"
            "4. Entity Self-Containment:\n"
            "   - Use specific, standardized biomedical terms where possible (e.g., 'acute renal failure' instead of 'failure').\n"
            "   - Use concise camelCase predicate names (e.g., inducesCondition, treatsCondition, administeredTo, inhibitsProcess).\n"
            "5. Entity Atomicity Rule:\n"
            "   - Entities must be atomic noun phrases (e.g., 'Cisplatin', 'Acute Kidney Injury', 'Rat').\n"
            "   - Never use clauses, prepositions, or descriptive verb phrases as entities.\n"
            "   - When a sentence mentions a list of specific chemicals/diseases (e.g., 'including X, Y, and Z'), extract individual triplets for EACH specific named entity, not just the parent category."
        ),
        "user_template": """\
Extract ALL relational triplets from the biomedical text below.
Each triplet captures one factual relationship between biomedical entities (chemicals, diseases, genes, phenotypes) or values.

--- Examples ---

Text: "Cisplatin was administered to adult male rats. Two weeks following administration, the animals developed acute renal failure. This compound also inhibited cellular proliferation."
Triplets:
  subject="Cisplatin"  relation="administeredTo"    object="adult male rats"
  subject="Cisplatin"  relation="inducesCondition"   object="acute renal failure"
  subject="Cisplatin"  relation="inhibitedProcess"   object="cellular proliferation"

Text: "Patients with hypertension were treated with Lisinopril. The drug significantly reduced systolic blood pressure, but several patients experienced persistent dry cough."
Triplets:
  subject="Lisinopril"  relation="treatsCondition"    object="hypertension"
  subject="Lisinopril"  relation="reducedMeasure"     object="systolic blood pressure"
  subject="Lisinopril"  relation="inducesCondition"   object="persistent dry cough"

--- End Examples ---

Now extract all triplets from:
Text: \"\"\"{chunk}\"\"\"
""",
    },
}


class TripletExtractor:
    """Extracts raw [Subject, Relation, Object] triplets from a text chunk.

    Uses LangChain's ``with_structured_output`` to get guaranteed JSON that
    conforms to :class:`models.RawTripletList`.

    Parameters
    ----------
    llm:
        A pre-built LangChain ``BaseChatModel`` instance.
    domain:
        Domain configuration preset: ``"general"`` or ``"biomedical"``.
    """

    def __init__(self, llm: BaseChatModel, domain: str = "general") -> None:
        self._chain = llm.with_structured_output(RawTripletList)
        self.domain = domain.lower() if domain else "general"

    def extract(self, chunk: str, chunk_index: int) -> list[Triplet]:
        """Return a list of :class:`Triplet` extracted from *chunk*.

        Falls back to an empty list on any LLM or parsing error so the
        pipeline can continue processing remaining chunks.
        """
        prompt_config = _PROMPTS.get(self.domain, _PROMPTS["general"])
        messages = [
            SystemMessage(content=prompt_config["system"]),
            HumanMessage(content=prompt_config["user_template"].format(chunk=chunk)),
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

