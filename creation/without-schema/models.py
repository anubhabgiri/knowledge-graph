"""Shared Pydantic data models for the without-schema KG pipeline."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# LLM-facing structured output models
# ---------------------------------------------------------------------------


class RawTriplet(BaseModel):
    """A single triplet as returned by Step 1 (extraction)."""

    subject: str = Field(description="The subject entity of the relation")
    relation: str = Field(description="A concise predicate name in camelCase, e.g. bornIn, ceoOf")
    object_entity: str = Field(
        alias="object",
        description="The object entity or value of the relation",
    )

    model_config = {"populate_by_name": True}


class RawTripletList(BaseModel):
    """Structured output wrapper — list of raw triplets from one chunk."""

    triplets: list[RawTriplet] = Field(
        description="All relational triplets extracted from the text chunk"
    )


class RelationDefinitionItem(BaseModel):
    """One relation paired with its natural-language definition."""

    relation_name: str = Field(description="The exact relation name as extracted")
    definition: str = Field(
        description="A concise, one-sentence definition of the relation's general semantic meaning"
    )


class RelationDefinitionList(BaseModel):
    """Structured output wrapper — definitions for a batch of relations."""

    definitions: list[RelationDefinitionItem]


class CanonicalizationDecision(BaseModel):
    """LLM's choice of canonical relation for a raw relation."""

    canonical_relation: str = Field(
        description=(
            "The chosen canonical relation name from the provided choices. "
            "Use the exact original relation name if none of the choices are a good semantic match."
        )
    )
    rationale: str = Field(description="One-sentence explanation of the choice")


# ---------------------------------------------------------------------------
# Internal / output models
# ---------------------------------------------------------------------------


class Triplet(BaseModel):
    """A fully processed triplet, optionally with a canonical relation."""

    subject: str
    relation: str  # raw extracted relation
    object_entity: str = Field(alias="object")
    source_chunk: int = Field(description="Zero-based index of the source chunk")
    canonical_relation: Optional[str] = None

    model_config = {"populate_by_name": True}


class GraphOutput(BaseModel):
    """The final knowledge graph written to the output JSON file."""

    triplets: list[Triplet]
    relation_definitions: dict[str, str] = Field(
        description="Canonical relation name → its natural-language definition"
    )
    metadata: dict[str, Any] = Field(
        description="Run metadata: timestamp, model, chunk_count, raw_triplet_count"
    )

