"""Hidden Intermediate Representation (HIR) schemas.

All pipeline modules consume and produce these Pydantic v2 models.
Boundary validators enforce that constrained fields reject malformed
payloads at construction time rather than deep inside pipeline logic.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

class FeatureVector(BaseModel):
    """Lexical, syntactic, and structural metrics for a single paragraph."""

    lexical_diversity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Type-Token Ratio [0.0, 1.0]",
    )
    readability_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized Flesch-Kincaid / textstat metric",
    )
    avg_sentence_length: float = Field(
        ...,
        ge=0.0,
        description="Average number of words per sentence",
    )
    sentence_variance: float = Field(
        ...,
        ge=0.0,
        description="Variance in sentence lengths",
    )
    passive_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Percentage of passive constructions [0.0, 1.0]",
    )
    repetition_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Frequency score of duplicated word transitions",
    )


# ---------------------------------------------------------------------------
# Transformation Planning
# ---------------------------------------------------------------------------

class TransformationPlan(BaseModel):
    """Output of the planner: binds a style profile to concrete
    rewrite instructions and LLM generation parameters."""

    target_profile: str
    selected_instructions: List[str]
    temperature_override: float
    top_p_override: float


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class EntityPreservationReport(BaseModel):
    """Entity-level diff between original and rewritten text."""

    original_count: int
    rewritten_count: int
    missing_critical_entities: List[str]
    is_valid: bool


class ValidationReport(BaseModel):
    """Composite validation gate for a single rewrite."""

    structural_integrity_passed: bool
    semantic_similarity_score: float
    entity_report: EntityPreservationReport
    asymmetric_contradiction_detected: bool
    overall_valid: bool


# ---------------------------------------------------------------------------
# Intermediate Representation Nodes
# ---------------------------------------------------------------------------

class ParagraphIR(BaseModel):
    """A single structural leaf node in the document IR.

    Non-editable nodes (code, math, tables, headings) are locked and pass
    through the pipeline untouched.  Editable paragraphs carry optional
    feature vectors, transformation plans, and validation reports populated
    by successive pipeline stages.
    """

    id: UUID
    node_index: int
    node_type: Literal["paragraph", "heading", "table", "code", "math"]
    editable: bool
    original_text: str
    rewritten_text: Optional[str] = None
    context_before: List[str] = Field(default_factory=list)
    context_after: List[str] = Field(default_factory=list)
    feature_vector: Optional[FeatureVector] = None
    transformation_plan: Optional[TransformationPlan] = None
    validation_report: Optional[ValidationReport] = None


class DocumentIR(BaseModel):
    """Root container for the full document intermediate representation."""

    document_id: UUID
    metadata: Dict[str, str]
    nodes: List[ParagraphIR]
