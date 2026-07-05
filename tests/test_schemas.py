"""Defensive validation tests for the HIR Pydantic schemas.

Covers happy-path construction, boundary rejection on constrained
fields, missing-required-field errors, Literal enforcement, and
round-trip serialization for every model in ``src.schemas``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from uuid import uuid4

from src.schemas import (
    DocumentIR,
    EntityPreservationReport,
    FeatureVector,
    ParagraphIR,
    TransformationPlan,
    ValidationReport,
)
from src.config import STYLE_PROFILES, get_profile


# ====================================================================
# FeatureVector
# ====================================================================

class TestFeatureVector:
    """FeatureVector must enforce [0, 1] on ratio fields and ≥ 0 on
    length / variance fields."""

    @staticmethod
    def _valid_kwargs() -> dict:
        return dict(
            lexical_diversity=0.5,
            readability_score=0.7,
            avg_sentence_length=15.0,
            sentence_variance=5.0,
            passive_ratio=0.1,
            repetition_score=0.3,
        )

    def test_valid_construction(self) -> None:
        fv = FeatureVector(**self._valid_kwargs())
        assert fv.lexical_diversity == 0.5
        assert fv.passive_ratio == 0.1

    def test_boundary_zero(self) -> None:
        kw = self._valid_kwargs()
        kw["lexical_diversity"] = 0.0
        kw["passive_ratio"] = 0.0
        fv = FeatureVector(**kw)
        assert fv.lexical_diversity == 0.0

    def test_boundary_one(self) -> None:
        kw = self._valid_kwargs()
        kw["lexical_diversity"] = 1.0
        kw["readability_score"] = 1.0
        fv = FeatureVector(**kw)
        assert fv.lexical_diversity == 1.0

    @pytest.mark.parametrize(
        "field,value",
        [
            ("lexical_diversity", 1.5),
            ("lexical_diversity", -0.01),
            ("readability_score", 2.0),
            ("readability_score", -1.0),
            ("passive_ratio", 1.1),
            ("passive_ratio", -0.5),
            ("repetition_score", 5.0),
            ("repetition_score", -0.1),
            ("avg_sentence_length", -1.0),
            ("sentence_variance", -0.001),
        ],
    )
    def test_out_of_range_rejected(self, field: str, value: float) -> None:
        kw = self._valid_kwargs()
        kw[field] = value
        with pytest.raises(ValidationError):
            FeatureVector(**kw)

    def test_missing_required_field(self) -> None:
        kw = self._valid_kwargs()
        del kw["readability_score"]
        with pytest.raises(ValidationError):
            FeatureVector(**kw)

    def test_round_trip_serialization(self) -> None:
        fv = FeatureVector(**self._valid_kwargs())
        fv2 = FeatureVector.model_validate(fv.model_dump())
        assert fv == fv2


# ====================================================================
# TransformationPlan
# ====================================================================

class TestTransformationPlan:
    def test_valid_construction(self) -> None:
        plan = TransformationPlan(
            target_profile="academic",
            selected_instructions=["Rewrite for clarity"],
            temperature_override=0.25,
            top_p_override=0.85,
        )
        assert plan.target_profile == "academic"
        assert len(plan.selected_instructions) == 1

    def test_missing_field(self) -> None:
        with pytest.raises(ValidationError):
            TransformationPlan(
                target_profile="academic",
                # missing selected_instructions
                temperature_override=0.25,
                top_p_override=0.85,
            )  # type: ignore[call-arg]


# ====================================================================
# EntityPreservationReport
# ====================================================================

class TestEntityPreservationReport:
    def test_valid(self) -> None:
        report = EntityPreservationReport(
            original_count=5,
            rewritten_count=4,
            missing_critical_entities=["ACME Corp"],
            is_valid=False,
        )
        assert not report.is_valid

    def test_empty_missing_is_valid(self) -> None:
        report = EntityPreservationReport(
            original_count=3,
            rewritten_count=3,
            missing_critical_entities=[],
            is_valid=True,
        )
        assert report.is_valid


# ====================================================================
# ValidationReport
# ====================================================================

class TestValidationReport:
    def test_valid(self) -> None:
        entity = EntityPreservationReport(
            original_count=2,
            rewritten_count=2,
            missing_critical_entities=[],
            is_valid=True,
        )
        report = ValidationReport(
            structural_integrity_passed=True,
            semantic_similarity_score=0.95,
            entity_report=entity,
            asymmetric_contradiction_detected=False,
            overall_valid=True,
        )
        assert report.overall_valid


# ====================================================================
# ParagraphIR
# ====================================================================

class TestParagraphIR:
    def test_valid_construction_defaults(self) -> None:
        node = ParagraphIR(
            id=uuid4(),
            node_index=0,
            node_type="paragraph",
            editable=True,
            original_text="Hello world.",
        )
        assert node.rewritten_text is None
        assert node.context_before == []
        assert node.context_after == []
        assert node.feature_vector is None

    def test_invalid_node_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ParagraphIR(
                id=uuid4(),
                node_index=0,
                node_type="blockquote",  # type: ignore[arg-type]
                editable=True,
                original_text="test",
            )

    def test_missing_original_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ParagraphIR(
                id=uuid4(),
                node_index=0,
                node_type="paragraph",
                editable=True,
                # original_text intentionally omitted
            )  # type: ignore[call-arg]

    @pytest.mark.parametrize("ntype", ["paragraph", "heading", "table", "code", "math"])
    def test_all_node_types_accepted(self, ntype: str) -> None:
        node = ParagraphIR(
            id=uuid4(),
            node_index=0,
            node_type=ntype,  # type: ignore[arg-type]
            editable=False,
            original_text="x",
        )
        assert node.node_type == ntype


# ====================================================================
# DocumentIR
# ====================================================================

class TestDocumentIR:
    def test_valid_construction(self) -> None:
        doc = DocumentIR(
            document_id=uuid4(),
            metadata={"key": "value"},
            nodes=[
                ParagraphIR(
                    id=uuid4(),
                    node_index=0,
                    node_type="paragraph",
                    editable=True,
                    original_text="Test.",
                )
            ],
        )
        assert len(doc.nodes) == 1

    def test_empty_nodes_accepted(self) -> None:
        doc = DocumentIR(
            document_id=uuid4(),
            metadata={},
            nodes=[],
        )
        assert doc.nodes == []

    def test_round_trip_serialization(self) -> None:
        doc = DocumentIR(
            document_id=uuid4(),
            metadata={"fmt": "md"},
            nodes=[
                ParagraphIR(
                    id=uuid4(),
                    node_index=0,
                    node_type="heading",
                    editable=False,
                    original_text="# Title",
                ),
                ParagraphIR(
                    id=uuid4(),
                    node_index=1,
                    node_type="paragraph",
                    editable=True,
                    original_text="Body text.",
                    context_before=["# Title"],
                ),
            ],
        )
        data = doc.model_dump(mode="json")
        doc2 = DocumentIR.model_validate(data)
        assert doc == doc2


# ====================================================================
# Config: STYLE_PROFILES
# ====================================================================

class TestStyleProfiles:
    @pytest.mark.parametrize("name", ["academic", "professional", "expressive"])
    def test_profile_exists(self, name: str) -> None:
        profile = get_profile(name)
        assert "temperature" in profile
        assert "top_p" in profile
        assert "validator_threshold" in profile
        assert "core_instructions" in profile
        assert isinstance(profile["core_instructions"], list)
        assert len(profile["core_instructions"]) >= 1

    def test_unknown_profile_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown style profile"):
            get_profile("nonexistent")

    def test_profiles_immutable_keys(self) -> None:
        expected_keys = {"temperature", "top_p", "validator_threshold", "core_instructions"}
        for name, profile in STYLE_PROFILES.items():
            assert set(profile.keys()) == expected_keys, (
                f"Profile {name!r} has unexpected keys"
            )
