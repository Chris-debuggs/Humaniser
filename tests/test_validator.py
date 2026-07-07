"""Tests for the Phase 6 Validation Engine.

Verifies structural integrity rejection (markdown hallucination, regurgitation),
entity preservation, and semantic similarity scoring.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.schemas import ParagraphIR, TransformationPlan
from src.validator import validate_rewrite


def create_mock_node(original_text: str, context_before: list[str] = None) -> ParagraphIR:
    return ParagraphIR(
        id=uuid.uuid4(),
        node_index=1,
        node_type="paragraph",
        editable=True,
        original_text=original_text,
        context_before=context_before or [],
        transformation_plan=TransformationPlan(
            target_profile="expressive",  # Threshold: 0.80
            selected_instructions=[],
            temperature_override=0.5,
            top_p_override=0.9,
        ),
    )


@patch("src.validator.SentenceTransformer")
@patch("src.validator.nlp")
def test_validation_passes(mock_nlp, mock_st) -> None:
    # Setup mocks
    mock_model = MagicMock()
    # Mock cosine similarity to return 0.95 (above 0.80 threshold)
    mock_model.encode.return_value = "tensor"
    mock_st.return_value = mock_model
    
    with patch("src.validator.util.cos_sim", return_value=MagicMock(item=lambda: 0.95)):
        node = create_mock_node("The company Apple was founded in 1976.")
        # Setup entity extraction mock for both original and rewritten text
        mock_doc = MagicMock()
        mock_ent = MagicMock()
        mock_ent.label_ = "ORG"
        mock_ent.text = "Apple"
        mock_doc.ents = [mock_ent]
        mock_nlp.return_value = mock_doc

        rewrite = "Apple, the tech giant, was established back in 1976."
        report = validate_rewrite(node, rewrite)

        assert report.structural_integrity_passed is True
        assert report.entity_report.is_valid is True
        assert report.overall_valid is True


@patch("src.validator.SentenceTransformer")
def test_structural_failure_markdown_hallucination(mock_st) -> None:
    # If structural check fails, similarity isn't even the bottleneck, but it computes it anyway
    mock_model = MagicMock()
    mock_st.return_value = mock_model
    with patch("src.validator.util.cos_sim", return_value=MagicMock(item=lambda: 0.90)):
        node = create_mock_node("Here is some text.")
        # Hallucinate a markdown block
        rewrite = "Here is some text.\n```python\nprint(1)\n```"
        
        report = validate_rewrite(node, rewrite)
        assert report.structural_integrity_passed is False
        assert report.overall_valid is False


@patch("src.validator.SentenceTransformer")
def test_structural_failure_context_regurgitation(mock_st) -> None:
    mock_model = MagicMock()
    mock_st.return_value = mock_model
    with patch("src.validator.util.cos_sim", return_value=MagicMock(item=lambda: 0.90)):
        node = create_mock_node(
            "This is the current paragraph.",
            context_before=["This is a very long sentence from the previous context block that shouldn't be here."]
        )
        
        rewrite = "This is a very long sentence from the previous context block that shouldn't be here. And also the rewrite."
        report = validate_rewrite(node, rewrite)
        
        assert report.structural_integrity_passed is False
        assert report.overall_valid is False


@patch("src.validator.SentenceTransformer")
@patch("src.validator.nlp")
def test_entity_preservation_failure(mock_nlp, mock_st) -> None:
    mock_model = MagicMock()
    mock_st.return_value = mock_model
    with patch("src.validator.util.cos_sim", return_value=MagicMock(item=lambda: 0.95)):
        node = create_mock_node("There are 500 apples.")
        
        # Original text mock returning NUM "500"
        orig_ent = MagicMock()
        orig_ent.label_ = "NUM"
        orig_ent.text = "500"
        orig_doc = MagicMock()
        orig_doc.ents = [orig_ent]

        # Rewritten text mock returning NO entities
        rewritten_doc = MagicMock()
        rewritten_doc.ents = []
        
        # Side effect to return orig_doc for the first call, rewritten_doc for the second
        mock_nlp.side_effect = [orig_doc, rewritten_doc]

        rewrite = "There are many apples."
        report = validate_rewrite(node, rewrite)

        assert report.entity_report.is_valid is False
        assert "500" in report.entity_report.missing_critical_entities
        assert report.overall_valid is False


@patch("src.validator.SentenceTransformer")
def test_semantic_similarity_failure(mock_st) -> None:
    mock_model = MagicMock()
    mock_st.return_value = mock_model
    # Drop similarity below 0.80 expressive threshold
    with patch("src.validator.util.cos_sim", return_value=MagicMock(item=lambda: 0.50)):
        node = create_mock_node("We went to the store.")
        # Completely unrelated rewrite
        rewrite = "Astronauts landed on the moon."
        
        report = validate_rewrite(node, rewrite)
        assert report.semantic_similarity_score == 0.50
        assert report.overall_valid is False
