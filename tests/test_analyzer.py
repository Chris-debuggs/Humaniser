"""Tests for the Phase 3 Deterministic Analyzer.

Verifies boundary constraints and metric calculations using mocked
textual inputs for edge cases (empty strings, single words, passives).
"""

from __future__ import annotations

import uuid

import pytest

from src.analyzer import analyze_paragraph
from src.schemas import ParagraphIR


def create_paragraph(text: str, editable: bool = True) -> ParagraphIR:
    """Helper to mock a ParagraphIR for analysis."""
    return ParagraphIR(
        id=uuid.uuid4(),
        node_index=0,
        node_type="paragraph",
        editable=editable,
        original_text=text,
    )


class TestAnalyzerEdgeCases:
    def test_not_editable_returns_zeros(self) -> None:
        node = create_paragraph("Some text", editable=False)
        vector = analyze_paragraph(node)
        assert vector.lexical_diversity == 0.0
        assert vector.readability_score == 0.0
        assert vector.avg_sentence_length == 0.0
        assert vector.sentence_variance == 0.0
        assert vector.passive_ratio == 0.0
        assert vector.repetition_score == 0.0

    def test_empty_string_returns_zeros(self) -> None:
        node = create_paragraph("   \n  ")
        vector = analyze_paragraph(node)
        assert vector.lexical_diversity == 0.0
        assert vector.readability_score == 0.0

    def test_single_word_paragraph(self) -> None:
        node = create_paragraph("Hello.")
        vector = analyze_paragraph(node)
        # Type-token ratio of 1 word is 1.0
        assert vector.lexical_diversity == 1.0
        # Average sentence length should be 1
        assert vector.avg_sentence_length == 1.0
        # Variance of one item is 0.0
        assert vector.sentence_variance == 0.0
        # No repetition for 1 word
        assert vector.repetition_score == 0.0
        assert vector.passive_ratio == 0.0


class TestAnalyzerMetrics:
    def test_repetition_score(self) -> None:
        # "test test test test" has 4 words, 3 bigrams.
        # The bigrams are all "test_test" (1 unique bigram).
        # Score = 1.0 - (1 / 3) = 0.666...
        node = create_paragraph("Test test test test.")
        vector = analyze_paragraph(node)
        assert vector.repetition_score > 0.6
        assert vector.lexical_diversity == 0.25  # 1 unique word / 4 total words

    def test_passive_ratio(self) -> None:
        # A classic passive sentence: "The ball was thrown by the boy."
        # spaCy should tag "was" as auxpass and "ball" as nsubjpass.
        node = create_paragraph("The ball was thrown by the boy.")
        vector = analyze_paragraph(node)
        # Should detect at least one passive construct
        assert vector.passive_ratio > 0.0

    def test_active_ratio(self) -> None:
        # Active sentence: "The boy threw the ball."
        node = create_paragraph("The boy threw the ball.")
        vector = analyze_paragraph(node)
        assert vector.passive_ratio == 0.0

    def test_readability_and_variance(self) -> None:
        # Two sentences, one long, one short.
        text = "This is a very long and complicated sentence that goes on for a while. Short one."
        node = create_paragraph(text)
        vector = analyze_paragraph(node)
        
        assert vector.readability_score > 0.0
        assert vector.avg_sentence_length > 0.0
        assert vector.sentence_variance > 0.0
