"""Tests for the Phase 4 Transformation Planner.

Verifies that multidimensional deltas map correctly to dynamic instructions
based on target style profiles.
"""

from __future__ import annotations

import pytest

from src.planner import generate_plan
from src.schemas import FeatureVector


def create_vector(**kwargs) -> FeatureVector:
    """Helper to mock a FeatureVector."""
    defaults = dict(
        lexical_diversity=0.5,
        readability_score=0.5,
        avg_sentence_length=15.0,
        sentence_variance=7.0,
        passive_ratio=0.1,
        repetition_score=0.1,
    )
    defaults.update(kwargs)
    return FeatureVector(**defaults)


class TestTransformationPlanner:
    def test_academic_low_diversity(self) -> None:
        # Academic profile with low lexical diversity should trigger vocabulary elevation.
        vector = create_vector(lexical_diversity=0.3)
        plan = generate_plan(vector, "academic")
        
        assert plan.target_profile == "academic"
        assert plan.temperature_override == 0.25
        assert plan.top_p_override == 0.85
        assert any("Elevate vocabulary" in inst for inst in plan.selected_instructions)

    def test_professional_high_complexity(self) -> None:
        # Professional profile with high readability score (complexity) should trigger simplification.
        vector = create_vector(readability_score=0.7, passive_ratio=0.25)
        plan = generate_plan(vector, "professional")
        
        assert any("Simplify complex sentences" in inst for inst in plan.selected_instructions)
        assert any("Eliminate passive voice" in inst for inst in plan.selected_instructions)

    def test_expressive_low_variance(self) -> None:
        # Expressive profile with low sentence variance should trigger variance instruction.
        vector = create_vector(sentence_variance=2.0)
        plan = generate_plan(vector, "expressive")
        
        assert any("Introduce high variance" in inst for inst in plan.selected_instructions)

    def test_repetition_penalty(self) -> None:
        # High repetition should trigger redundant phrasing removal across any profile.
        vector = create_vector(repetition_score=0.5)
        plan = generate_plan(vector, "professional")
        assert any("repetitive" in inst.lower() for inst in plan.selected_instructions)

    def test_invalid_profile_raises(self) -> None:
        vector = create_vector()
        with pytest.raises(KeyError):
            generate_plan(vector, "non_existent_profile")
