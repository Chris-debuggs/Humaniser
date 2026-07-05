"""Transformation planning module.

Translates a FeatureVector and target profile into a concrete
TransformationPlan with LLM generation parameters and rewrite
instructions.

Implementation scheduled for Phase 3.
"""

from __future__ import annotations

from .schemas import FeatureVector, TransformationPlan


def generate_plan(vector: FeatureVector, target_profile: str) -> TransformationPlan:
    """Generate a transformation plan for the given feature vector and profile.

    Raises:
        NotImplementedError: Always — implementation is Phase 3.
    """
    raise NotImplementedError("generate_plan is scheduled for Phase 3")
