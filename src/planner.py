"""Phase 4: Transformation Planner Implementation.

Maps continuous FeatureVector metrics into discrete LLM instructions
based on target style profiles.
"""

from __future__ import annotations

from typing import List

from src.config import get_profile
from src.schemas import FeatureVector, TransformationPlan


def _generate_dynamic_instructions(vector: FeatureVector, profile_name: str) -> List[str]:
    """Compute multidimensional deltas and map to discrete instructions."""
    instructions: List[str] = []

    # 1. Passive Ratio checks
    if vector.passive_ratio > 0.15:
        if profile_name in ("professional", "expressive"):
            instructions.append("Eliminate passive voice constructs and use active voice.")
    elif vector.passive_ratio < 0.05:
        if profile_name == "academic":
            instructions.append("Introduce passive voice where it improves objectivity.")

    # 2. Sentence Variance checks
    if vector.sentence_variance < 5.0:
        if profile_name == "expressive":
            instructions.append("Introduce high variance in sentence length to improve rhythm.")
    elif vector.sentence_variance > 10.0:
        if profile_name in ("professional", "academic"):
            instructions.append("Standardize sentence lengths to improve readability.")

    # 3. Lexical Diversity checks
    if vector.lexical_diversity < 0.4:
        if profile_name == "academic":
            instructions.append("Elevate vocabulary and use precise, formal terminology.")
        elif profile_name == "expressive":
            instructions.append("Use more varied and evocative vocabulary.")
    elif vector.lexical_diversity > 0.7:
        if profile_name == "professional":
            instructions.append("Simplify vocabulary for maximum executive clarity.")

    # 4. Readability / Complexity checks
    if vector.readability_score > 0.6: # roughly grade 12+
        if profile_name == "professional":
            instructions.append("Simplify complex sentences; aim for an 8th-grade reading level.")
    elif vector.readability_score < 0.4:
        if profile_name == "academic":
            instructions.append("Increase syntactic complexity suitable for academic publication.")

    # 5. Repetition checks
    if vector.repetition_score > 0.2:
        instructions.append("Remove repetitive word sequences and redundant phrasing.")

    return instructions


def generate_plan(vector: FeatureVector, target_profile: str) -> TransformationPlan:
    """Generate a TransformationPlan bridging the input vector to the target profile."""
    # Fetch target parameters from config
    profile = get_profile(target_profile)

    # Combine profile's core instructions with dynamically generated ones
    core_instructions = list(profile["core_instructions"])
    dynamic_instructions = _generate_dynamic_instructions(vector, target_profile)
    
    selected_instructions = core_instructions + dynamic_instructions

    return TransformationPlan(
        target_profile=target_profile,
        selected_instructions=selected_instructions,
        temperature_override=profile["temperature"],
        top_p_override=profile["top_p"],
    )
