"""Immutable style-profile configuration.

All generation and validation parameters are fetched from this module.
No hardcoded values inside LLM calls.
"""

from __future__ import annotations

from typing import Any, Dict

STYLE_PROFILES: Dict[str, Dict[str, Any]] = {
    "academic": {
        "temperature": 0.25,
        "top_p": 0.85,
        "validator_threshold": 0.90,
        "core_instructions": [
            "De-cluster complex subordinate clauses",
            "Maintain formal nominalizations",
        ],
    },
    "professional": {
        "temperature": 0.40,
        "top_p": 0.90,
        "validator_threshold": 0.88,
        "core_instructions": [
            "Shift passive to active voice where appropriate",
            "Maximize executive clarity",
        ],
    },
    "expressive": {
        "temperature": 0.75,
        "top_p": 0.95,
        "validator_threshold": 0.80,
        "core_instructions": [
            "Increase sentence length variation",
            "Enhance rhythmic transitions",
        ],
    },
}


def get_profile(name: str) -> Dict[str, Any]:
    """Return the profile dict for *name*, or raise ``KeyError``."""
    if name not in STYLE_PROFILES:
        raise KeyError(
            f"Unknown style profile {name!r}. "
            f"Available: {', '.join(sorted(STYLE_PROFILES))}"
        )
    return STYLE_PROFILES[name]
