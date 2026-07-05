"""Paragraph feature extraction module.

Exposes a single public function to analyze a paragraph node and produce
a FeatureVector capturing lexical, syntactic, and structural metrics.

Implementation scheduled for Phase 3.
"""

from __future__ import annotations

from .schemas import FeatureVector, ParagraphIR


def analyze_paragraph(node: ParagraphIR) -> FeatureVector:
    """Analyze a paragraph and return its feature vector.

    Raises:
        NotImplementedError: Always — implementation is Phase 3.
    """
    raise NotImplementedError("analyze_paragraph is scheduled for Phase 3")
