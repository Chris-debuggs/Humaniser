"""Phase 6: Validation Engine.

Ensures LLM outputs maintain structural integrity, preserve critical entities,
and meet semantic similarity thresholds before acceptance.
"""

from __future__ import annotations

import re
from typing import Set

import spacy
from sentence_transformers import SentenceTransformer, util

from src.config import STYLE_PROFILES
from src.schemas import EntityPreservationReport, ParagraphIR, ValidationReport

# Global model cache to avoid reloading on every validation
nlp = None
sim_model = None


def _load_models() -> None:
    """Initialize spaCy and SentenceTransformer lazily."""
    global nlp, sim_model
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback for testing environments without the full model
            from spacy.lang.en import English
            nlp = English()
            nlp.add_pipe("ner") # Will be unpopulated, but allows test execution

    if sim_model is None:
        sim_model = SentenceTransformer("all-MiniLM-L6-v2")


def _check_structural_integrity(original: ParagraphIR, rewritten_text: str) -> bool:
    """Verify the rewrite does not contain structural markdown or regurgitate context."""
    # 1. Check for hallucinatory markdown injection
    if re.search(r"```", rewritten_text) or re.search(r"^#+\s", rewritten_text, flags=re.MULTILINE):
        return False

    # 2. Check for context wholesale regurgitation
    context_sentences: Set[str] = set()
    for block in original.context_before:
        # Simple split by punctuation to avoid full spaCy parse on context again
        for sentence in re.split(r"[.!?]+", block):
            cleaned = sentence.strip()
            if len(cleaned) > 20:  # Only track substantial phrases
                context_sentences.add(cleaned)
                
    for sentence in context_sentences:
        if sentence in rewritten_text:
            return False

    return True


def _extract_critical_entities(text: str) -> Set[str]:
    """Extract NUM, DATE, and ORG entities using spaCy."""
    doc = nlp(text)
    entities = set()
    if hasattr(doc, "ents"):
        for ent in doc.ents:
            if ent.label_ in ("NUM", "DATE", "ORG"):
                entities.add(ent.text.lower())
    return entities


def validate_rewrite(original: ParagraphIR, rewritten_text: str) -> ValidationReport:
    """Execute the composite validation pipeline on a candidate rewrite."""
    if not original.transformation_plan:
        raise ValueError("Cannot validate rewrite without a TransformationPlan.")

    _load_models()

    # 1. Structural Integrity Check
    struct_pass = _check_structural_integrity(original, rewritten_text)

    # 2. Semantic Similarity Calculation
    emb_orig = sim_model.encode(original.original_text, convert_to_tensor=True)
    emb_rewritten = sim_model.encode(rewritten_text, convert_to_tensor=True)
    similarity = float(util.cos_sim(emb_orig, emb_rewritten).item())

    # 3. Entity Preservation Analysis
    orig_ents = _extract_critical_entities(original.original_text)
    rewritten_ents = _extract_critical_entities(rewritten_text)
    
    missing = list(orig_ents - rewritten_ents)
    ent_report = EntityPreservationReport(
        original_count=len(orig_ents),
        rewritten_count=len(rewritten_ents),
        missing_critical_entities=missing,
        is_valid=len(missing) == 0,
    )

    # 4. Aggregation and Profile Threshold Verification
    target_profile = original.transformation_plan.target_profile
    profile_data = STYLE_PROFILES.get(target_profile, STYLE_PROFILES["academic"])
    threshold = profile_data["validator_threshold"]

    overall_valid = (
        struct_pass 
        and ent_report.is_valid
        and (similarity >= threshold)
    )

    return ValidationReport(
        structural_integrity_passed=struct_pass,
        semantic_similarity_score=similarity,
        entity_report=ent_report,
        asymmetric_contradiction_detected=False,  # Stubbed for future NLI integration
        overall_valid=overall_valid,
    )
