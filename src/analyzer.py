"""Phase 3: Deterministic Analyzer Implementation.

Extracts deterministic metrics using spaCy and textstat to populate
the FeatureVector for editable paragraphs.
"""

from __future__ import annotations

import math
from typing import List

import spacy
import textstat

from src.schemas import FeatureVector, ParagraphIR

# Load spaCy model globally (lazy loading can be added if needed)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Fallback for testing environments where model might not be downloaded
    # Note: Dependency parsing requires the full model to compute passive ratio.
    import spacy.lang.en
    nlp = spacy.lang.en.English()
    nlp.add_pipe("sentencizer")


def _calculate_variance(values: List[float], mean: float) -> float:
    if not values:
        return 0.0
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance


def analyze_paragraph(node: ParagraphIR) -> FeatureVector:
    """Analyze an editable paragraph and return its FeatureVector.

    If the node is not editable or contains no text, returns a zeroed vector.
    """
    if not node.editable or not node.original_text.strip():
        return FeatureVector(
            lexical_diversity=0.0,
            readability_score=0.0,
            avg_sentence_length=0.0,
            sentence_variance=0.0,
            passive_ratio=0.0,
            repetition_score=0.0,
        )

    text = node.original_text
    doc = nlp(text)

    # Filter for actual words (no punctuation, no spaces)
    words = [token.text.lower() for token in doc if token.is_alpha]
    total_words = len(words)

    # 1. Lexical Diversity (Type-Token Ratio)
    if total_words > 0:
        unique_words = len(set(words))
        lexical_diversity = unique_words / total_words
    else:
        lexical_diversity = 0.0

    lexical_diversity = max(0.0, min(lexical_diversity, 1.0))

    # 2. Readability Score (Flesch-Kincaid Grade Level normalized to 20)
    grade = textstat.flesch_kincaid_grade(text)
    readability_score = grade / 20.0
    readability_score = max(0.0, min(readability_score, 1.0))

    # 3. Sentence Length & Variance
    sentences = list(doc.sents)
    total_sentences = len(sentences)
    
    if total_sentences > 0:
        sentence_lengths = [
            len([t for t in sent if t.is_alpha]) for sent in sentences
        ]
        avg_sentence_length = sum(sentence_lengths) / total_sentences
        sentence_variance = _calculate_variance(sentence_lengths, avg_sentence_length)
    else:
        avg_sentence_length = 0.0
        sentence_variance = 0.0

    # 4. Passive Ratio
    # Count clauses by approximating with verbs.
    # Count passive constructs by looking for nsubjpass or auxpass.
    passive_constructs = 0
    total_clauses = 0
    
    for token in doc:
        if token.pos_ == "VERB" or token.dep_ == "ROOT":
            total_clauses += 1
        if token.dep_ in ("nsubjpass", "auxpass"):
            passive_constructs += 1

    # Normalize by total clauses (or 1 if none found but passives exist)
    if total_clauses > 0:
        passive_ratio = passive_constructs / total_clauses
    elif passive_constructs > 0:
        passive_ratio = 1.0
    else:
        passive_ratio = 0.0
        
    passive_ratio = max(0.0, min(passive_ratio, 1.0))

    # 5. Repetition Score (Bigram Redundancy)
    if total_words > 1:
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        unique_bigrams = len(set(bigrams))
        repetition_score = 1.0 - (unique_bigrams / len(bigrams))
    else:
        repetition_score = 0.0
        
    repetition_score = max(0.0, min(repetition_score, 1.0))

    return FeatureVector(
        lexical_diversity=lexical_diversity,
        readability_score=readability_score,
        avg_sentence_length=avg_sentence_length,
        sentence_variance=sentence_variance,
        passive_ratio=passive_ratio,
        repetition_score=repetition_score,
    )
