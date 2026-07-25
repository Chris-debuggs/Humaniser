"""Pipeline entry-point and Orchestrator.

Coordinates the full parse → analyze → plan → execute → validate pipeline.
"""
from __future__ import annotations

from typing import Literal

from src.analyzer import analyze_paragraph
from src.executor import execute_rewrite
from src.parser import parse_document, reconstruct_document
from src.planner import generate_plan
from src.schemas import DocumentIR
from src.validator import validate_rewrite


async def transform_document(
    source_text: str,
    source_format: Literal["markdown", "latex"],
    target_profile: str,
    api_key: str,
    max_retries: int = 3,
) -> tuple[str, DocumentIR]:
    """Execute the full optimization compiler pipeline on a document.
    
    Args:
        source_text: The raw input string.
        source_format: 'markdown' or 'latex'.
        target_profile: Target style profile (e.g. 'academic', 'professional').
        api_key: The Gemini API key.
        max_retries: Maximum attempts to pass validation before fallback.
        
    Returns:
        A tuple of (reconstructed_document_string, final_document_ir).
    """
    # 1. Parse
    document_ir = parse_document(source_text, source_format)
    
    # Process editable paragraphs
    for node in document_ir.nodes:
        if not node.editable:
            continue
            
        # 2. Analyze
        node.feature_vector = analyze_paragraph(node)
        
        # 3. Plan
        node.transformation_plan = generate_plan(node.feature_vector, target_profile)
        
        # 4 & 5. Execute and Validate (with Retry loop)
        attempt = 0
        success = False
        
        while attempt < max_retries:
            attempt += 1
            
            try:
                candidate_text = await execute_rewrite(node, api_key)
                validation = validate_rewrite(node, candidate_text)
                
                if validation.overall_valid:
                    node.rewritten_text = candidate_text
                    node.validation_report = validation
                    success = True
                    break
                else:
                    # Retain the last validation failure report for observability
                    node.validation_report = validation
                    
            except Exception as e:
                # If network fails or rate-limits, we don't crash the whole compiler
                print(f"Warning: Node {node.node_index} execution failed on attempt {attempt} - {e}")
                
        # Silent fallback to original text if validation fails consistently
        if not success:
            node.rewritten_text = node.original_text
            
    # 6. Reconstruct
    final_text = reconstruct_document(document_ir)
    return final_text, document_ir
