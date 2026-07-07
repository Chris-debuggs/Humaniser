"""Phase 5: Prompt Builder & LLM Execution Layer.

Compiles the strict markdown prompt and orchestrates the async call to the Gemini API.
"""

from __future__ import annotations

import google.generativeai as genai

from src.schemas import ParagraphIR


async def execute_rewrite(node: ParagraphIR, api_key: str) -> str:
    """Execute the rewrite for a single node via the Gemini API."""
    if not node.transformation_plan:
        raise ValueError("Node is missing a TransformationPlan.")

    genai.configure(api_key=api_key)

    # Compile READ-ONLY CONTEXT
    context_blocks = node.context_before + node.context_after
    context_str = "\n\n".join(context_blocks).strip()
    if not context_str:
        context_str = "No surrounding context available."

    # Compile TRANSFORMATION PLAN
    instructions = "\n".join(
        f"- {inst}" for inst in node.transformation_plan.selected_instructions
    )

    # Construct the strict prompt layout
    prompt = (
        f"### READ-ONLY CONTEXT:\n"
        f"{context_str}\n"
        f"(Do NOT summarize or modify the context above)\n\n"
        f"### TRANSFORMATION PLAN:\n"
        f"{instructions}\n\n"
        f"### CONSTRAINTS:\n"
        f"Do not invent facts. Do not output markdown structural blocks. "
        f"Return ONLY the rewritten target paragraph. No preambles, no quotes.\n\n"
        f"### TARGET PARAGRAPH:\n"
        f"{node.original_text}"
    )

    # Initialize model and pass generation configs
    model = genai.GenerativeModel("gemini-1.5-flash")
    config = genai.GenerationConfig(
        temperature=node.transformation_plan.temperature_override,
        top_p=node.transformation_plan.top_p_override,
    )

    # Async generation
    response = await model.generate_content_async(
        contents=prompt,
        generation_config=config,
    )

    # Safety bounds for empty text
    if not response.text:
        return node.original_text

    return response.text.strip()
