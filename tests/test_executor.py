"""Tests for the Phase 5 LLM Execution Layer.

Verifies prompt compilation and async integration with the Gemini SDK.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.executor import execute_rewrite
from src.schemas import ParagraphIR, TransformationPlan


def create_node_with_plan() -> ParagraphIR:
    return ParagraphIR(
        id=uuid.uuid4(),
        node_index=1,
        node_type="paragraph",
        editable=True,
        original_text="The quick brown fox.",
        context_before=["Context A"],
        context_after=["Context B"],
        transformation_plan=TransformationPlan(
            target_profile="expressive",
            selected_instructions=["Be creative", "Vary sentence length"],
            temperature_override=0.8,
            top_p_override=0.9,
        ),
    )


@pytest.mark.asyncio
async def test_execute_rewrite_prompt_compilation() -> None:
    node = create_node_with_plan()
    api_key = "dummy_key"

    # Mock the GenerativeModel and its generate_content_async method
    mock_model_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "The agile fox."
    mock_model_instance.generate_content_async = AsyncMock(return_value=mock_response)

    with patch("src.executor.genai.GenerativeModel", return_value=mock_model_instance) as mock_gen_model:
        result = await execute_rewrite(node, api_key)

        assert result == "The agile fox."
        
        # Verify Model instantiation
        mock_gen_model.assert_called_once_with("gemini-1.5-flash")
        
        # Verify the prompt string compilation
        call_args = mock_model_instance.generate_content_async.call_args[1]
        prompt = call_args["contents"]
        config = call_args["generation_config"]

        # 1. READ-ONLY CONTEXT checks
        assert "Context A\n\nContext B" in prompt
        assert "(Do NOT summarize or modify the context above)" in prompt

        # 2. TRANSFORMATION PLAN checks
        assert "- Be creative" in prompt
        assert "- Vary sentence length" in prompt

        # 3. CONSTRAINTS checks
        assert "Do not invent facts." in prompt
        assert "Return ONLY the rewritten target paragraph." in prompt

        # 4. TARGET PARAGRAPH checks
        assert "The quick brown fox." in prompt
        
        # Verify config injection
        assert config.temperature == 0.8
        assert config.top_p == 0.9


@pytest.mark.asyncio
async def test_execute_rewrite_missing_plan_raises() -> None:
    node = ParagraphIR(
        id=uuid.uuid4(),
        node_index=1,
        node_type="paragraph",
        editable=True,
        original_text="Text",
        transformation_plan=None,
    )
    with pytest.raises(ValueError, match="missing a TransformationPlan"):
        await execute_rewrite(node, "dummy_key")
