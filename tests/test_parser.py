"""Tests for the deterministic AST parser (Phase 2).

Covers Markdown and LaTeX parsing, round-trip reconstruction,
locked-block isolation, context harvesting, and structural stability.
"""

from __future__ import annotations

import pytest

from src.parser import parse_document, reconstruct_document


# ====================================================================
# Markdown — sample fixtures
# ====================================================================

SAMPLE_MD = """\
# Introduction

This is the first paragraph of the document. It contains several
sentences to test the parser's handling of multi-line paragraphs.

More text in a second paragraph.

```python
def hello():
    print("Hello, world!")
```

## Results

The results are shown below.

| Metric | Value |
|--------|-------|
| Score  | 95    |
| Grade  | A     |

$$
E = mc^2
$$

Final paragraph here.\
"""

SAMPLE_MD_INLINE_MATH = """\
# Math Test

A paragraph with inline math $x^2 + y^2 = z^2$ embedded.

$$ E = mc^2 $$

Another paragraph.\
"""

SAMPLE_MD_NESTED_FENCES = """\
# Code Examples

Some intro text.

````markdown
Here is a nested code block:

```python
print("inner")
```
````

After the code.\
"""


# ====================================================================
# Markdown — basic classification
# ====================================================================

class TestMarkdownBasicParse:
    def test_non_empty_result(self) -> None:
        doc = parse_document(SAMPLE_MD, "md")
        assert len(doc.nodes) > 0

    def test_metadata_source_format(self) -> None:
        doc = parse_document(SAMPLE_MD, "md")
        assert doc.metadata["source_format"] == "md"

    def test_node_indices_sequential(self) -> None:
        doc = parse_document(SAMPLE_MD, "md")
        for i, node in enumerate(doc.nodes):
            assert node.node_index == i


class TestMarkdownHeadings:
    def test_heading_count(self) -> None:
        doc = parse_document(SAMPLE_MD, "md")
        headings = [n for n in doc.nodes if n.node_type == "heading"]
        assert len(headings) == 2

    def test_headings_not_editable(self) -> None:
        doc = parse_document(SAMPLE_MD, "md")
        for node in doc.nodes:
            if node.node_type == "heading":
                assert not node.editable

    def test_heading_text_preserved(self) -> None:
        doc = parse_document(SAMPLE_MD, "md")
        headings = [n for n in doc.nodes if n.node_type == "heading"]
        assert headings[0].original_text == "# Introduction"
        assert headings[1].original_text == "## Results"


class TestMarkdownCodeBlocks:
    def test_code_block_isolation(self) -> None:
        doc = parse_document(SAMPLE_MD, "md")
        code_nodes = [n for n in doc.nodes if n.node_type == "code"]
        assert len(code_nodes) >= 1

    def test_code_block_not_editable(self) -> None:
        doc = parse_document(SAMPLE_MD, "md")
        for node in doc.nodes:
            if node.node_type == "code":
                assert not node.editable

    def test_code_content_preserved(self) -> None:
        doc = parse_document(SAMPLE_MD, "md")
        code_nodes = [n for n in doc.nodes if n.node_type == "code"]
        assert 'def hello():' in code_nodes[0].original_text
        assert 'print("Hello, world!")' in code_nodes[0].original_text

    def test_code_block_includes_fences(self) -> None:
        doc = parse_document(SAMPLE_MD, "md")
        code_nodes = [n for n in doc.nodes if n.node_type == "code"]
        assert code_nodes[0].original_text.startswith("```python")
        assert code_nodes[0].original_text.endswith("```")

    def test_nested_fences(self) -> None:
        """Four-backtick fences containing three-backtick fences."""
        doc = parse_document(SAMPLE_MD_NESTED_FENCES, "md")
        code_nodes = [n for n in doc.nodes if n.node_type == "code"]
        assert len(code_nodes) == 1
        # The inner ``` must NOT close the outer ```` fence.
        assert '```python' in code_nodes[0].original_text
        assert 'print("inner")' in code_nodes[0].original_text


class TestMarkdownMathBlocks:
    def test_math_block_isolation(self) -> None:
        doc = parse_document(SAMPLE_MD, "md")
        math_nodes = [n for n in doc.nodes if n.node_type == "math"]
        assert len(math_nodes) >= 1

    def test_math_block_not_editable(self) -> None:
        doc = parse_document(SAMPLE_MD, "md")
        for node in doc.nodes:
            if node.node_type == "math":
                assert not node.editable

    def test_math_content_preserved(self) -> None:
        doc = parse_document(SAMPLE_MD, "md")
        math_nodes = [n for n in doc.nodes if n.node_type == "math"]
        assert "E = mc^2" in math_nodes[0].original_text

    def test_single_line_math(self) -> None:
        doc = parse_document(SAMPLE_MD_INLINE_MATH, "md")
        math_nodes = [n for n in doc.nodes if n.node_type == "math"]
        assert len(math_nodes) >= 1
        assert "E = mc^2" in math_nodes[0].original_text


class TestMarkdownTables:
    def test_table_isolation(self) -> None:
        doc = parse_document(SAMPLE_MD, "md")
        table_nodes = [n for n in doc.nodes if n.node_type == "table"]
        assert len(table_nodes) == 1

    def test_table_not_editable(self) -> None:
        doc = parse_document(SAMPLE_MD, "md")
        for node in doc.nodes:
            if node.node_type == "table":
                assert not node.editable

    def test_table_content_preserved(self) -> None:
        doc = parse_document(SAMPLE_MD, "md")
        table_nodes = [n for n in doc.nodes if n.node_type == "table"]
        assert "Metric" in table_nodes[0].original_text
        assert "Score" in table_nodes[0].original_text
        assert "Grade" in table_nodes[0].original_text


class TestMarkdownParagraphs:
    def test_editable_paragraphs_exist(self) -> None:
        doc = parse_document(SAMPLE_MD, "md")
        editable = [n for n in doc.nodes if n.editable]
        assert len(editable) >= 3  # first, second, results-intro, final


# ====================================================================
# Context Harvesting
# ====================================================================

class TestContextHarvesting:
    def test_first_editable_has_limited_before(self) -> None:
        doc = parse_document(SAMPLE_MD, "md")
        editable = [n for n in doc.nodes if n.editable]
        first = editable[0]
        # First editable is at index 1 (after heading at 0), so at most
        # 1 preceding node.
        assert len(first.context_before) <= 2

    def test_context_before_contents(self) -> None:
        doc = parse_document(SAMPLE_MD, "md")
        editable = [n for n in doc.nodes if n.editable]
        first = editable[0]
        # The node before the first paragraph should be the heading.
        if first.context_before:
            assert "# Introduction" in first.context_before[0]

    def test_interior_node_has_context(self) -> None:
        doc = parse_document(SAMPLE_MD, "md")
        editable = [n for n in doc.nodes if n.editable]
        # Pick a node that's not first or last.
        if len(editable) >= 3:
            middle = editable[1]
            assert len(middle.context_before) >= 1
            assert len(middle.context_after) >= 1

    def test_context_max_two(self) -> None:
        doc = parse_document(SAMPLE_MD, "md")
        for node in doc.nodes:
            if node.editable:
                assert len(node.context_before) <= 2
                assert len(node.context_after) <= 2

    def test_non_editable_has_no_context(self) -> None:
        doc = parse_document(SAMPLE_MD, "md")
        for node in doc.nodes:
            if not node.editable:
                assert node.context_before == []
                assert node.context_after == []


# ====================================================================
# Markdown — round-trip reconstruction
# ====================================================================

class TestMarkdownRoundTrip:
    def test_locked_blocks_byte_exact(self) -> None:
        """Every non-editable block must appear verbatim in the
        reconstructed output."""
        doc = parse_document(SAMPLE_MD, "md")
        reconstructed = reconstruct_document(doc)
        for node in doc.nodes:
            if not node.editable:
                assert node.original_text in reconstructed, (
                    f"Locked block lost: {node.original_text!r}"
                )

    def test_structural_stability(self) -> None:
        """parse → reconstruct → re-parse must yield identical node
        types, editability flags, and original_text for every node."""
        doc = parse_document(SAMPLE_MD, "md")
        reconstructed = reconstruct_document(doc)
        doc2 = parse_document(reconstructed, "md")

        assert len(doc.nodes) == len(doc2.nodes), (
            f"Node count changed: {len(doc.nodes)} → {len(doc2.nodes)}"
        )
        for n1, n2 in zip(doc.nodes, doc2.nodes):
            assert n1.node_type == n2.node_type
            assert n1.editable == n2.editable
            assert n1.original_text == n2.original_text

    def test_rewritten_text_used_for_editable(self) -> None:
        """When ``rewritten_text`` is set on an editable node, the
        reconstructor must emit it instead of ``original_text``."""
        doc = parse_document(SAMPLE_MD, "md")
        # Pick the first editable node and set a rewrite.
        editable = [n for n in doc.nodes if n.editable]
        assert len(editable) > 0
        editable[0].rewritten_text = "REPLACED_PARAGRAPH"
        reconstructed = reconstruct_document(doc)
        assert "REPLACED_PARAGRAPH" in reconstructed
        # Original should NOT appear (it was fully replaced).
        assert editable[0].original_text not in reconstructed

    def test_locked_block_ignores_rewritten_text(self) -> None:
        """Even if ``rewritten_text`` is set on a non-editable node, the
        reconstructor must use ``original_text``."""
        doc = parse_document(SAMPLE_MD, "md")
        locked = [n for n in doc.nodes if not n.editable]
        assert len(locked) > 0
        original = locked[0].original_text
        locked[0].rewritten_text = "SHOULD_BE_IGNORED"
        reconstructed = reconstruct_document(doc)
        assert original in reconstructed


# ====================================================================
# LaTeX — sample fixture
# ====================================================================

SAMPLE_LATEX = r"""\
\documentclass{article}
\usepackage{amsmath}

\begin{document}

\section{Introduction}

This is the introductory paragraph of the LaTeX document.

A second paragraph follows here.

\begin{equation}
E = mc^2
\end{equation}

\subsection{Details}

Some detailed text.

\begin{lstlisting}
print("hello from LaTeX")
\end{lstlisting}

\begin{tabular}{|c|c|}
\hline
A & B \\
\hline
1 & 2 \\
\hline
\end{tabular}

Final remarks.

\end{document}
"""


# ====================================================================
# LaTeX — classification tests
# ====================================================================

class TestLatexParse:
    def test_non_empty(self) -> None:
        doc = parse_document(SAMPLE_LATEX, "latex")
        assert len(doc.nodes) > 0

    def test_preamble_captured(self) -> None:
        """The preamble (\\documentclass, \\usepackage) must appear as a
        non-editable node."""
        doc = parse_document(SAMPLE_LATEX, "latex")
        preamble_nodes = [
            n for n in doc.nodes
            if "documentclass" in n.original_text
        ]
        assert len(preamble_nodes) >= 1
        assert not preamble_nodes[0].editable

    def test_heading_detected(self) -> None:
        doc = parse_document(SAMPLE_LATEX, "latex")
        headings = [n for n in doc.nodes if n.node_type == "heading"]
        assert len(headings) >= 2  # \section + \subsection

    def test_math_detected(self) -> None:
        doc = parse_document(SAMPLE_LATEX, "latex")
        math_nodes = [n for n in doc.nodes if n.node_type == "math"]
        assert len(math_nodes) >= 1
        assert "E = mc^2" in math_nodes[0].original_text

    def test_code_detected(self) -> None:
        doc = parse_document(SAMPLE_LATEX, "latex")
        code_nodes = [n for n in doc.nodes if n.node_type == "code"]
        # At least preamble + lstlisting
        lstlisting = [
            n for n in code_nodes
            if "print" in n.original_text
        ]
        assert len(lstlisting) >= 1

    def test_table_detected(self) -> None:
        doc = parse_document(SAMPLE_LATEX, "latex")
        table_nodes = [n for n in doc.nodes if n.node_type == "table"]
        assert len(table_nodes) >= 1

    def test_editable_paragraphs(self) -> None:
        doc = parse_document(SAMPLE_LATEX, "latex")
        editable = [n for n in doc.nodes if n.editable]
        assert len(editable) >= 2


class TestLatexRoundTrip:
    def test_locked_blocks_byte_exact(self) -> None:
        doc = parse_document(SAMPLE_LATEX, "latex")
        reconstructed = reconstruct_document(doc)
        for node in doc.nodes:
            if not node.editable:
                assert node.original_text in reconstructed

    def test_structural_stability(self) -> None:
        doc = parse_document(SAMPLE_LATEX, "latex")
        reconstructed = reconstruct_document(doc)
        doc2 = parse_document(reconstructed, "latex")

        assert len(doc.nodes) == len(doc2.nodes), (
            f"Node count changed: {len(doc.nodes)} → {len(doc2.nodes)}"
        )
        for n1, n2 in zip(doc.nodes, doc2.nodes):
            assert n1.node_type == n2.node_type
            assert n1.editable == n2.editable
            assert n1.original_text == n2.original_text


# ====================================================================
# Edge-cases
# ====================================================================

class TestEdgeCases:
    def test_empty_document(self) -> None:
        doc = parse_document("", "md")
        assert doc.nodes == []

    def test_whitespace_only(self) -> None:
        doc = parse_document("   \n\n   \n", "md")
        assert doc.nodes == []

    def test_single_heading(self) -> None:
        doc = parse_document("# Solo Heading", "md")
        assert len(doc.nodes) == 1
        assert doc.nodes[0].node_type == "heading"

    def test_single_paragraph(self) -> None:
        doc = parse_document("Just a plain paragraph.", "md")
        assert len(doc.nodes) == 1
        assert doc.nodes[0].node_type == "paragraph"
        assert doc.nodes[0].editable

    def test_unsupported_file_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported file_type"):
            parse_document("text", "rst")  # type: ignore[arg-type]
