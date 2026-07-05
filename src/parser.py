"""Deterministic AST parser for Markdown and LaTeX documents.

Produces a ``DocumentIR`` from raw text by classifying structural blocks
and harvesting context windows for downstream transformation.

Public API
----------
- ``parse_document``        — raw text → DocumentIR
- ``reconstruct_document``  — DocumentIR → raw text (round-trip)
"""

from __future__ import annotations

import re
from typing import List, Literal, Tuple
from uuid import uuid4

from .schemas import DocumentIR, ParagraphIR

# Type alias used internally for classified raw blocks.
_Block = Tuple[str, str, bool]  # (node_type, raw_text, editable)


# ========================================================================
# Public API
# ========================================================================

def parse_document(
    raw_text: str,
    file_type: Literal["md", "latex"],
) -> DocumentIR:
    """Parse a raw document string into a :class:`DocumentIR`.

    Parameters
    ----------
    raw_text:
        Full document text (Markdown or LaTeX source).
    file_type:
        ``"md"`` for Markdown, ``"latex"`` for LaTeX.

    Returns
    -------
    DocumentIR
        Ordered list of classified :class:`ParagraphIR` nodes with
        context windows populated for every editable node.
    """
    if file_type == "md":
        blocks = _parse_markdown_blocks(raw_text)
    elif file_type == "latex":
        blocks = _parse_latex_blocks(raw_text)
    else:
        raise ValueError(f"Unsupported file_type: {file_type!r}")

    nodes: List[ParagraphIR] = []
    for idx, (node_type, text, editable) in enumerate(blocks):
        nodes.append(
            ParagraphIR(
                id=uuid4(),
                node_index=idx,
                node_type=node_type,  # type: ignore[arg-type]
                editable=editable,
                original_text=text,
            )
        )

    _harvest_context(nodes)

    return DocumentIR(
        document_id=uuid4(),
        metadata={"source_format": file_type},
        nodes=nodes,
    )


def reconstruct_document(doc: DocumentIR) -> str:
    """Reassemble a document from its IR.

    * **Non-editable** nodes always emit ``original_text`` (byte-exact
      preservation of locked blocks).
    * **Editable** nodes emit ``rewritten_text`` when available,
      falling back to ``original_text``.

    Nodes are joined with ``\\n\\n`` (the universal Markdown / LaTeX
    paragraph separator).  This guarantees structural stability across
    a parse → reconstruct → re-parse cycle.
    """
    parts: List[str] = []
    for node in doc.nodes:
        if node.editable and node.rewritten_text is not None:
            parts.append(node.rewritten_text)
        else:
            parts.append(node.original_text)
    return "\n\n".join(parts)


# ========================================================================
# Context Harvesting
# ========================================================================

def _harvest_context(nodes: List[ParagraphIR]) -> None:
    """Populate ``context_before`` / ``context_after`` for editable nodes.

    Each editable node receives the ``original_text`` of up to two
    preceding and two following nodes (of any type) to give the
    downstream LLM local structural awareness.
    """
    n = len(nodes)
    for i, node in enumerate(nodes):
        if not node.editable:
            continue
        node.context_before = [
            nodes[j].original_text for j in range(max(0, i - 2), i)
        ]
        node.context_after = [
            nodes[j].original_text for j in range(i + 1, min(n, i + 3))
        ]


# ========================================================================
# Markdown Block Scanner
# ========================================================================
#
# A line-based state machine that splits Markdown into typed blocks.
# Block-level Markdown syntax has clear, unambiguous delimiters (``#``,
# fences, ``$$``, pipe-tables) which makes a line scanner both correct
# and trivially invertible for round-trip reconstruction.
#
# We intentionally do NOT rely on mistune's internal token offsets for
# raw-text extraction because they are not guaranteed stable across
# versions.  Instead, mistune is available as an optional secondary
# classifier if needed in future phases.

_FENCE_OPEN_RE = re.compile(r"^(`{3,}|~{3,})(.*)")
_HEADING_RE = re.compile(r"^(#{1,6})\s")


def _is_table_separator(line: str) -> bool:
    """Return ``True`` if *line* is a GFM table separator row.

    A valid separator row consists of pipe-delimited cells where each
    cell contains only dashes and optional alignment colons.
    """
    stripped = line.strip()
    if "|" not in stripped or "-" not in stripped:
        return False
    cells = [c.strip() for c in stripped.split("|") if c.strip()]
    return len(cells) >= 1 and all(
        re.match(r"^:?-+:?$", cell) for cell in cells
    )


def _parse_markdown_blocks(raw_text: str) -> List[_Block]:
    """Parse Markdown source into a flat list of classified blocks."""
    lines = raw_text.split("\n")
    n = len(lines)
    blocks: List[_Block] = []
    i = 0
    para_buf: List[str] = []

    def _flush_paragraph() -> None:
        nonlocal para_buf
        text = "\n".join(para_buf).strip()
        if text:
            blocks.append(("paragraph", text, True))
        para_buf = []

    while i < n:
        line = lines[i]

        # ---- blank line → flush current paragraph ----
        if line.strip() == "":
            _flush_paragraph()
            i += 1
            continue

        # ---- fenced code block ----
        fence_match = _FENCE_OPEN_RE.match(line)
        if fence_match:
            _flush_paragraph()
            fence_marker = fence_match.group(1)
            fence_char = fence_marker[0]
            fence_len = len(fence_marker)
            code_lines = [line]
            i += 1
            closed = False
            # The closing fence must use the same character and be at
            # least as long as the opening fence.
            close_re = re.compile(
                rf"^{re.escape(fence_char)}{{{fence_len},}}\s*$"
            )
            while i < n:
                if close_re.match(lines[i]):
                    code_lines.append(lines[i])
                    i += 1
                    closed = True
                    break
                code_lines.append(lines[i])
                i += 1
            # An unclosed fence is still captured as a code block.
            blocks.append(("code", "\n".join(code_lines), False))
            continue

        # ---- display-math block ($$…$$) ----
        if line.strip().startswith("$$"):
            _flush_paragraph()
            stripped = line.strip()
            # Single-line: $$ content $$  (at least one char between)
            if (
                stripped.endswith("$$")
                and len(stripped) > 4
                and stripped != "$$"
            ):
                blocks.append(("math", stripped, False))
                i += 1
                continue
            # Multi-line: opening $$ on its own line
            math_lines = [line]
            i += 1
            while i < n:
                math_lines.append(lines[i])
                if lines[i].strip().endswith("$$"):
                    i += 1
                    break
                i += 1
            blocks.append(("math", "\n".join(math_lines), False))
            continue

        # ---- ATX heading ----
        if _HEADING_RE.match(line):
            _flush_paragraph()
            blocks.append(("heading", line, False))
            i += 1
            continue

        # ---- GFM pipe-table ----
        if "|" in line and (i + 1 < n) and _is_table_separator(lines[i + 1]):
            _flush_paragraph()
            table_lines = [line]
            i += 1
            while i < n and lines[i].strip() != "" and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1
            blocks.append(("table", "\n".join(table_lines), False))
            continue

        # ---- regular paragraph text ----
        para_buf.append(line)
        i += 1

    # Flush any remaining text.
    _flush_paragraph()
    return blocks


# ========================================================================
# LaTeX Block Scanner  (TexSoup-based)
# ========================================================================
#
# Uses TexSoup to structurally parse the LaTeX source and walk the
# resulting tree.  This avoids the fragility of hand-written regex for
# nested environments, escaped braces, and macro expansion.

_LATEX_CODE_ENVS = frozenset(
    {"verbatim", "Verbatim", "lstlisting", "minted"}
)
_LATEX_MATH_ENVS = frozenset(
    {
        "equation",
        "equation*",
        "align",
        "align*",
        "gather",
        "gather*",
        "multline",
        "multline*",
        "math",
        "displaymath",
        "flalign",
        "flalign*",
    }
)
_LATEX_TABLE_ENVS = frozenset(
    {"tabular", "tabular*", "longtable", "table", "table*"}
)
_LATEX_HEADING_CMDS = frozenset(
    {
        "part",
        "chapter",
        "section",
        "subsection",
        "subsubsection",
        "paragraph",
        "subparagraph",
    }
)


def _parse_latex_blocks(raw_text: str) -> List[_Block]:
    """Parse LaTeX source into a flat list of classified blocks."""
    try:
        from TexSoup import TexSoup as TexSoupParser  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "TexSoup is required for LaTeX parsing.  "
            "Install it with:  pip install TexSoup"
        ) from exc

    soup = TexSoupParser(raw_text)
    blocks: List[_Block] = []

    # --- preamble (everything before \begin{document}) ---
    doc_env = soup.find("document")
    if doc_env is not None:
        preamble_marker = "\\begin{document}"
        preamble_idx = raw_text.find(preamble_marker)
        if preamble_idx > 0:
            preamble = raw_text[:preamble_idx].strip()
            if preamble:
                # Preamble is structural — never rewritten.
                blocks.append(("code", preamble, False))
        _walk_latex_children(doc_env, blocks)
        # Closing \end{document} is structural.
        blocks.append(("code", "\\end{document}", False))
    else:
        # No document environment; treat the whole file as body.
        _walk_latex_children(soup, blocks)

    return blocks


def _walk_latex_children(
    node: object,
    blocks: List[_Block],
) -> None:
    """Recursively classify TexSoup children into IR blocks."""
    children = getattr(node, "children", None)
    if children is None:
        return

    for child in children:
        name: str | None = getattr(child, "name", None)

        # --- plain text (NavigableString / RArg) ---
        if name is None:
            text = str(child)
            # Split into paragraphs on blank lines.
            paragraphs = re.split(r"\n\s*\n", text)
            for para in paragraphs:
                para = para.strip()
                if para:
                    blocks.append(("paragraph", para, True))
            continue

        # --- named environments / commands ---
        child_src = str(child).strip()
        if not child_src:
            continue

        if name in _LATEX_CODE_ENVS:
            blocks.append(("code", child_src, False))
        elif name in _LATEX_MATH_ENVS:
            blocks.append(("math", child_src, False))
        elif name in _LATEX_TABLE_ENVS:
            blocks.append(("table", child_src, False))
        elif name in _LATEX_HEADING_CMDS:
            blocks.append(("heading", child_src, False))
        elif name == "document":
            # Edge-case: nested document env → recurse.
            _walk_latex_children(child, blocks)
        else:
            # Unknown environments (itemize, figure, …) are structural
            # and should not be rewritten.
            blocks.append(("paragraph", child_src, False))
