"""Pipeline entry-point.

Orchestrates the full parse → analyze → plan → execute → validate
pipeline.  Downstream phases will populate this module.

Implementation scheduled for Phase 4+.
"""

from __future__ import annotations


def main() -> None:
    """Run the full document transformation pipeline.

    Raises:
        NotImplementedError: Always — implementation is Phase 4+.
    """
    raise NotImplementedError("Pipeline orchestration is scheduled for Phase 4+")


if __name__ == "__main__":
    main()
