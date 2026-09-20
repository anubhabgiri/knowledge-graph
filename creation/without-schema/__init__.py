"""Without-schema knowledge graph extraction pipeline.

Quick start
-----------
From Python::

    from creation.without-schema import run   # via importlib
    # or directly:
    import sys; sys.path.insert(0, 'creation/without-schema')
    from pipeline import run

    result = run(text=open("my_doc.txt").read(), output_path="graph.json")
    print(f"{len(result.triplets)} triplets extracted")

From the CLI::

    python creation/without-schema/main.py --input doc.txt --output graph.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import run  # noqa: F401  (re-export)

__all__ = ["run"]

