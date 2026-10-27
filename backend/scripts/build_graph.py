#!/usr/bin/env python
# backend/scripts/build_graph.py
# ─────────────────────────────────────────────────────────
# Builds the GR relationship graph from the existing vector index and
# writes the JSON snapshot. No MongoDB, no FastAPI, no LLM, no PDFs.
#
# Why this exists separately from build_graph():
#   The graph is otherwise only rebuilt as the tail end of an embedding
#   job, which re-embeds the whole corpus — minutes of Ollama time — to
#   change edges that are derived purely from chunk text that is already
#   on disk. This script rebuilds the graph alone, in seconds, as often
#   as the extraction rules are being tuned.
#
#   It deliberately does NOT write to MongoDB. The only Mongo writer is
#   build_graph() on the embedding path, so running this can never leave
#   the database holding a graph nobody asked for. What it writes is the
#   snapshot at DATA_DIR/gr_graph.json, which is what the retrieval path
#   reads anyway.
#
#   Nothing here re-embeds or rebuilds the index. The lexical index is
#   read from the FAISS store exactly as a search would read it.
#
# Usage:
#   python scripts/build_graph.py
#
# Prints node counts (resolved, external, and how many unresolved items
# were dropped as noise), the edge count, then one line per edge.
# ─────────────────────────────────────────────────────────

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.gr_graph import (            # noqa: E402
    build_graph_from_index,
    write_graph_snapshot,
    graph_snapshot_path,
)
from core.gr_meta import (             # noqa: E402
    extract_metadata,
    write_meta_snapshot,
)
from core.hybrid_search import get_lexical_index   # noqa: E402


def main(argv: list) -> int:
    if len(argv) > 1:
        print("usage: python scripts/build_graph.py")
        return 2

    print("reading the lexical index (no re-embedding, no PDF access)...")
    stats = {}
    nodes, edges = build_graph_from_index(stats=stats)

    if not nodes:
        print("no embedded chunks found — nothing to build.")
        print("embed the documents first, then run this again.")
        return 1

    resolved = [n for n in nodes if n.get("resolved")]
    external = [n for n in nodes if not n.get("resolved")]
    dropped  = stats.get("external_dropped", 0)

    path = write_graph_snapshot(nodes, edges)

    print()
    print(f"nodes:             {len(nodes)}")
    print(f"  resolved:        {len(resolved)}   (documents in the corpus)")
    print(f"  external:        {len(external)}   (cited but not uploaded)")
    print(f"external_dropped:  {dropped}   (unresolved items that did not "
          f"look like citations)")
    print(f"edges:             {len(edges)}")
    print()

    for edge in edges:
        print(
            f"  {edge.get('source', '?')} -> {edge.get('target', '?')} "
            f"[{edge.get('relation', '?')}, "
            f"{edge.get('ref_index', '?')}, "
            f"{edge.get('match_reason') or 'unresolved'}, "
            f"{bool(edge.get('ambiguous'))}]"
        )

    # ── Metadata ──────────────────────────────────────────
    # Derived from the same chunks, so it is rebuilt here too rather than
    # left to drift behind the graph.
    index = get_lexical_index()
    meta_path = None
    meta = {}
    if index is not None and index.documents:
        meta = extract_metadata(index.documents)
        meta_path = write_meta_snapshot(index.documents)

    if meta:
        print()
        print("metadata:")
        for fname in sorted(meta):
            entry = meta[fname]
            print(
                f"  {fname} | {entry.get('gr_number') or '-'} | "
                f"{entry.get('department') or '-'} | "
                f"{entry.get('doc_date') or '-'}"
            )

    print()
    print(f"snapshot written: {path}")
    if meta_path:
        print(f"metadata written: {meta_path}")
    print("mongodb was NOT written — that only happens on the embedding path.")
    return 0


if __name__ == "__main__":
    # The documents are Marathi; a cp1252 console raises on the first node
    # label printed, which would lose the whole report.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    raise SystemExit(main(sys.argv))
