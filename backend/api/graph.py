# backend/api/graph.py
# ─────────────────────────────────────────────────────────
# GR Relationship Graph endpoints.
#
# Endpoints:
#   GET /api/graph/          → full graph (nodes + edges)
#   GET /api/graph/{gr_id}   → relationships for one specific GR
#
# Data comes from MongoDB gr_graph collection.
# Built automatically when admin runs embedding.
# Frontend uses this data to render the graph visualization.
# ─────────────────────────────────────────────────────────

import sys
import os

from fastapi import APIRouter, Depends

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auth.router import get_current_user
from core.gr_graph import get_graph, get_relationships_for_gr

router = APIRouter(prefix="/api/graph", tags=["Graph"])


@router.get("/")
async def fetch_graph(
    current_user: dict = Depends(get_current_user),
):
    """
    Returns the full GR relationship graph.
    Available to all logged-in users — not admin only.
    Both users and admins can view the graph.

    Response shape:
        {
            "success": true,
            "built": true,
            "built_at": "2025-07-05T...",
            "nodes": [
                {
                    "id": "GR_2024_transfer.pdf",   <- the source_file
                    "label": "GR_2024_transfer",    <- the stem
                    "filename": "GR_2024_transfer.pdf",
                    "gr_number": "मराग्रं-2025/प्र.क्र.58",
                    "doc_date": "2026-08-14",       <- or null
                    "subject": "...",
                    "resolved": true               <- false for "ext:" nodes
                },
                ...
            ],
            "edges": [
                {
                    "source": "GR_2024_transfer.pdf",
                    "target": "GR_2021_transfer.pdf",
                    "relation": "supersedes",
                    "ref_index": 2,
                    "match_reason": "strong_key",   <- or "date_and_number", or ""
                    "ambiguous": false,
                    "evidence": "...text evidence..."
                },
                ...
            ],
            "stats": {
                "total_nodes": 5,
                "total_edges": 3,
                "supersedes_count": 2,
                "amends_count": 1,
                "refers_to_count": 0,
                "unknown_targets": 0,
                "references_count": 0,
                "resolved_nodes": 4,
                "external_nodes": 1,
                "ambiguous_edges": 0,
                "unresolved_edges": 1
            }
        }

    Every pre-existing key is still here. Nodes and edges carry the added
    fields documented above, and the graph is read from the JSON snapshot
    when MongoDB holds no graph — see core.gr_graph.get_graph().
    """
    graph = await get_graph()

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    # .get() throughout rather than e["relation"]: a graph written by an
    # older build has no ref_index or resolved, and a stats endpoint that
    # raises KeyError on last week's data is worse than one that counts a
    # missing field as absent.
    def count_relation(name):
        return sum(1 for e in edges if e.get("relation") == name)

    stats = {
        # ── pre-existing keys, unchanged ──────────────────
        "total_nodes":      len(nodes),
        "total_edges":      len(edges),
        "supersedes_count": count_relation("supersedes"),
        "amends_count":     count_relation("amends"),
        # Kept for the frontends already reading it. The current builder
        # labels a plain citation "references"; "refers_to" was the old
        # value and still counts if an older graph is being served.
        "refers_to_count":  count_relation("refers_to"),
        "unknown_targets":  sum(1 for e in edges if e.get("target") == "Unknown"),

        # ── added ─────────────────────────────────────────
        "references_count": count_relation("references"),
        "resolved_nodes":   sum(1 for n in nodes if n.get("resolved")),
        "external_nodes":   sum(1 for n in nodes if not n.get("resolved")),
        "ambiguous_edges":  sum(1 for e in edges if e.get("ambiguous")),
        "unresolved_edges": sum(1 for e in edges if not e.get("match_reason")),
    }

    return {
        "success":  True,
        "built":    graph.get("built", False),
        "built_at": graph.get("built_at", ""),
        "nodes":    nodes,
        "edges":    edges,
        "stats":    stats,
    }


@router.get("/{gr_id}")
async def fetch_gr_relationships(
    gr_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Returns all relationships for a specific GR.
    Called when user clicks a node in the graph visualization.

    Args:
        gr_id : the GR filename stem e.g. "GR_2024_transfer"

    Response shape:
        {
            "success": true,
            "gr_id": "GR_2024_transfer",
            "relationships": [
                {
                    "source": "GR_2024_transfer",
                    "target": "GR_2021_transfer",
                    "relation": "supersedes",
                    "snippet": "...evidence text..."
                }
            ],
            "total": 1
        }
    """
    relationships = await get_relationships_for_gr(gr_id)

    return {
        "success":       True,
        "gr_id":         gr_id,
        "relationships": relationships,
        "total":         len(relationships),
    }