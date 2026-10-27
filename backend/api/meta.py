# backend/api/meta.py
# ─────────────────────────────────────────────────────────
# Structured GR metadata endpoint.
#
# Endpoints:
#   GET /api/meta/   → one row per embedded document
#
# Data comes from the JSON snapshot at DATA_DIR/gr_meta.json, written by
# core.gr_graph.build_graph() after embedding and by
# scripts/build_graph.py offline. No MongoDB and no model call: every
# field was read off the documents' letterheads when the snapshot was
# built.
#
# The registry page uses this as its table, and the graph page uses the
# department field for node colouring.
# ─────────────────────────────────────────────────────────

import sys
import os

from fastapi import APIRouter, Depends

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auth.router import get_current_user
from core.gr_meta import load_meta

router = APIRouter(prefix="/api/meta", tags=["Metadata"])

# Dropped from the response: it is up to 400 characters of raw letterhead
# per document, useful when debugging extraction and pure weight over the
# wire for every consumer that exists.
_OMITTED_FIELDS = ("header_chunk_text",)


@router.get("/")
async def fetch_metadata(
    current_user: dict = Depends(get_current_user),
):
    """
    Returns the structured metadata for every embedded document.

    Available to all logged-in users — same dependency as /api/graph,
    which this is read alongside.

    Response shape:
        {
            "files": [
                {
                    "source_file": "GR1.pdf",
                    "stem": "GR1",
                    "gr_number": "मराग्रं-2025/प्र.क्र.58",
                    "subject": "...",
                    "doc_date": "2026-08-14",      <- or null
                    "doc_date_raw": "14 ऑगस्ट, 2026",
                    "department": "सामान्य प्रशासन विभाग",   <- or null
                    "department_raw": "...",
                    "header_page": 0,
                    "header_fallback": false
                },
                ...
            ],
            "count": 19
        }

    An empty "files" list means no snapshot has been built yet — ask an
    admin to run embedding, or run scripts/build_graph.py. It is not an
    error, so it is not reported as one.
    """
    meta = load_meta()

    files = []
    for fname in sorted(meta):
        entry = meta[fname]
        if not isinstance(entry, dict):
            continue
        files.append({k: v for k, v in entry.items() if k not in _OMITTED_FIELDS})

    return {
        "files": files,
        "count": len(files),
    }
