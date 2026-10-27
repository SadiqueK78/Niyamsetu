# backend/api/files.py
# ─────────────────────────────────────────────────────────
# Serves the original uploaded GR PDFs.
#
# Endpoints:
#   GET /api/files/{filename}   → the PDF itself
#
# Why this exists:
#   A citation says "GR1.pdf, page 4". Until now that was a dead end —
#   the user could read the 200-character preview and nothing else. This
#   endpoint is what lets the citation open the actual page it came from,
#   which is the difference between a claim and a verifiable claim on a
#   corpus of government orders.
#
# ── SECURITY ─────────────────────────────────────────────
# This is the only route that turns user input into a filesystem path, so
# the rules are deliberately strict and all of them are enforced before
# anything is read:
#
#   * the name must end in .pdf
#   * it may not contain a path separator, a parent reference, or a NUL
#   * the resolved path must sit inside grdocs, checked after resolution
#     so a symlink cannot point out of the directory
#   * every rejection is the same 404 with the same message, so the
#     endpoint cannot be used to probe what exists
#   * no error ever contains a filesystem path
#   * there is no listing endpoint here; a caller must already know the
#     name, which it gets from a citation
# ─────────────────────────────────────────────────────────

import sys
import os

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auth.router import get_current_user
from config import settings

router = APIRouter(prefix="/api/files", tags=["Files"])

# Characters and sequences that can only be an attempt to leave the
# directory. Checked on the raw path parameter, before any resolution.
_FORBIDDEN = ("/", "\\", "..", "\x00")

# One message for every rejection. A "not found" that differs from a
# "forbidden" tells a caller which filenames exist.
_NOT_FOUND = "Document not found."


def _safe_grdocs_path(filename: str):
    """
    The on-disk path for a requested document, or None if it is not
    allowed.

    Returns None rather than raising so the caller produces one identical
    404 for every failure, and so no branch can accidentally leak a path
    into an error message.
    """
    if not filename:
        return None

    # Extension first: it is the cheapest check and it alone rules out
    # every non-document in the directory.
    if not filename.lower().endswith(".pdf"):
        return None

    if any(bad in filename for bad in _FORBIDDEN):
        return None

    base = settings.GRDOCS_PATH.resolve()
    try:
        candidate = (base / filename).resolve()
    except (OSError, ValueError):
        return None

    # Containment is verified AFTER resolution, which is what makes a
    # symlink pointing outside grdocs fail too. is_relative_to is exact:
    # a sibling directory whose name merely starts with the same text
    # ("grdocs_old") does not pass.
    if not candidate.is_relative_to(base):
        return None

    if not candidate.is_file():
        return None

    return candidate


@router.get("/{filename}")
async def fetch_document(
    filename: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Returns one uploaded GR PDF.

    Available to all logged-in users — the same dependency every other
    protected route uses. Anyone who can read a citation can read the
    document the citation points at; withholding it would make the
    citation unverifiable, which is worse than the alternative on a
    corpus of public government orders.

    Args:
        filename : bare PDF filename exactly as a citation reports it,
                   e.g. "GR1.pdf"

    Returns:
        the PDF as application/pdf, inline.

    Raises:
        404 for anything that is not a readable PDF inside grdocs. The
        message and status are identical in every case, deliberately: a
        distinguishable response would let a caller map the filesystem.
    """
    path = _safe_grdocs_path(filename)
    if path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=_NOT_FOUND,
        )

    # inline, not attachment: the viewer renders this in an iframe rather
    # than downloading it.
    return FileResponse(
        path=str(path),
        media_type="application/pdf",
        filename=path.name,
        content_disposition_type="inline",
    )
