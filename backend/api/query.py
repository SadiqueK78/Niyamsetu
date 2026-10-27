# backend/api/query.py
# ─────────────────────────────────────────────────────────
# Chat query endpoint — the core of the application.
#
# Endpoints:
#   POST /api/query/chat    → main RAG query with history
#   POST /api/query/search  → cosine similarity search with scores
#   GET  /api/query/health  → check if Ollama + vector store are ready
#
# Every chat message flows through /chat:
#   frontend → POST /api/query/chat → rag.query() → LLM → response
# ─────────────────────────────────────────────────────────

import sys
import os
import asyncio
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auth.router import get_current_user
from core.rag import query as rag_query
from core.trace import trace
from core.hybrid_search import hybrid_search_explained
from core.vectorstore import is_ready
from db.logs import save_query_log
from config import settings

router = APIRouter(prefix="/api/query", tags=["Query"])


# ── Request / Response schemas ────────────────────────────

class ChatMessage(BaseModel):
    """One turn in the conversation history."""
    role: str       # "user" or "assistant"
    content: str    # the message text


class ChatRequest(BaseModel):
    """
    What the frontend sends for every chat message.

    query        : the current question
    history      : previous turns for context (can be empty list)
    language     : force language, or omit for auto-detect
    top_k        : how many chunks to retrieve (optional)
    focus_file   : document the question is about (optional)
    """
    query:    str
    history:  list[ChatMessage] = []
    language: Optional[str]     = None
    top_k:    Optional[int]     = None
    # Optional hint naming the document the question is about. Used only
    # by the metadata fast path, which needs to know WHICH document's
    # header to read; retrieval ignores it entirely.
    focus_file: Optional[str]   = None


class SearchRequest(BaseModel):
    """What the frontend sends for cosine similarity search."""
    query:       str
    top_k:       int = 5
    filter_file: Optional[str] = None   # filter results to one file


# ── Routes ────────────────────────────────────────────────

@router.post("/chat")
async def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Main RAG chat endpoint.
    Called by the frontend on every message send.

    Flow:
        1. Convert history to plain dicts for rag.query()
        2. Run RAG pipeline — retrieve chunks → build prompt → LLM
        3. Save query log to MongoDB
        4. Return answer + citations + timing

    Protected — requires valid JWT token.
    Any logged-in user can query (not admin-only).
    """
    # ── Arrival trace ─────────────────────────────────────
    # First thing in the handler, before any work: uvicorn's access log
    # only prints once the response is sent, so without this the terminal
    # shows nothing at all while a slow query is in flight.
    trace(
        "REQUEST  POST /api/query/chat",
        f"user={current_user['username']} "
        f"history={len(request.history)} turn(s) "
        f"query={request.query[:80]!r}",
    )

    # Convert Pydantic ChatMessage objects to plain dicts
    # rag.query() expects list of {"role": ..., "content": ...}
    history_dicts = [
        {"role": msg.role, "content": msg.content}
        for msg in request.history
    ]

    # ── Run RAG pipeline ──────────────────────────────────
    result = await rag_query(
        user_query=request.query,
        chat_history=history_dicts,
        top_k=request.top_k,
        language=request.language,
        focus_file=request.focus_file,
    )

    # ── Save to query log ─────────────────────────────────
    # Save regardless of success — failed queries are useful for debugging
    await save_query_log(
        username=current_user["username"],
        query=request.query,
        answer=result["answer"],
        citations=result["citations"],
        language=result["language"],
        elapsed_sec=result["elapsed_sec"],
        was_successful=result["success"],
    )

    # Return the full result to frontend
    return {
        "success":     result["success"],
        "answer":      result["answer"],
        "citations":   result["citations"],
        "language":    result["language"],
        "elapsed_sec": result["elapsed_sec"],
    }


@router.post("/search")
async def similarity_search(
    request: SearchRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Hybrid search endpoint. No LLM involved.
    Used on the Search page.

    Returns the same ranking /chat retrieves on — dense and BM25 fused by
    RRF — with the per-retriever ranks exposed, so a result that looks
    wrong can be attributed: a chunk BM25 alone found is a keyword match, a
    chunk only the dense side found shares no wording with the question,
    and a chunk both ranked highly is a real match. Cosine scores could not
    express that, because they only exist on one of the two sides.

    Response shape:
        {
            "success": true,
            "query": "...",
            "results": [
                {
                    "score":   0.82,            <- kept: now the fused relevance
                    "file":    "GR1.pdf",
                    "page":    "4",             <- kept: 1-based for display
                    "preview": "...",
                    "section_type":  "शासन निर्णय",
                    "dense_rank":    2,         <- or null
                    "lexical_rank":  1,         <- or null
                    "fused_rank":    1,
                    "rrf_score":     0.0163,
                    "relevance":     0.82
                },
                ...
            ],
            "total": 5,
            "dense_ok": true          <- false means the embedding endpoint
                                         is down and these are BM25 only
        }

    Every pre-existing key is still present and still means what it meant;
    "score" now carries the fused relevance rather than a cosine distance,
    which is the number the page was already displaying as a percentage.
    """
    if not is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not ready. Ask admin to embed documents first.",
        )

    # hybrid_search_explained() is synchronous — it embeds the query via a
    # blocking Ollama HTTP call (and loads the FAISS index from disk on the
    # first call), which pins the event loop and stalls every other
    # request. Same offload as the chat path uses.
    results, info = await asyncio.to_thread(
        hybrid_search_explained,
        request.query,
        top_k=request.top_k,
    )

    # Filter by filename if requested
    if request.filter_file:
        results = [
            r for r in results
            if r.get("source_file", "") == request.filter_file
        ]

    # Format results for frontend
    formatted = []
    for row in results:
        raw_page = row.get("page", None)
        page_display = str(int(raw_page) + 1) if raw_page is not None else "?"

        formatted.append({
            # Pre-existing keys, unchanged in name and meaning.
            "score":        row["relevance"],
            "file":         row.get("source_file", "Unknown"),
            "page":         page_display,
            "preview":      row.get("preview", ""),
            # Added: where each retriever put this chunk.
            "section_type": row.get("section_type"),
            "dense_rank":   row.get("dense_rank"),
            "lexical_rank": row.get("lexical_rank"),
            "fused_rank":   row.get("fused_rank"),
            "rrf_score":    row.get("rrf_score"),
            "relevance":    row.get("relevance"),
        })

    return {
        "success":  True,
        "query":    request.query,
        "results":  formatted,
        "total":    len(formatted),
        "dense_ok": info.get("dense_ok", False),
    }


@router.get("/health")
async def system_health(
    current_user: dict = Depends(get_current_user),
):
    """
    Returns system readiness status.
    Frontend calls this on load to show status indicators.

    Checks:
        - Is FAISS vector store ready?
        - Is Ollama reachable at localhost:11434?
    """
    # Check Ollama
    ollama_online = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(settings.OLLAMA_BASE_URL)
            ollama_online = response.status_code == 200
    except Exception:
        ollama_online = False

    vector_ready = is_ready()

    return {
        "success":      True,
        "vector_ready": vector_ready,
        "ollama_online": ollama_online,
        "llm_model":    settings.LLM_MODEL,
        "embed_model":  settings.EMBEDDING_MODEL,
    }