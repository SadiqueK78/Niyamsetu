# backend/core/reranker.py
# ─────────────────────────────────────────────────────────
# Cross-encoder reranking of retrieved chunks.
#
# Hybrid retrieval (dense + BM25 fused by RRF) ranks chunks by ORDER
# signals: neither side ever reads the question and the chunk together.
# A cross-encoder does — it scores each (question, chunk) pair jointly —
# so it can put the chunk that actually answers the question first,
# which matters most when only a handful of chunks reach the LLM.
#
# Enabled by settings.RERANK_ENABLED (see core/rag.py query()); with the
# flag off nothing here is imported or loaded.
#
# The model loads lazily on first use and is kept as a module-level
# singleton. sentence-transformers and torch are imported only then, so
# this module imports cheaply and the tests never touch them.
#
# Never fails the request: if the model cannot load or scoring raises,
# the original order is kept and cut to top_n.
# ─────────────────────────────────────────────────────────

import sys
import os
import threading
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from core.trace import trace

MAX_LENGTH = 512
BATCH_SIZE = 8

_model = None
_model_name = None
_lock = threading.Lock()

# Counters for diagnostics: how many rerank calls ran, and how many fell
# back to the original order. Read by tests/score_retrieval.py.
stats = {"calls": 0, "failures": 0}


def _load_cross_encoder(name: str):
    """Builds the CrossEncoder. Separate so tests can replace it."""
    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossEncoder(name, max_length=MAX_LENGTH, device=device)


def get_model():
    """
    The CrossEncoder for settings.RERANK_MODEL, loaded once.

    Reloaded only if RERANK_MODEL changes. Raises whatever loading raises;
    rerank() is the caller that turns that into a fallback.
    """
    global _model, _model_name
    name = settings.RERANK_MODEL
    with _lock:
        if _model is None or _model_name != name:
            _model = _load_cross_encoder(name)
            _model_name = name
        return _model


def reset_model() -> None:
    """Drops the loaded model. For tests."""
    global _model, _model_name
    with _lock:
        _model, _model_name = None, None


def rerank(query: str, chunks: list, top_n: int) -> list:
    """
    The top_n chunks by cross-encoder score against `query`, best first.

    Scores every (query, chunk text) pair in batches of BATCH_SIZE. Ties
    keep their original order (the sort is stable). The chunks returned
    are the same objects passed in, metadata untouched — page stays the
    0-based index and citations still display page + 1.

    On any failure, loading or scoring, logs a WARNING and returns the
    first top_n chunks in their original order.
    """
    chunks = list(chunks or [])
    stats["calls"] += 1
    if not chunks:
        return []

    started = time.monotonic()
    try:
        model = get_model()
        pairs = [(query, doc.page_content) for doc in chunks]
        scores = [float(s) for s in model.predict(pairs, batch_size=BATCH_SIZE)]
        if len(scores) != len(chunks):
            raise ValueError(f"{len(scores)} scores for {len(chunks)} chunks")
        order = sorted(range(len(chunks)), key=lambda i: -scores[i])
        kept = [chunks[i] for i in order[:top_n]]
    except Exception as e:
        stats["failures"] += 1
        trace("WARNING", f"reranker failed, keeping original order ({type(e).__name__}: {e})")
        return chunks[:top_n]

    seconds = round(time.monotonic() - started, 2)
    trace("RERANK done", f"{len(chunks)} candidates -> {len(kept)} in {seconds}s")
    return kept
