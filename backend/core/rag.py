# backend/core/rag.py
# ─────────────────────────────────────────────────────────
# RAG (Retrieval Augmented Generation) query engine.
#
# Flow for every chat message:
#   1. Detect language (English or Marathi)
#   2. Search FAISS for relevant chunks
#   3. Extract citations from chunk metadata
#   4. Build prompt with context + history + query
#   5. Send to LLM via Ollama
#   6. Return answer + citations + timing
#
# Called by:
#   api/query.py → passes user query and chat history here
# ─────────────────────────────────────────────────────────

import sys
import os
import time
import asyncio
import re
import threading

import httpx
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from core.gr_graph import fold_digits, graph_snapshot_path, read_graph_snapshot
from core.gr_meta import header_chunk_for, load_meta
from core.hybrid_search import hybrid_search, get_lexical_index, lexical_search
from core.language import detect_language, format_chat_history, clean_text
from core.trace import trace
from core.condense import condense_query
from core.reranker import rerank


# Do NOT pass think=False here to speed qwen3 up — it does not work.
# Measured on qwen3:4b, same prompt, both with a warm prompt cache:
#
#   think ON  : 36.1s, 181 generated tokens, answer "Divisional Commissioner"
#   think OFF : 40.1s, 181 generated tokens, answer = the full reasoning
#               monologue, ending in a stray "■■■" then the answer
#
# The model reasons either way. think=False only moves the chain of thought
# out of the response's `thinking` field and into `content`, so the token
# count — and therefore the time, at ~5 tok/s on CPU — is unchanged, while
# the answer text becomes unusable for the capture harness.
#
# Leaving thinking ON is both faster and cleaner here.


def get_llm() -> ChatOllama:
    """
    Creates a ChatOllama instance pointing to local Ollama server.
    temperature=0 means deterministic responses — same question
    gets consistent answers, important for government documents.
    """
    return ChatOllama(
        model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0,
        # Pinned so Ollama never resizes the context between requests — see
        # LLM_NUM_CTX in config.py for why a resize is what actually OOMs.
        # langchain-ollama folds this into the Ollama "options" payload.
        num_ctx=settings.LLM_NUM_CTX,
        # Keeps the model resident across idle gaps, so a question after a
        # pause reuses the loaded context instead of forcing a reload.
        keep_alive=settings.LLM_KEEP_ALIVE,
        # client_kwargs goes straight to ollama.Client, which forwards it to
        # httpx.Client. Ollama's own default is timeout=None, so without this
        # a generation can run indefinitely.
        # Closing the socket is what actually stops the work: Ollama ends the
        # request and frees the model, so the next queued request can start.
        # Only bounds total generation together with stream=False at the
        # invoke() call below — see the note there before changing either.
        client_kwargs={"timeout": settings.LLM_TIMEOUT},
    )


def get_chat_llm() -> ChatOllama:
    """
    The client for chat ANSWER generation: get_llm() with num_predict set
    to settings.LLM_NUM_PREDICT and every other option unchanged.

    A copy (model_copy), not a change to get_llm(): the summary, condense,
    rewrite and verify calls share get_llm() and keep its settings. Same
    technique as core/condense.py. LLM_NUM_PREDICT <= 0 returns get_llm()
    as it is — no ceiling.
    """
    llm = get_llm()
    if settings.LLM_NUM_PREDICT > 0:
        llm = llm.model_copy(update={"num_predict": settings.LLM_NUM_PREDICT})
    return llm


# Margin under LLM_NUM_CTX at which a prompt is reported as possibly
# truncated: Ollama silently drops the start of a prompt that does not fit.
PROMPT_TRUNCATION_MARGIN = 64


def log_token_usage(response) -> None:
    """
    Logs the token counts Ollama reports for one generation.

    INFO "llm prompt_tokens=<n> completion_tokens=<m>" when both counts are
    present; WARNING "prompt may be truncated: ..." when the prompt came
    within PROMPT_TRUNCATION_MARGIN tokens of LLM_NUM_CTX. Reads
    response_metadata (prompt_eval_count, eval_count), falling back to
    usage_metadata. Never raises: this is a diagnostic.
    """
    try:
        meta = getattr(response, "response_metadata", None) or {}
        prompt_tokens = meta.get("prompt_eval_count")
        completion_tokens = meta.get("eval_count")
        usage = getattr(response, "usage_metadata", None) or {}
        if prompt_tokens is None:
            prompt_tokens = usage.get("input_tokens")
        if completion_tokens is None:
            completion_tokens = usage.get("output_tokens")

        if prompt_tokens is not None and completion_tokens is not None:
            trace("INFO", f"llm prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}")
        if prompt_tokens is not None and \
                prompt_tokens >= settings.LLM_NUM_CTX - PROMPT_TRUNCATION_MARGIN:
            trace("WARNING", f"prompt may be truncated: prompt_tokens={prompt_tokens} "
                             f"num_ctx={settings.LLM_NUM_CTX}")
    except Exception:
        pass


# ── Graph-expanded retrieval ──────────────────────────────

# Relations that change what a document MEANS, ranked above a plain
# citation. If the retrieved GR has been superseded or amended, the
# document saying so is the one the answer is wrong without; a document
# merely referenced is ordinary background.
_GRAPH_RELATION_PRIORITY = {
    "supersedes": 0,
    "amends":     0,
    "references": 1,
}
_GRAPH_RELATION_DEFAULT = 2

# How deep lexical_search reads before its results are filtered down to
# one neighbour's chunks. The filter can discard almost everything — a
# neighbour may own three chunks out of a thousand — so the pool has to be
# far larger than the number of chunks actually wanted.
_GRAPH_LEXICAL_POOL = 200

# The snapshot is re-read only when the file's mtime and size change, so a
# rebuilt graph is picked up without a restart while a normal request pays
# one stat() call. Keyed on (mtime, size) rather than mtime alone because
# two writes inside one filesystem timestamp tick are possible.
_graph_cache_lock = threading.Lock()
_graph_cache = None
_graph_cache_key = None


def load_graph_snapshot():
    """
    The graph snapshot as {"nodes": [...], "edges": [...]}, or None when
    there is no snapshot file.

    Reads the JSON written by core.gr_graph, never MongoDB: this runs
    inside a chat request, and a Motor client is bound to the loop that
    created it. The file is also what an offline rebuild writes, so the
    retrieval path and scripts/build_graph.py always agree.
    """
    global _graph_cache, _graph_cache_key

    path = graph_snapshot_path()
    try:
        stat = path.stat()
    except OSError:
        return None

    key = (stat.st_mtime_ns, stat.st_size)
    with _graph_cache_lock:
        if _graph_cache is not None and _graph_cache_key == key:
            return _graph_cache

    snapshot = read_graph_snapshot()

    with _graph_cache_lock:
        _graph_cache, _graph_cache_key = snapshot, key
    return snapshot


def _neighbor_relations(snapshot: dict, seeds: list, present: list = None) -> dict:
    """
    Resolved documents linked to any of `seeds`, in both directions.

    Args:
        snapshot : the graph snapshot
        seeds    : source_files whose edges are followed
        present  : source_files already in the chunk list, which are never
                   returned as neighbours. Defaults to seeds.

    Returns:
        {neighbour source_file: {"relation": str, "links": int, "order": int}}
        where relation is the highest-priority relation seen on any edge to
        that neighbour, links counts how many of `seeds` reach it, and
        order is first appearance.

    Seeds and exclusions are separate lists because they answer different
    questions. Following edges from every retrieved file lets a file that
    merely ranked ninth drag in its own neighbours, so seeds are capped by
    the caller. Excluding only those same few would then let a file already
    sitting at rank nine be re-added as a neighbour of rank one, duplicating
    it in the context — so the exclusion set stays the whole retrieved set.
    """
    resolved = {
        n.get("id") for n in snapshot.get("nodes", [])
        if n.get("resolved") and n.get("id")
    }
    inside = set(seeds)
    already = set(present if present is not None else seeds)

    found = {}
    for edge in snapshot.get("edges", []):
        source, target = edge.get("source"), edge.get("target")
        relation = edge.get("relation") or "references"

        # Both directions: the superseding document cites the superseded
        # one, so following edges forwards only would find the old order
        # from the new one and never the other way round — which is the
        # direction that matters when the old one is what got retrieved.
        for near, far in ((source, target), (target, source)):
            if near not in inside or far in already:
                continue
            if far not in resolved:
                continue

            entry = found.get(far)
            if entry is None:
                found[far] = {"relation": relation, "links": 1, "order": len(found)}
                continue

            entry["links"] += 1
            if (_GRAPH_RELATION_PRIORITY.get(relation, _GRAPH_RELATION_DEFAULT)
                    < _GRAPH_RELATION_PRIORITY.get(entry["relation"], _GRAPH_RELATION_DEFAULT)):
                entry["relation"] = relation

    return found


def _neighbor_chunks(source_file: str, question: str, wanted: int) -> list:
    """
    The chunks to borrow from one neighbouring document.

    Question-relevant chunks first, via BM25 over the whole corpus
    filtered to this file — a neighbour contributes its passage about the
    question, not its letterhead, whenever it has one. Short of that, the
    header comes first (it carries the GR number and date, which is what
    identifies the document that supersedes or amends) and then the
    earliest body chunks.

    Returns:
        Up to `wanted` Documents, originals from the docstore. The caller
        copies them before writing metadata.
    """
    if wanted <= 0:
        return []

    picked, seen = [], set()

    for doc, _score in lexical_search(question, top_k=_GRAPH_LEXICAL_POOL):
        if doc.metadata.get("source_file") != source_file:
            continue
        if id(doc) in seen:
            continue
        seen.add(id(doc))
        picked.append(doc)
        if len(picked) >= wanted:
            return picked

    # Not enough question-relevant chunks — fall back to the document's own
    # opening, header first.
    index = get_lexical_index()
    if index is None:
        return picked

    own = [(i, d) for i, d in enumerate(index.documents)
           if d.metadata.get("source_file") == source_file]

    headers = [pair for pair in own
               if pair[1].metadata.get("section_type") == HEADER_SECTION_TYPE]
    rest = sorted(
        (pair for pair in own if pair not in headers),
        key=lambda pair: _header_fallback_key(pair[1], pair[0]),
    )

    for _, doc in headers + rest:
        if id(doc) in seen:
            continue
        seen.add(id(doc))
        picked.append(doc)
        if len(picked) >= wanted:
            break

    return picked


def expand_with_graph(chunks: list, question: str) -> list:
    """
    Appends chunks from documents linked to the retrieved ones.

    Retrieval matches wording, and a superseding GR shares almost none
    with the order it cancels — it names it by number and date and then
    states the new rule. So the document that makes the retrieved answer
    obsolete is exactly the document retrieval cannot find. The graph knows
    the link; this follows it.

    Only the top settings.GRAPH_SEED_DOCS distinct files are used as
    seeds. Following edges from everything retrieved meant a file that
    ranked ninth — there on a keyword coincidence, not because the
    question was about it — contributed its own neighbours, and those
    arrived in the context looking exactly as authoritative as a
    superseding order. The documents the question is actually about are at
    the front of the ranking, so that is where the edges worth following
    start.

    Args:
        chunks   : the final retrieved chunks, after routing and headers
        question : the user's question, used to pick which of a
                   neighbour's chunks to borrow

    Returns:
        chunks with neighbour chunks appended. Returns chunks unchanged
        when there is no snapshot, nothing is linked, or seeding is off.
        Never mutates the input list or the docstore's Documents.
    """
    if not chunks:
        return chunks

    if settings.GRAPH_SEED_DOCS <= 0:
        trace("GRAPH    skip ", "GRAPH_SEED_DOCS <= 0, no seeds to follow")
        return chunks

    snapshot = load_graph_snapshot()
    if snapshot is None:
        trace("GRAPH    skip ", f"no snapshot at {graph_snapshot_path()}")
        return chunks

    present = []
    for doc in chunks:
        fname = doc.metadata.get("source_file")
        if fname and fname not in present:
            present.append(fname)

    # First appearance order, which is rank order — chunks arrive best
    # first — so these are the top-ranked files, not the alphabetically
    # first ones.
    seeds = present[:settings.GRAPH_SEED_DOCS]

    found = _neighbor_relations(snapshot, seeds, present=present)
    if not found:
        trace("GRAPH    done ",
              f"no linked documents outside the retrieved set "
              f"(seeds: {', '.join(seeds)})")
        return chunks

    # Relation first, then how many of the retrieved documents agree on the
    # neighbour, then first appearance for a stable order.
    ranked = sorted(
        found.items(),
        key=lambda pair: (
            _GRAPH_RELATION_PRIORITY.get(pair[1]["relation"], _GRAPH_RELATION_DEFAULT),
            -pair[1]["links"],
            pair[1]["order"],
        ),
    )[:settings.GRAPH_MAX_NEIGHBOR_DOCS]

    added, summary = [], []
    for fname, entry in ranked:
        borrowed = _neighbor_chunks(fname, question, settings.GRAPH_CHUNKS_PER_NEIGHBOR)
        if not borrowed:
            continue

        for doc in borrowed:
            # COPIES. These Documents are the docstore's own objects, shared
            # with the BM25 index and every other request; stamping metadata
            # on them would leak graph_expanded into unrelated answers and
            # persist for the life of the process.
            clone = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "graph_expanded": True,
                    "graph_relation": entry["relation"],
                },
            )
            added.append(clone)

        summary.append(f"{fname} ({entry['relation']}, {len(borrowed)} chunk(s))")

    if not added:
        trace("GRAPH    done ", "linked documents found but no chunks available")
        return chunks

    trace("GRAPH    done ",
          f"added {len(added)} chunk(s) from " + "; ".join(summary)
          + f" (seeds: {', '.join(seeds)})")

    # Appended, so the question's own best matches stay at the front of the
    # context and the borrowed material reads as supporting background.
    return list(chunks) + added


# ── Metadata fast path ────────────────────────────────────
# WHY THIS EXISTS
#
# "What is the date of this GR?" has one correct answer, written in one
# place, in a fixed layout. Sending it through retrieval and a 14B model
# costs tens of seconds and can still come back with the date of a GR the
# letterhead happened to reference — which is exactly the failure the
# harness kept recording. The metadata table already holds the answer.
#
# Scope is deliberately narrow. The guards below reject anything that is
# not unmistakably a header question about one identifiable document, and
# every rejection falls through to the normal path. A fast path that
# answers a question it misunderstood is far worse than no fast path.

# The three fields this can answer, in the order they are reported.
_META_INTENT_ORDER = ("date", "department", "gr_number")

# Trigger words per intent. English included because the corpus is
# bilingual and the chat box is not.
_META_INTENT_TRIGGERS = {
    "date":       ("दिनांक", "तारीख", "कधी", "date"),
    "department": ("विभाग",),
    "gr_number":  ("क्रमांक", "number"),
}

# A header question is short. Anything longer is asking for reasoning,
# whatever words it happens to contain.
_META_MAX_QUESTION_CHARS = 80

# Words that mean the question is about relationships between documents,
# not about one document's own letterhead.
_META_BLOCKING_ROOTS = ("संदर्भ", "वाचा", "संबंध")

# At least one of these must appear, so a bare "कधी?" with no document
# type named never reaches the fast path.
_META_DOCTYPE_ROOTS = ("शासन", "परिपत्रक", "आदेश", "GR")

# Labels for the answer lines. Bilingual so one line serves both
# languages, and fixed strings rather than generated text.
_META_LABELS = {
    "date":       "दिनांक (Date)",
    "department": "विभाग (Department)",
    "gr_number":  "शासन निर्णय क्रमांक (GR number)",
}

# A GR number shares this many digits with the question before it counts
# as naming that document. Same threshold the graph uses for a strong key.
_META_NUMBER_RUN = 6

_META_DIGIT_RUN = re.compile(r"\d{%d,}" % _META_NUMBER_RUN)


def _strip_known_stems(folded: str, known_stems) -> str:
    """
    The question with every known document stem removed.

    Args:
        folded      : the question, digits already folded to ASCII
        known_stems : stems to remove, or None

    Returns:
        the question with each occurrence of each stem replaced by a
        space, matched case-insensitively.

    LONGEST FIRST, and that ordering is the whole point. Every stem in
    this corpus contains a digit, so naming a document is the one way a
    header question legitimately carries one. Removing "GR1" from a
    question about GR1.5 would leave ".5" behind, the digit guard would
    fire on that stray digit, and the question would be rejected for
    naming its own document. Removing the longest match first consumes
    "GR1.5" whole.

    A space rather than nothing, so removing a stem cannot weld its
    neighbours into a word that was never written.
    """
    if not known_stems:
        return folded

    out = folded
    for stem in sorted((s for s in known_stems if s), key=len, reverse=True):
        needle = fold_digits(stem).lower()
        if not needle:
            continue
        # Case-insensitive replace, done by hand because str.replace is
        # case-sensitive and a regex would need every stem escaped.
        lowered = out.lower()
        at = lowered.find(needle)
        while at != -1:
            out = out[:at] + " " + out[at + len(needle):]
            lowered = out.lower()
            at = lowered.find(needle, at + 1)

    return out


def _names_known_stem(folded: str, known_stems) -> bool:
    """
    Whether the question names any known document stem.

    Args:
        folded      : the question, digits already folded to ASCII
        known_stems : stems to look for, or None

    Returns:
        True as soon as one is found.

    Same normalisation as _strip_known_stems — fold, lowercase, substring,
    longest stem first — so the two cannot disagree about what counts as
    naming a document. The ordering does not change this function's
    answer, since any match returns True; it is kept so the two read as
    one convention rather than two.
    """
    if not known_stems:
        return False

    lowered = folded.lower()
    for stem in sorted((s for s in known_stems if s), key=len, reverse=True):
        needle = fold_digits(stem).lower()
        if needle and needle in lowered:
            return True

    return False


def detect_meta_intents(question: str, known_stems=None) -> list:
    """
    Which header fields a question is asking for, or [].

    Args:
        question    : the user's question
        known_stems : document stems to discount before the digit guard
                      runs; the metadata table's stems, in practice

    Returns:
        an ordered subset of ("date", "department", "gr_number"), empty
        unless every guard holds: short enough, no ASCII digit once known
        stems are discounted, no relationship word, and either a
        document-type word or a known document stem.

    NAMING A DOCUMENT IS NAMING A DOCUMENT. The document-type guard exists
    to reject a bare "कधी?" that identifies nothing, but it was written as
    a vocabulary test, so "aadesh 5 चा दिनांक काय आहे?" failed it — the
    question names one document unambiguously and none of शासन, परिपत्रक,
    आदेश or GR appears in it. Those questions then went to retrieval and
    the model, which is the path that answers them wrongly. A known stem
    now satisfies the guard on its own.

    The digit guard is what keeps "GR 1518 मध्ये किती सदस्य?" out: a
    question carrying numbers is almost always asking about the document's
    CONTENTS, where the metadata table has nothing to say.

    Stems are exempt from it because every stem in this corpus contains a
    digit — GR1, aadesh 1, letter 2 — so without the exemption the guard
    rejected exactly the questions that name the document they are about,
    which are the ones this path exists to answer. A number that is NOT
    part of a stem still rejects, so the guard keeps doing its job.

    The length guard is deliberately measured on the ORIGINAL question:
    it is asking how much the user wrote, and stripping stems out first
    would let an arbitrarily long question through by the accident of
    mentioning a document.
    """
    raw = question or ""
    if len(raw) > _META_MAX_QUESTION_CHARS:
        return []

    folded = fold_digits(raw)
    lowered = folded.lower()

    if any(ch.isdigit() and ch.isascii()
           for ch in _strip_known_stems(folded, known_stems)):
        return []
    if any(root in folded for root in _META_BLOCKING_ROOTS):
        return []
    # Either way of identifying a document will do. With no known_stems
    # the second test is False at once, so this reduces to exactly the
    # vocabulary check it has always been.
    if not (any(root.lower() in lowered for root in _META_DOCTYPE_ROOTS)
            or _names_known_stem(folded, known_stems)):
        return []

    return [
        intent for intent in _META_INTENT_ORDER
        if any(trigger.lower() in lowered
               for trigger in _META_INTENT_TRIGGERS[intent])
    ]


def _names_stem(question: str, stem: str) -> bool:
    """
    Whether the question names this document's stem as a whole token.

    "GR1" must not match a question about GR1.5: the character after the
    stem may not be a letter, a digit, or a dot followed by a digit, and
    the character before it may not be a letter or a digit. Without the
    dot-digit rule every question about GR1.5 would also resolve to GR1,
    and the fast path would answer confidently from the wrong document.
    """
    if not stem:
        return False

    lowered_q = question.lower()
    lowered_s = stem.lower()

    start = lowered_q.find(lowered_s)
    while start != -1:
        before = lowered_q[start - 1] if start > 0 else ""
        after_at = start + len(lowered_s)
        after = lowered_q[after_at] if after_at < len(lowered_q) else ""
        next_after = lowered_q[after_at + 1] if after_at + 1 < len(lowered_q) else ""

        before_ok = not (before.isalpha() or before.isdigit())
        after_ok = not (
            after.isalpha()
            or after.isdigit()
            or (after == "." and next_after.isdigit())
        )
        if before_ok and after_ok:
            return True

        start = lowered_q.find(lowered_s, start + 1)

    return False


def resolve_meta_target(question: str, focus_file, meta: dict):
    """
    Which document a header question is about, or None.

    Args:
        question   : the user's question
        focus_file : an explicit hint from the caller, or None
        meta       : the metadata table, keyed by source_file

    Returns:
        a source_file, or None when nothing or more than one thing matches.

    An explicit focus_file wins outright — the caller knows which document
    the user is looking at, and no amount of string matching beats that.
    Otherwise the question has to name the document itself, by stem as a
    whole token or by sharing a long digit run with its GR number.
    Ambiguity returns None rather than a guess.
    """
    if not meta:
        return None

    if focus_file and focus_file in meta:
        return focus_file

    folded_q = fold_digits(question or "")
    question_runs = set(_META_DIGIT_RUN.findall(folded_q))

    candidates = set()
    for fname, entry in meta.items():
        if not isinstance(entry, dict):
            continue

        if _names_stem(folded_q, entry.get("stem") or ""):
            candidates.add(fname)
            continue

        if question_runs:
            number_digits = "".join(
                ch for ch in fold_digits(entry.get("gr_number") or "")
                if ch.isdigit()
            )
            if any(run in number_digits for run in question_runs):
                candidates.add(fname)

    return candidates.pop() if len(candidates) == 1 else None


def build_meta_answer(entry: dict, intents: list):
    """
    The answer text for a metadata question, or None.

    Args:
        entry   : one metadata table row
        intents : detect_meta_intents() output

    Returns:
        one "<label>: <value>" line per intent in the fixed order, then an
        English source line. None when any requested field is missing —
        the fast path then falls through rather than reporting half an
        answer as though it were the whole one.

    The date prefers doc_date_raw, so the answer reads back the date the
    document actually prints rather than a normalised form the reader
    would have to trust.
    """
    if not entry or not intents:
        return None

    lines = []
    for intent in _META_INTENT_ORDER:
        if intent not in intents:
            continue

        if intent == "date":
            value = entry.get("doc_date_raw") or entry.get("doc_date")
        elif intent == "department":
            value = entry.get("department")
        else:
            value = entry.get("gr_number")

        if not value:
            return None
        lines.append(f"{_META_LABELS[intent]}: {value}")

    if not lines:
        return None

    # 1-based, matching what build_citations() reports and what the
    # citation pill displays. header_page is stored 0-based, as the index
    # holds it, so the conversion happens here at display time only —
    # without it the answer said "page 0" while its own citation said p.1.
    raw_page = entry.get("header_page")
    try:
        page_display = int(raw_page) + 1
    except (TypeError, ValueError):
        page_display = raw_page

    lines.append(
        f"Source: {entry.get('source_file')}, page {page_display}"
    )
    return "\n".join(lines)


def _meta_citation_docs(source_file: str) -> list:
    """
    The one chunk a metadata answer cites: that file's header.

    Returns [] when the index is unavailable, which makes the fast path
    fall through — an answer with no citation is not something this system
    should hand back.
    """
    index = get_lexical_index()
    if index is None:
        return []

    own = [d for d in index.documents
           if d.metadata.get("source_file") == source_file]
    if not own:
        return []

    chunk = header_chunk_for(own)
    return [chunk] if chunk is not None else []


def try_meta_fastpath(user_query: str, focus_file, language: str, start_time: float):
    """
    A complete response dict answered from metadata alone, or None.

    Returns None at every point where the question, the table or the index
    does not unambiguously support an answer, and the caller then runs the
    normal pipeline unchanged.

    Makes no model call of any kind and performs no retrieval.
    """
    meta = load_meta()
    if not meta:
        return None

    # The table's own stems, so a question naming the document it is about
    # is not rejected for the digit inside that name.
    known_stems = [
        entry.get("stem") for entry in meta.values()
        if isinstance(entry, dict) and entry.get("stem")
    ]

    intents = detect_meta_intents(user_query, known_stems=known_stems)
    if not intents:
        return None

    target = resolve_meta_target(user_query, focus_file, meta)
    if not target:
        return None

    answer = build_meta_answer(meta.get(target), intents)
    if not answer:
        return None

    docs = _meta_citation_docs(target)
    if not docs:
        return None

    trace("META     fastpath", f"file={target} intents={intents}")

    return {
        "success":     True,
        "answer":      answer,
        "citations":   build_citations(docs),
        "language":    language,
        "elapsed_sec": round(time.time() - start_time, 2),
        "query":       user_query,
    }


# Sent verbatim as the rewrite instruction. English on purpose even for a
# Marathi question: the instruction tells the model to answer in the
# question's own language, so the directive language and the output
# language are independent.
QUERY_REWRITE_INSTRUCTION = (
    "Rewrite the user question into up to 3 alternative phrasings that use "
    "different synonyms, in the same language as the question. Output only "
    "the alternatives, one per line, no numbering, no explanation."
)

# Ceiling on how many alternatives are used, however many come back.
QUERY_REWRITE_MAX_ALTERNATIVES = 3


async def expand_query(user_query: str) -> str:
    """
    Builds a wider retrieval query by asking the LLM for synonym phrasings.

    BM25 only matches the words it is given, and a Marathi GR corpus states
    the same concept several ways — a question phrased with one of them
    retrieves nothing from a document that uses another. Retrieving on the
    question plus its paraphrases widens lexical recall without changing
    what the user asked.

    The return value is for RETRIEVAL ONLY. The caller keeps user_query for
    the prompt, the history and the log, so an unhelpful rewrite can cost
    recall but can never change the question the model is asked or the
    question that is recorded.

    Returns:
        "<original> <alt1> <alt2> <alt3>", or user_query unchanged if the
        rewrite call fails, times out, or comes back empty.
    """
    try:
        llm    = get_llm()          # already temperature=0
        prompt = f"{QUERY_REWRITE_INSTRUCTION}\n\nQuestion: {user_query}"
        trace(
            "REWRITE  start",
            f"model={settings.LLM_MODEL} timeout={settings.QUERY_REWRITE_TIMEOUT:.0f}s",
        )
        # wait_for bounds how long THIS path waits; the worker thread cannot
        # be killed from here, so a hung generation is abandoned rather than
        # cancelled and is still bounded by LLM_TIMEOUT inside the client.
        # stream=False for the same reason as the generation call below.
        response = await asyncio.wait_for(
            asyncio.to_thread(
                llm.invoke, [HumanMessage(content=prompt)], stream=False
            ),
            timeout=settings.QUERY_REWRITE_TIMEOUT,
        )
        alternatives = [
            line.strip()
            for line in response.content.strip().splitlines()
            if line.strip()
        ][:QUERY_REWRITE_MAX_ALTERNATIVES]

    # Deliberately broad: a rewrite is an optimisation, and no failure of it
    # is worth failing the question over.
    except Exception as e:
        trace("REWRITE  failed", f"{type(e).__name__}: retrieving on the original question")
        return user_query

    if not alternatives:
        trace("REWRITE  failed", "no alternatives returned: retrieving on the original question")
        return user_query

    trace("REWRITE  done ", f"{len(alternatives)} alternative(s)")
    return user_query + " " + " ".join(alternatives)


def build_citations(docs: list) -> list:
    """
    Extracts citation information from retrieved document chunks.
    Deduplicates — same file+page combination appears only once.

    Args:
        docs : list of LangChain Document objects from retrieval

    Returns:
        List of citation dicts:
        [
            {
                "file": "GR_2024_transfer.pdf",
                "page": "4",           ← displayed as human-readable page number
                "preview": "First 200 chars of the chunk...",
                "relevance_score": 1.0 ← None when retrieval did not attach one
                # these two only when the chunk came from graph expansion:
                "via_graph": True,
                "graph_relation": "supersedes"
            },
            ...
        ]
    """
    citations = []
    seen      = set()   # tracks file+page combos we've already added

    for doc in docs:
        filename = doc.metadata.get("source_file", "Unknown Document")

        # LangChain pages are 0-indexed internally
        # We add 1 so users see "Page 4" instead of "Page 3"
        raw_page = doc.metadata.get("page", None)
        if raw_page is not None:
            page_display = str(int(raw_page) + 1)
        else:
            page_display = "?"

        # Create a unique key for deduplication
        key = f"{filename}_p{page_display}"

        if key not in seen:
            seen.add(key)
            citation = {
                "file":    filename,
                "page":    page_display,
                # Preview helps user understand why this source was cited
                "preview": doc.page_content[:200].strip(),
                # Attached by hybrid_search; absent when a retrieval path
                # does not score its results, so None rather than a number.
                "relevance_score": doc.metadata.get("relevance_score", None),
            }

            # Only present on chunks the graph pulled in, so a citation
            # from ordinary retrieval keeps exactly the keys it had before
            # graph expansion existed.
            if doc.metadata.get("graph_expanded"):
                citation["via_graph"]      = True
                citation["graph_relation"] = doc.metadata.get("graph_relation")

            citations.append(citation)

    return citations


# File-level fusion constant. Deliberately the same 60 as
# core.hybrid_search.RRF_K: this is the same reciprocal-rank trick applied
# one level up, over files instead of chunks, and a different constant here
# would make the two rankings incomparable for no reason.
ROUTE_RRF_K = 60


def route_documents(docs: list) -> list:
    """
    Narrows a retrieved chunk list to the few source files that dominate it.

    Retrieval ranks CHUNKS, so a question whose answer sits in one GR can
    still come back with eleven chunks spread over six documents, most of
    them incidental keyword matches. Scoring files by the reciprocal ranks
    of their chunks rewards a file that places several chunks high over one
    that places a single chunk high, which is what actually distinguishes
    the document the question is about from a document that merely shares
    its vocabulary.

    Args:
        docs : Documents from retrieval, best first

    Returns:
        The subset belonging to the top settings.ROUTE_TOP_DOCS files, in
        their original relative order, capped at settings.ROUTE_MAX_CHUNKS.
        Returns docs unchanged in shape when there is nothing to narrow.
    """
    def source_of(doc):
        return doc.metadata.get("source_file", "Unknown Document")

    scores     = {}
    first_seen = {}
    for rank, doc in enumerate(docs):
        fname = source_of(doc)
        scores[fname] = scores.get(fname, 0.0) + 1.0 / (ROUTE_RRF_K + rank)
        first_seen.setdefault(fname, len(first_seen))

    # Ties broken by first appearance, so two files that scored identically
    # resolve the same way on every run rather than by dict iteration luck.
    ranked = sorted(scores, key=lambda f: (-scores[f], first_seen[f]))
    keep   = set(ranked[:settings.ROUTE_TOP_DOCS])

    kept = [d for d in docs if source_of(d) in keep][:settings.ROUTE_MAX_CHUNKS]

    trace(
        "ROUTING  done ",
        f"kept {len(kept)}/{len(docs)} chunk(s) from "
        f"{', '.join(ranked[:settings.ROUTE_TOP_DOCS]) or 'no files'}",
    )
    return kept


# Section marker that core.vectorstore stamps on the block above a
# document's first structural marker — see _section_chunks() there. That
# block is where a GR's number, date, subject and department live, so it
# is the chunk that identifies the document.
HEADER_SECTION_TYPE = "__header__"

# At most three headers are ever prepended, so the context grows by a
# bounded amount no matter how many files routing let through.
HEADER_INJECT_MAX_FILES = 3


def _header_fallback_key(doc, position: int) -> tuple:
    """
    Sort key picking a document's most header-like chunk when no chunk is
    marked __header__: smallest page first, then the document's own
    section ordering, then docstore insertion order.

    section_index is the explicit index core.vectorstore writes for every
    section chunk, and section_part disambiguates the pieces of one
    oversize section that all share a section_index. Both fall back to
    position — the chunk's place in docstore order — when absent.
    """
    page = doc.metadata.get("page", None)
    try:
        # Missing or non-numeric pages sort last rather than crashing.
        page_key = (0, int(page))
    except (TypeError, ValueError):
        page_key = (1, 0)

    def idx(field):
        value = doc.metadata.get(field, None)
        try:
            return (0, int(value))
        except (TypeError, ValueError):
            return (1, 0)

    return (page_key, idx("section_index"), idx("section_part"), position)


def inject_headers(docs: list) -> list:
    """
    Prepends each source file's header chunk to a retrieved chunk list.

    Retrieval matches on body text, so the chunk that answers a question is
    usually from the middle of a GR and carries none of its identity — the
    GR number, date, subject and issuing department all sit in the header
    block, which rarely matches the question's wording and so rarely gets
    retrieved. Without it the LLM can state a fact correctly and attribute
    it to nothing.

    Args:
        docs : the final chunk list, after routing

    Returns:
        docs with up to HEADER_INJECT_MAX_FILES header chunks prepended, in
        the order their files first appear. Returns docs unchanged when the
        store is unavailable or every header is already present.
    """
    if not docs:
        return docs

    # The BM25 index already holds every chunk in docstore order and is
    # cached across requests, so this is a lookup rather than a second walk
    # of the docstore. None means no vector store, same as elsewhere.
    index = get_lexical_index()
    if index is None:
        return docs

    corpus = index.documents

    # Files in order of first appearance, capped.
    ordered_files = []
    for doc in docs:
        fname = doc.metadata.get("source_file", "Unknown Document")
        if fname not in ordered_files:
            ordered_files.append(fname)
    ordered_files = ordered_files[:HEADER_INJECT_MAX_FILES]

    def already_present(candidate):
        for doc in docs:
            if doc is candidate:
                return True
            if (doc.metadata.get("source_file") == candidate.metadata.get("source_file")
                    and doc.page_content == candidate.page_content):
                return True
        return False

    headers = []
    for fname in ordered_files:
        own = [(i, d) for i, d in enumerate(corpus)
               if d.metadata.get("source_file") == fname]
        if not own:
            continue

        marked = [(i, d) for i, d in own
                  if d.metadata.get("section_type") == HEADER_SECTION_TYPE]
        if marked:
            # Earliest in docstore order, in case a document somehow has
            # more than one chunk marked as its header.
            header = marked[0][1]
        else:
            header = min(own, key=lambda pair: _header_fallback_key(pair[1], pair[0]))[1]

        if not already_present(header):
            headers.append(header)

    # Prepended as one block so the headers keep their files' order rather
    # than reversing, which is what repeated insert(0, ...) would do.
    result = headers + docs

    trace(
        "HEADERS  done ",
        f"injected {len(headers)} header(s): "
        f"{', '.join(h.metadata.get('source_file', 'Unknown Document') for h in headers) or 'none'}",
    )
    return result


# The not-found refusals, lifted verbatim out of the two prompts below so
# the same strings can be reused outside them. The prompts interpolate
# these, so what the LLM is told to say and what the verifier substitutes
# are guaranteed to be the same text.
NOT_FOUND_MARATHI = "हे माहिती दिलेल्या दस्तऐवजात आढळली नाही."
NOT_FOUND_ENGLISH = "This information was not found in the uploaded documents."


def not_found_message(language: str) -> str:
    """The language-appropriate not-found refusal. Adds no new wording."""
    return NOT_FOUND_MARATHI if language == "marathi" else NOT_FOUND_ENGLISH


# Sent verbatim as the verification instruction.
ANSWER_VERIFY_INSTRUCTION = (
    "Check whether every factual claim in the Answer is directly supported "
    "by the Context. Reply with exactly one word: SUPPORTED or UNSUPPORTED."
)


async def verify_answer(
    user_query: str,
    context: str,
    answer: str,
    language: str,
) -> str:
    """
    Second pass that checks the generated answer against its own context.

    Grounding rules in the generation prompt are instructions, not
    guarantees — a model that has already drifted will not catch itself
    mid-answer. Asking a fresh call to compare the finished answer against
    the same context is a separate judgement, and refusing beats answering
    confidently from nothing on a corpus of government orders.

    Fails open in every direction: a timeout, an error, or any reply that is
    not UNSUPPORTED keeps the generated answer, so verification can only
    turn an ungrounded answer into a refusal, never lose a good answer to a
    flaky second call.

    Args:
        user_query : the question as the user asked it
        context    : the exact context block the answer was generated from
        answer     : the generated answer
        language   : 'english' or 'marathi', selects the refusal string

    Returns:
        answer unchanged, or the language-appropriate not-found refusal.
    """
    try:
        llm    = get_llm()          # already temperature=0
        prompt = (
            f"{ANSWER_VERIFY_INSTRUCTION}\n\n"
            f"Question: {user_query}\n\n"
            f"Context:\n{context}\n\n"
            f"Answer:\n{answer}"
        )
        trace(
            "VERIFY   start",
            f"model={settings.LLM_MODEL} timeout={settings.VERIFY_TIMEOUT:.0f}s",
        )
        # Same bounding as the rewrite call: wait_for caps this path, and
        # stream=False is what makes the client-side timeout cover the whole
        # call rather than the gap between tokens.
        response = await asyncio.wait_for(
            asyncio.to_thread(
                llm.invoke, [HumanMessage(content=prompt)], stream=False
            ),
            timeout=settings.VERIFY_TIMEOUT,
        )
        verdict = response.content.strip().upper()

    # Broad on purpose — see "fails open" above.
    except Exception as e:
        trace("VERIFY   failed", f"{type(e).__name__}: keeping the generated answer")
        return answer

    if verdict.startswith("UNSUPPORTED"):
        trace("VERIFY   done ", "UNSUPPORTED: answer replaced with the not-found refusal")
        return not_found_message(language)

    trace("VERIFY   done ", f"{verdict[:20]!r}: keeping the generated answer")
    return answer


def build_prompt(
    query: str,
    context: str,
    history_text: str,
    language: str,
) -> str:
    """
    Builds the full prompt sent to the LLM.
    Two versions — English and Marathi — with different instructions.

    The prompt has 4 parts:
        1. Role instruction  → tells LLM what it is
        2. Rules             → how to behave
        3. Conversation history → previous turns for context
        4. Retrieved context → the actual GR text chunks
        5. Current question  → what to answer right now

    Args:
        query        : current user question
        context      : concatenated FAISS chunks (the GR text)
        history_text : formatted previous conversation turns
        language     : 'english' or 'marathi'

    Returns:
        Complete prompt string ready to send to LLM
    """
    if language == "marathi":
        return f"""तुम्ही महाराष्ट्र शासन निर्णय (Government Resolution) तज्ञ आहात.

नियम:
- फक्त खालील संदर्भातून उत्तर द्या
- पूर्ण उत्तर अनेकदा वेगवेगळ्या स्रोतांत विभागलेले असते — एका स्रोतात एकूण संख्या किंवा नियम, तर दुसऱ्यात तपशील किंवा अपवाद. पहिला संबंधित स्रोत सापडल्यावर थांबू नका; उर्वरित स्रोतांतही संबंधित आकडे, नावे, तारखा किंवा अटी आहेत का ते तपासा
- उत्तरात कोणताही विशिष्ट आकडा देण्यापूर्वी, तो आकडा सांगणारे नेमके वाक्य संदर्भातून जसेच्या तसे उद्धृत करा, आणि मगच अंतिम उत्तर द्या
- या विशिष्ट प्रश्नाचे उत्तर म्हणून संदर्भात स्पष्टपणे न आलेला कोणताही आकडा सांगू नका. संदर्भात इतर आकडे असतात — बैठकीचा क्रमांक, कलम क्रमांक, वर्ष, दिनांक — ते उत्तर नाहीत. आकडा कशाची संख्या आहे ते तपासा
- ज्या आकड्यानंतर लगेच "बैठक", "बैठकीमध्ये", "सभा" किंवा "अधिवेशन" असे शब्द येतात, तो आकडा ती कितवी बैठक/सभा होती हे सांगतो — तो माणसांची, सदस्यांची, पदांची किंवा कोणत्याही वस्तूंची संख्या नाही. उदाहरणार्थ "39 व्या बैठकीमध्ये" म्हणजे 39 वी बैठक, 39 सदस्य नाही. अशा आकड्याचा वापर "किती" या प्रश्नाचे उत्तर म्हणून कधीही करू नका
- "किती" प्रश्नाचे उत्तर देताना, संख्या दाखवणारा शब्द त्याच वाक्यात ज्या गोष्टीची गणना होते तिच्या लगेच आधी असतो (उदा. "सहा सदस्यांचा", "चार निमंत्रित सदस्यांचा"). तो शब्द अंकात नसून अक्षरात असू शकतो, आणि त्या वाक्याचे स्पेलिंग OCR मुळे थोडे चुकीचे असू शकते — तरीही तेच उत्तर आहे. जवळचा, स्वच्छ लिहिलेला दुसरा आकडा निवडू नका
- मराठी मजकुरात संख्या आणि प्रमाण अनेकदा अंकांत नव्हे तर शब्दांत लिहिलेले असते — उदाहरणार्थ सहा, चार, दोन, एक, तीन, पाच, सात, आठ, नऊ, दहा. "किती" या प्रश्नाचे उत्तर देताना फक्त अंकचिन्हे शोधू नका; संदर्भातील हे मराठी संख्यावाचक शब्द सक्रियपणे शोधा. जवळ दिसणारा एखादा अंक — वर्ष, संदर्भ क्रमांक किंवा अनुक्रमांक — यापेक्षा अक्षरात लिहिलेला संख्याशब्द हेच योग्य उत्तर असण्याची शक्यता जास्त असते
- माहिती नसल्यास सांगा: "{NOT_FOUND_MARATHI}"
- उत्तर मराठीत द्या
- स्पष्ट उत्तर द्या; संक्षिप्त ठेवा, पण इतर स्रोतातील संबंधित तपशील वगळू नका

मागील संवाद:
{history_text if history_text else "नाही"}

शासन निर्णयातील संदर्भ:
{context}

प्रश्न: {query}

उत्तर:"""

    else:
        return f"""You are an expert assistant for Maharashtra Government Resolution (GR) documents.

Rules:
- Answer ONLY from the provided context below
- A complete answer is often split across sources — one may give a total or a rule, another the breakdown, exception, or detail. Do not stop at the first source that looks relevant; check the remaining sources for numbers, names, dates, or conditions that add to or qualify it
- If your answer includes a specific number, first identify and quote the exact phrase from the context that states that number, then give your final answer
- Do not state a number that does not appear explicitly in the context as the answer to this specific question. The context contains other numbers — meeting numbers, section and clause numbers, years, dates, GR reference numbers — that are NOT the answer. Check what each number counts before using it
- A number immediately followed by a word such as "बैठक", "बैठकीमध्ये", "सभा", "अधिवेशन", "meeting", or "session" identifies WHICH meeting or session it was — an ordinal. It is never a count of people, members, posts, or things. For example "39 व्या बैठकीमध्ये" means the 39th meeting, not 39 members. Never use such a number as the answer to a "how many" / "किती" question
- When answering "how many", the counting word sits immediately before the thing being counted in the same phrase (e.g. "सहा सदस्यांचा", "चार निमंत्रित सदस्यांचा"). It may be spelled out in words rather than digits, and OCR may have slightly misspelled the surrounding phrase — it is still the answer. Do not prefer a nearer, more cleanly spelled number over it
- In Marathi text, counts and quantities are frequently written as words, not digits — for example सहा (six), चार (four), दोन (two), एक (one), तीन (three), पाच (five), सात (seven), आठ (eight), नऊ (nine), दहा (ten). When answering a 'how many' question, actively look for these Marathi number-words in the context, not just digit characters, since the correct answer is more likely to be a spelled-out word than a nearby digit like a year, reference number, or serial number
- If the answer is not in the context, say exactly: "{NOT_FOUND_ENGLISH}"
- Be precise; stay brief, but never omit a relevant detail found in another source
- Use bullet points when listing multiple items
- Always refer to specific GR details when available

Previous conversation:
{history_text if history_text else "None"}

Context from GR documents:
{context}

Current question: {query}

Answer:"""


async def query(
    user_query: str,
    chat_history: list,
    top_k: int = None,
    language: str = None,
    focus_file: str = None,
) -> dict:
    """
    Main RAG pipeline — called for every chat message.

    Args:
        user_query   : the question the user typed
        chat_history : list of previous {"role", "content"} turns
        top_k        : chunks to retrieve (defaults to settings.TOP_K)
        language     : force a language, or None for auto-detect
        focus_file   : document the question is about, used only by the
                       metadata fast path; ignored by retrieval

    Returns dict:
        success      : bool
        answer       : the LLM's response text
        citations    : list of {file, page, preview}
        language     : detected or forced language
        elapsed_sec  : how long the full pipeline took
        query        : the original question (for logging)
    """
    start_time = time.time()

    if top_k is None:
        top_k = settings.TOP_K

    # ── Step 1: Detect language ───────────────────────────
    if language is None:
        language = detect_language(user_query)

    # ── Step 2: Retrieve (hybrid: dense + BM25, fused by RRF) ──
    # hybrid_search() runs FAISS and BM25 independently over the same
    # corpus and fuses the two rankings. It returns list[Document], the
    # identical shape vectorstore.search() returned, so Steps 3-4 below
    # are untouched — only the ORDER and membership of the chunks change.
    #
    # Still synchronous, and for the same reasons as before: the first
    # call loads the FAISS index from disk and builds the BM25 index, and
    # every call embeds the query via a blocking Ollama HTTP request
    # (~8s cold, ~0.3s warm). The added BM25 half needs no embedding call
    # and is CPU-only. All of it stalls the event loop, so it stays on a
    # worker thread exactly as search() did.
    #
    # Degrades rather than fails: if the dense side errors, hybrid_search
    # returns the lexical ranking alone instead of raising, so a
    # momentarily unreachable Ollama downgrades retrieval quality rather
    # than breaking the chat. Empty list still means "store not ready",
    # which the guard below handles unchanged.
    # ── Step 1a: Metadata fast path ───────────────────────
    # Returns a complete answer for a header question about one named
    # document, with no retrieval and no LLM call. Falls through on
    # anything it cannot answer unambiguously, so the normal pipeline
    # below is reached unchanged in every other case.
    #
    # Not logged here: this returns the same six-key dict the normal path
    # returns, and api/query.py logs whatever query() hands back — so the
    # fast path is already logged by the same call, and logging it here
    # too would double-count it.
    if settings.META_FASTPATH:
        fast = try_meta_fastpath(user_query, focus_file, language, start_time)
        if fast is not None:
            return fast

    # ── Step 1b: Widen the retrieval query ────────────────
    # Retrieval only. user_query stays the question everywhere else: the
    # prompt, the history and the returned "query" field all use it, so the
    # caller logs what the user actually typed.
    retrieval_query = user_query
    # Follow-up condensation (HISTORY_REWRITE): the follow-up and the
    # recent history become one standalone question, used for retrieval
    # only. With the flag off or no history, retrieval_query stays
    # user_query and everything below runs exactly as before.
    if settings.HISTORY_REWRITE and chat_history:
        retrieval_query = await asyncio.to_thread(
            condense_query, user_query, chat_history, language)
        trace("CONDENSE done", retrieval_query[:120])
    if settings.QUERY_REWRITE:
        retrieval_query = await expand_query(retrieval_query)

    #
    # hybrid_search already takes top_k, so routing needs no change there —
    # it just asks for a deeper pool. Routing has to measure which FILES
    # dominate the ranking before it can discard any, and TOP_K chunks is a
    # shortlist, not a sample. The pool is narrowed again immediately below,
    # so the LLM never sees ROUTE_POOL_K chunks.
    retrieval_k = settings.ROUTE_POOL_K if settings.ROUTE_ENABLED else top_k
    # The reranker needs a candidate pool to choose from; it narrows the
    # pool to RERANK_TOP_N right after retrieval below.
    if settings.RERANK_ENABLED:
        retrieval_k = settings.RERANK_CANDIDATES
    trace("RETRIEVAL start", f"top_k={retrieval_k} language={language} mode=hybrid")
    docs = await asyncio.to_thread(hybrid_search, retrieval_query, top_k=retrieval_k)
    retrieval_sec = round(time.time() - start_time, 2)
    trace("RETRIEVAL done ", f"{len(docs)} chunk(s) in {retrieval_sec}s")

    # If no docs returned, vector store is not ready
    if not docs:
        elapsed = round(time.time() - start_time, 2)
        no_docs_message = (
            "⚠️ माहिती संच अद्याप तयार नाही. कृपया प्रशासकाला शासन निर्णय दस्तऐवज अपलोड करण्यास सांगा."
            if language == "marathi"
            else "⚠️ Vector store is not ready. Please ask admin to embed the GR documents first."
        )
        return {
            "success":     False,
            "answer":      no_docs_message,
            "citations":   [],
            "language":    language,
            "elapsed_sec": elapsed,
            "query":       user_query,
        }

    # ── Step 2a: Rerank (RERANK_ENABLED) ─────────────────
    # Scored against the ORIGINAL question, not the condensed or expanded
    # retrieval query: those exist to find candidates, and the user's own
    # words are what the answer has to address. The kept chunks are both
    # the prompt context and the citations, since everything below reads
    # this one list. Before routing, header injection and graph expansion,
    # which then run on the reranked chunks exactly as they would on a
    # retrieved list.
    if settings.RERANK_ENABLED:
        docs = await asyncio.to_thread(rerank, user_query, docs, settings.RERANK_TOP_N)

    # ── Step 2b: Route to the dominant source files ──────
    if settings.ROUTE_ENABLED:
        docs = route_documents(docs)

    # ── Step 2c: Prepend the source files' header chunks ──
    # Runs whether or not routing did, and after it when both are on, so
    # the headers describe the files that actually survived.
    if settings.HEADER_INJECT:
        docs = inject_headers(docs)

    # ── Step 2d: Follow the GR relationship graph ────────
    # After routing and headers, so the neighbours are those of the
    # documents that actually survived into the context.
    if settings.GRAPH_EXPAND:
        docs = expand_with_graph(docs, user_query)

    # ── Step 3: Build context string ─────────────────────
    # Join all retrieved chunks into one context block
    # Each chunk separated by a divider so LLM can distinguish them
    context_parts = []
    for i, doc in enumerate(docs, 1):
        fname = doc.metadata.get("source_file", "Unknown")
        page  = doc.metadata.get("page", "?")
        # Label each chunk with its source for better LLM grounding
        context_parts.append(
            f"[Source {i}: {fname}, Page {page}]\n{doc.page_content}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # ── Step 4: Extract citations ─────────────────────────
    citations = build_citations(docs)

    # ── Step 5: Format history ────────────────────────────
    history_text = format_chat_history(
        chat_history,
        context_window=settings.CONTEXT_WINDOW,
    )

    # ── Step 6: Build prompt ──────────────────────────────
    prompt = build_prompt(
        query=user_query,
        context=context,
        history_text=history_text,
        language=language,
    )

    # ── Step 7: Call LLM ──────────────────────────────────
    try:
        llm      = get_chat_llm()
        # Marks the retrieval → generation boundary. If the terminal sits on
        # RETRIEVAL done the stall is in FAISS/embedding; if it sits here it
        # is Ollama generating (or queued behind another generation).
        trace(
            "LLM      start",
            f"model={settings.LLM_MODEL} prompt={len(prompt)} chars "
            f"timeout={settings.LLM_TIMEOUT:.0f}s",
        )
        # llm.invoke() is the SYNCHRONOUS entrypoint — ChatOllama drives
        # ollama.Client (blocking httpx), pinning the event loop for the full
        # generation. This runs directly on the request path, so a single
        # chat message froze the whole server. Offloaded to a worker thread;
        # exceptions still propagate to the except block below unchanged.
        # stream=False is what makes settings.LLM_TIMEOUT an actual ceiling.
        # ChatOllama streams even for .invoke() (chat_models.py: "stream":
        # kwargs.pop("stream", True)), and httpx's timeout is per-read — it
        # bounds the GAP BETWEEN TOKENS, not total time. A slow-but-steady
        # generation therefore never trips it and runs unbounded. Non-streaming
        # keeps the connection silent until the answer is complete, so the read
        # timeout covers the whole call. Nothing is streamed to the caller
        # today — _generate() aggregates the stream anyway — so this changes no
        # observable behaviour.
        response = await asyncio.to_thread(
            llm.invoke, [HumanMessage(content=prompt)], stream=False
        )
        answer   = response.content.strip()
        trace("LLM      done ", f"{round(time.time() - start_time, 2)}s total, {len(answer)} chars")
        log_token_usage(response)

    # Separated from the generic handler below because the cause is the
    # opposite one: Ollama is up and working, just not fast enough. Reporting
    # this as "is Ollama running?" sends you looking in the wrong place.
    except httpx.TimeoutException:
        elapsed = round(time.time() - start_time, 2)
        return {
            "success":     False,
            "answer":      (f"⚠️ LLM timed out after {settings.LLM_TIMEOUT:.0f}s "
                            f"({settings.LLM_MODEL}). The request was aborted so it "
                            f"cannot block the next one."),
            "citations":   citations,
            "language":    language,
            "elapsed_sec": elapsed,
            "query":       user_query,
        }

    except Exception as e:
        elapsed = round(time.time() - start_time, 2)
        return {
            "success":     False,
            "answer":      f"⚠️ LLM error: {str(e)}. Is Ollama running with {settings.LLM_MODEL}?",
            "citations":   citations,
            "language":    language,
            "elapsed_sec": elapsed,
            "query":       user_query,
        }

    # ── Step 7b: Verify the answer against its own context ──
    # Only the answer text can change here. The response schema, the
    # citations and success=True are all untouched — a refusal produced by
    # the model itself has always been a successful response, and one
    # produced here is the same kind of answer.
    if settings.ANSWER_VERIFY:
        answer = await verify_answer(
            user_query=user_query,
            context=context,
            answer=answer,
            language=language,
        )

    elapsed = round(time.time() - start_time, 2)

    return {
        "success":     True,
        "answer":      answer,
        "citations":   citations,
        "language":    language,
        "elapsed_sec": elapsed,
        "query":       user_query,
    }