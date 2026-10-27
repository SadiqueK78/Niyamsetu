# backend/core/gr_meta.py
# ─────────────────────────────────────────────────────────
# Structured per-document metadata, derived from chunk text.
#
# What this is for:
#   A GR's letterhead states four things that identify it — its number,
#   its date, the department that issued it, and its subject. Those facts
#   are not opinions and do not need a language model: they are at a known
#   place in a known layout. Reading them once into a table makes them
#   answerable directly, listable in a registry, and usable as node
#   attributes in the relationship graph.
#
# Relationship to core/gr_graph.py:
#   The graph owns the parsing rules. Every helper used here — digit
#   folding, the date regexes, the दिनांक rule, the doc_identity split —
#   is imported from gr_graph rather than reimplemented, so a document's
#   date in the registry can never disagree with its date in the graph.
#   The one thing added here is doc_date_raw, which the graph has no use
#   for: it keeps the date exactly as the document spells it, next to the
#   normalised form.
#
# Pure, except for the snapshot functions at the bottom. extract_metadata
# opens nothing, calls nothing and re-embeds nothing — it reads the chunks
# the index already holds.
# ─────────────────────────────────────────────────────────

import json
import os
import sys
import threading
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from core.gr_graph import (
    HEADER_SECTION_TYPE,
    _DATE_MONTH_NAME,
    _DATE_NUMERIC,
    _DATE_WORD,
    _group_by_source,
    _header_chunks,
    _iso_date,
    _month_from_token,
    _order_key,
    _split_identity,
    fold_digits,
)


# ── Department extraction ─────────────────────────────────

# The word a department line is built around.
_DEPARTMENT_ROOT = "विभाग"

# Roots that disqualify a line. A letterhead line carrying any of these is
# the number line, the subject line, or a reference entry — all of which
# can also contain विभाग, and none of which is the issuing department.
_DEPARTMENT_EXCLUDE = (
    "शासन निर्णय",
    "परिपत्रक",
    "आदेश",
    "क्रमांक",
    "संदर्भ",
    "वाचा",
)

# A department name is a few words. Anything longer is a sentence that
# happens to mention a department.
_DEPARTMENT_MAX_CHARS = 80

# The boilerplate line above the department in every Maharashtra GR. When
# OCR runs the two together, this is the prefix to drop.
_STATE_PREFIX = "महाराष्ट्र शासन"

# Stripped from both ends of a department line: whitespace plus the
# punctuation that letterheads decorate lines with. Devanagari danda and
# the ASCII set are both included.
_DEPARTMENT_TRIM = " \t\r\n,.;:|-–—_()[]{}\"'/\\*०।॥"

# How much of the header is kept verbatim on each entry.
_HEADER_PREVIEW_CHARS = 400

# Snapshot filename, beside the graph's own snapshot.
META_SNAPSHOT_NAME = "gr_meta.json"


def _header_text_and_page(chunks: list) -> tuple:
    """
    A document's letterhead text, the page it sits on, and whether it had
    to be guessed.

    Returns:
        (text, page, fallback) — fallback is True when the document has no
        chunk marked __header__ and the smallest-page chunks were used
        instead. A fallback-chunked document (OCR failure, no detectable
        section markers) still has a first page, and that first page is
        still where its letterhead is.
    """
    headers = _header_chunks(chunks)
    fallback = False

    if not headers:
        fallback = True
        pages = []
        for position, doc in enumerate(chunks):
            try:
                pages.append(int(doc.metadata.get("page")))
            except (TypeError, ValueError):
                continue
        if pages:
            smallest = min(pages)
            headers = [c for c in chunks
                       if str(c.metadata.get("page")) == str(smallest)]
        else:
            # No usable page numbers at all — the opening chunk is the
            # best available guess, and chunks arrive in reading order.
            headers = chunks[:1]

    if not headers:
        return "", None, fallback

    text = "\n".join(c.page_content for c in headers)

    page = None
    try:
        page = int(headers[0].metadata.get("page"))
    except (TypeError, ValueError):
        page = headers[0].metadata.get("page")

    return text, page, fallback


def extract_department(header_text: str) -> tuple:
    """
    The issuing department, read off the letterhead.

    Returns:
        (department, department_raw) — the cleaned name and the line it
        came from, or (None, None) when no line qualifies.

    The rule is positional, not semantic: the first line that is built
    around विभाग, carries none of the words that mark a different kind of
    line, and is short enough to be a name rather than a sentence.
    """
    for raw_line in (header_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _DEPARTMENT_ROOT not in line:
            continue
        if any(bad in line for bad in _DEPARTMENT_EXCLUDE):
            continue
        if len(line) > _DEPARTMENT_MAX_CHARS:
            continue

        cleaned = line
        if cleaned.startswith(_STATE_PREFIX):
            cleaned = cleaned[len(_STATE_PREFIX):]
        cleaned = cleaned.strip(_DEPARTMENT_TRIM).strip()

        if not cleaned:
            continue
        return cleaned, line

    return None, None


# ── Dates, with the raw substring kept ────────────────────

def find_dates_with_raw(text: str) -> list:
    """
    Every date in the text as (iso, raw), in order of appearance.

    Identical rules to gr_graph.find_dates() — the same two regexes, the
    same fuzzy month resolution, the same date validation — with the
    matched substring carried along.

    The raw substring is sliced out of the ORIGINAL text, not the folded
    copy the regexes ran on. fold_digits() is a one-to-one str.translate,
    so every offset in the folded string indexes the same character in the
    original, and a document that wrote its date in Devanagari digits gets
    those digits back here while doc_date stays ASCII and sortable.
    """
    source = text or ""
    folded = fold_digits(source)

    found = []
    for match in _DATE_NUMERIC.finditer(folded):
        iso = _iso_date(match.group(1), match.group(2), match.group(3))
        if iso:
            found.append((match.start(), iso, source[match.start():match.end()]))

    for match in _DATE_MONTH_NAME.finditer(folded):
        month = _month_from_token(match.group(2))
        if month is None:
            continue
        iso = _iso_date(match.group(1), month, match.group(3))
        if iso:
            found.append((match.start(), iso, source[match.start():match.end()]))

    found.sort(key=lambda triple: triple[0])

    ordered, seen = [], set()
    for _, iso, raw in found:
        if iso not in seen:
            seen.add(iso)
            ordered.append((iso, raw))
    return ordered


def header_date(header_text: str) -> tuple:
    """
    The document's own date as (iso, raw), or (None, None).

    Same precedence the graph uses: the first date after दिनांक, else the
    first date anywhere in the header. A letterhead lists the dates of the
    orders it references above its own, so position alone picks the wrong
    one and the label is what disambiguates.
    """
    source = header_text or ""
    folded = fold_digits(source)

    at = folded.find(_DATE_WORD)
    if at != -1:
        after = find_dates_with_raw(source[at:])
        if after:
            return after[0]

    anywhere = find_dates_with_raw(source)
    return anywhere[0] if anywhere else (None, None)


# ── Extraction ────────────────────────────────────────────

def extract_metadata(docs: list) -> dict:
    """
    Per-document metadata for every source_file in a chunk list.

    Pure and synchronous: nothing is opened, loaded, embedded or written.

    Args:
        docs : LangChain Documents as held by
               core.hybrid_search.get_lexical_index().documents

    Returns:
        {source_file: {
            source_file, stem, gr_number, subject,
            doc_date, doc_date_raw,
            department, department_raw,
            header_page, header_fallback, header_chunk_text,
        }}
    """
    grouped = _group_by_source(docs)

    meta = {}
    for fname, chunks in grouped.items():
        identity = chunks[0].metadata.get("doc_identity", "") if chunks else ""
        stem, gr_number, subject = _split_identity(identity)

        header_text, header_page, fallback = _header_text_and_page(chunks)
        doc_date, doc_date_raw = header_date(header_text)
        department, department_raw = extract_department(header_text)

        meta[fname] = {
            "source_file":       fname,
            "stem":              stem or Path(fname).stem,
            "gr_number":         gr_number,
            "subject":           subject,
            "doc_date":          doc_date,
            "doc_date_raw":      doc_date_raw,
            "department":        department,
            "department_raw":    department_raw,
            "header_page":       header_page,
            "header_fallback":   fallback,
            "header_chunk_text": header_text[:_HEADER_PREVIEW_CHARS],
        }

    return meta


def header_chunk_for(chunks: list):
    """
    The one chunk that best represents a document's letterhead.

    Prefers a chunk marked __header__, falling back to the smallest page.
    Used where a single Document is needed rather than text — building a
    citation for a metadata answer, for instance.

    Returns:
        a Document, or None when chunks is empty.
    """
    if not chunks:
        return None

    headers = _header_chunks(chunks)
    if headers:
        return headers[0]

    return min(
        chunks,
        key=lambda doc: _order_key(doc, chunks.index(doc)),
    )


# ── Snapshot ──────────────────────────────────────────────

def meta_snapshot_path() -> Path:
    """Where the metadata snapshot lives."""
    return Path(settings.DATA_DIR) / META_SNAPSHOT_NAME


def write_meta_snapshot(docs: list) -> Path:
    """
    Extracts the metadata and writes it atomically.

    Temp file in the same directory then os.replace, which is atomic on
    both POSIX and Windows, so a reader sees either the old snapshot or
    the new one and never a half-written file.

    Returns:
        the snapshot path.
    """
    meta = extract_metadata(docs)

    path = meta_snapshot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temp, path)
    return path


# Re-read only when the file's mtime or size changes, so the chat path
# pays one stat() call per question. Keyed on both because two writes
# inside one filesystem timestamp tick are possible.
_cache_lock = threading.Lock()
_cached_meta = None
_cached_key = None


def load_meta() -> dict:
    """
    The metadata snapshot, or {} when there is none.

    Never raises: a missing, truncated or hand-edited file degrades to an
    empty dict, because every caller treats "no metadata" as a normal
    state and none of them should fail a request over it.
    """
    global _cached_meta, _cached_key

    path = meta_snapshot_path()
    try:
        stat = path.stat()
    except OSError:
        return {}

    key = (stat.st_mtime_ns, stat.st_size)
    with _cache_lock:
        if _cached_meta is not None and _cached_key == key:
            return _cached_meta

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    if not isinstance(data, dict):
        return {}

    with _cache_lock:
        _cached_meta, _cached_key = data, key
    return data


def clear_meta_cache() -> None:
    """Drops the cached snapshot. For tests that rewrite the file."""
    global _cached_meta, _cached_key
    with _cache_lock:
        _cached_meta, _cached_key = None, None
