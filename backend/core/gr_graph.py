# backend/core/gr_graph.py
# ─────────────────────────────────────────────────────────
# GR Supersession and Amendment Graph Detection.
#
# What it does:
#   Scans GR text for phrases like "supersedes GR dated..."
#   or "रद्द करण्यात येत आहे" (Marathi: "is being cancelled")
#   and builds a relationship graph between GR documents.
#
# Why it matters:
#   Without this, if GR-2019 says "10 days leave" and
#   GR-2023 supersedes it with "12 days leave", a naive
#   RAG system might return the outdated GR-2019 answer.
#   This graph lets the system know GR-2023 is the valid one.
#
# Called by:
#   api/embed.py → runs build_graph() after embedding completes
#   api/graph.py → returns graph data to frontend for visualization
# ─────────────────────────────────────────────────────────

import sys
import os
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from datetime import datetime, timezone, date

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from core.keyword_match import strip_marks
from db.mongo import get_db


# ── Pure extraction (no I/O) ──────────────────────────────
# Everything from here to build_graph_from_docs() runs on Documents that
# are already in memory. No PDF is opened, no index is loaded, no Mongo
# call is made — which is what lets the graph be rebuilt from the existing
# index without re-embedding, and tested on synthetic chunks.


# The section marker core.vectorstore stamps on the letterhead block above
# a document's first structural marker.
HEADER_SECTION_TYPE = "__header__"

# Devanagari digits fold to ASCII before any number or date is read. The
# corpus mixes both scripts inside one line, so every pattern below can
# assume ASCII digits instead of carrying a second Devanagari variant.
_DEV_DIGITS = {0x0966 + i: str(i) for i in range(10)}

# Month names mapped to month number, Marathi and English. Matched
# fuzzily (see _MONTH_RATIO) so OCR damage inside a name still resolves.
# English abbreviations are listed explicitly rather than left to the
# fuzzy match: ratio("jan", "january") is 0.6, under the threshold, so an
# abbreviation would otherwise never resolve.
_MONTH_NAMES = {
    "जानेवारी":  1,
    "फेब्रुवारी": 2,
    "मार्च":      3,
    "एप्रिल":     4,
    "मे":         5,
    "जून":        6,
    "जुलै":       7,
    "ऑगस्ट":      8,
    "सप्टेंबर":   9,
    "ऑक्टोबर":   10,
    "नोव्हेंबर": 11,
    "डिसेंबर":   12,

    "january": 1, "february": 2, "march":     3, "april":    4,
    "may":     5, "june":     6, "july":      7, "august":   8,
    "september": 9, "october": 10, "november": 11, "december": 12,

    "jan": 1, "feb": 2, "mar": 3, "apr":  4, "jun":  6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Minimum difflib ratio for a month token to resolve to a month name.
_MONTH_RATIO = 0.75

# The word that labels the issuing date in a GR letterhead.
_DATE_WORD = "दिनांक"

# dd.mm.yyyy and dd/mm/yyyy. Spaces around the separators are tolerated
# because OCR inserts them. Deliberately NOT matching dd-mm-yyyy: the
# hyphen form was not specified, and in this corpus a hyphen also joins
# reference numbers and ranges, where reading one as a date would invent
# a document date out of a serial number.
_DATE_NUMERIC = re.compile(r"(\d{1,2})\s*[./]\s*(\d{1,2})\s*[./]\s*(\d{4})")

# "dd <month name>, yyyy" — comma optional, month token resolved fuzzily.
_DATE_MONTH_NAME = re.compile(r"(\d{1,2})\s+([^\s,.;:/0-9]{2,20})\s*,?\s*(\d{4})")


def fold_digits(text: str) -> str:
    """Devanagari digits rewritten as ASCII. Everything else passes through."""
    return (text or "").translate(_DEV_DIGITS)


def _month_from_token(token: str):
    """
    Month number for a month-name token, or None.

    Fuzzy because the names arrive OCR-damaged. The best candidate wins
    only if it clears _MONTH_RATIO, so a token resembling nothing is
    rejected rather than snapped to whichever name is nearest.
    """
    probe = (token or "").strip().lower()
    if not probe:
        return None
    if probe in _MONTH_NAMES:
        return _MONTH_NAMES[probe]

    best, best_ratio = None, 0.0
    for name, number in _MONTH_NAMES.items():
        ratio = SequenceMatcher(None, probe, name, autojunk=False).ratio()
        if ratio > best_ratio:
            best, best_ratio = number, ratio
    return best if best_ratio >= _MONTH_RATIO else None


def _iso_date(day, month, year):
    """
    "YYYY-MM-DD", or None when the parts do not form a real date.

    Validated by constructing a date rather than by range-checking, so
    31.02.2026 and 45.13.2026 are both rejected instead of normalised into
    something that looks authoritative and is not.
    """
    try:
        return date(int(year), int(month), int(day)).isoformat()
    except (TypeError, ValueError):
        return None


def find_dates(text: str) -> list:
    """
    Every date in the text as YYYY-MM-DD, in order of appearance,
    de-duplicated.

    Args:
        text : raw chunk text — digits are folded here, so callers need not

    Returns:
        list[str]; candidates that do not parse are dropped.
    """
    folded = fold_digits(text or "")

    found = []
    for match in _DATE_NUMERIC.finditer(folded):
        iso = _iso_date(match.group(1), match.group(2), match.group(3))
        if iso:
            found.append((match.start(), iso))

    for match in _DATE_MONTH_NAME.finditer(folded):
        month = _month_from_token(match.group(2))
        if month is None:
            continue
        iso = _iso_date(match.group(1), month, match.group(3))
        if iso:
            found.append((match.start(), iso))

    # Position order, so "the first date" is the first one a reader sees,
    # whichever pattern caught it.
    found.sort(key=lambda pair: pair[0])

    ordered, seen = [], set()
    for _, iso in found:
        if iso not in seen:
            seen.add(iso)
            ordered.append(iso)
    return ordered


def _first_date_after_date_word(text: str):
    """
    The first date following दिनांक, or None.

    A letterhead carries several dates — the GRs it references are dated
    above the issuing date — so position alone picks the wrong one. The
    label is what disambiguates.
    """
    folded = fold_digits(text or "")
    at = folded.find(_DATE_WORD)
    if at == -1:
        return None
    after = find_dates(folded[at:])
    return after[0] if after else None


def _order_key(doc, position: int) -> tuple:
    """
    Reading order for one file's chunks: page, then the document's own
    section ordering, then the order the docstore holds them in.

    Every component tolerates absence — section_index is written only for
    section-chunked documents and section_part only for the pieces of an
    oversize section — so a fallback-chunked file still sorts instead of
    raising.
    """
    def number(field):
        value = doc.metadata.get(field, None)
        try:
            return (0, int(value))
        except (TypeError, ValueError):
            return (1, 0)

    return (number("page"), number("section_index"), number("section_part"), position)


def _group_by_source(docs: list) -> dict:
    """{source_file: chunks in reading order}, files in first-appearance order."""
    grouped = {}
    for position, doc in enumerate(docs or []):
        fname = doc.metadata.get("source_file")
        if not fname:
            continue
        grouped.setdefault(fname, []).append((position, doc))

    return {
        fname: [d for _, d in sorted(pairs, key=lambda pair: _order_key(pair[1], pair[0]))]
        for fname, pairs in grouped.items()
    }


def _split_identity(identity: str) -> tuple:
    """
    doc_identity split into (stem, gr_number, subject).

    core.vectorstore._document_identity writes "stem | number | subject"
    and degrades field by field, so a document whose letterhead could not
    be read carries fewer than three parts. Missing parts come back "".
    """
    parts = [p.strip() for p in (identity or "").split(" | ")[:3]]
    while len(parts) < 3:
        parts.append("")
    return parts[0], parts[1], parts[2]


def _header_chunks(chunks: list) -> list:
    """The chunks a document's letterhead was split into."""
    return [c for c in chunks if c.metadata.get("section_type") == HEADER_SECTION_TYPE]


def _doc_date(chunks: list):
    """
    A document's own date: the first date after दिनांक in its header
    chunks, else the first date anywhere in them, else None.

    Scoped to the header deliberately. Body text quotes other documents'
    dates constantly, and a date taken from there would make this document
    look as if it were issued on the day of the order it cites.
    """
    headers = _header_chunks(chunks)
    if not headers:
        return None

    text = "\n".join(c.page_content for c in headers)
    labelled = _first_date_after_date_word(text)
    if labelled:
        return labelled

    anywhere = find_dates(text)
    return anywhere[0] if anywhere else None


# ── Relation types ────────────────────────────────────────
# The three relations an edge can carry. refers_to is NOT used: the stored
# shape's existing value for a plain citation stays as it is below so the
# /api/graph stats keys keep counting something real.
RELATION_SUPERSEDES = "supersedes"
RELATION_AMENDS     = "amends"
RELATION_REFERENCES = "references"


# ── Reference extraction ──────────────────────────────────

# The two markers that open a reference list in a GR. Both are already in
# core.keyword_match.SECTION_KEYWORDS and so appear as section_type values
# on real chunks; they are re-stated here because a chunk's section_type
# can be an OCR-damaged variant of either, and matching is fuzzy below.
_REFERENCE_MARKERS = ("संदर्भ", "वाचा")

# Minimum difflib ratio for a section_type to count as a reference marker,
# measured after combining marks are stripped — the mark layer is what OCR
# destroys first, so the consonant skeleton is the stable comparison.
_MARKER_RATIO = 0.75

# Numbered reference items: "1)" / "(1)" / "1." at the start of a line.
_REF_ITEM = re.compile(r"(?m)^\s*\(?(\d+)[\)\.]")

# A strong key is any run of six or more ASCII digits after folding. Six
# because GR reference numbers in this corpus carry 1026/1226/9005-style
# codes and year-plus-serial runs; shorter runs are dates, clause numbers
# and page numbers, which match across unrelated documents constantly.
_STRONG_KEY = re.compile(r"\d{6,}")

# Kept when normalising a reference number for comparison: ASCII letters
# and digits, plus Devanagari letters. Punctuation, spaces and combining
# marks go, because those are exactly what differ between two renderings
# of the same number.
_NUMBER_KEEP = re.compile(r"[A-Za-z0-9ऄ-हऽॐक़-ॡ]")

# Minimum difflib ratio between two normalised numbers for the
# date-plus-number fallback to accept a candidate.
_NUMBER_RATIO = 0.6

# Characters of normalised item text used to key an unresolved node.
_EXTERNAL_ID_CHARS = 40
# Characters of raw item text used as an unresolved node's label.
_EXTERNAL_LABEL_CHARS = 80


def _skeleton(text: str) -> str:
    """Combining marks removed, for comparing OCR-damaged Devanagari."""
    stripped, _ = strip_marks(text or "")
    return stripped


def _is_reference_marker(section_type: str) -> bool:
    """
    Whether a chunk's section_type is one of the reference markers.

    Exact first, then a mark-stripped difflib ratio, so a section_type that
    OCR rendered as a near-miss of संदर्भ still opens a reference list.
    """
    probe = (section_type or "").strip()
    if not probe:
        return False
    if probe in _REFERENCE_MARKERS:
        return True

    probe_skeleton = _skeleton(probe)
    for marker in _REFERENCE_MARKERS:
        ratio = SequenceMatcher(
            None, probe_skeleton, _skeleton(marker), autojunk=False
        ).ratio()
        if ratio >= _MARKER_RATIO:
            return True
    return False


def _reference_text(chunks: list) -> str:
    """
    The text a file's references live in: its header plus every chunk whose
    section marker opens a reference list, in reading order.

    The header is included because short GRs state their references inline
    in the letterhead rather than under a संदर्भ heading.
    """
    parts = [
        c.page_content for c in chunks
        if c.metadata.get("section_type") == HEADER_SECTION_TYPE
        or _is_reference_marker(c.metadata.get("section_type"))
    ]
    return "\n".join(parts)


def normalize_number(text: str) -> str:
    """
    A reference number reduced to its comparable skeleton: digits folded,
    then only letters, digits and Devanagari letters kept.

    Two renderings of one number differ in their punctuation and spacing
    far more than in their characters, so dropping both is what makes a
    difflib ratio between them mean anything.
    """
    folded = fold_digits(text or "")
    return "".join(_NUMBER_KEEP.findall(folded))


def _split_reference_items(text: str) -> list:
    """
    A reference block split into its numbered items.

    Returns:
        list of (index, item_text). An unnumbered block is one item with
        index 1, so a GR that states a single reference in prose is treated
        the same as the first entry of a numbered list.
    """
    matches = list(_REF_ITEM.finditer(text or ""))
    if not matches:
        body = (text or "").strip()
        return [(1, body)] if body else []

    items = []
    for n, match in enumerate(matches):
        start = match.end()
        end   = matches[n + 1].start() if n + 1 < len(matches) else len(text)
        items.append((int(match.group(1)), text[start:end].strip()))
    return items


def _item_features(item_text: str) -> dict:
    """Everything a reference item can be matched on."""
    folded = fold_digits(item_text or "")
    return {
        "text":        item_text,
        "dates":       find_dates(folded),
        "strong_keys": set(_STRONG_KEY.findall(folded)),
        "number":      normalize_number(folded),
    }


def _match_against(item: dict, info: dict):
    """
    How a reference item matches one candidate, or None.

    Returns:
        "strong_key"      a six-plus-digit run shared with the candidate's
                          identity. Long enough to be effectively unique to
                          a document, so a shared one is near proof.
        "date_and_number" the candidate's own date appears in the item AND
                          their normalised numbers are similar. Needed
                          because OCR mangles the digits of the very number
                          being cited; the date is the independent check
                          that keeps a fuzzy number match honest.
        None              neither.
    """
    if item["strong_keys"] & info["keys"]:
        return "strong_key"

    if info["date"] and info["date"] in item["dates"] and item["number"] and info["number"]:
        ratio = SequenceMatcher(
            None, item["number"], info["number"], autojunk=False
        ).ratio()
        if ratio >= _NUMBER_RATIO:
            return "date_and_number"

    return None


def _describes_self(item: dict, citing_file: str, files: dict) -> bool:
    """
    Whether a reference item is the citing document describing itself.

    The reference text includes the letterhead, and a letterhead states the
    document's own number and date — so every document hands at least one
    "reference" to its own identity to the resolver. Those are not
    citations, and without this check almost every document would acquire a
    phantom unresolved neighbour built out of its own letterhead.

    Uses the candidate criteria first, then one extra test they do not
    need: whether the citing file's own number appears INSIDE the item's
    normalised number. A letterhead item carries the labels around the
    number ("दिनांक ... शासन निर्णय क्रमांक ..."), which dilutes a difflib
    ratio below the threshold even though the number is sitting there
    verbatim. Containment is only used here, never to resolve a citation,
    so the candidate matching stays exactly as specified.
    """
    info = files.get(citing_file)
    if not info:
        return False

    if _match_against(item, info) is not None:
        return True

    own = info["number"]
    return bool(
        own
        and info["date"]
        and info["date"] in item["dates"]
        and own in item["number"]
    )


# Loosest number agreement accepted, and only when the item's date
# already singles out one document. Deliberately far below
# _NUMBER_RATIO: the date is carrying the identification here, and the
# number is only being asked not to contradict it.
_UNIQUE_DATE_NUMBER_RATIO = 0.4


def _match_by_unique_date(item: dict, citing_file: str, files: dict):
    """
    The one document whose date the item names, or None.

    Requires the date to belong to exactly one candidate — never the
    citing file — and the normalised numbers to agree at least loosely.

    A GR reference is a number and a date. When OCR has destroyed the
    number badly enough that even the date_and_number threshold fails, the
    date can still be decisive, but only if no other document shares it.
    Two documents on the same date return None rather than a coin flip.
    """
    if not item["dates"] or not item["number"]:
        return None

    for item_date in item["dates"]:
        # The citing file is excluded before counting, so a document
        # quoting its own date does not look like a unique match.
        sharing = [
            fname for fname, info in files.items()
            if fname != citing_file and info["date"] and info["date"] == item_date
        ]
        if len(sharing) != 1:
            continue

        candidate = sharing[0]
        number = files[candidate]["number"]
        if not number:
            continue

        ratio = SequenceMatcher(
            None, item["number"], number, autojunk=False
        ).ratio()
        if ratio >= _UNIQUE_DATE_NUMBER_RATIO:
            return candidate

    return None


def _resolve_item(item: dict, citing_file: str, files: dict) -> tuple:
    """
    Resolves one reference item to the file or files it cites.

    Two independent ways to match, in order of how much they prove:

      strong_key      a six-plus-digit run shared with the candidate's own
                      text. These runs are long enough to be effectively
                      unique to a document, so a shared one is near proof.

      date_and_number the candidate's document date equals a date in the
                      item AND their normalised numbers are similar. Needed
                      because OCR mangles the digits of the very number
                      being cited; the date is the independent check that
                      keeps a fuzzy number match honest.

      date_unique_fuzzy_number
                      the item's date belongs to exactly ONE document in
                      the corpus, and the numbers are loosely similar. Last
                      resort, and only safe because of the uniqueness: when
                      one date identifies one document, a weak number
                      agreement is confirmation rather than evidence. A
                      date shared by two documents gets nothing, which is
                      what keeps this from guessing.

    Args:
        item        : _item_features() output
        citing_file : the file holding the reference — never a candidate
        files       : {source_file: {"date": iso|None, "number": normalised,
                       "keys": set of strong keys}}

    Returns:
        (matched_files, match_reason). matched_files is empty when nothing
        resolves; match_reason is "" then.
    """
    by_key, by_date_number = [], []

    for fname, info in files.items():
        # A document citing itself is its reference block quoting its own
        # number, not a relationship. Never a candidate.
        if fname == citing_file:
            continue

        reason = _match_against(item, info)
        if reason == "strong_key":
            by_key.append(fname)
        elif reason == "date_and_number":
            by_date_number.append(fname)

    # Strong keys win outright — a shared long digit run is better evidence
    # than any similarity score, so a date-and-number candidate never
    # competes with one.
    matched, reason = (by_key, "strong_key") if by_key else (by_date_number, "date_and_number")

    # Third path, tried only when the two above found nothing, so their
    # priority and behaviour are untouched.
    if not matched:
        unique = _match_by_unique_date(item, citing_file, files)
        if unique:
            return [unique], "date_unique_fuzzy_number"

    if not matched:
        return [], ""

    # An item that names a date and still matches several files is narrowed
    # by that date before anything is called ambiguous.
    if len(matched) > 1 and item["dates"]:
        dated = [f for f in matched if files[f]["date"] in item["dates"]]
        if dated:
            matched = dated

    return matched, reason


# ── External item filtering ───────────────────────────────
# WHY THIS EXISTS
#
# Reference text includes the letterhead and every chunk under a संदर्भ or
# वाचा marker. When one of those sections is long enough to be split, the
# BODY prose that follows it inherits the marker — and body prose is full
# of numbered paragraphs. Every one of those arrives at the resolver as a
# "reference item", and anything the corpus cannot match becomes an
# external node. On the live corpus that produced 55 external nodes from
# 19 documents, most of them sentences, one of them a digital-signature
# hash.
#
# An external node is a claim: "this document cites a document we do not
# have". These six tests are what a candidate has to survive before that
# claim is made.

# Shortest item that can plausibly identify a document. A citation names
# at minimum a type and a number or a date; anything shorter is a fragment.
_EXTERNAL_MIN_CHARS = 20

# Document-type words. An item that names none of these and carries no
# date is not identifying a document, whatever else it says.
_EXTERNAL_ROOTS = (
    "शासन",
    "परिपत्रक",
    "आदेश",
    "शुद्धीपत्रक",
    "पत्र",
    "अधिनियम",
    "नियम",
)

# A run this long of unbroken letters-and-digits is not a reference
# number; it is a signature hash, a base64 blob or an OCR smear.
_EXTERNAL_OPAQUE_RUN = 16

# Reference lists in this corpus run to a handful of entries. A double
# digit index this high means the numbering came from body paragraphs, not
# from a reference list.
_EXTERNAL_MAX_INDEX = 15

# Runs of ASCII letters/digits, used to spot an opaque token.
_OPAQUE_TOKEN = re.compile(r"[A-Za-z0-9]{%d,}" % _EXTERNAL_OPAQUE_RUN)

# Boilerplate markers. Every GR closes with a line telling the reader the
# order is available on the state portal, and that line is not a citation
# of another document — it is the same sentence in every file.
_EXTERNAL_WEB_MARKERS = ("www.", "maharashtra.gov")

# The second opaque test, measured across ALL runs rather than the longest
# one. A digital signature block arrives as several medium-length runs
# separated by OCR noise, so no single run dominates and the
# longest-run test above misses it; their combined share does not.
_EXTERNAL_RUN_FLOOR = 8
_RUN_FLOOR_TOKEN = re.compile(r"[A-Za-z0-9]{%d,}" % _EXTERNAL_RUN_FLOOR)


def _keep_external_item(item: dict, ref_index: int) -> bool:
    """
    Whether an unresolved reference item deserves an external node.

    Applies only to items nothing in the corpus matched. A resolved item is
    never tested: its target is a document that demonstrably exists, so
    whatever the item's prose looks like, the link is real.

    Args:
        item      : _item_features() output
        ref_index : the item's number within its reference list

    Returns:
        True to create the external node, False to drop the item silently.
    """
    # (d) index — cheapest test, and the one that rejects whole runs of
    # body paragraphs at once.
    if ref_index > _EXTERNAL_MAX_INDEX:
        return False

    normalized = " ".join((item["text"] or "").split())

    # (a) length
    if len(normalized) < _EXTERNAL_MIN_CHARS:
        return False

    # (b) a date or a document-type word
    if not item["dates"] and not any(root in normalized for root in _EXTERNAL_ROOTS):
        return False

    # (c) not mostly one opaque token
    folded = fold_digits(normalized)
    half = len(normalized) / 2
    for match in _OPAQUE_TOKEN.finditer(folded):
        if len(match.group(0)) > half:
            return False

    # (e) not the portal boilerplate every document closes with
    if any(marker in normalized for marker in _EXTERNAL_WEB_MARKERS):
        return False

    # (f) not mostly opaque in total. Whitespace is removed first, so an
    # OCR-shattered signature blob cannot hide behind the spaces that
    # broke its runs apart.
    squeezed = "".join(fold_digits(normalized).split())
    if squeezed:
        run_chars = sum(len(m.group(0)) for m in _RUN_FLOOR_TOKEN.finditer(squeezed))
        if run_chars > len(squeezed) / 2:
            return False

    return True


def _external_node(item: dict) -> dict:
    """
    A node for a reference that resolves to nothing in the corpus.

    Created rather than dropped: a GR citing an order nobody uploaded is a
    real statement about a real document, and showing it as unresolved is
    the difference between "no relationship" and "a relationship whose
    other end is missing".
    """
    key = (item["number"] or _skeleton(item["text"]))[:_EXTERNAL_ID_CHARS]
    label = " ".join((item["text"] or "").split())[:_EXTERNAL_LABEL_CHARS]
    return {
        "id":        "ext:" + key,
        "label":     label or "ext:" + key,
        "filename":  "",
        "gr_number": "",
        "doc_date":  item["dates"][0] if item["dates"] else None,
        "subject":   "",
        "resolved":  False,
    }


# ── Relation typing ───────────────────────────────────────

# Locates the phrase that introduces reference N in the body, so the words
# AROUND that phrase can be read. "संदर्भ क्रमांक 2" and the clipped
# "सं. क्र. 2" are both written in this corpus, hence the \S* tails.
_REF_MENTION = re.compile(r"(?:संदर्भ|सं)\S*\s*(?:क्र|क्रमांक)\S*\s*(\d+)")

# How much text around the mention is read when typing the relation.
_RELATION_WINDOW = 300
# How much of it is kept as the edge's evidence.
_EVIDENCE_CHARS = 200

# Roots, not whole words. Marathi inflects these heavily — रद्द appears as
# रद्द, रद्दबातल, रद्दची; सुधार as सुधारणा, सुधारित — so matching the root
# covers the forms without listing them.
_SUPERSEDE_ROOTS = ("रद्द", "अधिक्रमित")
_AMEND_ROOTS     = ("सुधार", "शुद्धीपत्रक")


def _relation_from_window(window: str) -> str:
    """
    The relation the words around a reference mention imply.

    Supersession is checked first: a GR that both cancels and amends in the
    same breath has cancelled, and reporting that as an amendment would
    leave the superseded document looking live.
    """
    if any(root in window for root in _SUPERSEDE_ROOTS):
        return RELATION_SUPERSEDES
    if any(root in window for root in _AMEND_ROOTS):
        return RELATION_AMENDS
    return RELATION_REFERENCES


def _relation_for_item(full_text: str, ref_index: int) -> tuple:
    """
    Types one reference by reading the text around where the citing
    document mentions it.

    A reference list says WHICH documents are cited; only the body says
    what is being done to them. The list entry for a superseded GR looks
    exactly like the entry for one merely consulted, so the relation has to
    come from the sentence that mentions "संदर्भ क्रमांक N".

    Args:
        full_text : the citing file's whole text, digits already folded
        ref_index : the reference number to look for

    Returns:
        (relation, evidence). Falls back to (references, "") when the body
        never mentions this reference by number — which is the honest
        answer: the citation is real, its purpose is unstated.
    """
    for match in _REF_MENTION.finditer(full_text or ""):
        if int(match.group(1)) != int(ref_index):
            continue

        start  = max(0, match.start() - _RELATION_WINDOW // 2)
        end    = min(len(full_text), match.end() + _RELATION_WINDOW // 2)
        window = full_text[start:end]

        return _relation_from_window(window), window[:_EVIDENCE_CHARS].strip()

    return RELATION_REFERENCES, ""


def build_graph_from_docs(docs: list, stats: dict = None) -> tuple:
    """
    Builds the GR relationship graph from chunks already in memory.

    Pure and synchronous: nothing is opened, loaded or written. The whole
    graph is derived from chunk text and metadata, which is why it can be
    rebuilt from the existing index without re-embedding.

    Args:
        docs  : LangChain Documents as held by
                core.hybrid_search.get_lexical_index().documents, each
                carrying source_file, page, section_type, section_index and
                doc_identity in its metadata.
        stats : optional dict, filled in with build counters —
                resolved_nodes, external_nodes and external_dropped (how
                many unresolved items _keep_external_item() rejected).
                An out-parameter rather than a third return value so every
                existing (nodes, edges) caller keeps working.

    Returns:
        (nodes, edges) — shapes documented at the top of this module.
    """
    grouped = _group_by_source(docs)

    # Department and header page come from the metadata extractor, which
    # reads the same chunks. Imported here, not at module scope: gr_meta
    # imports this module for its parsing helpers, so a top-level import
    # would close the cycle. Guarded because the graph is the product of
    # this function and two optional node fields must never cost it.
    try:
        from core.gr_meta import extract_metadata
        meta = extract_metadata(docs)
    except Exception as e:
        print(f"  ⚠️  gr_graph: metadata fields unavailable "
              f"({type(e).__name__}: {e}); nodes will carry None for them")
        meta = {}

    nodes = []
    for fname, chunks in grouped.items():
        identity = chunks[0].metadata.get("doc_identity", "") if chunks else ""
        stem, gr_number, subject = _split_identity(identity)

        entry = meta.get(fname) or {}

        nodes.append({
            # id is the source_file, so edges key on the same value the
            # chunks themselves carry and nothing has to translate between
            # a stem and a filename.
            "id":        fname,
            "label":     stem or Path(fname).stem,
            "filename":  fname,
            "gr_number": gr_number,
            "doc_date":  _doc_date(chunks),
            "subject":   subject,
            "resolved":  True,
            # Added fields. The graph page colours nodes by department and
            # the side panel shows the page the letterhead sits on; both
            # come from the metadata table rather than being parsed twice.
            # None when extraction found nothing, which is a real answer.
            "department":  entry.get("department"),
            "header_page": entry.get("header_page"),
        })

    # Candidate index, built once: every file's date, normalised number and
    # strong keys, so resolution is a dict walk rather than a re-scan.
    files = {}
    full_texts = {}
    for fname, chunks in grouped.items():
        node = next(n for n in nodes if n["id"] == fname)
        full_text = fold_digits("\n".join(c.page_content for c in chunks))
        full_texts[fname] = full_text

        # A candidate's strong keys come from its IDENTITY region — its
        # letterhead plus its own number — not its whole text. A document
        # quotes the numbers of every order it cites, so keys taken from
        # the body make the cited document look as though it cites its
        # citer, and every resolved edge gains a false reverse twin.
        headers = _header_chunks(chunks)
        if headers:
            identity_text = fold_digits(
                "\n".join(c.page_content for c in headers) + "\n" + node["gr_number"]
            )
        else:
            # No letterhead was detected, so there is no identity region to
            # narrow to. Falling back to the full text keeps such a file
            # reachable as a target at all, at the cost of this precision.
            identity_text = full_text

        files[fname] = {
            "date":   node["doc_date"],
            "number": normalize_number(node["gr_number"]),
            "keys":   set(_STRONG_KEY.findall(identity_text)),
        }

    edges = []
    external = {}
    external_dropped = 0

    for fname, chunks in grouped.items():
        items = _split_reference_items(_reference_text(chunks))

        for ref_index, item_text in items:
            item = _item_features(item_text)
            matched, reason = _resolve_item(item, fname, files)

            # Typed once per item, not once per target: an ambiguous item is
            # one citation whose other end is unclear, and the body says one
            # thing about it.
            relation, evidence = _relation_for_item(full_texts[fname], ref_index)

            if matched:
                ambiguous = len(matched) > 1
                for target in matched:
                    edges.append({
                        "source":       fname,
                        "target":       target,
                        "relation":     relation,
                        "ref_index":    ref_index,
                        "match_reason": reason,
                        "ambiguous":    ambiguous,
                        "evidence":     evidence,
                    })
                continue

            if _describes_self(item, fname, files):
                continue

            # Nothing in the corpus matched. Before claiming this document
            # cites one we do not have, check the item actually looks like
            # a citation — see _keep_external_item().
            if not _keep_external_item(item, ref_index):
                external_dropped += 1
                continue

            # A real citation to a document outside the corpus.
            node = _external_node(item)
            external.setdefault(node["id"], node)
            edges.append({
                "source":       fname,
                "target":       node["id"],
                "relation":     relation,
                "ref_index":    ref_index,
                "match_reason": "",
                "ambiguous":    False,
                "evidence":     evidence,
            })

    nodes.extend(external.values())

    if stats is not None:
        stats["resolved_nodes"]   = sum(1 for n in nodes if n["resolved"])
        stats["external_nodes"]   = sum(1 for n in nodes if not n["resolved"])
        stats["external_dropped"] = external_dropped

    return nodes, edges


# ── Relationship detection patterns (UNUSED) ─────────────
# NO LONGER DRIVES THE GRAPH. build_graph() used to scan PDF prose for
# these phrases and guess the target from a filename or a year; it now
# derives edges from reference lists and document numbers instead, in
# build_graph_from_docs() above. Kept because the phrase lists are the
# accumulated record of how supersession is actually worded in this
# corpus, and _relation_from_window()'s roots were drawn from them.
# Nothing calls anything in this section.
#
# Each pattern is a regex that detects a specific relationship type.
# We scan the first few pages of each GR for these phrases.

SUPERSEDES_PATTERNS = [
    # English patterns
    r"supersedes?\s+(?:the\s+)?(?:GR|Government\s+Resolution|circular|order)[\s\w\-/\.,]+?dated",
    r"in\s+supersession\s+of[\s\w\-/\.,]+",
    r"this\s+(?:order|resolution|circular)\s+supersedes",
    r"(?:hereby\s+)?cancels?\s+(?:the\s+)?(?:GR|Government\s+Resolution|circular)",
    r"stands?\s+cancelled",
    r"(?:hereby\s+)?replaces?\s+(?:the\s+)?(?:GR|Government\s+Resolution)",
    r"previous\s+(?:GR|order|circular)\s+(?:is\s+)?hereby\s+(?:cancelled|superseded)",

    # Marathi patterns — common phrases in Maharashtra GRs
    r"रद्द\s+करण्यात\s+येत",          # "is being cancelled"
    r"अधिक्रमित\s+करण्यात\s+येत",     # "is being superseded"
    r"रद्दबातल\s+करण्यात\s+येत",      # "is being annulled"
    r"निरस्त\s+करण्यात\s+येत",        # "is being revoked"
]

AMENDS_PATTERNS = [
    # English patterns
    r"amends?\s+(?:the\s+)?(?:GR|Government\s+Resolution|circular)",
    r"partial\s+(?:modification|amendment)",
    r"modified\s+to\s+read\s+as",
    r"amendment\s+to\s+(?:the\s+)?(?:GR|Government\s+Resolution)",
    r"(?:hereby\s+)?modifies?\s+(?:the\s+)?(?:GR|Government\s+Resolution)",

    # Marathi patterns
    r"दुरुस्ती\s+करण्यात\s+येत",      # "amendment is being made"
    r"सुधारणा\s+करण्यात\s+येत",       # "modification is being made"
]

REFERS_PATTERNS = [
    # English patterns
    r"(?:with\s+)?reference\s+to\s+(?:GR|Government\s+Resolution|circular)",
    r"vide\s+(?:GR|Government\s+Resolution|circular)",
    r"as\s+per\s+(?:GR|Government\s+Resolution|circular)",
    r"in\s+pursuance\s+of\s+(?:GR|Government\s+Resolution)",
    r"referred\s+to\s+in\s+(?:GR|Government\s+Resolution)",

    # Marathi patterns
    r"संदर्भ\s+क्र",                   # "Reference No."
    r"उपरोक्त\s+शासन\s+निर्णय",       # "above mentioned GR"
    r"संदर्भाधीन\s+शासन\s+निर्णय",    # "referenced GR"
]


def _detect_relationships(text: str, source_gr_id: str) -> list:
    """
    Scans text for relationship phrases and returns detected relationships.

    Args:
        text         : the GR document text (first few pages)
        source_gr_id : identifier of the GR being scanned (filename stem)

    Returns:
        List of relationship dicts:
        [
            {
                "source":       "GR_2024_transfer",
                "relation":     "supersedes",
                "target_hint":  "the matched text snippet",
                "snippet":      "...full context around the match..."
            },
            ...
        ]
    """
    relationships = []

    def scan_patterns(patterns: list, relation_type: str):
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.UNICODE)
            for match in matches:
                # Get surrounding context — 100 chars before and after match
                start   = max(0, match.start() - 50)
                end     = min(len(text), match.end() + 100)
                snippet = text[start:end].strip()

                relationships.append({
                    "source":      source_gr_id,
                    "relation":    relation_type,
                    "target_hint": match.group(0)[:120],
                    "snippet":     snippet[:300],
                })

    scan_patterns(SUPERSEDES_PATTERNS, "supersedes")
    scan_patterns(AMENDS_PATTERNS,     "amends")
    scan_patterns(REFERS_PATTERNS,     "refers_to")

    return relationships


def _match_target_to_known_gr(
    hint: str,
    all_gr_ids: list,
    current_id: str,
) -> str:
    """
    Tries to match a detected relationship hint to a known GR filename.

    For example if we detect "supersedes GR dated 15 March 2023"
    we try to find which uploaded GR file matches that date.

    Args:
        hint       : the text snippet that triggered detection
        all_gr_ids : list of all known GR filename stems
        current_id : the source GR (to avoid self-reference)

    Returns:
        Matched GR id string, or "Unknown" if no match found
    """
    hint_lower = hint.lower()

    for gr_id in all_gr_ids:
        if gr_id == current_id:
            continue   # skip self

        # Check if the GR filename or parts of it appear in the hint
        gr_lower = gr_id.lower()

        # Try matching by filename stem
        if gr_lower in hint_lower:
            return gr_id

        # Try matching by year — if GR filename contains a year
        # and the hint mentions that year, it's likely the same GR
        year_match = re.search(r'(20\d{2})', gr_id)
        if year_match:
            year = year_match.group(1)
            if year in hint_lower:
                return gr_id

    return "Unknown"


# ── Snapshot ──────────────────────────────────────────────

# Filename of the JSON snapshot written beside the index. The retrieval
# path reads this instead of Mongo: graph expansion runs inside a chat
# request, and a Motor client is bound to the loop that created it, so a
# file is the one source both the request path and an offline script can
# read without caring which loop they are on.
GRAPH_SNAPSHOT_NAME = "gr_graph.json"


def graph_snapshot_path() -> Path:
    """Where the JSON snapshot lives."""
    return Path(settings.DATA_DIR) / GRAPH_SNAPSHOT_NAME


def write_graph_snapshot(nodes: list, edges: list) -> Path:
    """
    Writes {"nodes": [...], "edges": [...]} atomically.

    Temp file in the same directory, then os.replace, which is atomic on
    both POSIX and Windows. A reader therefore sees either the previous
    snapshot or the new one, never a half-written file — and the retrieval
    path reads this on live requests, so a torn read would surface as a
    broken chat rather than a broken build.

    Returns:
        The snapshot path.
    """
    path = graph_snapshot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temp, path)
    return path


def read_graph_snapshot() -> dict:
    """
    The snapshot as {"nodes": [...], "edges": [...]}, or empty lists.

    Never raises: a missing, truncated or hand-edited file degrades to an
    empty graph, because every caller treats "no graph" as a normal state
    and none of them should fail a request over it.
    """
    path = graph_snapshot_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            "nodes": data.get("nodes", []) or [],
            "edges": data.get("edges", []) or [],
        }
    except (OSError, ValueError):
        return {"nodes": [], "edges": []}


def build_graph_from_index(stats: dict = None) -> tuple:
    """
    Builds the graph from the vector store's chunks, with no Mongo and no
    PDF access.

    Args:
        stats : optional dict, filled in with build counters — see
                build_graph_from_docs().

    Returns:
        (nodes, edges) — both empty when there is no vector store yet.
    """
    # Imported here rather than at module scope: core.hybrid_search imports
    # core.vectorstore, which imports this module's siblings, and a
    # top-level import would close that circle at startup.
    from core.hybrid_search import get_lexical_index

    index = get_lexical_index()
    if index is None or not index.documents:
        return [], []

    return build_graph_from_docs(index.documents, stats=stats)


async def build_graph() -> dict:
    """
    Main graph building function — called after embedding completes.

    Steps:
        1. Read every chunk from the vector store's lexical index
        2. Derive nodes and edges from chunk text and metadata
        3. Write the JSON snapshot atomically
        4. Replace the stored graph in MongoDB
        5. Return summary stats

    No PDF is reopened. The chunks already carry the letterhead, the
    reference lists and the body text, so rebuilding the graph needs the
    index that embedding just wrote and nothing else — which also means a
    graph rebuild never re-embeds anything.

    Returns:
        dict with success, nodes count, edges count (shape unchanged)
    """
    database = get_db()

    stats = {}
    nodes, edges = build_graph_from_index(stats=stats)

    if not nodes:
        return {
            "success": False,
            "message": "No embedded chunks found — embed the documents first.",
            "nodes":   0,
            "edges":   0,
        }

    # Snapshot first. It is what the retrieval path reads, and it is the
    # half that works without a database, so a Mongo failure below leaves a
    # usable graph on disk rather than nothing.
    snapshot = write_graph_snapshot(nodes, edges)

    # Metadata rides along on the same rebuild, since it is derived from
    # the same chunks. Guarded: the metadata table is a convenience for
    # the registry and the fast path, and the graph is the thing this
    # function exists to produce — a metadata failure must not cost the
    # caller a graph it has already computed.
    #
    # Imported here rather than at module scope: core.gr_meta imports this
    # module for its parsing helpers, so a top-level import would close
    # the cycle at startup.
    try:
        from core.gr_meta import write_meta_snapshot
        from core.hybrid_search import get_lexical_index

        index = get_lexical_index()
        if index is not None and index.documents:
            meta_path = write_meta_snapshot(index.documents)
            print(f"     metadata: {meta_path}")
    except Exception as e:
        print(f"  ⚠️  gr_graph: metadata snapshot failed "
              f"({type(e).__name__}: {e}); the graph itself is unaffected")

    resolved_nodes = [n for n in nodes if n.get("resolved")]

    # Same document id and same field names as before; the new per-node and
    # per-edge fields ride along inside the existing nodes/edges lists.
    graph_doc = {
        "nodes":     nodes,
        "edges":     edges,
        "built_at":  datetime.now(timezone.utc),
        "pdf_count": len(resolved_nodes),
    }

    await database.gr_graph.replace_one(
        {"_id": "main_graph"},              # always use same document id
        {"_id": "main_graph", **graph_doc},
        upsert=True,
    )

    matched_edges  = [e for e in edges if e.get("match_reason")]
    external_edges = [e for e in edges if not e.get("match_reason")]

    external_dropped = stats.get("external_dropped", 0)

    print(f"  🕸️ Graph built: {len(nodes)} nodes "
          f"({len(resolved_nodes)} documents, {len(nodes) - len(resolved_nodes)} external), "
          f"{len(edges)} edges ({len(matched_edges)} resolved, "
          f"{len(external_edges)} unresolved), "
          f"{external_dropped} noisy item(s) dropped")
    print(f"     snapshot: {snapshot}")

    # external_dropped rides on the BUILD result, which is what the embed
    # job reports. It is deliberately not part of the /api/graph response.
    return {
        "success":          True,
        "message":          (f"Graph built with {len(resolved_nodes)} GRs and "
                             f"{len(edges)} relationships detected."),
        "nodes":            len(nodes),
        "edges":            len(edges),
        "matched_edges":    len(matched_edges),
        "resolved_nodes":   len(resolved_nodes),
        "external_nodes":   len(nodes) - len(resolved_nodes),
        "external_dropped": external_dropped,
    }


def _read_snapshot_if_valid():
    """
    The snapshot as {"nodes": [...], "edges": [...]}, or None when the file
    is missing or unreadable.

    Deliberately stricter than read_graph_snapshot(), which flattens both
    "no file" and "a file holding an empty graph" into empty lists. The
    difference matters to get_graph(): a snapshot that exists and parses is
    the authoritative graph even when it is empty, while a missing one is a
    reason to look elsewhere.

    Requires both keys to be present AND to be lists, so a truncated write
    or a hand-edited file falls back rather than serving half a graph.
    """
    path = graph_snapshot_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"  ⚠️ gr_graph: snapshot unreadable at {path} "
              f"({type(e).__name__}: {e})")
        return None

    if not isinstance(data, dict):
        print(f"  ⚠️ gr_graph: snapshot at {path} is not a JSON object")
        return None

    nodes, edges = data.get("nodes"), data.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        print(f"  ⚠️ gr_graph: snapshot at {path} has no usable "
              f"'nodes'/'edges' lists")
        return None

    return {"nodes": nodes, "edges": edges}


async def get_graph() -> dict:
    """
    Retrieves the stored graph, from the JSON snapshot or from MongoDB.

    THE SNAPSHOT WINS. It is the only source every writer keeps current:
    scripts/build_graph.py writes it and never touches Mongo, build_graph()
    writes it before Mongo, and the retrieval path reads it. Mongo is
    written only at the end of an embedding run, so preferring it served a
    graph that could be weeks stale — and a stale graph here is worse than
    no graph, because the old node shape carries no "resolved" flag, which
    makes every real document render as an unresolved external reference
    and drops every edge.

    Mongo remains the fallback for the one case the snapshot cannot cover:
    a deployment whose data directory was replaced or never populated, but
    whose database still holds the last graph an embedding run wrote.

    Returns:
        dict with nodes list and edges list, or an empty graph when neither
        source has one. Keys are unchanged; "source" says which one
        answered — "snapshot" or "mongo".
    """
    snapshot = _read_snapshot_if_valid()
    if snapshot is not None:
        path = graph_snapshot_path()
        try:
            built_at = str(datetime.fromtimestamp(path.stat().st_mtime, timezone.utc))
        except OSError:
            built_at = ""
        return {
            "nodes":    snapshot["nodes"],
            "edges":    snapshot["edges"],
            # An empty snapshot is a real answer — the graph has not been
            # built — so "built" tracks the content, not the file.
            "built":    bool(snapshot["nodes"]),
            "built_at": built_at,
            "source":   "snapshot",
        }

    graph = None
    try:
        database = get_db()
        graph    = await database.gr_graph.find_one({"_id": "main_graph"})
    except Exception as e:
        # No snapshot and no database is the only way to reach here, and
        # an empty graph is still a usable response.
        print(f"  ⚠️ gr_graph: Mongo read failed ({type(e).__name__}: {e}); "
              f"no snapshot either, returning an empty graph")

    if graph and graph.get("nodes"):
        return {
            "nodes":    graph.get("nodes", []),
            "edges":    graph.get("edges", []),
            "built":    True,
            "built_at": str(graph.get("built_at", "")),
            "source":   "mongo",
        }

    return {"nodes": [], "edges": [], "built": False}


async def remove_from_graph(filenames: list) -> dict:
    """
    Removes deleted GRs from the stored relationship graph.

    build_graph() rewrites the whole graph from whatever PDFs are on disk,
    so it would eventually drop these on its own. That is not good enough
    here: the graph is read by api/graph.py on every citation click, and
    between a delete and the next embed it would keep offering nodes for
    documents that no longer exist and edges that resolve to nothing.

    Two different removals are needed, because a GR appears in the graph in
    two roles:

      - as a NODE, and as the SOURCE of edges it declared. Both describe a
        document that is gone, so both are dropped outright.

      - as the TARGET of some *other* GR's edge. That edge belongs to a
        document that still exists and still genuinely says "supersedes
        ...". Deleting it would erase a real statement from a surviving
        document. Instead the target is reset to "Unknown", which is the
        same value build_graph() assigns when it cannot resolve a hint —
        so the edge stays visible and honest, just unresolved.

    Edges are matched on every form a document's identity can take. Node
    ids used to be the filename STEM and are now the source FILENAME, so a
    stored graph can legitimately hold either, and a graph written before
    the change sits next to edges written after it. Matching all three
    forms — id, filename and stem — means a delete cleans up whichever
    vocabulary the stored edges happen to use, instead of silently leaving
    edges pointing at a document that no longer exists.

    Args:
        filenames : bare PDF filenames being deleted, e.g. ["GR1.3.pdf"]

    Returns:
        dict with success, nodes_removed, edges_removed, edges_unlinked
    """
    database = get_db()
    graph    = await database.gr_graph.find_one({"_id": "main_graph"})

    if not graph:
        return {
            "success":        True,
            "nodes_removed":  0,
            "edges_removed":  0,
            "edges_unlinked": 0,
            "message":        "No graph stored — nothing to remove.",
        }

    targets = set(filenames)
    stems   = {Path(name).stem for name in targets}

    old_nodes = graph.get("nodes", [])
    old_edges = graph.get("edges", [])

    # Match on filename where the node carries one, falling back to the id,
    # so nodes written before "filename" existed are still cleaned up.
    # Unchanged — this is the node rule that has always applied.
    def is_removed(node) -> bool:
        return node.get("filename") in targets or node.get("id") in stems

    nodes = [n for n in old_nodes if not is_removed(n)]

    # Every spelling of the deleted documents that an edge might use: the
    # filename (current node ids), the stem (older node ids), and the id
    # actually stored on the nodes being dropped, which covers a graph
    # written by any version including one neither form predicts.
    edge_keys = set(targets) | set(stems)
    for node in old_nodes:
        if is_removed(node) and node.get("id"):
            edge_keys.add(node["id"])

    edges          = [e for e in old_edges if e.get("source") not in edge_keys]
    edges_unlinked = 0
    for edge in edges:
        if edge.get("target") in edge_keys:
            edge["target"] = "Unknown"
            edges_unlinked += 1

    await database.gr_graph.update_one(
        {"_id": "main_graph"},
        {"$set": {
            "nodes":     nodes,
            "edges":     edges,
            # Real documents only, matching what build_graph() writes.
            # External "ext:" nodes are not documents and must not inflate
            # this; a legacy node carries no "resolved" key at all and was
            # always a real document, hence the default.
            "pdf_count": sum(1 for n in nodes if n.get("resolved", True)),
        }},
    )

    result = {
        "success":        True,
        "nodes_removed":  len(old_nodes) - len(nodes),
        "edges_removed":  len(old_edges) - len(edges),
        "edges_unlinked": edges_unlinked,
    }
    result["message"] = (
        f"Graph: removed {result['nodes_removed']} node(s) and "
        f"{result['edges_removed']} edge(s), unlinked {edges_unlinked} incoming edge(s)."
    )
    return result


async def get_relationships_for_gr(gr_id: str) -> list:
    """
    Returns all edges where this GR is source or target.
    Used when user clicks a node in the graph visualization.

    Args:
        gr_id : the GR filename stem to look up

    Returns:
        List of edge dicts related to this GR
    """
    database = get_db()
    graph    = await database.gr_graph.find_one({"_id": "main_graph"})

    if not graph:
        return []

    edges = graph.get("edges", [])

    # Return edges where this GR is either the source or target
    related = [
        e for e in edges
        if e["source"] == gr_id or e["target"] == gr_id
    ]

    return related