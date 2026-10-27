# backend/core/summary_v2.py
# ─────────────────────────────────────────────────────────
# Map-reduce document summaries with a numeric faithfulness check.
#
# Enabled by settings.SUMMARY_V2_ENABLED; with the flag off nothing here
# runs and core/summarizer.py is used exactly as before.
#
# Why map-reduce: the original summarizer truncates every document to
# 12,000 characters and asks for six sections in one call, so the tail of
# a long order is never read and the model fills sections it has no
# evidence for. Here every part of the document is read (MAP: a few facts
# per batch of chunks), and the summary is written only from those facts
# (REDUCE). The header — department, number, date — is never generated:
# it comes from gr_meta.json, which is extracted deterministically.
#
# Then every number in the summary is checked against the source text
# (core/summary_check.py). If any is missing from the source, the REDUCE
# call is repeated once with those numbers named, and the better of the
# two summaries by that check is kept.
#
# Uses the chat path's own LLM helper (core.rag.get_llm: temperature 0,
# the same model, num_ctx, keep_alive and timeout), called the same way
# the chat path calls it. The chat prompt is not touched.
# ─────────────────────────────────────────────────────────

import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from core.summary_check import check_faithfulness
from core.trace import trace

# ── Prompts (exact wording; braces filled with str.format) ─

MAP_PROMPT = (
    "You are extracting facts from part {i} of {n} of a Maharashtra Government "
    "document. Write at most 6 short bullet points, in {language_name}, listing "
    "only facts that are explicitly stated in the text below: who issued or "
    "decided what, amounts, dates, numbers, named persons and posts, references "
    "to other orders. Do not add anything that is not in the text. Copy numbers "
    "exactly as written. If the part contains no facts, write: NONE.\n\n"
    "TEXT:\n{batch}"
)

REDUCE_PROMPT = (
    "Combine the bullet points below into one summary of the document in "
    "{language_name}. Use exactly these headings: {headings}. Under each heading "
    "write short bullet points; write '{not_stated}' if the points contain "
    "nothing for that heading. Use only facts present in the bullet points. Copy "
    "numbers exactly. Do not add a date, order number or department; they are "
    "added separately.\n\n"
    "BULLET POINTS:\n{points}"
)

REPAIR_SENTENCE = (
    "These figures are not in the source and must be removed or corrected: {list}."
)

LANGUAGES = {
    "english": {
        "name": "English",
        "headings": ["Subject", "Key decisions", "Amounts and numbers",
                     "Referenced orders", "Action required"],
        "not_stated": "Not stated",
    },
    "marathi": {
        "name": "Marathi",
        "headings": ["विषय", "मुख्य निर्णय", "रक्कम व आकडे",
                     "संदर्भ आदेश", "आवश्यक कार्यवाही"],
        "not_stated": "नमूद नाही",
    },
}

# Accepted spellings for the language override.
_LANGUAGE_ALIASES = {"english": "english", "en": "english",
                     "marathi": "marathi", "mr": "marathi"}

# A document is Marathi when at least this share of its letters is Devanagari.
MARATHI_SHARE = 0.5

_DEVANAGARI_FIRST, _DEVANAGARI_LAST = chr(0x0900), chr(0x097F)

# Context size the LLM runs with. get_llm() reads settings.LLM_NUM_CTX
# itself; this is the same value, for the trace line and for older
# settings that predate the field.
DEFAULT_NUM_CTX = 8192


# ── Collaborators (module-level so tests can replace them) ─

def _store_chunks(filename: str) -> list:
    """
    Every indexed chunk of one document, in vector-store order.

    Reads the FAISS docstore through index_to_docstore_id, the same order
    the index itself uses. No embedding call: loading the store does not
    touch the embedding endpoint.
    """
    from langchain_core.documents import Document
    from core.vectorstore import load_store

    db = load_store()
    if db is None:
        return []
    chunks = []
    for i in range(db.index.ntotal):
        doc_id = db.index_to_docstore_id.get(i)
        if doc_id is None:
            continue
        doc = db.docstore.search(doc_id)
        if isinstance(doc, Document) and doc.metadata.get("source_file") == filename:
            chunks.append(doc)
    return chunks


def _load_meta() -> dict:
    """The gr_meta.json snapshot, keyed by file name ({} when absent)."""
    from core.gr_meta import load_meta
    return load_meta()


def _llm(prompt: str) -> str:
    """
    One generation through the chat path's helper, called as the chat
    path calls it: a single HumanMessage, stream=False so the configured
    timeout bounds the whole call.
    """
    from langchain_core.messages import HumanMessage
    from core.rag import get_llm

    response = get_llm().invoke([HumanMessage(content=prompt)], stream=False)
    return (response.content or "").strip()


# ── Pure helpers ──────────────────────────────────────────

def devanagari_share(text: str) -> float:
    """Share of letters (str.isalpha) in U+0900-U+097F; 0.0 with no letters."""
    letters = devanagari = 0
    for ch in (text or ""):
        if ch.isalpha():
            letters += 1
            if _DEVANAGARI_FIRST <= ch <= _DEVANAGARI_LAST:
                devanagari += 1
    return devanagari / letters if letters else 0.0


def detect_language(text: str) -> str:
    """"marathi" when the Devanagari share of letters is at least 0.5, else "english"."""
    return "marathi" if devanagari_share(text) >= MARATHI_SHARE else "english"


def order_chunks(chunks: list) -> list:
    """Page ascending, then position in the store. A missing page sorts first."""
    def key(item):
        position, doc = item
        page = doc.metadata.get("page")
        return (page if isinstance(page, int) else -1, position)
    return [doc for _, doc in sorted(enumerate(chunks), key=key)]


def build_batches(texts: list, batch_chars: int, max_batches: int) -> list:
    """
    Consecutive texts joined into batches of at most batch_chars characters.

    A text is added to the current batch unless that would take the batch
    over batch_chars; a single text longer than the limit becomes a batch
    on its own rather than being cut. When more than max_batches result,
    everything past the first max_batches - 1 is merged into the last one,
    so no text is ever dropped.
    """
    separator = "\n\n"
    batches, current = [], ""
    for text in texts:
        if not text:
            continue
        candidate = f"{current}{separator}{text}" if current else text
        if current and len(candidate) > batch_chars:
            batches.append(current)
            current = text
        else:
            current = candidate
    if current:
        batches.append(current)

    if max_batches >= 1 and len(batches) > max_batches:
        head = batches[:max_batches - 1]
        tail = separator.join(batches[max_batches - 1:])
        batches = head + [tail]
    return batches


def is_none_output(text: str) -> bool:
    """
    Whether a MAP output is the "no facts" answer.

    Tolerates the decoration a model adds around a one-word answer — a
    bullet, quotes, a full stop, surrounding whitespace — but nothing more:
    "NONE of the amounts are stated" is a fact and is kept.
    """
    return (text or "").strip().strip("-*•\"'`").strip().rstrip(".").strip().upper() == "NONE"


def _resolve_language(language) -> str:
    if language is None:
        return None
    resolved = _LANGUAGE_ALIASES.get(str(language).strip().lower())
    if resolved is None:
        raise ValueError(f"unsupported summary language {language!r}; use english or marathi")
    return resolved


def build_header(filename: str, meta: dict, not_stated: str) -> dict:
    """
    Department, number and date for one file, from the metadata snapshot.
    No LLM call. A field that is missing or empty becomes not_stated.
    """
    entry = (meta or {}).get(filename) or {}

    def field(*keys):
        for key in keys:
            value = entry.get(key)
            if value not in (None, ""):
                return value
        return not_stated

    return {
        "department": field("department"),
        "number":     field("gr_number"),
        "date":       field("doc_date"),
    }


def _all_not_stated(spec: dict) -> str:
    return "\n\n".join(f"{heading}\n- {spec['not_stated']}" for heading in spec["headings"])


# ── Pipeline ──────────────────────────────────────────────

def summarize_document(filename: str, language: str | None = None) -> dict:
    """
    Map-reduce summary of one indexed document, with a faithfulness check.

    Args:
        filename : the document's source_file, e.g. "GR5.pdf"
        language : "english"/"en" or "marathi"/"mr" to force the OUTPUT
                   language; None writes in the document's own language.

    Returns:
        {"header": {department, number, date}, "summary": text,
         "faithfulness": {score, checked, unsupported}, "seconds": float,
         "batches": int, "language": "english" | "marathi"}

    Raises:
        ValueError when the document has no indexed chunks, or for an
        unsupported language.
    """
    started = time.monotonic()
    forced = _resolve_language(language)

    chunks = order_chunks(_store_chunks(filename))
    texts = [doc.page_content for doc in chunks if doc.page_content]
    if not texts:
        raise ValueError(f"no indexed chunks for {filename!r}")
    source_text = "\n\n".join(texts)

    output_language = forced or detect_language(source_text)
    spec = LANGUAGES[output_language]
    header = build_header(filename, _load_meta(), spec["not_stated"])

    batches = build_batches(texts, settings.SUMMARY_BATCH_CHARS, settings.SUMMARY_MAX_BATCHES)
    num_ctx = getattr(settings, "LLM_NUM_CTX", DEFAULT_NUM_CTX)
    trace("SUMMARY  start ",
          f"{filename} chunks={len(chunks)} batches={len(batches)} "
          f"language={output_language} num_ctx={num_ctx}")

    # ── MAP ───────────────────────────────────────────────
    points = []
    for i, batch in enumerate(batches, 1):
        output = _llm(MAP_PROMPT.format(i=i, n=len(batches),
                                        language_name=spec["name"], batch=batch))
        trace("SUMMARY  map   ", f"{filename} part {i}/{len(batches)} {len(output)} chars")
        if not is_none_output(output):
            points.append(output.strip())

    # ── REDUCE (+ one repair) ─────────────────────────────
    if not points:
        # Every part answered NONE: there are no facts to combine, so the
        # summary is every heading marked not stated, with no LLM call.
        summary = _all_not_stated(spec)
        check = check_faithfulness(summary, source_text)
    else:
        reduce_prompt = REDUCE_PROMPT.format(
            language_name=spec["name"],
            headings="; ".join(spec["headings"]),
            not_stated=spec["not_stated"],
            points="\n".join(points),
        )
        summary = _llm(reduce_prompt)
        check = check_faithfulness(summary, source_text)

        if check["unsupported"]:
            trace("SUMMARY  repair", f"{filename} unsupported={check['unsupported']}")
            repair_prompt = (reduce_prompt + "\n\n"
                             + REPAIR_SENTENCE.format(list=", ".join(check["unsupported"])))
            repaired = _llm(repair_prompt)
            repaired_check = check_faithfulness(repaired, source_text)
            # Strictly better only: on a tie the first summary stands.
            if repaired_check["score"] > check["score"]:
                summary, check = repaired, repaired_check

    seconds = round(time.monotonic() - started, 2)
    trace("SUMMARY  done  ",
          f"{filename} {seconds}s faithfulness={check['score']:.2f} "
          f"({check['checked']} checked)")
    return {
        "header":       header,
        "summary":      summary,
        "faithfulness": check,
        "seconds":      seconds,
        "batches":      len(batches),
        "language":     output_language,
    }


# ── Route adapter ─────────────────────────────────────────

async def process_gr_v2(pdf_path: str, progress_callback=None) -> dict:
    """
    Drop-in for core.summarizer.process_gr() when SUMMARY_V2_ENABLED is on.

    Returns every key process_gr() returns — success, message, metadata,
    summary, json_path, txt_path — filled from summarize_document(), plus
    the new keys header and faithfulness. Writes the same
    <stem>_summary.json and <stem>_summary.txt files, so the list and
    download routes and the frontend keep working unchanged; the JSON gains
    header, faithfulness and pipeline fields.

    metadata keeps the old keys. Values come from gr_meta.json instead of
    an LLM call; a value the snapshot does not have is null, as the old
    extractor reported it.
    """
    import asyncio
    import json
    import traceback
    from datetime import datetime, timezone
    from pathlib import Path

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return {"success": False, "message": f"File not found: {pdf_path.name}",
                "metadata": {}, "summary": ""}

    try:
        if progress_callback:
            progress_callback("Generating summary (map-reduce)...")
        result = await asyncio.to_thread(summarize_document, pdf_path.name, None)

        entry = _load_meta().get(pdf_path.name) or {}
        metadata = {
            "gr_number":         entry.get("gr_number") or None,
            "department":        entry.get("department") or None,
            "issue_date":        entry.get("doc_date") or None,
            "subject":           entry.get("subject") or None,
            "applicable_region": None,
            "signatory":         None,
        }

        if progress_callback:
            progress_callback("Saving report...")
        settings.SUMMARIES_PATH.mkdir(parents=True, exist_ok=True)
        base_name = pdf_path.stem
        result_data = {
            "filename":     pdf_path.name,
            "processed_at": str(datetime.now(timezone.utc)),
            "metadata":     metadata,
            "summary":      result["summary"],
            "header":       result["header"],
            "faithfulness": result["faithfulness"],
            "pipeline": {
                "version":  2,
                "batches":  result["batches"],
                "language": result["language"],
                "seconds":  result["seconds"],
            },
        }
        json_path = settings.SUMMARIES_PATH / f"{base_name}_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)

        header = result["header"]
        check = result["faithfulness"]
        txt_path = settings.SUMMARIES_PATH / f"{base_name}_summary.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"GR DOCUMENT: {pdf_path.name}\n")
            f.write(f"Processed: {result_data['processed_at']}\n")
            f.write("=" * 60 + "\n\n")
            f.write("METADATA\n")
            f.write("-" * 40 + "\n")
            f.write(json.dumps(metadata, indent=2, ensure_ascii=False))
            f.write("\n\n" + "=" * 60 + "\n\n")
            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"{header['department']}\n{header['number']}\n{header['date']}\n\n")
            f.write(result["summary"])
            f.write("\n\n" + "=" * 60 + "\n\n")
            f.write("FAITHFULNESS (numbers only)\n")
            f.write("-" * 40 + "\n")
            f.write(f"score {check['score']:.2f}, {check['checked']} checked, "
                    f"unsupported: {', '.join(check['unsupported']) or 'none'}\n")

        return {
            "success":      True,
            "message":      "Summary generated successfully.",
            "metadata":     metadata,
            "summary":      result["summary"],
            "json_path":    str(json_path),
            "txt_path":     str(txt_path),
            "header":       header,
            "faithfulness": check,
        }

    except Exception as e:
        print(f"\n{'=' * 60}", flush=True)
        print(f"process_gr_v2 FAILED for {pdf_path.name}", flush=True)
        print(f"  {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        print(f"{'=' * 60}\n", flush=True)
        return {
            "success":  False,
            "message":  f"Summary generation failed: {type(e).__name__}: {e}",
            "metadata": {},
            "summary":  "",
        }
