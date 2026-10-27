#!/usr/bin/env python
# backend/tests/eval_summaries.py
# ─────────────────────────────────────────────────────────
# Runs the map-reduce summarizer (core/summary_v2.py) in process over the
# indexed documents and records how each one went: time, batches, the
# numeric faithfulness check, and whether the deterministic header found
# its number and date in the stored metadata.
#
# NOT a grade. The faithfulness check only asks whether every number in a
# summary also occurs in the source; a summary can pass it and still be
# wrong, vague or incomplete. Read a sample by hand before reporting.
#
# Calls the LLM (through core.rag.get_llm, as the chat path does) and
# reads the vector store and data/gr_meta.json. Needs the LLM endpoint up;
# needs no running backend server. SUMMARY_V2_ENABLED is forced on in
# memory for this run only — .env is not touched.
#
# Usage:
#   python tests/eval_summaries.py
#   python tests/eval_summaries.py --files "GR5.pdf,aadesh 1.pdf"
#   python tests/eval_summaries.py --language english
#
# Writes:
#   tests/summary_eval_<timestamp>.md
#   tests/summary_eval_<timestamp>.json
# ─────────────────────────────────────────────────────────

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.append(str(TESTS_DIR.parent))

from config import settings          # noqa: E402
import core.summary_v2 as summary_v2  # noqa: E402

NOTE = "faithfulness checks numbers only; read a sample by hand before reporting"


def header_in_meta(header: dict, entry: dict) -> dict:
    """
    Whether the header's number and date are the stored metadata values.

    The header is built from gr_meta.json, so a False here means the
    snapshot has no value for that field and the header fell back to the
    not-stated word.
    """
    number, date = entry.get("gr_number"), entry.get("doc_date")
    return {
        "header_number_in_meta": bool(number) and header.get("number") == number,
        "header_date_in_meta":   bool(date) and header.get("date") == date,
    }


def evaluate(files: list, meta: dict, language=None) -> list:
    """One row per file. A file that raises is recorded with its error."""
    rows = []
    for n, filename in enumerate(files, 1):
        print(f"[{n}/{len(files)}] {filename} ... ", end="", flush=True)
        try:
            result = summary_v2.summarize_document(filename, language)
        except Exception as e:
            print(f"ERROR {type(e).__name__}: {e}")
            rows.append({"file": filename, "error": f"{type(e).__name__}: {e}"})
            continue
        check = result["faithfulness"]
        row = {
            "file":         filename,
            "error":        "",
            "seconds":      result["seconds"],
            "batches":      result["batches"],
            "language":     result["language"],
            "score":        round(check["score"], 3),
            "checked":      check["checked"],
            "unsupported":  check["unsupported"],
            "header":       result["header"],
            "summary":      result["summary"],
        }
        row.update(header_in_meta(result["header"], meta.get(filename) or {}))
        rows.append(row)
        print(f"{row['seconds']}s batches={row['batches']} score={row['score']:.2f} "
              f"({row['checked']} checked, {len(row['unsupported'])} unsupported)")
    return rows


def averages(rows: list) -> dict:
    ok = [r for r in rows if not r["error"]]

    def mean(key):
        return round(sum(r[key] for r in ok) / len(ok), 3) if ok else None

    return {
        "files":               len(rows),
        "errors":              len(rows) - len(ok),
        "seconds":             mean("seconds"),
        "batches":             mean("batches"),
        "score":               mean("score"),
        "checked":             mean("checked"),
        "header_number_in_meta": sum(1 for r in ok if r["header_number_in_meta"]),
        "header_date_in_meta":   sum(1 for r in ok if r["header_date_in_meta"]),
        "with_unsupported":    sum(1 for r in ok if r["unsupported"]),
    }


def _cell(value) -> str:
    return " ".join(str(value).split()).replace("|", "\\|")


def report(rows: list, avg: dict, run_at: str) -> str:
    yn = lambda v: "Y" if v else "n"
    lines = [
        "# Summary evaluation (map-reduce)",
        "",
        f"- run at: {run_at}",
        f"- SUMMARY_BATCH_CHARS: {settings.SUMMARY_BATCH_CHARS}, "
        f"SUMMARY_MAX_BATCHES: {settings.SUMMARY_MAX_BATCHES}, LLM: {settings.LLM_MODEL}",
        "",
        f"Note: {NOTE}.",
        "",
        "| file | seconds | batches | faithfulness | checked | unsupported tokens "
        "| number in meta | date in meta |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["error"]:
            lines.append(f"| {_cell(r['file'])} | ERROR: {_cell(r['error'])} |  |  |  |  |  |  |")
            continue
        lines.append("| " + " | ".join([
            _cell(r["file"]), f"{r['seconds']:.1f}", str(r["batches"]), f"{r['score']:.2f}",
            str(r["checked"]), _cell(", ".join(r["unsupported"]) or "-"),
            yn(r["header_number_in_meta"]), yn(r["header_date_in_meta"]),
        ]) + " |")
    ok = avg["files"] - avg["errors"]
    lines += [
        "",
        "## Averages",
        "",
        f"- files: {avg['files']} ({avg['errors']} errors)",
        f"- seconds: {avg['seconds']}",
        f"- batches: {avg['batches']}",
        f"- faithfulness score: {avg['score']}",
        f"- tokens checked: {avg['checked']}",
        f"- files with unsupported tokens: {avg['with_unsupported']}/{ok}",
        f"- header number in metadata: {avg['header_number_in_meta']}/{ok}",
        f"- header date in metadata: {avg['header_date_in_meta']}/{ok}",
        "",
    ]
    return "\n".join(lines)


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description="Evaluate map-reduce summaries in process.")
    ap.add_argument("--files", default=None,
                    help="comma-separated file names (default: every file in gr_meta.json)")
    ap.add_argument("--language", default=None,
                    help="force the output language: english or marathi")
    ap.add_argument("--out-dir", type=Path, default=TESTS_DIR)
    args = ap.parse_args(argv[1:])

    # Forced on for this process only; the setting is never written back.
    settings.SUMMARY_V2_ENABLED = True

    meta = summary_v2._load_meta()
    if args.files:
        files = [f.strip() for f in args.files.split(",") if f.strip()]
    else:
        files = sorted(meta)
    if not files:
        print("no files: gr_meta.json is empty or missing, and --files was not given")
        return 1

    run_at = datetime.now(timezone.utc)
    rows = evaluate(files, meta, args.language)
    avg = averages(rows)

    stamp = run_at.strftime("%Y%m%d_%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.out_dir / f"summary_eval_{stamp}.md"
    json_path = args.out_dir / f"summary_eval_{stamp}.json"
    text = report(rows, avg, run_at.isoformat())
    md_path.write_text(text, encoding="utf-8")
    json_path.write_text(json.dumps({
        "run_at":   run_at.isoformat(),
        "note":     NOTE,
        "settings": {"SUMMARY_BATCH_CHARS": settings.SUMMARY_BATCH_CHARS,
                     "SUMMARY_MAX_BATCHES": settings.SUMMARY_MAX_BATCHES,
                     "LLM_MODEL": settings.LLM_MODEL},
        "averages": avg,
        "results":  rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(text)
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    # Marathi file names and summaries; a cp1252 console would raise.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main(sys.argv))
