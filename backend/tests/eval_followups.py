#!/usr/bin/env python
# backend/tests/eval_followups.py
# ─────────────────────────────────────────────────────────
# Follow-up question evaluation against the LIVE POST /api/query/chat.
#
# Each gold conversation has two turns. Turn 2 depends on turn 1 — "who
# signed it?", "त्यासाठी किती खर्च मंजूर झाला?" — and cannot be answered
# on its own. For every conversation this:
#
#   1. sends turn 1 with an empty history
#   2. sends turn 2 WITH the history the chat page would send: turn 1's
#      question and the system's answer, as
#      [{"role": "user", ...}, {"role": "assistant", ...}]
#   3. sends turn 2 again with an EMPTY history, as a baseline
#
# and compares where turn 2's citations point in the two cases. The
# difference is what the conversation history buys retrieval.
#
# Answers that are infrastructure failures (ERR_NGROK, WinError, "Server
# disconnected", "LLM error") say nothing about follow-up handling and are
# left out of the rates; how many were left out is printed.
#
# Not a grade: the .md sheet has an empty Correct column for a human.
#
# Requires the backend running: uvicorn main:app
#
# Usage:
#   python tests/eval_followups.py --tag rewrite_off
#   python tests/eval_followups.py --gold tests/gold_followups.json --tag rewrite_on
#
# Writes:
#   tests/followups_<tag>_<timestamp>.json
#   tests/followups_<tag>_<timestamp>.md
# ─────────────────────────────────────────────────────────

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))

import run_harness as rh   # noqa: E402  — login, ask, citation helpers

# Substrings that mark an answer (or error text) as an infrastructure
# failure rather than something the system said about the question.
INFRA_MARKERS = ("ERR_NGROK", "WinError", "Server disconnected", "LLM error")

DEFAULT_TIMEOUT = 300.0


# ── Requests ──────────────────────────────────────────────

def build_history(question: str, outcome: dict) -> list:
    """
    The history the chat page sends after one exchange.

    Mirrors frontend/src/context/ChatContext.jsx completeExchange(): the
    user's question and the assistant's answer, roles "user" and
    "assistant". A failed turn adds nothing — the page leaves history
    untouched on an error — so a failed turn 1 yields [].
    """
    if outcome.get("error"):
        return []
    return [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": outcome.get("answer", "")},
    ]


def ask_with_history(client, base_url: str, headers: dict, query: str,
                     history: list, timeout: float) -> dict:
    """
    One /api/query/chat request carrying a history.

    run_harness.ask() always sends history=[], so a request WITH history
    needs this. The payload is the chat page's — {query, history} — and
    the outcome has exactly ask()'s shape and failure handling. With an
    empty history this delegates to ask() itself.
    """
    if not history:
        return rh.ask(client, base_url, headers, {"question": query},
                      top_k=None, language=None, timeout=timeout)

    started = time.time()
    try:
        res = client.post(f"{base_url}/api/query/chat",
                          json={"query": query, "history": history},
                          headers=headers, timeout=timeout)
    except httpx.RequestError as e:
        return {"http_status": None, "error": f"{type(e).__name__}: {e}",
                "answer": "", "citations": [], "language": "", "success": False,
                "server_elapsed_sec": None,
                "client_elapsed_sec": round(time.time() - started, 2)}

    client_elapsed = round(time.time() - started, 2)
    if res.status_code != 200:
        return {"http_status": res.status_code, "error": res.text[:500],
                "answer": "", "citations": [], "language": "", "success": False,
                "server_elapsed_sec": None, "client_elapsed_sec": client_elapsed}

    body = res.json()
    return {"http_status": 200, "error": "",
            "answer": body.get("answer", ""), "citations": body.get("citations", []),
            "language": body.get("language", ""), "success": body.get("success", False),
            "server_elapsed_sec": body.get("elapsed_sec"),
            "client_elapsed_sec": client_elapsed}


def is_infra_failure(outcome: dict) -> bool:
    """Whether the answer or error text carries an infrastructure marker."""
    text = f"{outcome.get('answer') or ''}\n{outcome.get('error') or ''}"
    return any(marker in text for marker in INFRA_MARKERS)


def turn_record(outcome: dict, expected_file=None, expected_page=None) -> dict:
    """What one request returned, plus its citation checks."""
    pairs = rh.cited_pairs(outcome.get("citations"))
    record = {
        "answer":        outcome.get("answer", ""),
        "citations":     pairs,
        "seconds":       outcome.get("client_elapsed_sec"),
        "http_status":   outcome.get("http_status"),
        "error":         outcome.get("error", ""),
        "infra_failure": is_infra_failure(outcome),
    }
    if expected_file is not None:
        checks = rh.citation_checks(pairs, expected_file, expected_page)
        record["citation_file_hit"] = checks["citation_file_hit"]
        record["citation_page_hit"] = checks["citation_page_hit"]
    return record


def _safe(send, query, history) -> dict:
    """send(), with anything it raises turned into an error outcome."""
    try:
        return send(query, history)
    except Exception as e:
        return {"http_status": None, "error": f"{type(e).__name__}: {e}",
                "answer": "", "citations": [], "success": False,
                "client_elapsed_sec": None}


def run_conversation(conv: dict, send) -> dict:
    """
    The three requests for one conversation.

    send(query, history) returns an outcome shaped like run_harness.ask().
    Order: turn 1, turn 2 with turn 1's history, turn 2 without history.
    """
    first = _safe(send, conv["turn1_question"], [])
    history = build_history(conv["turn1_question"], first)
    with_history = _safe(send, conv["turn2_question"], history)
    without_history = _safe(send, conv["turn2_question"], [])

    return {
        "id":                  conv["id"],
        "language":            conv["language"],
        "turn1_question":      conv["turn1_question"],
        "turn1_expected_file": conv["turn1_expected_file"],
        "turn2_question":      conv["turn2_question"],
        "expected_file":       conv["expected_file"],
        "expected_page":       conv["expected_page"],
        "history_sent":        history,
        "turn1":               turn_record(first, conv["turn1_expected_file"], None),
        "turn2_with_history":  turn_record(with_history, conv["expected_file"],
                                           conv["expected_page"]),
        "turn2_without_history": turn_record(without_history, conv["expected_file"],
                                             conv["expected_page"]),
    }


# ── Metrics ───────────────────────────────────────────────

def _rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else None}


def summarise(records: list) -> dict:
    """
    Turn-2 citation hits with and without history, and the difference.

    with history    : conversations where neither turn 1 nor turn 2 (with
                      history) was an infrastructure failure — a dead
                      turn 1 means no real history was sent.
    without history : conversations where the baseline turn 2 was not.
    improvement     : paired, over conversations valid for BOTH, so the
                      two rates compare the same questions.
    """
    with_ok = [r for r in records
               if not r["turn1"]["infra_failure"] and not r["turn2_with_history"]["infra_failure"]]
    without_ok = [r for r in records if not r["turn2_without_history"]["infra_failure"]]
    with_ids = {r["id"] for r in with_ok}
    paired = [r for r in without_ok if r["id"] in with_ids]

    def hits(rows, side, key):
        return sum(1 for r in rows if r[side].get(key))

    out = {"conversations": len(records)}
    for key in ("citation_file_hit", "citation_page_hit"):
        out[key] = {
            "with_history":    _rate(hits(with_ok, "turn2_with_history", key), len(with_ok)),
            "without_history": _rate(hits(without_ok, "turn2_without_history", key), len(without_ok)),
        }
        k_with = hits(paired, "turn2_with_history", key)
        k_without = hits(paired, "turn2_without_history", key)
        out[key]["improvement"] = {
            "k": k_with - k_without,
            "n": len(paired),
            "rate": round((k_with - k_without) / len(paired), 3) if paired else None,
        }
    out["excluded_with_history"] = len(records) - len(with_ok)
    out["excluded_without_history"] = len(records) - len(without_ok)
    out["excluded_paired"] = len(records) - len(paired)
    out["infra_failures"] = sum(
        1 for r in records for t in ("turn1", "turn2_with_history", "turn2_without_history")
        if r[t]["infra_failure"])
    return out


# ── Output ────────────────────────────────────────────────

def read_query_rewrite() -> str:
    """
    QUERY_REWRITE as backend/config.py reads it from backend/.env.

    This is this process's reading of the same .env the backend uses; the
    running server only sees a change after a restart.
    """
    backend_dir = str(TESTS_DIR.parent)
    if backend_dir not in sys.path:
        sys.path.append(backend_dir)
    try:
        from config import settings
        return str(settings.QUERY_REWRITE).lower()
    except Exception as e:
        return f"unknown ({type(e).__name__}: {e})"


def _pct(rate: dict) -> str:
    if rate["rate"] is None:
        return f"n/a ({rate['k']}/{rate['n']})"
    return f"{rate['rate'] * 100:.1f}% ({rate['k']}/{rate['n']})"


def summary_lines(summary: dict) -> list:
    lines = [
        "| turn 2 check | with history | without history | improvement (paired) |",
        "|---|---|---|---|",
    ]
    for key in ("citation_file_hit", "citation_page_hit"):
        s = summary[key]
        imp = s["improvement"]
        imp_text = (f"{imp['k']:+d} of {imp['n']}"
                    + (f" ({imp['rate'] * 100:+.1f} pts)" if imp["rate"] is not None else ""))
        lines.append(f"| {key} | {_pct(s['with_history'])} | {_pct(s['without_history'])} "
                     f"| {imp_text} |")
    lines += [
        "",
        f"Excluded as infrastructure failures: {summary['excluded_with_history']} "
        f"(with history), {summary['excluded_without_history']} (without history), "
        f"{summary['excluded_paired']} (paired); {summary['infra_failures']} failed "
        f"request(s) in total.",
    ]
    return lines


def _cell(text) -> str:
    return " ".join(str(text if text is not None else "").split()).replace("|", "\\|")


def grading_markdown(meta: dict, records: list, summary: dict) -> str:
    lines = [
        "# Follow-up questions — grading sheet",
        "",
        f"- run at: {meta['run_at']}",
        f"- backend: {meta['base_url']}",
        f"- QUERY_REWRITE: {meta['query_rewrite']}",
        f"- tag: {meta['tag'] or '-'}",
        f"- conversations: {len(records)}",
        "",
        "Citation checks are mechanical. Fill in Correct (Y/P/N) for the answer "
        "WITH history by hand.",
        "",
        "## Summary",
        "",
        *summary_lines(summary),
        "",
        "## Answers",
        "",
        "| id | turn2 question | expected file and page | cited files with history "
        "| cited files without history | answer with history (first 400 chars) | Correct (Y/P/N) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in records:
        def cited(turn):
            t = r[turn]
            if t["error"] and not t["citations"]:
                return f"ERROR: {t['error'][:80]}"
            return "; ".join(f"{f} p{p}" for f, p in t["citations"]) or "-"

        t2 = r["turn2_with_history"]
        answer = t2["answer"][:400] if not t2["error"] else f"ERROR: {t2['error'][:200]}"
        lines.append("| " + " | ".join(_cell(x) for x in (
            r["id"], r["turn2_question"], f"{r['expected_file']} p{r['expected_page']}",
            cited("turn2_with_history"), cited("turn2_without_history"), answer,
        )) + " |  |")
    lines.append("")
    return "\n".join(lines)


def write_outputs(out_dir: Path, stem: str, meta: dict, records: list) -> tuple:
    summary = summarise(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    payload = dict(meta)
    payload["completed"] = len(records)
    payload["summary"] = summary
    payload["results"] = records
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(grading_markdown(meta, records, summary), encoding="utf-8")
    return json_path, md_path, summary


def load_gold(path: Path) -> list:
    raw = json.loads(path.read_text(encoding="utf-8"))
    conversations = raw.get("conversations", []) if isinstance(raw, dict) else raw
    if not conversations:
        sys.exit(f"error: {path} holds no conversations")
    return conversations


# ── Main ──────────────────────────────────────────────────

def main(argv: list, client_factory=httpx.Client) -> int:
    ap = argparse.ArgumentParser(description="Evaluate two-turn follow-up questions.")
    ap.add_argument("--gold", type=Path, default=TESTS_DIR / "gold_followups.json")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--username", default="admin")
    ap.add_argument("--password", default="admin123")
    ap.add_argument("--out-dir", type=Path, default=TESTS_DIR)
    args = ap.parse_args(argv[1:])

    conversations = load_gold(args.gold)
    query_rewrite = read_query_rewrite()
    run_at = datetime.now(timezone.utc)
    stamp = run_at.strftime("%Y%m%d_%H%M%S")
    stem = f"followups_{args.tag}_{stamp}" if args.tag else f"followups_{stamp}"

    meta = {
        "run_at":        run_at.isoformat(),
        "base_url":      args.base_url,
        "gold":          str(args.gold),
        "tag":           args.tag,
        "query_rewrite": query_rewrite,
        "timeout_sec":   args.timeout,
        "total":         len(conversations),
        "note": ("query_rewrite is this script's reading of backend/.env; the running "
                 "backend uses the value it was started with."),
    }

    print(f"QUERY_REWRITE = {query_rewrite}   (from backend config; the server "
          f"must have been restarted after changing it)")
    print(f"gold: {args.gold} ({len(conversations)} conversations)  backend: {args.base_url}  "
          f"timeout: {args.timeout:.0f}s\n")

    records = []
    with client_factory() as client:
        token = rh.login(client, args.base_url, args.username, args.password)
        headers = {"Authorization": f"Bearer {token}"}

        def send(query, history):
            return ask_with_history(client, args.base_url, headers, query, history, args.timeout)

        for n, conv in enumerate(conversations, 1):
            print(f"[{n}/{len(conversations)}] {conv['id']} ({conv['language']}) ... ",
                  end="", flush=True)
            record = run_conversation(conv, send)
            records.append(record)
            write_outputs(args.out_dir, stem, meta, records)
            w, b = record["turn2_with_history"], record["turn2_without_history"]
            flag = lambda t: "INFRA" if t["infra_failure"] else (
                f"file={'Y' if t['citation_file_hit'] else 'n'} "
                f"page={'Y' if t['citation_page_hit'] else 'n'}")
            print(f"with history: {flag(w)} | without: {flag(b)}")

    json_path, md_path, summary = write_outputs(args.out_dir, stem, meta, records)
    print("\n" + "\n".join(summary_lines(summary)))
    print(f"\n  -> {json_path}\n  -> {md_path}")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main(sys.argv))
