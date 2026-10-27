#!/usr/bin/env python
# backend/tests/quick_eval.py
# ─────────────────────────────────────────────────────────
# Fast iteration loop for answer quality.
#
# Asks a subset of the natural gold set (tests/gold_natural.json) against
# the LIVE POST /api/query/chat, then has a local LLM judge each answer
# against the gold evidence snippet: Y (same key fact), P (partly), N
# (wrong, unrelated, or "not found"). Trap questions are graded without
# the judge — the right answer to them is the application's refusal.
#
# The grades are automatic and approximate. They exist to tell two runs
# apart quickly (--compare), not to be reported as accuracy; verify by
# hand before reporting anything.
#
# Reuses tests/run_harness.py for login, the ask call (natural-mode
# payload: {query, history: []}), the LLM preflight, infrastructure-failure
# classification, refusal detection and citation checks.
#
# Requires the backend running (uvicorn main:app) and the LLM endpoint at
# OLLAMA_BASE_URL (from backend config) serving the judge model.
#
# Usage:
#   python tests/quick_eval.py --tag baseline_7b
#   python tests/quick_eval.py --tag fix_x --compare tests/quick_eval_baseline_7b_<ts>.json
#   python tests/quick_eval.py --tag full --subset all
#   python tests/quick_eval.py --tag two --ids nat-002,nat-058
#
# Exit codes: 0 done, 2 preflight failed, 4 backend uses a stale LLM URL.
#
# Writes:
#   tests/quick_eval_<tag>_<timestamp>.json
#   tests/quick_eval_<tag>_<timestamp>.md
# ─────────────────────────────────────────────────────────

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import httpx

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))

import run_harness as rh   # noqa: E402

SMOKE_IDS = (
    "nat-002", "nat-005", "nat-007", "nat-008", "nat-010", "nat-014", "nat-019",
    "nat-020", "nat-023", "nat-029", "nat-030", "nat-034", "nat-038", "nat-041",
    "nat-042", "nat-047", "nat-058", "nat-060", "nat-062",
)

ASK_TIMEOUT   = 300.0
RETRY_DELAY   = 5.0
JUDGE_TIMEOUT = 300.0
JUDGE_NUM_CTX = 8192

# httpx exception names whose request never reached the server — the
# only failures retried. A ReadTimeout means the server was working on
# it, and asking again would queue a second generation behind the first.
CONNECTION_ERRORS = ("ConnectError", "ConnectTimeout")

# Strings that mean the backend is calling an old tunnel URL.
STALE_MARKERS = ("viewing-unretired-jazz", "ERR_NGROK")
STALE_MSG = ("BACKEND USES A STALE OLLAMA_BASE_URL: restart the backend and clear "
             "OLLAMA_BASE_URL environment variables")
EXIT_PREFLIGHT_FAILED = 2
EXIT_STALE_BACKEND    = 4

NOTE = "judge grades are automatic and approximate; verify by hand before reporting"

GRADES = ("Y", "P", "N", "ERR", "INFRA")
GRADED = ("Y", "P", "N")
_RANK = {"N": 0, "P": 1, "Y": 2}

JUDGE_PROMPT = (
    "You are grading an answer from a question-answering system about Maharashtra "
    "Government documents. The answer may be in Marathi or English.\n\n"
    "QUESTION:\n{question}\n\n"
    "EVIDENCE (the passage of the document that answers the question):\n{evidence}\n\n"
    "SYSTEM ANSWER:\n{answer}\n\n"
    "Grade the system answer against the evidence:\n"
    "Y = the answer states the same key fact(s) as the evidence.\n"
    "P = the answer is partly right or incomplete.\n"
    "N = the answer is wrong, unrelated, garbled, or says the information was not found.\n\n"
    'Return only JSON, nothing else: {{"grade":"Y" or "P" or "N","reason":"<one short sentence>"}}'
)


class StaleBackend(Exception):
    """An answer showed the backend is calling a stale LLM URL."""


# ── Selection ─────────────────────────────────────────────

def select_questions(questions: list, subset: str = "smoke", ids: str = None) -> list:
    """
    The questions to ask, in run order.

    ids (comma-separated) overrides subset and keeps the order given.
    "smoke" is SMOKE_IDS in its fixed order; "all" is the gold file order.
    An unknown id is an error rather than a silently shorter run.
    """
    by_id = {q["id"]: q for q in questions}
    if ids:
        wanted = [i.strip() for i in ids.split(",") if i.strip()]
    elif subset == "smoke":
        wanted = list(SMOKE_IDS)
    else:
        return list(questions)
    missing = [i for i in wanted if i not in by_id]
    if missing:
        raise SystemExit(f"error: unknown id(s) in the gold file: {', '.join(missing)}")
    return [by_id[i] for i in wanted]


# ── Asking ────────────────────────────────────────────────

def is_connection_error(outcome: dict) -> bool:
    """Whether the request failed before reaching the server."""
    return (outcome.get("http_status") is None
            and (outcome.get("error") or "").startswith(CONNECTION_ERRORS))


def is_stale(outcome: dict) -> bool:
    """Whether the answer (or error text) carries a stale-tunnel marker."""
    text = f"{outcome.get('answer') or ''}\n{outcome.get('error') or ''}"
    return any(marker in text for marker in STALE_MARKERS)


def ask_with_policy(q: dict, ask_one, sleep=time.sleep) -> tuple:
    """
    One question: retried once after RETRY_DELAY on a connection error
    only. Never retried after a ReadTimeout, an HTTP error or an answer.
    Anything ask_one raises becomes an error outcome. Returns (outcome, attempts).
    """
    def once():
        try:
            return ask_one(q)
        except Exception as e:
            return {"http_status": None, "error": f"{type(e).__name__}: {e}",
                    "answer": "", "citations": [], "success": False,
                    "client_elapsed_sec": None}

    outcome = once()
    if is_connection_error(outcome):
        sleep(RETRY_DELAY)
        return once(), 2
    return outcome, 1


def collect_answers(questions: list, ask_one, workers: int = 1, sleep=time.sleep) -> list:
    """
    Asks every question on a pool of `workers` threads; returns one record
    per question in question order.

    Raises StaleBackend as soon as any answer carries a stale-URL marker;
    questions not yet started are cancelled.
    """
    records = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(ask_with_policy, q, ask_one, sleep): q for q in questions}
        try:
            for done, future in enumerate(as_completed(futures), 1):
                q = futures[future]
                outcome, attempts = future.result()
                if is_stale(outcome):
                    raise StaleBackend(q["id"])
                records[q["id"]] = answer_record(q, outcome, attempts)
                r = records[q["id"]]
                print(f"  [{done}/{len(questions)}] {q['id']} {r['seconds']}s"
                      + (" INFRA" if r["infra_failure"] else "")
                      + (f" ERROR {r['error'][:60]}" if r["error"] else ""), flush=True)
        except StaleBackend:
            for future in futures:
                future.cancel()
            raise
    return [records[q["id"]] for q in questions]


def answer_record(q: dict, outcome: dict, attempts: int) -> dict:
    pairs = rh.cited_pairs(outcome.get("citations"))
    trap = q["tier"] == rh.NEGATIVE_TIER
    file_hit = (None if trap else
                rh.citation_checks(pairs, q["expected_file"], q["expected_page"])["citation_file_hit"])
    return {
        "id":                q["id"],
        "language":          q["language"],
        "tier":              q["tier"],
        "question":          q["question"],
        "expected_file":     q["expected_file"],
        "expected_page":     q["expected_page"],
        "evidence_snippet":  q.get("evidence_snippet"),
        "answer":            outcome.get("answer", ""),
        "citations":         pairs,
        "seconds":           outcome.get("client_elapsed_sec"),
        "http_status":       outcome.get("http_status"),
        "error":             outcome.get("error", ""),
        "attempts":          attempts,
        "infra_failure":     rh.is_infra_failure(outcome),
        "refused":           rh.is_refusal(outcome.get("answer", "")),
        "citation_file_hit": file_hit,
    }


# ── Judging ───────────────────────────────────────────────

def make_judge(ollama_url: str, model: str, post=httpx.post):
    """
    judge(prompt) -> raw text, via POST {ollama_url}/api/chat with
    stream false, temperature 0 and num_ctx 8192.
    """
    url = f"{ollama_url.rstrip('/')}/api/chat"

    def judge(prompt: str) -> str:
        res = post(url, json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "num_ctx": JUDGE_NUM_CTX},
        }, headers=rh.PREFLIGHT_HEADERS, timeout=JUDGE_TIMEOUT)
        res.raise_for_status()
        return (res.json().get("message") or {}).get("content", "")
    return judge


def parse_judge(text: str):
    """
    {"grade", "reason"} from the judge's output, or None if it is not the
    JSON asked for. Tolerates code fences and text around one JSON object.
    """
    raw = (text or "").strip().replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    grade = str(data.get("grade", "")).strip().upper()
    if grade not in GRADED:
        return None
    return {"grade": grade, "reason": str(data.get("reason", "")).strip()}


def grade_record(record: dict, judge) -> dict:
    """
    The grade for one record. Order matters:
      INFRA  infrastructure failure — excluded from rates, never judged
      trap   Y if the refusal detector fired, else N — no judge call
      N      a failed request with no answer — nothing to judge
      judge  Y/P/N from the judge; unparseable twice -> ERR
    """
    if record["infra_failure"]:
        return {"grade": "INFRA", "reason": "infrastructure failure"}
    if record["tier"] == rh.NEGATIVE_TIER:
        return ({"grade": "Y", "reason": "refused, as a trap question should be"}
                if record["refused"] else
                {"grade": "N", "reason": "answered a question the documents do not cover"})
    if record["error"] and not record["answer"]:
        return {"grade": "N", "reason": f"no answer: {record['error'][:120]}"}

    prompt = JUDGE_PROMPT.format(question=record["question"],
                                 evidence=record["evidence_snippet"] or "",
                                 answer=record["answer"])
    last = ""
    for _ in range(2):
        try:
            last = judge(prompt)
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            continue
        parsed = parse_judge(last)
        if parsed is not None:
            return parsed
    return {"grade": "ERR", "reason": f"judge output not parseable: {last[:120]}"}


def grade_all(records: list, judge) -> list:
    """Sequential, after every answer is in: one judge call at a time."""
    for n, record in enumerate(records, 1):
        record.update(grade_record(record, judge))
        print(f"  judged [{n}/{len(records)}] {record['id']} {record['grade']}", flush=True)
    return records


# ── Summary ───────────────────────────────────────────────

def _rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else None}


def _breakdown(records: list) -> dict:
    counts = {g: sum(1 for r in records if r.get("grade") == g) for g in GRADES}
    graded = sum(counts[g] for g in GRADED)
    return {
        "total":    len(records),
        "counts":   counts,
        "rates":    {g: _rate(counts[g], graded) for g in ("Y", "P", "N")},
        "graded":   graded,
        "strict":   _rate(counts["Y"], graded),
        "lenient":  _rate(counts["Y"] + counts["P"], graded),
    }


def summarise(records: list) -> dict:
    """
    Counts of every grade; rates and accuracies over GRADED records only
    (Y, P, N) — INFRA and ERR are counted but excluded from every rate.
    Latency is over records that were not infrastructure failures.
    """
    times = sorted(r["seconds"] for r in records
                   if r.get("grade") != "INFRA" and r["seconds"] is not None)
    median = None
    if times:
        mid = len(times) // 2
        median = times[mid] if len(times) % 2 else round((times[mid - 1] + times[mid]) / 2, 2)
    out = _breakdown(records)
    out["marathi"] = _breakdown([r for r in records if r["language"] == "mr"])
    out["english"] = _breakdown([r for r in records if r["language"] == "en"])
    out["median_sec"] = median
    out["max_sec"] = times[-1] if times else None
    return out


def _pct(rate: dict) -> str:
    return "n/a" if rate["rate"] is None else f"{rate['rate'] * 100:.1f}% ({rate['k']}/{rate['n']})"


def _cell(text) -> str:
    return " ".join(str(text if text is not None else "").split()).replace("|", "\\|")


def result_table(records: list) -> list:
    lines = ["| id | language | tier | grade | seconds | refused | citation_file_hit | reason |",
             "|---|---|---|---|---|---|---|---|"]
    for r in records:
        hit = "-" if r["citation_file_hit"] is None else ("Y" if r["citation_file_hit"] else "n")
        lines.append("| " + " | ".join(_cell(x) for x in (
            r["id"], r["language"], r["tier"], r["grade"], r["seconds"],
            "Y" if r["refused"] else "n", hit, r["reason"])) + " |")
    return lines


def summary_lines(s: dict) -> list:
    def block(label, b):
        c = b["counts"]
        return [f"| {label} | {c['Y']} | {c['P']} | {c['N']} | {c['ERR']} | {c['INFRA']} "
                f"| {_pct(b['strict'])} | {_pct(b['lenient'])} |"]
    lines = ["| scope | Y | P | N | ERR | INFRA | strict (Y/graded) | lenient ((Y+P)/graded) |",
             "|---|---|---|---|---|---|---|---|"]
    lines += block("all", s) + block("Marathi", s["marathi"]) + block("English", s["english"])
    lines += [
        "",
        f"Rates over graded (Y+P+N = {s['graded']}): "
        f"Y {_pct(s['rates']['Y'])}, P {_pct(s['rates']['P'])}, N {_pct(s['rates']['N'])}. "
        f"ERR and INFRA are excluded from every rate.",
        f"Seconds per question (infra failures excluded): median {s['median_sec']}, "
        f"max {s['max_sec']}.",
    ]
    return lines


# ── Compare ───────────────────────────────────────────────

def compare(previous: dict, records: list, summary: dict) -> dict:
    """
    Per-id grade changes against a previous quick_eval JSON, and the change
    in strict and lenient accuracy. Only ids graded Y/P/N in both runs are
    ranked; any other pairing is reported as not comparable.
    """
    before = {r["id"]: r.get("grade") for r in previous.get("results", [])}
    changes = {"improved": [], "regressed": [], "unchanged": [], "not_comparable": []}
    for r in records:
        if r["id"] not in before:
            continue
        old, new = before[r["id"]], r["grade"]
        entry = {"id": r["id"], "before": old, "after": new}
        if old in _RANK and new in _RANK:
            key = ("improved" if _RANK[new] > _RANK[old] else
                   "regressed" if _RANK[new] < _RANK[old] else "unchanged")
        else:
            key = "unchanged" if old == new else "not_comparable"
        changes[key].append(entry)

    prev = previous.get("summary", {})

    def delta(name):
        a = (prev.get(name) or {}).get("rate")
        b = summary[name]["rate"]
        return None if a is None or b is None else round(b - a, 3)

    return {"previous": previous.get("tag"), "changes": changes,
            "strict_delta": delta("strict"), "lenient_delta": delta("lenient")}


def compare_lines(c: dict) -> list:
    fmt = lambda d: "n/a" if d is None else f"{d * 100:+.1f} pts"
    lines = [f"Compared with: {c['previous']}"]
    for key in ("improved", "regressed", "unchanged", "not_comparable"):
        items = c["changes"][key]
        lines.append(f"- {key} ({len(items)}): "
                     + (", ".join(f"{e['id']} {e['before']}->{e['after']}" for e in items) or "-"))
    lines.append(f"- strict accuracy delta: {fmt(c['strict_delta'])}; "
                 f"lenient accuracy delta: {fmt(c['lenient_delta'])}")
    return lines


# ── Output ────────────────────────────────────────────────

def markdown(meta: dict, records: list, summary: dict, comparison) -> str:
    lines = [f"# quick_eval — {meta['tag']}", "",
             f"- run at: {meta['run_at']}", f"- backend: {meta['base_url']}",
             f"- judge: {meta['judge_model']} at {meta['llm_endpoint']}",
             f"- questions: {len(records)} ({meta['selection']}), workers: {meta['workers']}",
             "", f"Note: {NOTE}.", "", "## Results", "", *result_table(records),
             "", "## Summary", "", *summary_lines(summary), ""]
    if comparison:
        lines += ["## Comparison", "", *compare_lines(comparison), ""]
    lines += ["## Full answers", ""]
    for r in records:
        lines += [f"### {r['id']} — {r['grade']} ({r['language']}, {r['tier']})", "",
                  f"**Question:** {r['question']}", "",
                  f"**Expected:** {r['expected_file'] or '— (trap)'}"
                  + (f" p{r['expected_page']}" if r["expected_page"] else ""), "",
                  f"**Evidence:** {r['evidence_snippet'] or '-'}", "",
                  f"**Cited:** {'; '.join(f'{f} p{p}' for f, p in r['citations']) or '-'}", "",
                  "**Answer:**", "", r["answer"] or f"(no answer) {r['error']}", "",
                  f"**Judge:** {r['reason']}", ""]
    return "\n".join(lines)


def write_outputs(out_dir: Path, meta: dict, records: list, summary: dict, comparison) -> tuple:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"quick_eval_{meta['tag']}_{meta['stamp']}"
    json_path, md_path = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    payload = dict(meta, note=NOTE, summary=summary, results=records)
    if comparison:
        payload["comparison"] = comparison
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown(meta, records, summary, comparison), encoding="utf-8")
    return json_path, md_path


# ── Main ──────────────────────────────────────────────────

def parse_args(argv: list):
    ap = argparse.ArgumentParser(description="Fast answer-quality loop with an LLM judge.")
    ap.add_argument("--gold", type=Path, default=TESTS_DIR / "gold_natural.json")
    ap.add_argument("--subset", choices=("smoke", "all"), default="smoke")
    ap.add_argument("--ids", default=None, help="comma-separated ids; overrides --subset")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--judge-model", default="qwen2.5:14b-instruct")
    ap.add_argument("--compare", type=Path, default=None)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--username", default="admin")
    ap.add_argument("--password", default="admin123")
    ap.add_argument("--out-dir", type=Path, default=TESTS_DIR)
    return ap.parse_args(argv[1:])


def main(argv: list, client_factory=httpx.Client, preflight=None, read_url=None,
         ask_one=None, judge=None, sleep=time.sleep) -> int:
    """
    ask_one(question) and judge(prompt) default to the live backend and
    the LLM endpoint; tests pass fakes. Nothing is written unless every
    answer came back without a stale-URL marker.
    """
    args = parse_args(argv)
    questions = select_questions(rh.load_natural(args.gold), args.subset, args.ids)
    selection = f"ids {args.ids}" if args.ids else f"subset {args.subset}"

    ollama_url = (read_url or rh.read_ollama_base_url)()
    ok, reason = (preflight or rh.preflight)(ollama_url)
    if not ok:
        print(rh.PREFLIGHT_FAILED_MSG)
        print(f"  reason: {reason}")
        return EXIT_PREFLIGHT_FAILED
    print(f"preflight: {reason}")
    print(f"{len(questions)} question(s) ({selection}), workers={args.workers}, "
          f"judge={args.judge_model}\n")

    run_at = datetime.now(timezone.utc)
    with client_factory() as client:
        if ask_one is None:
            token = rh.login(client, args.base_url, args.username, args.password)
            headers = {"Authorization": f"Bearer {token}"}
            ask_one = lambda q: rh.ask_natural(client, args.base_url, headers, q, ASK_TIMEOUT)
        try:
            records = collect_answers(questions, ask_one, args.workers, sleep)
        except StaleBackend:
            print(STALE_MSG)
            return EXIT_STALE_BACKEND

    print("\njudging ...")
    grade_all(records, judge or make_judge(ollama_url, args.judge_model))
    summary = summarise(records)

    comparison = None
    if args.compare:
        comparison = compare(json.loads(args.compare.read_text(encoding="utf-8")),
                             records, summary)

    meta = {"tag": args.tag, "stamp": run_at.strftime("%Y%m%d_%H%M%S"),
            "run_at": run_at.isoformat(), "base_url": args.base_url,
            "llm_endpoint": ollama_url, "judge_model": args.judge_model,
            "gold": str(args.gold), "selection": selection, "workers": args.workers}
    json_path, md_path = write_outputs(args.out_dir, meta, records, summary, comparison)

    print("\n" + "\n".join(result_table(records)))
    print("\n" + "\n".join(summary_lines(summary)))
    if comparison:
        print("\n" + "\n".join(compare_lines(comparison)))
    print(f"\nNote: {NOTE}")
    print(f"  -> {json_path}\n  -> {md_path}")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main(sys.argv))
