#!/usr/bin/env python
# backend/tests/run_harness.py
# ─────────────────────────────────────────────────────────
# RAG answer-capture harness. NOT a unittest suite, and
# deliberately separate from json_test_runner.py, which only
# understands binary pass/fail raised from assertions.
#
# What this does:
#   reads tests/harness_questions.json
#   → asks each question against the LIVE POST /api/query/chat
#   → records the system's actual answer + citations next to the
#     original question, gr_file, tier and gold_answer
#   → writes tests/harness_results_<timestamp>.json
#
# What this deliberately does NOT do:
#   ANY grading. It never compares `answer` to `gold_answer`, never
#   scores, never marks pass/fail. gold_answer is copied through
#   untouched so a human — or a validated LLM judge — can grade in a
#   separate later step. Adding "is the answer right?" logic here
#   would prejudge exactly the thing the harness exists to measure.
#
# Requires the backend to already be running:
#   uvicorn main:app
#
# Usage:
#   python tests/run_harness.py
#   python tests/run_harness.py --validate            # check the file, ask nothing
#   python tests/run_harness.py --limit 3             # smoke test on 3 questions
#   python tests/run_harness.py --base-url http://localhost:8000
#   python tests/run_harness.py --gold tests/gold_natural.json --tag natural
#       natural mode: unscoped questions, auto-checks on citations, script
#       and refusals, writes tests/harness_natural_<timestamp>.json and
#       tests/harness_natural_<timestamp>_grading.md
# ─────────────────────────────────────────────────────────

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

TESTS_DIR = Path(__file__).resolve().parent

# Recognised tier values. Unknown tiers are reported but NOT rejected —
# extending the taxonomy should not require editing this script.
KNOWN_TIERS = {"direct_lookup", "synthesis", "cross_reference", "negative_trap"}

# Fields every question must carry. `id` is filled in positionally if absent.
REQUIRED_FIELDS = ("question", "gr_file", "tier", "gold_answer")

# Seconds between "still waiting" lines while a question is in flight.
HEARTBEAT_INTERVAL = 15.0


class _Heartbeat:
    """
    Prints an elapsed-time line every `interval` seconds while a request is
    in flight, so a slow question is distinguishable from a stuck one.

    Mirrors the heartbeat in core/vectorstore.py rather than importing it:
    that module pulls in FAISS and langchain at import time, which this
    standalone script has no reason to load.

    Observes only — it never cancels, times out, or touches the request.
    Daemon thread, so it can never hold the process open by itself.

    `ticks` counts the lines printed, which the caller uses to decide
    whether the result needs its own line.
    """

    def __init__(self, label: str, interval: float = HEARTBEAT_INTERVAL):
        self.label    = label
        self.interval = interval
        self.ticks    = 0
        self._stop    = threading.Event()
        self._thread  = None
        self.started  = None

    def _tick(self):
        while not self._stop.wait(self.interval):
            self.ticks += 1
            waited = time.time() - self.started
            # Leading newline: the caller left the cursor mid-line after
            # printing the question header with end="".
            print(f"\n       ... still waiting on {self.label} "
                  f"— {waited:.0f}s elapsed", end="", flush=True)

    def __enter__(self):
        self.started = time.time()
        self._thread = threading.Thread(target=self._tick, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        return False


def load_questions(path: Path) -> list:
    """
    Reads the question set.

    Accepts either shape:
        [ {...}, {...} ]                  — a bare list
        { "questions": [ {...}, ... ] }   — wrapped, any sibling keys ignored

    Each question needs: question, gr_file, tier, gold_answer.
    `id` is optional — index-based ids (Q01, Q02, ...) are assigned if missing.

    Raises:
        SystemExit with a readable message if the file is unusable.
    """
    if not path.exists():
        sys.exit(f"error: question file not found: {path}\n"
                 f"       place harness_questions.json in {TESTS_DIR}")

    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"error: {path.name} is not valid JSON — {e}")

    if isinstance(raw, dict):
        questions = raw.get("questions")
        if questions is None:
            sys.exit(f"error: {path.name} is an object but has no 'questions' key")
    elif isinstance(raw, list):
        questions = raw
    else:
        sys.exit(f"error: {path.name} must be a list or an object with 'questions'")

    if not questions:
        sys.exit(f"error: {path.name} contains no questions")

    problems = []
    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            problems.append(f"  [{i}] not an object")
            continue
        q.setdefault("id", f"Q{i + 1:02d}")
        missing = [f for f in REQUIRED_FIELDS if not str(q.get(f, "")).strip()]
        if missing:
            problems.append(f"  [{i}] id={q['id']!r} missing/empty: {', '.join(missing)}")

    if problems:
        sys.exit(f"error: {path.name} has {len(problems)} malformed question(s):\n"
                 + "\n".join(problems))

    return questions


def describe_set(questions: list) -> None:
    """Prints a breakdown of the question set — tiers, files, unknown tiers."""
    by_tier, by_file = {}, {}
    for q in questions:
        by_tier[q["tier"]] = by_tier.get(q["tier"], 0) + 1
        by_file[q["gr_file"]] = by_file.get(q["gr_file"], 0) + 1

    print(f"  questions : {len(questions)}")
    print(f"  tiers     : " + ", ".join(f"{t}={n}" for t, n in sorted(by_tier.items())))
    print(f"  gr_files  : " + ", ".join(f"{f}={n}" for f, n in sorted(by_file.items())))

    unknown = set(by_tier) - KNOWN_TIERS
    if unknown:
        print(f"  note      : unrecognised tier(s) {sorted(unknown)} — recorded as-is")

    dupes = [i for i, c in
             {q["id"]: sum(1 for x in questions if x["id"] == q["id"]) for q in questions}.items()
             if c > 1]
    if dupes:
        print(f"  WARNING   : duplicate id(s) {sorted(dupes)} — results will be ambiguous")


def login(client: httpx.Client, base_url: str, username: str, password: str) -> str:
    """Authenticates and returns a bearer token. Exits with guidance on failure."""
    try:
        res = client.post(f"{base_url}/api/auth/login",
                          json={"username": username, "password": password})
    except httpx.RequestError as e:
        sys.exit(f"error: cannot reach the backend at {base_url} — {type(e).__name__}: {e}\n"
                 f"       is it running?  uvicorn main:app")

    if res.status_code != 200:
        sys.exit(f"error: login failed for {username!r} "
                 f"(HTTP {res.status_code}: {res.text[:200]})\n"
                 f"       is MongoDB running and seeded? (db/users.py seed_default_users)")

    return res.json()["access_token"]


def ask(client: httpx.Client, base_url: str, headers: dict, question: dict,
        top_k, language, timeout: float, scoped: bool = False) -> dict:
    """
    Sends ONE question to /api/query/chat and returns what came back.

    History is always empty — each question is judged standalone, so a
    previous answer cannot influence the next one.

    Never raises: transport and HTTP errors are recorded in the result so a
    single failure cannot abort a long run.
    """
    payload = {"query": question["question"], "history": []}
    if top_k is not None:
        payload["top_k"] = top_k
    if language is not None:
        payload["language"] = language
    # Scoped runs tell the backend which document the question is about,
    # which is what the metadata fast path needs to read the right
    # letterhead. It measures the ceiling: retrieval is no longer allowed
    # to pick the wrong document, so what is left is the answer quality.
    if scoped and question.get("gr_file"):
        payload["focus_file"] = question["gr_file"]

    started = time.time()
    try:
        res = client.post(f"{base_url}/api/query/chat", json=payload,
                          headers=headers, timeout=timeout)
    except httpx.RequestError as e:
        return {
            "http_status": None,
            "error":       f"{type(e).__name__}: {e}",
            "answer":      "",
            "citations":   [],
            "language":    "",
            "success":     False,
            "server_elapsed_sec": None,
            "client_elapsed_sec": round(time.time() - started, 2),
        }

    client_elapsed = round(time.time() - started, 2)

    if res.status_code != 200:
        return {
            "http_status": res.status_code,
            "error":       res.text[:500],
            "answer":      "",
            "citations":   [],
            "language":    "",
            "success":     False,
            "server_elapsed_sec": None,
            "client_elapsed_sec": client_elapsed,
        }

    body = res.json()
    return {
        "http_status": 200,
        "error":       "",
        # Recorded verbatim — never trimmed, normalised or compared.
        "answer":      body.get("answer", ""),
        "citations":   body.get("citations", []),
        "language":    body.get("language", ""),
        "success":     body.get("success", False),
        "server_elapsed_sec": body.get("elapsed_sec"),
        "client_elapsed_sec": client_elapsed,
    }


def write_output(path: Path, meta: dict, records: list) -> None:
    """
    Writes the results file. Called after EVERY question, not just at the
    end, so a long run that dies partway still leaves usable data on disk.
    """
    payload = dict(meta)
    payload["completed"] = len(records)
    payload["results"]   = records
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ── Natural gold set ──────────────────────────────────────
# A gold file whose entries carry "expected_file" (tests/gold_natural.json)
# runs in natural mode: every question is asked unscoped, exactly as a user
# would type it into the chat page — {query, history} and nothing else —
# and a few mechanical checks are computed on what came back. Those checks
# are NOT a grade: they say whether the right document was cited and
# whether the system refused, not whether the answer is right. The grading
# sheet written alongside is where a human decides that.

# Fields every natural gold entry must carry. expected_file and
# expected_page may be null (negative_trap), but the keys must exist.
NATURAL_FIELDS = ("id", "question", "tier", "language", "expected_file", "expected_page")

NEGATIVE_TIER = "negative_trap"

# Per-question timeout and retry policy for natural runs.
NATURAL_TIMEOUT = 120.0
RETRY_DELAY     = 5.0

# Preflight: the LLM endpoint must list a qwen2.5 model before anything is
# asked, so a dead Colab tunnel fails in 15 seconds instead of producing a
# results file of 63 "LLM error" answers.
PREFLIGHT_TIMEOUT      = 15.0
PREFLIGHT_MODEL_PREFIX = "qwen2.5"
PREFLIGHT_HEADERS      = {"ngrok-skip-browser-warning": "1"}
PREFLIGHT_FAILED_MSG   = "PREFLIGHT FAILED: LLM endpoint not reachable, fix Colab and rerun"

# Circuit breaker: after this many consecutive infrastructure failures the
# run stops asking and polls the preflight instead.
BREAKER_THRESHOLD = 3
RECOVERY_INTERVAL = 20.0
RECOVERY_WINDOW   = 600.0

# Exit codes for natural mode.
EXIT_PREFLIGHT_FAILED = 2
EXIT_ABORTED          = 3

# Substrings that mark an answer or error as an infrastructure failure
# rather than something the model said. See is_infra_failure().
INFRA_MARKERS = ("ERR_NGROK", "WinError", "Server disconnected")

# core/rag.py answers f"\u26a0\ufe0f LLM error: ..." when the LLM call raises.
_WARNING_SIGN, _VARIATION_SELECTOR = "\u26a0", "\ufe0f"

# The application's fixed not-found refusals, copied verbatim from
# core/rag.py (NOT_FOUND_MARATHI, NOT_FOUND_ENGLISH). Both prompts tell the
# model to say exactly these, and verify_answer() substitutes them for an
# unsupported answer. Copied rather than imported because importing
# core.rag loads the whole backend and its settings; the unit tests check
# the copies against core/rag.py's source so they cannot drift.
#
# The "⚠️ Vector store is not ready" and "⚠️ LLM error / timed out"
# answers are deliberately NOT here: they are system failures, not the
# model declining to answer, and a trap question "passing" because the
# LLM was down would be a false pass.
REFUSAL_MESSAGES = (
    "हे माहिती दिलेल्या दस्तऐवजात आढळली नाही.",
    "This information was not found in the uploaded documents.",
)

# Script thresholds for language_match, on the Devanagari share of letters.
MARATHI_MIN_SHARE = 0.5
ENGLISH_MAX_SHARE = 0.2

_DEVANAGARI_FIRST, _DEVANAGARI_LAST = chr(0x0900), chr(0x097F)


def is_natural_gold(path: Path) -> bool:
    """Whether a question file is a natural gold set (entries carry expected_file)."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    items = raw.get("questions", []) if isinstance(raw, dict) else raw
    return isinstance(items, list) and any(
        isinstance(q, dict) and "expected_file" in q for q in items)


def load_natural(path: Path) -> list:
    """Reads a natural gold set. Exits with a readable message if unusable."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    questions = raw.get("questions", []) if isinstance(raw, dict) else raw
    if not questions:
        sys.exit(f"error: {path.name} contains no questions")
    problems = [f"  [{i}] id={q.get('id')!r} missing: {', '.join(m)}"
                for i, q in enumerate(questions)
                for m in [[f for f in NATURAL_FIELDS if f not in q]] if m]
    if problems:
        sys.exit(f"error: {path.name} has {len(problems)} malformed entr(ies):\n"
                 + "\n".join(problems))
    return questions


def devanagari_share(text: str):
    """
    Share of letters (str.isalpha) in the Devanagari block U+0900-U+097F.
    None when the text has no letters at all. Vowel signs are combining
    marks rather than letters, so they count on neither side.
    """
    letters = devanagari = 0
    for ch in (text or ""):
        if ch.isalpha():
            letters += 1
            if _DEVANAGARI_FIRST <= ch <= _DEVANAGARI_LAST:
                devanagari += 1
    return devanagari / letters if letters else None


def language_match(answer: str, language: str) -> bool:
    """
    Whether the answer is written in the question's script: a Devanagari
    share of at least 0.5 for "mr", below 0.2 for "en". An answer with no
    letters matches neither; an unknown language code never matches.
    """
    share = devanagari_share(answer)
    if share is None:
        return False
    if language == "mr":
        return share >= MARATHI_MIN_SHARE
    if language == "en":
        return share < ENGLISH_MAX_SHARE
    return False


def is_refusal(answer: str) -> bool:
    """
    Whether the answer contains one of the fixed not-found refusals.

    Whitespace is collapsed and the closing full stop is optional, so a
    model that wraps the sentence in quotes, adds a line break or drops the
    final "." is still recognised. Nothing looser than that: a paraphrased
    refusal is the human grader's call, not this check's.
    """
    text = " ".join((answer or "").split())
    for message in REFUSAL_MESSAGES:
        core = message.rstrip(".")
        if core in text or core.casefold() in text.casefold():
            return True
    return False


def cited_pairs(citations: list) -> list:
    """
    [[file, page], ...] in the order the API returned them, values exactly
    as returned — page is the API's 1-based display string ("4", or "?"
    when the chunk had no page).
    """
    return [[c.get("file"), c.get("page")] for c in (citations or []) if isinstance(c, dict)]


def _same_page(cited_page, expected_page) -> bool:
    """
    Page equality with NO offset applied.

    The API already reports the displayed, 1-based page: core/rag.py
    build_citations() stores str(int(metadata["page"]) + 1), and the UI's
    CitationPill shows that value and opens the viewer at it unchanged.
    expected_page in the gold set is the same 1-based page. So the only
    conversion is string to int; a "?" or any non-number never matches.
    """
    if expected_page is None:
        return False
    try:
        return int(str(cited_page).strip()) == int(expected_page)
    except (TypeError, ValueError):
        return False


def citation_checks(pairs: list, expected_file, expected_page) -> dict:
    """citation_file_hit, citation_page_hit, top_citation_is_expected."""
    files = [f for f, _ in pairs]
    return {
        "citation_file_hit":        bool(expected_file) and expected_file in files,
        "citation_page_hit":        bool(expected_file) and any(
            f == expected_file and _same_page(p, expected_page) for f, p in pairs),
        "top_citation_is_expected": bool(expected_file) and bool(files) and files[0] == expected_file,
    }


def auto_checks(record: dict) -> dict:
    """Every auto-check for one natural record, as booleans."""
    checks = citation_checks(record["citations"], record["expected_file"], record["expected_page"])
    checks["language_match"] = language_match(record["answer"], record["language"])
    refused = is_refusal(record["answer"])
    checks["refused"] = refused
    if record["tier"] == NEGATIVE_TIER:
        checks["correct_refusal"] = refused
    else:
        checks["false_refusal"] = refused
    return checks


def read_ollama_base_url() -> str:
    """
    OLLAMA_BASE_URL from the backend's own settings (backend/config.py),
    so the preflight probes the endpoint the server will actually call.
    Imported here rather than at module level: loading settings needs a
    valid backend .env, and nothing else in this script does.
    """
    backend_dir = str(TESTS_DIR.parent)
    if backend_dir not in sys.path:
        sys.path.append(backend_dir)
    from config import settings
    return settings.OLLAMA_BASE_URL


def preflight(base_url: str, get=httpx.get) -> tuple:
    """
    GET {base_url}/api/tags and check a qwen2.5 model is listed.

    Returns:
        (ok, reason). ok only for HTTP 200 with a JSON body whose
        "models" list has a name (or model) starting with qwen2.5.
        Never raises.
    """
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        res = get(url, headers=PREFLIGHT_HEADERS, timeout=PREFLIGHT_TIMEOUT)
    except Exception as e:
        return False, f"{url}: {type(e).__name__}: {e}"
    if res.status_code != 200:
        return False, f"{url}: HTTP {res.status_code}"
    try:
        body = res.json()
    except Exception:
        return False, f"{url}: response is not valid JSON"
    models = body.get("models") if isinstance(body, dict) else None
    names = [str(m.get("name") or m.get("model") or "")
             for m in (models if isinstance(models, list) else []) if isinstance(m, dict)]
    if any(name.startswith(PREFLIGHT_MODEL_PREFIX) for name in names):
        return True, f"{url}: ok"
    return False, f"{url}: no {PREFLIGHT_MODEL_PREFIX} model listed (found {names or 'none'})"


def is_infra_failure(outcome: dict) -> bool:
    """
    Whether an outcome is an infrastructure failure rather than an answer:

        - the answer starts with the warning emoji and "LLM error"
          (core/rag.py's answer when the LLM call raised)
        - the answer or error text contains ERR_NGROK, WinError or
          "Server disconnected"
        - the request itself raised ReadTimeout

    These outcomes say nothing about the system's quality, so they are
    excluded from the answer metrics and feed the circuit breaker.
    """
    answer = outcome.get("answer") or ""
    error  = outcome.get("error") or ""
    head = answer.lstrip()
    if head.startswith(_WARNING_SIGN):
        head = head[1:].lstrip(_VARIATION_SELECTOR).lstrip()
        if head.startswith("LLM error"):
            return True
    text = answer + "\n" + error
    if any(marker in text for marker in INFRA_MARKERS):
        return True
    return error.startswith("ReadTimeout")


def wait_for_recovery(check, sleep=None, clock=None,
                      interval: float = None, window: float = None) -> bool:
    """
    Runs check() now and then every `interval` seconds until it passes or
    `window` seconds have gone by. check() returns (ok, reason).

    sleep, clock, interval and window default to time.sleep,
    time.monotonic, RECOVERY_INTERVAL and RECOVERY_WINDOW, resolved at
    call time so a test can replace them.
    """
    sleep    = sleep or time.sleep
    clock    = clock or time.monotonic
    interval = RECOVERY_INTERVAL if interval is None else interval
    window   = RECOVERY_WINDOW if window is None else window
    deadline = clock() + window
    while True:
        ok, reason = check()
        print(f"\n       preflight: {'PASSED' if ok else 'failed'} - {reason}", end="", flush=True)
        if ok:
            return True
        if clock() + interval > deadline:
            return False
        sleep(interval)


class RunAborted(Exception):
    """The circuit breaker gave up; .records holds everything asked so far."""

    def __init__(self, records: list):
        super().__init__(f"aborted after {len(records)} question(s)")
        self.records = records


def ask_natural(client: httpx.Client, base_url: str, headers: dict,
                question: dict, timeout: float) -> dict:
    """
    One natural question, sent the way the chat page sends it: the query
    as typed, an empty history, nothing else — no language, no top_k, no
    focus_file.
    """
    return ask(client, base_url, headers, {"question": question["question"]},
               top_k=None, language=None, timeout=timeout, scoped=False)


def _ask_once(q: dict, ask_one) -> dict:
    """ask_one(q), with anything it raises turned into an error outcome."""
    try:
        with _Heartbeat(q["id"]):
            return ask_one(q)
    except Exception as e:   # anything ask() did not already absorb
        return {"http_status": None, "error": f"{type(e).__name__}: {e}",
                "answer": "", "citations": [], "success": False,
                "client_elapsed_sec": None}


def _ask_with_retry(q: dict, ask_one, sleep, retry_delay: float) -> tuple:
    """
    One question with the ordinary retry: a failed attempt — an error in
    the outcome, or ask_one raising — is retried once after retry_delay.
    Returns (outcome, attempts).
    """
    outcome = _ask_once(q, ask_one)
    if not outcome.get("error"):
        return outcome, 1
    print(f"failed ({outcome['error'][:80]}), retrying in {retry_delay:.0f}s ... ",
          end="", flush=True)
    sleep(retry_delay)
    return _ask_once(q, ask_one), 2


def _natural_record(q: dict, outcome: dict, attempts: int) -> dict:
    """One result record: the question, what came back, and the auto-checks."""
    record = {
        "id":            q["id"],
        "tier":          q["tier"],
        "language":      q["language"],
        "question":      q["question"],
        "expected_file": q["expected_file"],
        "expected_page": q["expected_page"],
        "answer":        outcome.get("answer", ""),
        "citations":     cited_pairs(outcome.get("citations")),
        "elapsed_sec":   outcome.get("client_elapsed_sec"),
        "http_status":   outcome.get("http_status"),
        "error":         outcome.get("error", ""),
        "api_success":   outcome.get("success", False),
        "attempts":      attempts,
        "infra_failure": is_infra_failure(outcome),
    }
    record.update(auto_checks(record))
    return record


def run_natural(questions: list, ask_one, sleep=None,
                retry_delay: float = None, on_record=None,
                preflight_check=None, clock=None,
                threshold: int = None, interval: float = None,
                window: float = None) -> list:
    """
    Asks every question and returns one record per question, in order.

    ask_one(question) returns an outcome shaped like ask()'s. A failed
    attempt is retried once after retry_delay seconds; if that fails too
    the error is recorded and the run moves on.

    Circuit breaker, when preflight_check is given: `threshold`
    consecutive infrastructure failures (is_infra_failure) stop the run
    from asking, and preflight_check() is polled every `interval` seconds
    for up to `window` seconds. If it passes, the question that tripped
    the breaker is asked once more and the run continues; if it never
    passes, RunAborted is raised carrying every record so far, the
    tripping question included.

    on_record(records) is called after every question, so the caller can
    write partial results as it goes.

    Timing defaults (time.sleep, time.monotonic, RETRY_DELAY,
    BREAKER_THRESHOLD, RECOVERY_INTERVAL, RECOVERY_WINDOW) are resolved at
    call time, so a test can replace them.
    """
    sleep       = sleep or time.sleep
    clock       = clock or time.monotonic
    retry_delay = RETRY_DELAY if retry_delay is None else retry_delay
    threshold   = BREAKER_THRESHOLD if threshold is None else threshold
    interval    = RECOVERY_INTERVAL if interval is None else interval
    window      = RECOVERY_WINDOW if window is None else window

    records = []
    consecutive = 0
    for n, q in enumerate(questions, 1):
        print(f"[{n}/{len(questions)}] {q['id']} ({q['tier']}, {q['language']}) ... ",
              end="", flush=True)

        outcome, attempts = _ask_with_retry(q, ask_one, sleep, retry_delay)
        record = _natural_record(q, outcome, attempts)
        consecutive = consecutive + 1 if record["infra_failure"] else 0

        if preflight_check is not None and consecutive >= threshold:
            print(f"\n       {consecutive} consecutive infrastructure failures - "
                  f"polling the LLM endpoint every {interval:.0f}s for up to "
                  f"{window / 60:.0f} min", end="", flush=True)
            if not wait_for_recovery(preflight_check, sleep=sleep, clock=clock,
                                     interval=interval, window=window):
                records.append(record)
                if on_record is not None:
                    on_record(records)
                print("\n       endpoint never recovered - aborting the run")
                raise RunAborted(records)
            print(f"\n       retrying {q['id']} ... ", end="", flush=True)
            record = _natural_record(q, _ask_once(q, ask_one), attempts + 1)
            record["breaker_retry"] = True
            # The streak restarts: the preflight just passed, so only this
            # retry's own outcome counts towards the next trip.
            consecutive = 1 if record["infra_failure"] else 0

        records.append(record)
        if on_record is not None:
            on_record(records)

        if record["error"]:
            print(f"ERROR ({record['http_status']}): {record['error'][:120]}")
        else:
            print(f"{record['elapsed_sec']}s [{len(record['citations'])} cite] "
                  f"file={'Y' if record['citation_file_hit'] else 'n'} "
                  f"page={'Y' if record['citation_page_hit'] else 'n'} "
                  f"refused={'Y' if record['refused'] else 'n'}"
                  + (" INFRA" if record["infra_failure"] else ""))
    return records


def _rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else None}


def summarise_natural(records: list) -> dict:
    """
    Counts behind every auto-check.

    Citation checks come from retrieval, so they are over ALL answerable
    questions (traps have no expected file). language_match, the refusal
    checks and the latency stats describe the answer, so they are over the
    records that were NOT infrastructure failures — an "LLM error" answer
    says nothing about either. infra_excluded is how many were left out.
    A question that errored for any other reason stays in its denominators
    as a miss, and every error is counted under errors.
    """
    answerable       = [r for r in records if r["tier"] != NEGATIVE_TIER]
    valid            = [r for r in records if not r.get("infra_failure")]
    valid_answerable = [r for r in valid if r["tier"] != NEGATIVE_TIER]
    valid_traps      = [r for r in valid if r["tier"] == NEGATIVE_TIER]
    times = sorted(r["elapsed_sec"] for r in valid if r["elapsed_sec"] is not None)

    def count(rows, key):
        return _rate(sum(1 for r in rows if r.get(key)), len(rows))

    median = None
    if times:
        mid = len(times) // 2
        median = times[mid] if len(times) % 2 else round((times[mid - 1] + times[mid]) / 2, 2)

    return {
        "citation_file_hit":        count(answerable, "citation_file_hit"),
        "citation_page_hit":        count(answerable, "citation_page_hit"),
        "top_citation_is_expected": count(answerable, "top_citation_is_expected"),
        "language_match":           count(valid, "language_match"),
        "correct_refusal":          count(valid_traps, "correct_refusal"),
        "false_refusal":            count(valid_answerable, "false_refusal"),
        "median_sec":               median,
        "max_sec":                  times[-1] if times else None,
        "errors":                   sum(1 for r in records if r["error"]),
        "infra_excluded":           len(records) - len(valid),
        "questions":                len(records),
    }


def format_summary(summary: dict) -> list:
    """The summary as markdown lines, every rate shown with its k/n."""
    def line(label, key, scope):
        s = summary[key]
        pct = f"{s['rate'] * 100:.1f}%" if s["rate"] is not None else "n/a"
        return f"| {label} | {pct} ({s['k']}/{s['n']}) | {scope} |"

    return [
        "| check | result | over |",
        "|---|---|---|",
        line("citation_file_hit", "citation_file_hit", "answerable questions"),
        line("citation_page_hit", "citation_page_hit", "answerable questions"),
        line("top_citation_is_expected", "top_citation_is_expected", "answerable questions"),
        line("language_match", "language_match", "all questions, infra failures excluded"),
        line("correct_refusal", "correct_refusal",
             f"{NEGATIVE_TIER} questions, infra failures excluded"),
        line("false_refusal", "false_refusal", "answerable questions, infra failures excluded"),
        f"| median seconds per question | {summary['median_sec']} "
        f"| all questions, infra failures excluded |",
        f"| max seconds per question | {summary['max_sec']} "
        f"| all questions, infra failures excluded |",
        f"| excluded (infra_failure) | {summary['infra_excluded']} | all questions |",
        f"| errors | {summary['errors']} | all questions |",
    ]


def _cell(text) -> str:
    """One markdown table cell: single line, pipes escaped."""
    return " ".join(str(text if text is not None else "").split()).replace("|", "\\|")


def grading_markdown(meta: dict, records: list, summary: dict) -> str:
    """The hand-grading sheet: summary first, then one row per question."""
    lines = [
        "# Natural gold set — grading sheet",
        "",
        f"- run at: {meta['run_at']}",
        f"- backend: {meta['base_url']}",
        f"- question file: `{meta['question_file']}`"
        + (f", tag: `{meta['tag']}`" if meta.get("tag") else ""),
        f"- completed: {len(records)}/{meta['total']}",
        "",
        "Auto-checks are mechanical (cited file and page, script, fixed refusal "
        "string). They do not judge the answer; fill in Correct (Y/P/N) by hand.",
        "",
        "## Summary",
        "",
        *format_summary(summary),
        "",
        "## Answers",
        "",
        "| id | tier | language | question | expected file and page | cited files and pages "
        "| answer (first 400 characters) | Correct (Y/P/N) | Notes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        expected = (f"{r['expected_file']} p{r['expected_page']}"
                    if r["expected_file"] else "— (none: trap)")
        cited = "; ".join(f"{f} p{p}" for f, p in r["citations"]) or "—"
        answer = r["answer"][:400] if not r["error"] else f"ERROR: {r['error'][:200]}"
        lines.append("| " + " | ".join(_cell(x) for x in (
            r["id"], r["tier"], r["language"], r["question"], expected, cited, answer,
        )) + " |  |  |")
    lines.append("")
    return "\n".join(lines)


def write_natural(json_path: Path, md_path: Path, meta: dict, records: list) -> dict:
    """Writes the results JSON and the grading sheet; returns the summary."""
    summary = summarise_natural(records)
    payload = dict(meta)
    payload["completed"] = len(records)
    payload["summary"]   = summary
    payload["results"]   = records
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    md_path.write_text(grading_markdown(meta, records, summary), encoding="utf-8")
    return summary


def main_natural(args) -> int:
    """Natural mode: unscoped questions, auto-checks, grading sheet."""
    if args.scoped:
        sys.exit("error: --scoped does not apply to a natural gold set; "
                 "its questions are always asked unscoped")

    questions = load_natural(args.questions)
    by_tier = {}
    for q in questions:
        by_tier[q["tier"]] = by_tier.get(q["tier"], 0) + 1
    print(f"  questions : {len(questions)} (natural gold set)")
    print(f"  tiers     : " + ", ".join(f"{t}={n}" for t, n in sorted(by_tier.items())))

    if args.validate:
        print("\nvalidate only — no questions sent.")
        return 0
    if args.limit:
        questions = questions[:args.limit]
        print(f"  limited to first {len(questions)} question(s)")

    # Preflight before the first question and before any file is written:
    # a dead LLM endpoint exits here with nothing on disk.
    try:
        ollama_url = read_ollama_base_url()
    except Exception as e:
        print(PREFLIGHT_FAILED_MSG)
        print(f"  reason: cannot read OLLAMA_BASE_URL from backend settings - "
              f"{type(e).__name__}: {e}")
        return EXIT_PREFLIGHT_FAILED
    ok, reason = preflight(ollama_url)
    if not ok:
        print(PREFLIGHT_FAILED_MSG)
        print(f"  reason: {reason}")
        return EXIT_PREFLIGHT_FAILED
    print(f"\npreflight: {reason}")

    timeout = args.timeout if args.timeout is not None else NATURAL_TIMEOUT
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # The file names are fixed for natural runs; --tag is recorded in the
    # results instead, so "--tag natural" does not produce harness_natural_natural_.
    json_path = args.out_dir / f"harness_natural_{timestamp}.json"
    md_path   = args.out_dir / f"harness_natural_{timestamp}_grading.md"

    meta = {
        "run_at":        str(datetime.now(timezone.utc)),
        "base_url":      args.base_url,
        "question_file": str(args.questions),
        "total":         len(questions),
        "mode":          "natural",
        "tag":           args.tag,
        "timeout_sec":   timeout,
        "retry":         f"once, after {RETRY_DELAY:.0f}s",
        "page_convention": ("citation pages are the API's 1-based display pages "
                            "(build_citations: metadata page + 1), compared to the "
                            "gold expected_page (1-based) with no further offset"),
        "refusal_messages": list(REFUSAL_MESSAGES),
        "llm_endpoint":  ollama_url,
        "graded":        False,
        "aborted":       False,
    }

    print(f"\nbackend: {args.base_url}  (timeout {timeout:.0f}s, one retry)")
    with httpx.Client() as client:
        token = login(client, args.base_url, args.username, args.password)
        headers = {"Authorization": f"Bearer {token}"}
        print(f"logged in as {args.username!r}\n")

        started_all = time.time()
        try:
            records = run_natural(
                questions,
                lambda q: ask_natural(client, args.base_url, headers, q, timeout),
                on_record=lambda recs: write_natural(json_path, md_path, meta, recs),
                preflight_check=lambda: preflight(ollama_url),
            )
        except RunAborted as stop:
            meta["aborted"] = True
            meta["elapsed_sec"] = round(time.time() - started_all, 2)
            summary = write_natural(json_path, md_path, meta, stop.records)
            print("\n" + "\n".join(format_summary(summary)))
            print(f"\nABORTED: LLM endpoint did not recover within "
                  f"{RECOVERY_WINDOW / 60:.0f} min; partial results written")
            print(f"  -> {json_path}")
            print(f"  -> {md_path}")
            return EXIT_ABORTED
        meta["elapsed_sec"] = round(time.time() - started_all, 2)

    summary = write_natural(json_path, md_path, meta, records)
    print("\n" + "\n".join(format_summary(summary)))
    print(f"\n  -> {json_path}")
    print(f"  -> {md_path}")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Capture live RAG answers for a graded question set. Does not grade.",
    )
    # --gold and --questions are the same option; --questions keeps
    # existing invocations working.
    ap.add_argument("--gold", "--questions", dest="questions", type=Path,
                    default=TESTS_DIR / "harness_questions.json",
                    help="question set (default: tests/harness_questions.json); a file "
                         "whose entries carry expected_file, such as "
                         "tests/gold_natural.json, runs in natural mode")
    ap.add_argument("--out-dir", type=Path, default=TESTS_DIR,
                    help="where to write harness_results_<timestamp>.json (default: tests/)")
    ap.add_argument("--base-url", default="http://localhost:8000",
                    help="running backend (default: http://localhost:8000)")
    ap.add_argument("--username", default="admin")
    ap.add_argument("--password", default="admin123")
    ap.add_argument("--top-k", type=int, default=None,
                    help="override retrieval depth; omit to use the server default")
    ap.add_argument("--language", default=None,
                    help="force 'english' or 'marathi'; omit for per-question auto-detect")
    ap.add_argument("--timeout", type=float, default=None,
                    help="per-question timeout in seconds "
                         f"(default: 600, or {NATURAL_TIMEOUT:.0f} in natural mode)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N questions — useful for a smoke test")
    ap.add_argument("--preview-chars", type=int, default=100,
                    help="answer preview length on the progress line (default: 100)")
    ap.add_argument("--scoped", action="store_true",
                    help="send focus_file with each question, naming its own "
                         "gr_file, so the backend need not infer the document")
    ap.add_argument("--tag", default=None,
                    help="label inserted into the output filename, e.g. "
                         "--tag routing -> harness_results_routing_<timestamp>.json")
    ap.add_argument("--validate", action="store_true",
                    help="check the question file and exit without asking anything")
    args = ap.parse_args()

    print(f"question set: {args.questions}")
    if is_natural_gold(args.questions):
        return main_natural(args)
    if args.timeout is None:
        args.timeout = 600.0
    questions = load_questions(args.questions)
    describe_set(questions)

    if args.validate:
        print("\nvalidate only — no questions sent.")
        return 0

    if args.limit:
        questions = questions[:args.limit]
        print(f"  limited to first {len(questions)} question(s)")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Omitted entirely when no tag is given, so the filename is unchanged.
    tag      = f"{args.tag}_" if args.tag else ""
    out_path = args.out_dir / f"harness_results_{tag}{timestamp}.json"

    meta = {
        "run_at":        str(datetime.now(timezone.utc)),
        "base_url":      args.base_url,
        "question_file": str(args.questions),
        "total":         len(questions),
        "top_k":         args.top_k,
        "language":      args.language,
        "graded":        False,
        "note": ("Raw capture only. 'answer' and 'citations' are what the live "
                 "system returned; 'gold_answer' is the untouched expected answer. "
                 "No comparison or scoring has been applied — grade separately."),
    }

    # Added only when the flag is set, so an unscoped run writes exactly
    # the keys it has always written.
    if args.scoped:
        meta["scoped"] = True

    print(f"\nbackend: {args.base_url}")
    with httpx.Client() as client:
        token = login(client, args.base_url, args.username, args.password)
        headers = {"Authorization": f"Bearer {token}"}
        print(f"logged in as {args.username!r}\n")

        records = []
        started_all = time.time()

        for n, q in enumerate(questions, 1):
            print(f"[{n}/{len(questions)}] {q['id']} ({q['tier']}) ... ", end="", flush=True)

            with _Heartbeat(f"{q['id']}") as hb:
                outcome = ask(client, args.base_url, headers, q,
                              args.top_k, args.language, args.timeout,
                              scoped=args.scoped)

            # If the heartbeat printed, the cursor is at the end of its last
            # line — start a fresh, indented line so the result stays readable.
            if hb.ticks:
                print("\n       -> ", end="", flush=True)

            records.append({
                # ── the question, carried through unchanged ──
                "id":          q["id"],
                "question":    q["question"],
                "gr_file":     q["gr_file"],
                "tier":        q["tier"],
                "gold_answer": q["gold_answer"],
                # ── what the live system actually said ──
                "actual_answer":       outcome["answer"],
                "actual_citations":    outcome["citations"],
                "detected_language":   outcome["language"],
                "api_success":         outcome["success"],
                "http_status":         outcome["http_status"],
                "error":               outcome["error"],
                "server_elapsed_sec":  outcome["server_elapsed_sec"],
                "client_elapsed_sec":  outcome["client_elapsed_sec"],
                # deliberately absent: any score, verdict or pass/fail field
            })

            # Written every iteration — a run that dies at question 17 keeps 16.
            write_output(out_path, meta, records)

            if outcome["error"]:
                print(f"ERROR ({outcome['http_status']}): {outcome['error'][:120]}")
            else:
                preview = " ".join(outcome["answer"].split())[:args.preview_chars]
                print(f"{outcome['client_elapsed_sec']}s "
                      f"[{len(outcome['citations'])} cite] {preview!r}")

        total_elapsed = round(time.time() - started_all, 2)

    meta["elapsed_sec"] = total_elapsed
    write_output(out_path, meta, records)

    failed = [r for r in records if r["error"]]
    print(f"\ncaptured {len(records)} answer(s) in {total_elapsed}s")
    if failed:
        print(f"  {len(failed)} request(s) failed: {[r['id'] for r in failed]}")
    print(f"  -> {out_path}")
    print("  not graded — 'gold_answer' is stored alongside for a separate grading pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
