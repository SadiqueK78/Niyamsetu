#!/usr/bin/env python
# backend/tests/score_retrieval.py
# ─────────────────────────────────────────────────────────
# Retrieval-only scoring. No chat model is called anywhere in this
# script; the only model it touches is the EMBEDDING model, and only for
# the configurations that need a dense ranking.
#
# What it measures:
#   For each harness question, which SOURCE FILES does retrieval put in
#   front of the model, and where in that list does the question's own
#   document land. Five configurations are compared, so a change to
#   routing or graph expansion can be judged on retrieval alone, without
#   waiting on generation and without generation quality masking the
#   result.
#
# ── WHY THESE NUMBERS ARE A LOWER BOUND ──────────────────
#
# Many harness questions do not name a document. "शासन निर्णयाचा दिनांक
# काय आहे?" is a reasonable question against a corpus of one GR and
# inherently ambiguous against a corpus of nineteen — every one of them
# has a date. Scoring it against a single expected gr_file therefore
# counts a defensible retrieval as a miss whenever another document would
# have answered the question just as well.
#
# So treat every number here as a FLOOR on retrieval quality, not a
# measurement of it. The numbers are comparable BETWEEN configurations,
# which is what they exist for: the ambiguity is identical across
# configurations, so a difference between them is real even though the
# absolute values are pessimistic. Per-tier figures matter for the same
# reason — cross_reference and synthesis questions legitimately need more
# than one document, and a single expected file cannot express that.
#
# Nothing here writes .env, rebuilds the index, or re-embeds anything.
# Settings that a configuration needs to change are overridden in memory
# and restored afterwards.
#
# Usage:
#   python tests/score_retrieval.py
#   python tests/score_retrieval.py --questions tests/harness_questions.json --top-k 12
#   python tests/score_retrieval.py --gold tests/gold_natural.json
#
# Writes:
#   tests/retrieval_scores_<timestamp>.json
#   tests/retrieval_scores_<timestamp>.md
#
# ── NATURAL GOLD SET ─────────────────────────────────────
#
# A gold file whose entries carry "expected_file" (tests/gold_natural.json)
# is scored in natural mode. Those questions are written to be answerable
# from exactly one document, and each names the 1-based page holding its
# evidence, so two things change:
#
#   - page-hit@12 is reported: does a chunk of the expected file whose
#     page is expected_page appear among the first 12 retrieved chunks.
#     A chunk's page is the page it STARTS on (metadata "page" + 1), so
#     evidence near the top of a page can sit in a chunk that began on the
#     previous one; the gold set keeps that case rare, not impossible.
#   - negative_trap entries have no answer in the corpus. They have no
#     expected file to rank, so they are left out of every metric.
#
# Natural mode writes tests/retrieval_scores_natural_<timestamp>.{md,json}
# and shows each rate as a percentage with its k/n count.
# ─────────────────────────────────────────────────────────

import argparse
import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.append(str(TESTS_DIR.parent))

from config import settings                      # noqa: E402
from core.hybrid_search import hybrid_search, lexical_search   # noqa: E402
from core.rag import expand_with_graph, route_documents        # noqa: E402
from core.vectorstore import search as dense_search            # noqa: E402
from core.reranker import rerank, get_model as get_rerank_model  # noqa: E402
from core.reranker import stats as rerank_stats                # noqa: E402

# Ranks reported in the metric tables.
_HIT_CUTOFFS = (1, 3, 5, 12)

# Natural mode: file-level cutoffs, and how many retrieved chunks the
# page-level check looks at.
_NATURAL_CUTOFFS = (1, 3, 12)
_PAGE_HIT_CHUNKS = 12

# Tier excluded from metrics in natural mode — nothing to retrieve.
_NEGATIVE_TIER = "negative_trap"

# Marker used when a configuration could not run.
UNAVAILABLE = "unavailable (embedding endpoint down)"

# The reranked configuration and the metrics reported for it. Its list is
# RERANK_TOP_N chunks long, so hit@12 is computed over what exists.
RERANK_CONFIG = "hybrid_script_aware_rerank"
_RERANK_CUTOFFS = (1, 3, 5, 12)
_RERANK_PAGE_CHUNKS = 5

# Configurations that need a query embedding, and so are the ones that
# fail when the embedding endpoint is unreachable.
_NEEDS_EMBEDDING = ("dense_only", "hybrid", "hybrid_routed",
                    "hybrid_routed_graph", "hybrid_graph",
                    "hybrid_script_aware", RERANK_CONFIG)


@contextmanager
def overridden(**values):
    """
    Temporarily set settings attributes, in memory only.

    Restores the previous values on the way out, including on an
    exception. Never touches .env — the whole point is that a scoring run
    cannot change how the server behaves afterwards.
    """
    previous = {}
    try:
        for name, value in values.items():
            previous[name] = getattr(settings, name)
            setattr(settings, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(settings, name, value)


def _language_of(record: dict) -> str:
    """A record's question language, "unknown" when the gold entry has none."""
    return record.get("language") or "unknown"


def files_in_order(docs) -> list:
    """Distinct source_files in order of first appearance."""
    seen, ordered = set(), []
    for doc in docs or []:
        fname = doc.metadata.get("source_file")
        if fname and fname not in seen:
            seen.add(fname)
            ordered.append(fname)
    return ordered


def page_hit(docs, expected_file: str, expected_page, limit: int = _PAGE_HIT_CHUNKS) -> bool:
    """
    Whether one of the first `limit` chunks is from the expected file and
    starts on the expected page.

    expected_page is 1-based, as the gold file and every citation show it;
    chunk metadata "page" is PyPDFLoader's 0-based index.
    """
    if not expected_file or expected_page is None:
        return False
    for doc in (docs or [])[:limit]:
        page = doc.metadata.get("page")
        if (doc.metadata.get("source_file") == expected_file
                and page is not None and int(page) + 1 == int(expected_page)):
            return True
    return False


# ── Configurations ────────────────────────────────────────

def run_dense_only(question: str, top_k: int) -> list:
    """FAISS alone, through the same function the system has always used."""
    return dense_search(question, top_k=top_k)


def run_bm25_only(question: str, top_k: int) -> list:
    """BM25 alone. The only configuration that needs no embedding call."""
    return [doc for doc, _score in lexical_search(question, top_k=top_k)]


def run_hybrid(question: str, top_k: int) -> list:
    """Dense and lexical fused by RRF — the deployed default."""
    return hybrid_search(question, top_k=top_k)


def run_hybrid_routed(question: str, top_k: int) -> list:
    """
    Hybrid over a deeper pool, then narrowed to the dominant files.

    Reads the pool depth and the routing parameters from settings exactly
    as core.rag.query() does, so this measures the configuration the flags
    actually produce rather than a reimplementation of it.

    ROUTE_ENABLED is forced on for the call. route_documents() does not
    itself gate on that flag — the gate lives in query() — but forcing it
    makes the configuration independent of whatever the local .env happens
    to say, so two machines scoring the same corpus get the same numbers.
    """
    docs = hybrid_search(question, top_k=settings.ROUTE_POOL_K)
    with overridden(ROUTE_ENABLED=True):
        return route_documents(docs)


def run_hybrid_routed_graph(question: str, top_k: int) -> list:
    """
    Routed hybrid, then neighbours followed through the GR graph.

    Both flags forced on for the same reason as above: the numbers must
    describe the configuration, not the machine's .env.
    """
    docs = hybrid_search(question, top_k=settings.ROUTE_POOL_K)
    with overridden(ROUTE_ENABLED=True):
        routed = route_documents(docs)
    with overridden(GRAPH_EXPAND=True):
        return expand_with_graph(routed, question)


def run_hybrid_graph(question: str, top_k: int) -> list:
    """
    Plain hybrid, then neighbours followed through the GR graph — the same
    expansion hybrid_routed_graph applies, with routing left out.

    Isolates what the graph contributes. hybrid_routed_graph changes two
    things at once against hybrid, so a difference between them cannot be
    attributed: routing narrows the file set and expansion widens it
    again. This configuration moves only the second, so comparing it
    against hybrid measures graph expansion alone, and comparing it
    against hybrid_routed_graph shows whether routing helps or hurts once
    the graph is in play.

    Identical retrieval to run_hybrid — the same hybrid_search(top_k) call,
    no routing, no header injection — followed by the identical expansion
    call hybrid_routed_graph makes, under the same flag override and the
    same GRAPH_MAX_NEIGHBOR_DOCS / GRAPH_CHUNKS_PER_NEIGHBOR from settings.
    """
    docs = hybrid_search(question, top_k=top_k)
    with overridden(GRAPH_EXPAND=True):
        return expand_with_graph(docs, question)


def run_hybrid_script_aware(question: str, top_k: int) -> list:
    """
    Hybrid with SCRIPT_AWARE_FUSION forced on for this call only: a
    Latin-script question gets the dense ranking alone, everything else
    the same RRF fusion as hybrid. Compared against hybrid, the difference
    is exactly what dropping BM25 for English questions does.
    """
    with overridden(SCRIPT_AWARE_FUSION=True):
        return hybrid_search(question, top_k=top_k)


def run_hybrid_script_aware_rerank(question: str, top_k: int) -> list:
    """
    hybrid_script_aware, then the cross-encoder: SCRIPT_AWARE_FUSION and
    RERANK_ENABLED forced on for this call only. Retrieves
    RERANK_CANDIDATES chunks (not top_k), reranks them against the
    question, and keeps RERANK_TOP_N — the same steps core.rag.query()
    takes with both flags on.
    """
    with overridden(SCRIPT_AWARE_FUSION=True, RERANK_ENABLED=True):
        docs = hybrid_search(question, top_k=settings.RERANK_CANDIDATES)
        return rerank(question, docs, settings.RERANK_TOP_N)


# Column labels for the progress lines.
_SHORT = {
    "dense_only":          "dense",
    "bm25_only":           "bm25",
    "hybrid":              "hyb",
    "hybrid_routed":       "route",
    "hybrid_routed_graph": "graph",
    "hybrid_graph":        "hyb+g",
    "hybrid_script_aware": "hyb+s",
    RERANK_CONFIG:         "hyb+s+r",
}

CONFIGURATIONS = (
    ("dense_only",           run_dense_only),
    ("bm25_only",            run_bm25_only),
    ("hybrid",               run_hybrid),
    ("hybrid_routed",        run_hybrid_routed),
    ("hybrid_routed_graph",  run_hybrid_routed_graph),
    # Last, so it is the rightmost column in every table: the tables are
    # ordered by this tuple.
    ("hybrid_graph",         run_hybrid_graph),
)

# Natural mode scores one configuration more, appended so the default-mode
# tables keep their columns and order.
NATURAL_CONFIGURATIONS = CONFIGURATIONS + (
    ("hybrid_script_aware",  run_hybrid_script_aware),
    (RERANK_CONFIG,          run_hybrid_script_aware_rerank),
)


# ── Metrics ───────────────────────────────────────────────

def rank_of(expected: str, ranked: list):
    """1-based rank of the expected file, or None when absent."""
    if not expected or not ranked:
        return None
    try:
        return ranked.index(expected) + 1
    except ValueError:
        return None


def summarise(rows: list) -> dict:
    """
    Metrics over a list of per-question results.

    Args:
        rows : [{"rank": int|None, "files": int}, ...]

    Returns:
        hit@k for each cutoff, MRR, mean distinct files, and the count.
    """
    total = len(rows)
    if not total:
        return {f"hit@{k}": 0.0 for k in _HIT_CUTOFFS} | {
            "mrr": 0.0, "mean_files": 0.0, "questions": 0}

    out = {}
    for cutoff in _HIT_CUTOFFS:
        hits = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= cutoff)
        out[f"hit@{cutoff}"] = round(hits / total, 3)

    # Absent expected file contributes 0, which is what makes MRR a
    # single number rather than a number plus a coverage caveat.
    out["mrr"] = round(
        sum((1.0 / r["rank"]) if r["rank"] else 0.0 for r in rows) / total, 3)
    out["mean_files"] = round(sum(r["files"] for r in rows) / total, 2)
    out["questions"] = total
    return out


def summarise_natural(rows: list) -> dict:
    """
    Natural-mode metrics, with the k behind every rate kept alongside it.

    Args:
        rows : [{"rank": int|None, "files": int, "page_hit": bool}, ...]

    Returns:
        hit@k and page_hit@12 as rates plus their counts, MRR by file,
        mean distinct files, and the count.
    """
    total = len(rows)
    out = {}
    for cutoff in _NATURAL_CUTOFFS:
        hits = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= cutoff)
        out[f"hit@{cutoff}"] = round(hits / total, 3) if total else 0.0
        out[f"hit@{cutoff}_count"] = hits

    pages = sum(1 for r in rows if r["page_hit"])
    out[f"page_hit@{_PAGE_HIT_CHUNKS}"] = round(pages / total, 3) if total else 0.0
    out[f"page_hit@{_PAGE_HIT_CHUNKS}_count"] = pages

    out["mrr"] = round(
        sum((1.0 / r["rank"]) if r["rank"] else 0.0 for r in rows) / total, 3) if total else 0.0
    out["mean_files"] = round(sum(r["files"] for r in rows) / total, 2) if total else 0.0
    out["questions"] = total
    return out


def summarise_rerank(rows: list) -> dict:
    """
    Metrics for the reranked configuration: hit@1/3/5/12 and MRR by file,
    page_hit@5, each rate with its count. A list shorter than a cutoff is
    scored over what it has: the expected file must simply be in it.
    """
    total = len(rows)
    out = {}
    for cutoff in _RERANK_CUTOFFS:
        hits = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= cutoff)
        out[f"hit@{cutoff}"] = round(hits / total, 3) if total else 0.0
        out[f"hit@{cutoff}_count"] = hits
    pages = sum(1 for r in rows if r["page_hit_5"])
    key = f"page_hit@{_RERANK_PAGE_CHUNKS}"
    out[key] = round(pages / total, 3) if total else 0.0
    out[f"{key}_count"] = pages
    out["mrr"] = round(
        sum((1.0 / r["rank"]) if r["rank"] else 0.0 for r in rows) / total, 3) if total else 0.0
    out["mean_files"] = round(sum(r["files"] for r in rows) / total, 2) if total else 0.0
    out["questions"] = total
    return out


def rerank_metric_table(title: str, value) -> list:
    """One-row natural-style table for the reranked configuration."""
    page_key = f"page_hit@{_RERANK_PAGE_CHUNKS}"
    header = (["configuration"] + [f"hit@{k}" for k in _RERANK_CUTOFFS]
              + ["MRR", page_key, "mean files", "n"])
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    if isinstance(value, str):
        lines.append(f"| {RERANK_CONFIG} | " + " | ".join([value] + [""] * (len(header) - 2)) + " |")
    elif value:
        def rate(key):
            return f"{value[key] * 100:.1f}% ({value[key + '_count']}/{value['questions']})"
        cells = [RERANK_CONFIG] + [rate(f"hit@{k}") for k in _RERANK_CUTOFFS]
        cells += [f"{value['mrr']:.3f}", rate(page_key), f"{value['mean_files']:.2f}",
                  str(value["questions"])]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


# ── Reporting ─────────────────────────────────────────────

def metric_table(title: str, per_config: dict, order: list) -> list:
    """Markdown table of metrics, one row per configuration."""
    lines = [f"### {title}", ""]
    header = ["configuration"] + [f"hit@{k}" for k in _HIT_CUTOFFS] + ["MRR", "mean files", "n"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")

    for name in order:
        value = per_config.get(name)
        if isinstance(value, str):
            lines.append(f"| {name} | " + " | ".join([value] + [""] * (len(header) - 2)) + " |")
            continue
        if not value:
            continue
        cells = [name] + [f"{value[f'hit@{k}']:.3f}" for k in _HIT_CUTOFFS]
        cells += [f"{value['mrr']:.3f}", f"{value['mean_files']:.2f}", str(value["questions"])]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return lines


def natural_metric_table(title: str, per_config: dict, order: list) -> list:
    """
    Natural-mode markdown table: every rate as a percentage with its k/n
    count beside it, so a small tier cannot pass a single question off as
    a trend.
    """
    page_key = f"page_hit@{_PAGE_HIT_CHUNKS}"
    lines = [f"### {title}", ""]
    header = (["configuration"] + [f"hit@{k}" for k in _NATURAL_CUTOFFS]
              + ["MRR", page_key, "mean files", "n"])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")

    def rate(value, key):
        return f"{value[key] * 100:.1f}% ({value[key + '_count']}/{value['questions']})"

    for name in order:
        value = per_config.get(name)
        if isinstance(value, str):
            lines.append(f"| {name} | " + " | ".join([value] + [""] * (len(header) - 2)) + " |")
            continue
        if not value:
            continue
        cells = [name] + [rate(value, f"hit@{k}") for k in _NATURAL_CUTOFFS]
        cells += [f"{value['mrr']:.3f}", rate(value, page_key),
                  f"{value['mean_files']:.2f}", str(value["questions"])]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return lines


def per_question_table(records: list, order: list, natural: bool = False) -> list:
    """
    Markdown table of the expected file's rank under each configuration.

    In natural mode a rank is suffixed "p" when an expected-page chunk was
    also retrieved, and the expected page is shown with the file.
    """
    lines = ["### Per question — rank of the expected file", ""]
    if natural:
        lines += [f"`p` after a rank: a chunk from the expected page was among the "
                  f"first {_PAGE_HIT_CHUNKS} retrieved chunks.", ""]
    header = ["id", "tier", "expected"] + order
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")

    for rec in records:
        expected = rec["expected"] or "-"
        if natural and rec.get("expected_page") is not None:
            expected = f"{expected} p{rec['expected_page']}"
        cells = [rec["id"], rec["tier"], expected]
        for name in order:
            entry = rec["configurations"].get(name)
            if isinstance(entry, str):
                cells.append("n/a")
            elif entry is None:
                cells.append("-")
            else:
                cell = str(entry["rank"]) if entry["rank"] else "miss"
                if natural and entry.get("page_hit"):
                    cell += " p"
                cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return lines


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(
        description="Score retrieval configurations against the harness "
                    "question set. Calls no chat model.",
    )
    # --gold and --questions are the same option; --questions is kept so
    # existing invocations keep working.
    ap.add_argument("--gold", "--questions", dest="questions", type=Path,
                    default=TESTS_DIR / "harness_questions.json",
                    help="gold question set (default: tests/harness_questions.json); "
                         "a file whose entries carry expected_file, such as "
                         "tests/gold_natural.json, is scored in natural mode")
    ap.add_argument("--top-k", type=int, default=12,
                    help="chunks requested per query (default: 12)")
    args = ap.parse_args(argv[1:])

    if not args.questions.is_file():
        print(f"not a file: {args.questions}")
        return 1

    raw = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = raw if isinstance(raw, list) else raw.get("questions", [])
    if not questions:
        print(f"{args.questions.name} holds no questions")
        return 1

    # The natural gold set names its answer "expected_file"; the harness
    # set calls it "gr_file". The key decides the mode, not the file name,
    # so a copy of either set scores the way its schema says.
    natural = any("expected_file" in q for q in questions)
    excluded = []
    if natural:
        excluded = [q for q in questions if q.get("tier") == _NEGATIVE_TIER]
        questions = [q for q in questions if q.get("tier") != _NEGATIVE_TIER]
        if not questions:
            print(f"{args.questions.name} holds only {_NEGATIVE_TIER} entries")
            return 1

    print(f"questions: {len(questions)} from {args.questions.name}")
    if natural:
        print(f"mode:      natural ({len(excluded)} {_NEGATIVE_TIER} entries "
              f"excluded from metrics)")
    print(f"top_k:     {args.top_k}")
    print(f"pool:      ROUTE_POOL_K={settings.ROUTE_POOL_K} "
          f"ROUTE_TOP_DOCS={settings.ROUTE_TOP_DOCS} "
          f"ROUTE_MAX_CHUNKS={settings.ROUTE_MAX_CHUNKS}")
    print()

    # One probe decides whether the embedding-dependent configurations can
    # run at all, so a dead endpoint costs one failed call rather than one
    # per question per configuration.
    configurations = NATURAL_CONFIGURATIONS if natural else CONFIGURATIONS
    order = [name for name, _ in configurations]

    unavailable = set()
    try:
        dense_search(questions[0]["question"], top_k=1)
        print("embedding endpoint: reachable")
    except Exception as e:
        unavailable = set(_NEEDS_EMBEDDING) & set(order)
        print(f"embedding endpoint: UNREACHABLE ({type(e).__name__}: {e})")
        print("  dense_only, hybrid, hybrid_routed and hybrid_routed_graph "
              "will be reported as unavailable")
        print("  bm25_only needs no embedding and still runs")

    # The reranked configuration needs the cross-encoder. Probed once, so a
    # missing package or model reports the configuration as unavailable
    # rather than silently scoring the fallback (unreranked) order.
    reasons = {}
    if RERANK_CONFIG in order and RERANK_CONFIG not in unavailable:
        try:
            with overridden(RERANK_ENABLED=True):
                get_rerank_model()
            print(f"reranker: {settings.RERANK_MODEL} loaded "
                  f"(candidates={settings.RERANK_CANDIDATES}, top_n={settings.RERANK_TOP_N})")
        except Exception as e:
            unavailable.add(RERANK_CONFIG)
            reasons[RERANK_CONFIG] = (f"unavailable (reranker failed to load: "
                                      f"{type(e).__name__})")
            print(f"reranker: FAILED TO LOAD ({type(e).__name__}: {e}) - "
                  f"{RERANK_CONFIG} will be reported as unavailable")
    print()

    records = []

    for n, question in enumerate(questions, 1):
        text = question.get("question", "")
        expected = question.get("expected_file") if natural else question.get("gr_file")
        expected_page = question.get("expected_page") if natural else None
        record = {
            "id":       question.get("id", f"q{n}"),
            "tier":     question.get("tier", "unknown"),
            "question": text,
            "expected": expected,
            "configurations": {},
        }
        if natural:
            record["expected_page"] = expected_page
            record["language"] = question.get("language")

        for name, runner in configurations:
            if name in unavailable:
                record["configurations"][name] = reasons.get(name, UNAVAILABLE)
                continue
            try:
                docs = runner(text, args.top_k)
                ranked = files_in_order(docs)
                record["configurations"][name] = {
                    "rank":  rank_of(expected, ranked),
                    "files": len(ranked),
                    "ranked": ranked,
                }
                if natural:
                    record["configurations"][name]["page_hit"] = page_hit(
                        docs, expected, expected_page)
                if name == RERANK_CONFIG:
                    record["configurations"][name]["page_hit_5"] = page_hit(
                        docs, expected, expected_page, limit=_RERANK_PAGE_CHUNKS)
            except Exception as e:
                # One configuration failing on one question must not lose
                # the run; it is recorded as a miss with the reason.
                record["configurations"][name] = {
                    "rank": None, "files": 0, "ranked": [],
                    "error": f"{type(e).__name__}: {e}",
                }
                if natural:
                    record["configurations"][name]["page_hit"] = False
                if name == RERANK_CONFIG:
                    record["configurations"][name]["page_hit_5"] = False

        records.append(record)

        # Short per-configuration labels, distinct enough to read at a
        # glance: three of the five names start with "hybrid".
        def shown(name):
            entry = record["configurations"][name]
            if not isinstance(entry, dict):
                return "n/a"
            cell = str(entry["rank"]) if entry["rank"] else "miss"
            return cell + "p" if entry.get("page_hit") else cell

        ranks = "  ".join(f"{_SHORT[name]}={shown(name)}" for name in order)
        print(f"  [{n:3}/{len(questions)}] {record['id']:10} {ranks}", flush=True)

    # ── Aggregate ─────────────────────────────────────────
    tiers = sorted({r["tier"] for r in records})

    def rows_for(name, tier=None, language=None):
        rows = []
        for rec in records:
            if tier is not None and rec["tier"] != tier:
                continue
            if language is not None and _language_of(rec) != language:
                continue
            entry = rec["configurations"].get(name)
            if not isinstance(entry, dict):
                continue
            row = {"rank": entry["rank"], "files": entry["files"]}
            if natural:
                row["page_hit"] = entry.get("page_hit", False)
            if name == RERANK_CONFIG:
                row["page_hit_5"] = entry.get("page_hit_5", False)
            rows.append(row)
        return rows

    summary = summarise_natural if natural else summarise
    table = natural_metric_table if natural else metric_table

    overall, per_tier = {}, {}
    for name in order:
        if name in unavailable:
            overall[name] = reasons.get(name, UNAVAILABLE)
            continue
        overall[name] = summary(rows_for(name))

    for tier in tiers:
        per_tier[tier] = {}
        for name in order:
            if name in unavailable:
                per_tier[tier][name] = reasons.get(name, UNAVAILABLE)
                continue
            per_tier[tier][name] = summary(rows_for(name, tier))

    # Natural gold entries carry a language; the harness set does not, so
    # the breakdown exists in natural mode only.
    per_language = {}
    if natural:
        for language in sorted({_language_of(r) for r in records}):
            per_language[language] = {}
            for name in order:
                if name in unavailable:
                    per_language[language][name] = reasons.get(name, UNAVAILABLE)
                    continue
                per_language[language][name] = summary(rows_for(name, language=language))

    rerank_report = None
    if RERANK_CONFIG in order:
        def rerank_summary(**where):
            if RERANK_CONFIG in unavailable:
                return reasons.get(RERANK_CONFIG, UNAVAILABLE)
            return summarise_rerank(rows_for(RERANK_CONFIG, **where))
        rerank_report = {
            "configuration": RERANK_CONFIG,
            "settings": {"RERANK_MODEL": settings.RERANK_MODEL,
                         "RERANK_CANDIDATES": settings.RERANK_CANDIDATES,
                         "RERANK_TOP_N": settings.RERANK_TOP_N},
            "fallbacks": dict(rerank_stats),
            "overall": rerank_summary(),
            "per_tier": {tier: rerank_summary(tier=tier) for tier in tiers},
            "per_language": {lang: rerank_summary(language=lang)
                             for lang in sorted({_language_of(r) for r in records})},
        }

    # ── Write and print ───────────────────────────────────
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = f"retrieval_scores_natural_{timestamp}" if natural else f"retrieval_scores_{timestamp}"
    json_path = TESTS_DIR / f"{stem}.json"
    md_path   = TESTS_DIR / f"{stem}.md"

    if natural:
        note = ("Each question is written to be answerable from exactly one "
                "document, so a miss here is a retrieval miss rather than an "
                "ambiguous question. page_hit@12 counts a chunk by the page it "
                f"starts on. {_NEGATIVE_TIER} entries ({len(excluded)}) have no "
                "answer in the corpus and are excluded from every metric.")
    else:
        note = ("Questions that do not name a document are inherently ambiguous — "
                "every GR in the corpus has a date — so these figures are a LOWER "
                "BOUND on retrieval quality. They are comparable between "
                "configurations, which is what they are for.")

    lines = [
        "# Retrieval scores" + (" — natural gold set" if natural else ""),
        "",
        f"- run at: {datetime.now(timezone.utc).isoformat()}",
        f"- questions: {len(questions)} from `{args.questions.name}`"
        + (f" ({len(excluded)} {_NEGATIVE_TIER} excluded)" if natural else ""),
        f"- top_k: {args.top_k}, ROUTE_POOL_K: {settings.ROUTE_POOL_K}, "
        f"ROUTE_TOP_DOCS: {settings.ROUTE_TOP_DOCS}, "
        f"ROUTE_MAX_CHUNKS: {settings.ROUTE_MAX_CHUNKS}",
        "",
        note,
        "",
    ]
    lines += table("Overall", overall, order)
    for tier in tiers:
        lines += table(f"Tier: {tier}", per_tier[tier], order)
    if natural:
        lines += ["## By question language", ""]
        for language in per_language:
            lines += table(f"Language: {language}", per_language[language], order)
    if rerank_report is not None:
        lines += [f"## Reranked ({RERANK_CONFIG})", "",
                  f"{settings.RERANK_MODEL}: {settings.RERANK_CANDIDATES} candidates "
                  f"-> top {settings.RERANK_TOP_N}. hit@12 is over the "
                  f"{settings.RERANK_TOP_N}-chunk list that exists. Reranker fallbacks "
                  f"to the original order: {rerank_report['fallbacks']['failures']} of "
                  f"{rerank_report['fallbacks']['calls']} calls.", ""]
        lines += rerank_metric_table("Overall", rerank_report["overall"])
        for tier in tiers:
            lines += rerank_metric_table(f"Tier: {tier}", rerank_report["per_tier"][tier])
        for language, value in rerank_report["per_language"].items():
            lines += rerank_metric_table(f"Language: {language}", value)
    lines += per_question_table(records, order, natural=natural)

    report = "\n".join(lines)
    md_path.write_text(report, encoding="utf-8")
    payload = {
        "run_at":    str(datetime.now(timezone.utc)),
        "questions": len(questions),
        "top_k":     args.top_k,
        "settings": {
            "ROUTE_POOL_K":     settings.ROUTE_POOL_K,
            "ROUTE_TOP_DOCS":   settings.ROUTE_TOP_DOCS,
            "ROUTE_MAX_CHUNKS": settings.ROUTE_MAX_CHUNKS,
        },
        "unavailable": sorted(unavailable),
        "note": note if natural else (
            "Questions that do not name a document are inherently "
            "ambiguous, so these metrics are a lower bound."),
        "overall":  overall,
        "per_tier": per_tier,
        "results":  records,
    }
    if natural:
        payload["per_language"] = per_language
        if rerank_report is not None:
            payload["rerank"] = rerank_report
        payload["gold"] = args.questions.name
        payload["excluded"] = [{"id": q.get("id"), "tier": q.get("tier"),
                                "question": q.get("question")} for q in excluded]
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(report)
    print()
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    # The questions are Marathi; a cp1252 console raises on the first one
    # printed, which would lose the whole run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    raise SystemExit(main(sys.argv))
