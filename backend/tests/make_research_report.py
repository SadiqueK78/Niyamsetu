#!/usr/bin/env python
# backend/tests/make_research_report.py
# ─────────────────────────────────────────────────────────
# Turns the raw evaluation outputs for the 25-GR corpus into the tables
# and figures used in the research write-up. Calls no model and reads
# nothing it is not given; every number in the output is traced to one
# of the input files.
#
# Inputs (all JSON):
#   --retrieval  retrieval_scores_natural_<ts>.json  (score_retrieval.py --gold gold_gad25.json)
#   --harness    harness_natural_<ts>.json            (run_harness.py --gold gold_gad25.json)
#   --keys       gold_gad25_keys.json                 (answer keys)
#   --units      unit test summary (one row per module)
#   --corpus     corpus statistics
#   --review     optional manual overrides {id: "Y"|"P"|"N"} for answer grades
#
# Writes into --out (default tests/research_results/):
#   fig*.png (300 dpi), table*.csv, RESULTS.md, results.json
#
# Needs matplotlib for the figures; tables and RESULTS.md are written
# without it.
# ─────────────────────────────────────────────────────────

import argparse
import csv
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

# Display names for the retrieval configurations score_retrieval.py runs.
CONFIG_LABELS = {
    "dense_only": "Dense (bge-m3)",
    "bm25_only": "BM25",
    "hybrid": "Hybrid (RRF)",
    "hybrid_routed": "Hybrid + routing",
    "hybrid_routed_graph": "Hybrid + routing + graph",
    "hybrid_graph": "Hybrid + graph",
    "hybrid_script_aware": "Hybrid, script-aware",
    "hybrid_script_aware_rerank": "Script-aware + reranker",
}

# Reference categorical palette (dataviz skill, light mode), fixed order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
REFUSALS = ("हे माहिती दिलेल्या दस्तऐवजात आढळली नाही", "This information was not found")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").translate(DEV_DIGITS).lower())


def grade(answer: str, facts: list) -> str:
    """Y when every fact matches, P when some do, N when none do."""
    a = norm(answer)
    hits = [any(norm(form) in a for form in forms) for forms in facts]
    if hits and all(hits):
        return "Y"
    return "P" if any(hits) else "N"


def pct(k, n):
    return f"{100 * k / n:.1f}% ({k}/{n})" if n else "n/a"


def rate(k, n):
    return round(k / n, 3) if n else None


def write_csv(path: Path, header: list, rows: list, bom: bool = True):
    with path.open("w", newline="", encoding="utf-8-sig" if bom else "utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def md_table(header: list, rows: list) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ── answer-level scoring ─────────────────────────────────

def score_answers(harness: dict, keys: dict, review: dict) -> list:
    rows = []
    for r in harness["results"]:
        neg = r["tier"] == "negative_trap"
        refused = bool(r.get("refused"))
        if r.get("infra_failure"):
            g = "INFRA"
        elif neg:
            g = "Y" if refused else "N"
        elif refused:
            g = "N"
        else:
            g = grade(r.get("answer", ""), keys.get(r["id"], []))
        auto = g
        g = review.get(r["id"], g)
        cited = [c[0] for c in r.get("citations") or []]
        rows.append(dict(
            id=r["id"], tier=r["tier"], language=r["language"], question=r["question"],
            expected_file=r.get("expected_file"), answer=r.get("answer", ""),
            cited=cited, n_citations=len(cited), auto_grade=auto, grade=g,
            citation_file_hit=r.get("citation_file_hit"),
            top_citation_is_expected=r.get("top_citation_is_expected"),
            language_match=r.get("language_match"), refused=refused,
            false_refusal=r.get("false_refusal"), elapsed_sec=r.get("elapsed_sec"),
            infra_failure=bool(r.get("infra_failure")),
        ))
    return rows


def answer_summary(rows: list) -> dict:
    ok = [r for r in rows if not r["infra_failure"]]
    ans = [r for r in ok if r["tier"] != "negative_trap"]
    neg = [r for r in ok if r["tier"] == "negative_trap"]
    lat = sorted(r["elapsed_sec"] for r in ok if r["elapsed_sec"] is not None)

    def c(rs, f):
        return sum(1 for r in rs if f(r))

    s = dict(
        n=len(ok), answerable=len(ans), negatives=len(neg),
        infra_failures=len(rows) - len(ok),
        correct=c(ans, lambda r: r["grade"] == "Y"),
        partial=c(ans, lambda r: r["grade"] == "P"),
        wrong=c(ans, lambda r: r["grade"] == "N"),
        citation_file_hit=c(ans, lambda r: r["citation_file_hit"]),
        top_citation=c(ans, lambda r: r["top_citation_is_expected"]),
        false_refusal=c(ans, lambda r: r["refused"]),
        correct_refusal=c(neg, lambda r: r["refused"]),
        language_match=c(ok, lambda r: r["language_match"]),
        lang_n=c(ok, lambda r: r["language_match"] is not None),
        mean_citations=round(st.mean(r["n_citations"] for r in ans), 2) if ans else None,
    )
    if lat:
        s.update(median_sec=round(st.median(lat), 2), mean_sec=round(st.mean(lat), 2),
                 p90_sec=round(lat[min(len(lat) - 1, int(0.9 * len(lat)))], 2), max_sec=round(lat[-1], 2))
    return s


# ── figures ──────────────────────────────────────────────

def _style(plt):
    plt.rcParams.update({
        # Nirmala UI (Windows) supplies Devanagari glyphs for Marathi cells.
        "font.family": ["DejaVu Sans", "Nirmala UI"], "font.size": 9, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def grouped_bars(plt, path, groups, series, values, title, ylabel="%", ylim=100, note=None):
    """values[series_index][group_index]; one colour per series, fixed order."""
    import numpy as np
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(groups) * max(1, len(series) / 2)), 3.6))
    x = np.arange(len(groups))
    w = 0.8 / len(series)
    for i, name in enumerate(series):
        bars = ax.bar(x + (i - (len(series) - 1) / 2) * w, values[i], w * 0.92,
                      color=PALETTE[i], label=name, edgecolor="white", linewidth=0.8)
        for b, v in zip(bars, values[i]):
            if v is not None:
                ax.text(b.get_x() + b.get_width() / 2, v + ylim * 0.01, f"{v:.0f}" if ylim == 100 else f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, ylim * 1.1)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="bold")
    ax.legend(ncol=min(4, len(series)), loc="upper left", bbox_to_anchor=(0, -0.14 if len(groups) > 4 else -0.1), fontsize=8)
    if note:
        fig.text(0.0, -0.02, note, fontsize=7, color=INK2)
    ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)


def make_figures(out: Path, retr: dict, ans_rows: list, units: dict, corpus: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - figures skipped")
        return []
    _style(plt)
    made = []
    configs = [c for c in CONFIG_LABELS if c in retr["overall"]]
    labels = [CONFIG_LABELS[c] for c in configs]

    # Fig 1 - retrieval accuracy per configuration
    vals = [[100 * retr["overall"][c][m] for c in configs] for m in ("hit@1", "hit@3", "hit@12")]
    vals.append([100 * retr["overall"][c]["mrr"] for c in configs])
    grouped_bars(plt, out / "fig1_retrieval_by_config.png", [l.replace(", ", ",\n").replace(" + ", "\n+ ") for l in labels],
                 ["Hit@1", "Hit@3", "Hit@12", "MRR ×100"], vals,
                 f"Retrieval accuracy by configuration (n = {retr['questions']} answerable questions, 25 GRs)")
    made.append("fig1_retrieval_by_config.png")

    # Fig 2 - Hit@1 by query language
    key = [c for c in ("dense_only", "bm25_only", "hybrid", "hybrid_script_aware", "hybrid_script_aware_rerank") if c in configs]
    langs = [l for l in ("en", "mr") if l in retr.get("per_language", {})]
    vals = [[100 * retr["per_language"][l][c]["hit@1"] for l in langs] for c in key]
    grouped_bars(plt, out / "fig2_hit1_by_language.png",
                 [f"{'English' if l == 'en' else 'Marathi'} queries (n={retr['per_language'][l][key[0]]['questions']})" for l in langs],
                 [CONFIG_LABELS[c] for c in key], vals, "Cross-lingual retrieval: Hit@1 by query language")
    made.append("fig2_hit1_by_language.png")

    # Fig 3 - Hit@1 by question tier
    tiers = [t for t in ("situation", "detail", "disambiguation") if t in retr.get("per_tier", {})]
    vals = [[100 * retr["per_tier"][t][c]["hit@1"] for t in tiers] for c in key]
    grouped_bars(plt, out / "fig3_hit1_by_tier.png",
                 [f"{t.capitalize()} (n={retr['per_tier'][t][key[0]]['questions']})" for t in tiers],
                 [CONFIG_LABELS[c] for c in key], vals, "Retrieval Hit@1 by question type")
    made.append("fig3_hit1_by_tier.png")

    # Fig 4 - end-to-end answer accuracy by tier and language
    groups, vy, vp, vc = [], [], [], []
    for name, sel in [("All", lambda r: True), ("Situation", lambda r: r["tier"] == "situation"),
                      ("Detail", lambda r: r["tier"] == "detail"), ("Disambig.", lambda r: r["tier"] == "disambiguation"),
                      ("English", lambda r: r["language"] == "en"), ("Marathi", lambda r: r["language"] == "mr")]:
        rs = [r for r in ans_rows if sel(r) and r["tier"] != "negative_trap" and not r["infra_failure"]]
        if not rs:
            continue
        groups.append(f"{name}\n(n={len(rs)})")
        vy.append(100 * sum(r["grade"] == "Y" for r in rs) / len(rs))
        vp.append(100 * sum(r["grade"] in ("Y", "P") for r in rs) / len(rs))
        vc.append(100 * sum(bool(r["citation_file_hit"]) for r in rs) / len(rs))
    grouped_bars(plt, out / "fig4_answer_accuracy.png", groups,
                 ["Fully correct answer", "Correct or partial", "Correct GR cited"], [vy, vp, vc],
                 "End-to-end answer accuracy (live system, phi4-mini)")
    made.append("fig4_answer_accuracy.png")

    # Fig 5 - latency distribution
    lat = [r["elapsed_sec"] for r in ans_rows if r["elapsed_sec"] is not None and not r["infra_failure"]]
    if lat:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(lat, bins=20, color=PALETTE[0], edgecolor="white", linewidth=1)
        med = st.median(lat)
        ax.axvline(med, color=INK2, linestyle="--", linewidth=1)
        ax.text(med, ax.get_ylim()[1] * 0.92, f"  median {med:.1f}s", color=INK2, fontsize=8)
        ax.set_xlabel("Response time per question (s)")
        ax.set_ylabel("Questions")
        ax.set_title(f"End-to-end response time (n = {len(lat)})", loc="left", fontsize=10.5, fontweight="bold")
        ax.grid(axis="x", visible=False)
        fig.savefig(out / "fig5_latency.png")
        plt.close(fig)
        made.append("fig5_latency.png")

    # Fig 6 - corpus: chunks per document
    rows = corpus["rows"]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    names = [f"{r['date']}  {r['short']}" for r in rows]
    ax.barh(range(len(rows)), [r["chunks"] for r in rows], color=PALETTE[0], height=0.7)
    ax.set_yticks(range(len(rows)), names, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("Indexed chunks")
    ax.set_title(f"Corpus: {len(rows)} GAD documents, {sum(r['chunks'] for r in rows)} chunks", loc="left", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", visible=False)
    fig.savefig(out / "fig6_corpus_chunks.png")
    plt.close(fig)
    made.append("fig6_corpus_chunks.png")

    # Fig 7 - unit tests per module
    mods = sorted((m for m in units if units[m]["run"]), key=lambda m: -units[m]["run"])
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    passed = [units[m]["run"] - units[m]["failures"] - units[m]["errors"] for m in mods]
    failed = [units[m]["failures"] + units[m]["errors"] for m in mods]
    ax.barh(mods, passed, color=PALETTE[2], label="Passed", height=0.7)
    ax.barh(mods, failed, left=passed, color=PALETTE[7], label="Failed", height=0.7)
    for i, p in enumerate(passed):
        ax.text(p + 1, i, str(p), va="center", fontsize=7, color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel("Test cases")
    ax.set_title(f"Automated test suite: {sum(passed)}/{sum(passed) + sum(failed)} passed", loc="left", fontsize=10.5, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    fig.savefig(out / "fig7_unit_tests.png")
    plt.close(fig)
    made.append("fig7_unit_tests.png")
    return made


# ── tables as images ─────────────────────────────────────

HEAD_BG, ZEBRA, RULE = "#eef3fa", "#f8f8f6", "#d9d8d3"
STATUS_INK = {"Pass": "#008300", "Fail": "#c62f2e", "Correct": "#008300", "Partial": "#a36a00", "Wrong": "#c62f2e"}


def draw_table(plt, path, title, header, rows, widths, note=None, numeric=(), highlight=None, font=7.5):
    """
    One table as a PNG. widths are relative column widths; text wraps to
    them. numeric: column indexes right-aligned. highlight(row) -> True
    bolds that row (e.g. the deployed configuration).
    """
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    total_in = 10.0 if sum(widths) > 6 else 7.0
    scale = total_in / sum(widths)
    cols_in = [w * scale for w in widths]
    pad = 0.06
    fp = {False: FontProperties(family=["DejaVu Sans", "Nirmala UI"], size=font),
          True: FontProperties(family=["DejaVu Sans", "Nirmala UI"], size=font, weight="bold")}
    width_cache = {}

    def text_in(t, bold):
        # Rendered width in inches, measured on the real glyph outlines.
        key = (t, bold)
        if key not in width_cache:
            width_cache[key] = TextPath((0, 0), t, prop=fp[bold]).get_extents().width / 72 if t else 0.0
        return width_cache[key]

    def wrap_to(text, max_in, bold=False):
        lines = []
        for part in str(text).split("\n"):
            cur = ""
            for word in part.split(" "):
                cand = (cur + " " + word).strip()
                if text_in(cand, bold) <= max_in:
                    cur = cand
                    continue
                if cur:
                    lines.append(cur)
                # a single word wider than the column is split by characters
                while text_in(word, bold) > max_in and len(word) > 1:
                    k = len(word)
                    while k > 1 and text_in(word[:k], bold) > max_in:
                        k -= 1
                    lines.append(word[:k])
                    word = word[k:]
                cur = word
            lines.append(cur)
        return lines or [""]

    usable = [w - 2 * pad - 0.02 for w in cols_in]
    note_in = total_in - 0.05

    def wrap(text, n, bold=False):
        return wrap_to(text, n, bold)

    head_lines = [wrap(h, usable[i], True) for i, h in enumerate(header)]
    body_lines = [[wrap("" if c is None else c, usable[i]) for i, c in enumerate(r)] for r in rows]
    line_h = font * 1.45 / 72
    head_h = max(len(x) for x in head_lines) * line_h + 2 * pad
    row_h = [max(len(x) for x in r) * line_h + 2 * pad for r in body_lines]
    title_h = 0.38
    note_lines = wrap_to(note, note_in * font / 6.5) if note else []
    note_h = (len(note_lines) * 0.13 + 0.08) if note else 0
    H = title_h + head_h + sum(row_h) + note_h + 0.05

    fig = plt.figure(figsize=(total_in, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, total_in)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.text(0, 0.22, title, fontsize=10, fontweight="bold", color=INK, va="center")

    xs = [0]
    for w in cols_in:
        xs.append(xs[-1] + w)

    def put(y, h, cells, bold=False, bg=None, color_fn=None):
        if bg:
            ax.add_patch(plt.Rectangle((0, y), total_in, h, color=bg, lw=0))
        for i, lines in enumerate(cells):
            right = i in numeric
            x = xs[i + 1] - pad if right else xs[i] + pad
            txt = "\n".join(lines)
            ink = color_fn(i, txt) if color_fn else INK
            ax.text(x, y + pad, txt, fontsize=font, va="top", ha="right" if right else "left",
                    color=ink, fontweight="bold" if bold else "normal", linespacing=1.3)

    y = title_h
    put(y, head_h, head_lines, bold=True, bg=HEAD_BG)
    y += head_h
    ax.plot([0, total_in], [y, y], color=INK2, lw=0.8)
    for k, (cells, h) in enumerate(zip(body_lines, row_h)):
        hl = bool(highlight and highlight(rows[k]))
        put(y, h, cells, bold=hl, bg=ZEBRA if k % 2 else None,
            color_fn=lambda i, t: STATUS_INK.get(t.strip(), INK))
        y += h
        ax.plot([0, total_in], [y, y], color=RULE, lw=0.4)
    ax.plot([0, total_in], [y, y], color=INK2, lw=0.8)
    if note:
        ax.text(0, y + 0.08, "\n".join(wrap(note, int(total_in / (6.5 * 0.0078)))), fontsize=6.5, color=INK2, va="top")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def make_table_images(out: Path, t: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    _style(plt)
    made = []
    for name, spec in t.items():
        if not spec["rows"]:
            continue
        draw_table(plt, out / f"{name}.png", spec["title"], spec["header"], spec["rows"], spec["widths"],
                   note=spec.get("note"), numeric=spec.get("numeric", ()), highlight=spec.get("highlight"),
                   font=spec.get("font", 7.5))
        made.append(f"{name}.png")
    return made


# ── machine-readable data (numbers as numbers, rates as 0-1) ──

def write_dash_data(d: Path, retr, ans_rows, s, units, corpus, configs):
    d.mkdir(exist_ok=True)
    write_csv(d / "retrieval_overall.csv", ["config", "label", "hit1", "hit3", "hit12", "mrr", "mean_files", "n"],
              [[c, CONFIG_LABELS[c], o["hit@1"], o["hit@3"], o["hit@12"], o["mrr"], o["mean_files"], o["questions"]]
               for c in configs for o in [retr["overall"][c]]], bom=False)
    for section, name, col in (("per_language", "retrieval_by_language.csv", "language"),
                               ("per_tier", "retrieval_by_tier.csv", "tier")):
        write_csv(d / name, [col, "config", "label", "hit1", "hit3", "mrr", "n"],
                  [[g, c, CONFIG_LABELS[c], o["hit@1"], o["hit@3"], o["mrr"], o["questions"]]
                   for g, cfgs in retr.get(section, {}).items() for c in configs for o in [cfgs.get(c)] if o], bom=False)
    subsets = [("all", "All answerable", lambda r: True)] + \
              [(t, t.capitalize(), (lambda t: lambda r: r["tier"] == t)(t)) for t in ("situation", "detail", "disambiguation")] + \
              [(l, n, (lambda l: lambda r: r["language"] == l)(l)) for l, n in (("en", "English"), ("mr", "Marathi"))]
    rows = []
    for key, label, sel in subsets:
        rs = [r for r in ans_rows if sel(r) and r["tier"] != "negative_trap" and not r["infra_failure"]]
        n = len(rs)
        if n:
            rows.append([key, label, n, rate(sum(r["grade"] == "Y" for r in rs), n),
                         rate(sum(r["grade"] in ("Y", "P") for r in rs), n),
                         rate(sum(bool(r["citation_file_hit"]) for r in rs), n),
                         rate(sum(bool(r["top_citation_is_expected"]) for r in rs), n),
                         rate(sum(r["refused"] for r in rs), n)])
    write_csv(d / "answers_by_subset.csv", ["subset", "label", "n", "correct", "correct_or_partial",
                                            "cited_correct_gr", "cited_correct_gr_first", "false_refusal"], rows, bom=False)
    write_csv(d / "answer_summary.csv", ["metric", "value", "k", "n"], [
        ["answer_correct", rate(s["correct"], s["answerable"]), s["correct"], s["answerable"]],
        ["answer_correct_or_partial", rate(s["correct"] + s["partial"], s["answerable"]), s["correct"] + s["partial"], s["answerable"]],
        ["citation_file_hit", rate(s["citation_file_hit"], s["answerable"]), s["citation_file_hit"], s["answerable"]],
        ["top_citation", rate(s["top_citation"], s["answerable"]), s["top_citation"], s["answerable"]],
        ["false_refusal", rate(s["false_refusal"], s["answerable"]), s["false_refusal"], s["answerable"]],
        ["correct_refusal", rate(s["correct_refusal"], s["negatives"]), s["correct_refusal"], s["negatives"]],
        ["language_match", rate(s["language_match"], s["lang_n"]), s["language_match"], s["lang_n"]],
        ["median_sec", s.get("median_sec"), "", ""], ["p90_sec", s.get("p90_sec"), "", ""], ["max_sec", s.get("max_sec"), "", ""],
    ], bom=False)
    retr_by_id = {r["id"]: r for r in retr["results"]}
    rk = lambda q, c: (retr_by_id.get(q, {}).get("configurations", {}).get(c, {}).get("rank") or "")
    write_csv(d / "per_question.csv", ["id", "tier", "language", "question", "expected_gr", "dense_rank", "hybrid_rank",
                                       "script_aware_rank", "rerank_rank", "cited_first", "grade", "seconds", "answer"],
              [[r["id"], r["tier"], r["language"], r["question"], (r["expected_file"] or "").replace(".pdf.txt", ""),
                rk(r["id"], "dense_only"), rk(r["id"], "hybrid"), rk(r["id"], "hybrid_script_aware"),
                rk(r["id"], "hybrid_script_aware_rerank"), (r["cited"] or [""])[0].replace(".pdf.txt", ""),
                r["grade"], r["elapsed_sec"], r["answer"].replace("\n", " ")[:400]] for r in ans_rows], bom=False)
    write_csv(d / "unit_tests.csv", ["module", "cases", "passed", "failed", "seconds"],
              [[m, u["run"], u["run"] - u["failures"] - u["errors"], u["failures"] + u["errors"], u["secs"]]
               for m, u in units.items() if u["run"]], bom=False)
    write_csv(d / "corpus.csv", ["doc_id", "date", "type", "subject", "short", "characters", "chunks", "mean_chunk_chars"],
              [[r["file"].replace(".pdf.txt", ""), r["date"], r["type"], r["title"], r["short"], r["chars"], r["chunks"], r["avg_chunk"]]
               for r in corpus["rows"]], bom=False)


# ── main ─────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retrieval", type=Path, required=True)
    ap.add_argument("--harness", type=Path, required=True)
    ap.add_argument("--keys", type=Path, default=TESTS_DIR / "gold_gad25_keys.json")
    ap.add_argument("--units", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--review", type=Path, default=None)
    ap.add_argument("--functional", type=Path, default=None,
                    help="live API test cases: [{id, area, description, input, expected, actual, status, note?}]")
    ap.add_argument("--out", type=Path, default=TESTS_DIR / "research_results")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    load = lambda p: json.loads(p.read_text(encoding="utf-8"))
    retr, harness, units, corpus = load(args.retrieval), load(args.harness), load(args.units), load(args.corpus)
    keys = load(args.keys)["keys"]
    review = load(args.review) if args.review else {}

    ans_rows = score_answers(harness, keys, review)
    s = answer_summary(ans_rows)

    # Table 1 - corpus
    t1 = [[i + 1, r["file"].replace(".pdf.txt", ""), r["date"], r["type"], r["title"], r["chars"], r["chunks"], r["avg_chunk"]]
          for i, r in enumerate(corpus["rows"])]
    h1 = ["#", "Document ID", "Issue date", "Type", "Subject (short)", "Characters", "Chunks", "Mean chunk (chars)"]
    write_csv(out / "table1_corpus.csv", h1, t1)

    # Table 3 - retrieval
    configs = [c for c in CONFIG_LABELS if c in retr["overall"]]
    h3 = ["Configuration", "Hit@1", "Hit@3", "Hit@12", "MRR", "Mean distinct files in top-12"]
    t3 = [[CONFIG_LABELS[c], pct(o["hit@1_count"], o["questions"]), pct(o["hit@3_count"], o["questions"]),
           pct(o["hit@12_count"], o["questions"]), f"{o['mrr']:.3f}", o["mean_files"]]
          for c in configs for o in [retr["overall"][c]]]
    write_csv(out / "table3_retrieval_overall.csv", h3, t3)

    def breakdown(section):
        rows = []
        for grp, cfgs in retr.get(section, {}).items():
            for c in configs:
                o = cfgs.get(c)
                if o:
                    rows.append([grp, CONFIG_LABELS[c], o["questions"], pct(o["hit@1_count"], o["questions"]),
                                 pct(o["hit@3_count"], o["questions"]), f"{o['mrr']:.3f}"])
        return rows
    h4 = ["Group", "Configuration", "n", "Hit@1", "Hit@3", "MRR"]
    t4, t5 = breakdown("per_tier"), breakdown("per_language")
    write_csv(out / "table4_retrieval_by_tier.csv", h4, t4)
    write_csv(out / "table5_retrieval_by_language.csv", h4, t5)

    # Table 6 - end to end
    def e2e(rs_all, label):
        rs = [r for r in rs_all if not r["infra_failure"] and r["tier"] != "negative_trap"]
        n = len(rs)
        if not n:
            return None
        return [label, n, pct(sum(r["grade"] == "Y" for r in rs), n), pct(sum(r["grade"] in ("Y", "P") for r in rs), n),
                pct(sum(bool(r["citation_file_hit"]) for r in rs), n), pct(sum(bool(r["top_citation_is_expected"]) for r in rs), n),
                pct(sum(r["refused"] for r in rs), n)]
    h6 = ["Subset", "n", "Fully correct", "Correct or partial", "Correct GR cited", "Correct GR cited first", "False refusal"]
    t6 = [x for x in [
        e2e(ans_rows, "All answerable"),
        *[e2e([r for r in ans_rows if r["tier"] == t], f"Tier: {t}") for t in ("situation", "detail", "disambiguation")],
        *[e2e([r for r in ans_rows if r["language"] == l], f"Language: {'English' if l == 'en' else 'Marathi'}") for l in ("en", "mr")],
    ] if x]
    write_csv(out / "table6_end_to_end.csv", h6, t6)

    # Table 7 - unit tests
    h7 = ["Test module", "Test cases", "Passed", "Failed/errors", "Time (s)"]
    t7 = [[m, u["run"], u["run"] - u["failures"] - u["errors"], u["failures"] + u["errors"], u["secs"]]
          for m, u in units.items() if u["run"]]
    t7.append(["TOTAL", sum(r[1] for r in t7), sum(r[2] for r in t7), sum(r[3] for r in t7), round(sum(r[4] for r in t7), 2)])
    write_csv(out / "table7_unit_tests.csv", h7, t7)

    # Appendix - per question
    retr_by_id = {r["id"]: r for r in retr["results"]}
    ha = ["ID", "Tier", "Lang", "Question", "Expected GR", "Dense rank", "Hybrid rank", "Script-aware rank",
          "Cited GRs", "Grade (auto)", "Grade (final)", "Time (s)", "Answer"]
    ta = []
    for r in ans_rows:
        rr = retr_by_id.get(r["id"], {}).get("configurations", {})
        rk = lambda c: (rr.get(c, {}).get("rank") or "miss") if rr else "-"
        ta.append([r["id"], r["tier"], r["language"], r["question"], (r["expected_file"] or "— (no answer)").replace(".pdf.txt", ""),
                   rk("dense_only"), rk("hybrid"), rk("hybrid_script_aware"),
                   "; ".join(dict.fromkeys(c.replace(".pdf.txt", "") for c in r["cited"])), r["auto_grade"], r["grade"],
                   r["elapsed_sec"], r["answer"].replace("\n", " ")[:500]])
    write_csv(out / "appendix_per_question.csv", ha, ta)

    # Table 8 - functional test cases against the live API
    functional = load(args.functional) if args.functional else []
    h8 = ["Test ID", "Area", "Test case", "Input", "Expected", "Actual", "Status", "Note"]
    t8 = [[f["id"], f["area"], f["description"], f["input"], f["expected"], f["actual"], f["status"], f.get("note", "")]
          for f in functional]
    if t8:
        write_csv(out / "table8_functional_tests.csv", h8, t8)

    # Table 2 - experimental setup (values read from the run inputs where they exist)
    cs = corpus["summary"]
    h2 = ["Component", "Setting used in this evaluation"]
    t2 = [
        ["Corpus", f"{cs['docs']} GAD documents (2018-2026), bilingual text: Marathi original + English translation"],
        ["Document types", ", ".join(f"{k} {v}" for k, v in sorted(cs["types"].items(), key=lambda kv: -kv[1]))],
        ["Chunking", f"Section-aware (GR headings), fallback 800 chars / 150 overlap; {cs['chunks']} chunks, mean {cs['mean_chunk']} chars"],
        ["Embedding model", f"bge-m3 (multilingual), {cs['dim']}-dimensional"],
        ["Vector index", f"FAISS {cs['index_type']} (exact search), {cs['ntotal']} vectors"],
        ["Lexical retrieval", "BM25 over the same chunks"],
        ["Fusion", "Reciprocal rank fusion of dense + BM25 (deployed configuration)"],
        ["Reranker (offline test)", "BAAI/bge-reranker-v2-m3 cross-encoder, 20 candidates -> top 5, CPU"],
        ["Retrieved chunks per query", f"top-k = {retr.get('top_k', 12)}"],
        ["Generator LLM", f"{harness.get('llm_model') or 'phi4-mini'} (3.8B, Q4_K_M) via Ollama, context 8192, max 800 output tokens"],
        ["Answer test", f"{harness['total']} questions through live POST /api/query/chat, timeout {harness.get('timeout_sec')} s"],
        ["Grading", "Key facts per question (auto) + manual review; refusal = fixed refusal string"],
    ]
    write_csv(out / "table2_setup.csv", h2, t2)

    # Table 2b - test question set composition
    comp = Counter((r["tier"], r["language"]) for r in ans_rows)
    tiers = ["situation", "detail", "disambiguation", "negative_trap"]
    h2b = ["Question type", "What it tests", "English", "Marathi", "Total"]
    what = {"situation": "Plain-language need, no GR terminology", "detail": "One specific fact (date, number, name)",
            "disambiguation": "Telling near-identical GRs apart", "negative_trap": "No answer in corpus; system must refuse"}
    t2b = [[t.replace("_", " ").capitalize(), what[t], comp[(t, "en")], comp[(t, "mr")], comp[(t, "en")] + comp[(t, "mr")]] for t in tiers]
    t2b.append(["Total", "", sum(r[2] for r in t2b), sum(r[3] for r in t2b), sum(r[4] for r in t2b)])
    write_csv(out / "table2b_question_set.csv", h2b, t2b)

    # Compact pivots of tables 4/5 for the paper (the CSVs keep the long form)
    def pivot(section, groups, names):
        rows = []
        for c in configs:
            row = [CONFIG_LABELS[c]]
            for g in groups:
                o = retr.get(section, {}).get(g, {}).get(c)
                row += [f"{100 * o['hit@1']:.1f}", f"{100 * o['hit@3']:.1f}", f"{o['mrr']:.3f}"] if o else ["-", "-", "-"]
            rows.append(row)
        present = [g for g in groups if g in retr.get(section, {})]
        head = ["Configuration"] + [f"{names[g]}\n{m}" for g in present for m in ("Hit@1 %", "Hit@3 %", "MRR")]
        ns = "; ".join(f"{names[g]} n = {retr[section][g][configs[0]]['questions']}" for g in present)
        return head, rows, ns
    h4p, t4p, n4 = pivot("per_tier", ["situation", "detail", "disambiguation"],
                     {"situation": "Situation", "detail": "Detail", "disambiguation": "Disambig."})
    h5p, t5p, n5 = pivot("per_language", ["en", "mr"], {"en": "English", "mr": "Marathi"})

    deployed = lambda r: str(r[0]).startswith("Hybrid (RRF)")
    rerank_note = "Hit@k = expected GR among the first k distinct documents retrieved. The reranker keeps 5 chunks, so its Hit@12 is effectively Hit@5. Bold row = configuration deployed in the live system."
    graded_note = "Fully correct: every key fact present (auto-check, then manual review). Partial: some key facts. Correct GR cited: expected GR among the answer's citations."
    tables = {
        "table1_corpus": dict(title="Table 1. Corpus of indexed Government Resolutions", header=h1, rows=t1,
                              widths=[0.3, 1.8, 0.85, 0.95, 3.9, 0.85, 0.6, 0.85], numeric=(0, 5, 6, 7), font=7),
        "table2_setup": dict(title="Table 2. Experimental setup", header=h2, rows=t2, widths=[1.6, 5.4]),
        "table2b_question_set": dict(title="Table 2b. Test question set", header=h2b, rows=t2b,
                                     widths=[1.4, 3.0, 0.8, 0.8, 0.7], numeric=(2, 3, 4), highlight=lambda r: r[0] == "Total"),
        "table3_retrieval_overall": dict(title="Table 3. Retrieval accuracy by configuration (75 answerable questions)", header=h3, rows=t3,
                                         widths=[2.0, 1.25, 1.25, 1.25, 0.7, 1.1], numeric=(1, 2, 3, 4, 5), highlight=deployed, note=rerank_note),
        "table4_retrieval_by_tier": dict(title="Table 4. Retrieval accuracy by question type", header=h4p, rows=t4p,
                                         widths=[1.9] + [0.62] * (len(h4p) - 1), numeric=tuple(range(1, len(h4p))), highlight=deployed, font=7,
                                         note=f"Questions per type: {n4}. Bold row = deployed configuration."),
        "table5_retrieval_by_language": dict(title="Table 5. Retrieval accuracy by query language", header=h5p, rows=t5p,
                                             widths=[1.9] + [0.75] * (len(h5p) - 1), numeric=tuple(range(1, len(h5p))), highlight=deployed,
                                             note=f"Questions per language: {n5}. Bold row = deployed configuration."),
        "table6_end_to_end": dict(title="Table 6. End-to-end answer accuracy (live system, phi4-mini)", header=h6, rows=t6,
                                  widths=[1.7, 0.4, 1.15, 1.15, 1.15, 1.15, 1.0], numeric=(1, 2, 3, 4, 5, 6), note=graded_note +
                                  f" Negative traps refused: {s['correct_refusal']}/{s['negatives']}. Median latency {s.get('median_sec')} s."),
        "table7_unit_tests": dict(title="Table 7. Automated test suite (unittest)", header=h7, rows=t7,
                                  widths=[2.2, 0.9, 0.9, 1.1, 0.9], numeric=(1, 2, 3, 4), highlight=lambda r: r[0] == "TOTAL"),
        "table8_functional_tests": dict(title="Table 8. Functional test cases", header=h8, rows=t8,
                                        widths=[0.5, 0.95, 1.4, 1.6, 1.25, 1.75, 0.5, 1.6], font=6.5),
    }
    # Appendix in pages of 27 rows so each image stays readable
    app_h = ["ID", "Type", "Lang", "Question", "Expected GR", "Rank*", "Grade"]
    grade_word = {"Y": "Correct", "P": "Partial", "N": "Wrong"}
    app = [[r[0], r[1].replace("negative_trap", "negative"), r[2], r[3], r[4], r[6],
            ("Refused" if r[10] == "Y" else "Answered") if r[1] == "negative_trap" else grade_word.get(r[10], r[10])] for r in ta]
    for p in range(0, len(app), 27):
        k = p // 27 + 1
        tables[f"appendix_questions_p{k}"] = dict(
            title=f"Appendix A{k}. Test questions {p + 1}-{min(p + 27, len(app))} of {len(app)}", header=app_h, rows=app[p:p + 27],
            widths=[0.55, 0.95, 0.4, 4.6, 1.55, 0.5, 0.75], numeric=(5,), font=6.5,
            note="*Rank of the expected GR under the deployed hybrid retrieval (score_retrieval.py). Grade from the live answer run." if k == 1 else None)
    table_imgs = make_table_images(out, tables)

    write_dash_data(out / "data", retr, ans_rows, s, units, corpus, configs)
    if t8:
        write_csv(out / "data" / "functional_tests.csv", ["id", "area", "description", "input", "expected", "actual", "status", "note"],
                  [r for r in t8], bom=False)
    figs = make_figures(out, retr, ans_rows, units, corpus)

    results = dict(answer_summary=s, retrieval_overall=retr["overall"], retrieval_per_tier=retr.get("per_tier"),
                   retrieval_per_language=retr.get("per_language"), corpus=corpus["summary"], units=units,
                   answers=ans_rows, sources=dict(retrieval=args.retrieval.name, harness=args.harness.name))
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    md = [f"# Niyamsetu — evaluation results on the 25-GR corpus\n",
          f"Sources: `{args.retrieval.name}`, `{args.harness.name}`. Figures: {', '.join(figs) or 'not generated'}.\n",
          "## Table 1. Corpus\n", md_table(h1, t1),
          "\n## Table 3. Retrieval accuracy (file-level, top-12)\n", md_table(h3, t3),
          "\n## Table 4. Retrieval by question tier\n", md_table(h4, t4),
          "\n## Table 5. Retrieval by query language\n", md_table(h4, t5),
          "\n## Table 6. End-to-end answer accuracy\n", md_table(h6, t6),
          f"\nNegative traps (no answer in corpus): correct refusal {pct(s['correct_refusal'], s['negatives'])}. "
          f"Response language matched question: {pct(s['language_match'], s['lang_n'])}. "
          f"Latency median {s.get('median_sec')} s, p90 {s.get('p90_sec')} s, max {s.get('max_sec')} s.\n",
          "\n## Table 7. Automated test suite\n", md_table(h7, t7)]
    if t8:
        md += ["\n## Table 8. Functional test cases (live API)\n", md_table(h8, t8)]
    (out / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print("table images:", ", ".join(table_imgs))
    print(json.dumps(s, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
