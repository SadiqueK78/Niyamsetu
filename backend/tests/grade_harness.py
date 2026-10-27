#!/usr/bin/env python
# backend/tests/grade_harness.py
# ─────────────────────────────────────────────────────────
# LLM grader for a harness capture.
#
# run_harness.py captures only — it records what the live system answered
# and the untouched gold answer, and deliberately scores nothing. This
# script is the scoring half: it asks the LLM to compare each captured
# answer against its gold answer and label the result.
#
# Why a model judges instead of a string comparison:
#   The gold answers are short Marathi facts ("14 ऑगस्ट, 2026") and the
#   system answers are sentences that contain them. Exact match scores a
#   correct answer wrong, and substring match scores "16-02-2018 नाही, तर
#   14 ऑगस्ट 2026" correct whichever way it is phrased. The interesting
#   distinctions here — right fact / partial fact / wrong fact / invented
#   fact — are judgements, so they are made by a judge.
#
# The input file is never modified. Results are written to a sibling copy
# with a _graded.json suffix, so a capture can be re-graded as often as
# needed and the original stays the record of what the system said.
#
# Usage:
#   python tests/grade_harness.py tests/harness_results_20260827_120429.json
#
# Writes:
#   tests/harness_results_20260827_120429_graded.json
# ─────────────────────────────────────────────────────────

import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings                      # noqa: E402
from core.rag import get_llm                     # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402


# The four labels the judge may return, plus the one this script assigns
# when it cannot read a label out of the reply.
LABELS = ["CORRECT", "PARTIAL", "WRONG", "HALLUCINATED"]
UNPARSED = "UNPARSED"
ALL_LABELS = LABELS + [UNPARSED]

# Word-bounded so "INCORRECT" is not read as "CORRECT" — the substring is
# there, and taking it would invert the verdict. A reply that says only
# "INCORRECT" therefore lands in UNPARSED rather than being scored
# backwards, which is the safer way to be wrong.
LABEL_PATTERN = re.compile(r"\b(" + "|".join(LABELS) + r")\b")

# The tier whose correct behaviour is a refusal rather than an answer.
NEGATIVE_TIER = "negative_trap"

JUDGE_INSTRUCTION = (
    "You are grading one answer produced by a document question-answering "
    "system over Maharashtra Government Resolution (GR) documents.\n"
    "Compare the Actual answer against the Gold answer and reply with "
    "exactly one of these labels, and nothing else:\n"
    "  CORRECT      - the Actual answer states the same fact as the Gold "
    "answer, whatever its wording or language\n"
    "  PARTIAL      - the Actual answer is on the right fact but incomplete, "
    "or it is correct while omitting a required part of the Gold answer\n"
    "  WRONG        - the Actual answer states a different fact, or does not "
    "answer the question\n"
    "  HALLUCINATED - the Actual answer states a specific fact that is not "
    "the Gold answer and appears to be invented\n"
    "Reply with the single label only. No explanation, no punctuation."
)

# Appended for the refusal tier, where the gold behaviour is inverted: the
# question deliberately asks for something the corpus does not contain, so
# an answer is the failure and a refusal is the pass.
NEGATIVE_TRAP_NOTE = (
    "\nIMPORTANT - this question is a negative trap. The information it asks "
    "for is absent from the documents, so the correct behaviour is DECLINING "
    "to answer (for example, saying the information was not found). Label a "
    "decline or a statement that the information is not present as CORRECT. "
    "Label any specific fabricated answer as HALLUCINATED."
)


def build_judge_prompt(record: dict) -> str:
    """
    The full judge prompt for one record.

    Includes the tier verbatim — it tells the judge what kind of question
    it is looking at (a direct lookup, a synthesis across sources, a
    cross-reference, a trap), which changes what counts as complete.
    """
    instruction = JUDGE_INSTRUCTION
    if record.get("tier") == NEGATIVE_TIER:
        instruction += NEGATIVE_TRAP_NOTE

    return (
        f"{instruction}\n\n"
        f"Question: {record.get('question', '')}\n\n"
        f"Tier: {record.get('tier', 'unknown')}\n\n"
        f"Gold answer: {record.get('gold_answer', '')}\n\n"
        f"Actual answer: {record.get('actual_answer', '')}\n\n"
        f"Label:"
    )


def parse_label(reply: str) -> str:
    """
    First label appearing in the judge's reply, or UNPARSED.

    "First" by position, not by preference: a judge that reasons before
    answering tends to name the label it settles on last, but one that
    answers then explains names it first, and only the position is
    knowable without reading the prose.
    """
    match = LABEL_PATTERN.search((reply or "").upper())
    return match.group(1) if match else UNPARSED


def grade_record(llm, record: dict) -> tuple:
    """
    Judges one record.

    Returns:
        (label, error) — error is "" on success. A failed call is labelled
        UNPARSED rather than skipped, so the counts always add up to the
        number of records and a broken run cannot quietly look like a
        clean one.
    """
    try:
        response = llm.invoke(
            [HumanMessage(content=build_judge_prompt(record))],
            stream=False,
        )
        return parse_label(response.content), ""
    except Exception as e:
        return UNPARSED, f"{type(e).__name__}: {e}"


def print_table(records: list) -> None:
    """Counts per label, overall and per tier, then the ids under each."""
    tiers = sorted({r.get("tier", "unknown") for r in records})

    def count(label, tier=None):
        return sum(
            1 for r in records
            if r.get("verdict") == label and (tier is None or r.get("tier", "unknown") == tier)
        )

    width = max([len(t) for t in tiers] + [len("tier")]) + 2

    print()
    print("─" * (width + 14 * len(ALL_LABELS) + 8))
    header = "tier".ljust(width) + "".join(l[:12].rjust(14) for l in ALL_LABELS) + "total".rjust(8)
    print(header)
    print("─" * (width + 14 * len(ALL_LABELS) + 8))

    for tier in tiers:
        row = tier.ljust(width)
        row += "".join(str(count(l, tier)).rjust(14) for l in ALL_LABELS)
        row += str(sum(1 for r in records if r.get("tier", "unknown") == tier)).rjust(8)
        print(row)

    print("─" * (width + 14 * len(ALL_LABELS) + 8))
    total = "ALL".ljust(width) + "".join(str(count(l)).rjust(14) for l in ALL_LABELS)
    total += str(len(records)).rjust(8)
    print(total)
    print("─" * (width + 14 * len(ALL_LABELS) + 8))

    print()
    for label in ALL_LABELS:
        ids = [str(r.get("id", "?")) for r in records if r.get("verdict") == label]
        print(f"{label:14} ({len(ids):3})  {', '.join(ids) if ids else '-'}")


def main(argv: list) -> int:
    if len(argv) != 2:
        print(__doc__ or "", end="")
        print("usage: python tests/grade_harness.py <results_json_path>")
        return 2

    in_path = Path(argv[1])
    if not in_path.is_file():
        print(f"not a file: {in_path}")
        return 1

    data = json.loads(in_path.read_text(encoding="utf-8"))
    records = data.get("results")
    if not isinstance(records, list):
        print(f"{in_path.name} has no 'results' list — is this a harness capture?")
        return 1

    out_path = in_path.with_name(in_path.stem + "_graded.json")

    print(f"grading {len(records)} record(s) from {in_path.name}")
    print(f"judge:  {settings.LLM_MODEL} at {settings.OLLAMA_BASE_URL}")
    print(f"output: {out_path.name}")
    print()

    llm = get_llm()

    # Judged into copies, so the records that are written are built here and
    # the parsed input is never mutated in place.
    graded = []
    for n, record in enumerate(records, 1):
        label, error = grade_record(llm, record)
        entry = dict(record)
        entry["verdict"] = label
        if error:
            entry["verdict_error"] = error
        graded.append(entry)

        note = f"  ({error})" if error else ""
        print(f"  [{n:3}/{len(records)}] {str(record.get('id', '?')):10} "
              f"{str(record.get('tier', '?')):16} -> {label}{note}", flush=True)

    out = dict(data)
    out["results"] = graded
    out["graded"] = True

    # "llm" holds the models involved in a run. The capture does not write
    # one today, so this creates it rather than assuming it is there, and
    # only ever adds the judge key — whatever else a future capture records
    # about its models is left alone.
    llm_info = dict(out.get("llm") or {})
    llm_info["judge"] = settings.LLM_MODEL
    out["llm"] = llm_info

    out_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print_table(graded)
    print()
    print(f"wrote {out_path}")
    print(f"input untouched: {in_path}")
    return 0


if __name__ == "__main__":
    # The questions and answers are Marathi; a cp1252 console raises on the
    # first one printed. Nothing below prints document text, but an error
    # path might, and losing a whole grading run to a console encoding would
    # be absurd.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    raise SystemExit(main(sys.argv))
