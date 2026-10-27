#!/usr/bin/env python
# backend/tests/test_quick_eval.py
# ─────────────────────────────────────────────────────────
# Offline unit tests for tests/quick_eval.py. Fake ask and fake judge
# functions stand in for the backend and the LLM; nothing touches the
# network.
#
# Run:
#   python -m unittest tests.test_quick_eval -q
# ─────────────────────────────────────────────────────────

import io
import json
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))

import quick_eval as qe   # noqa: E402

GOLD = TESTS_DIR / "gold_natural.json"
REFUSAL_EN = "This information was not found in the uploaded documents."


def ok(answer="The rate is five times the fee.", cite=("GR5.pdf", "1")):
    return {"http_status": 200, "error": "", "answer": answer,
            "citations": [{"file": cite[0], "page": cite[1]}] if cite else [],
            "success": True, "client_elapsed_sec": 2.0}


def failed(error, status=None):
    return {"http_status": status, "error": error, "answer": "", "citations": [],
            "success": False, "client_elapsed_sec": 1.0}


def question(qid="nat-001", tier="detail", language="en", expected_file="GR5.pdf"):
    return {"id": qid, "tier": tier, "language": language, "question": "q?",
            "expected_file": expected_file, "expected_page": 1 if expected_file else None,
            "evidence_snippet": "पाच पट" if expected_file else None}


def judge_returning(*outputs):
    calls = []
    outputs = list(outputs)

    def judge(prompt):
        calls.append(prompt)
        return outputs.pop(0) if outputs else '{"grade":"Y","reason":"same fact"}'
    judge.calls = calls
    return judge


def quiet(fn, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class SelectionTests(unittest.TestCase):

    def setUp(self):
        self.gold = json.loads(GOLD.read_text(encoding="utf-8"))["questions"]

    def test_smoke_subset_ids_and_order(self):
        picked = [q["id"] for q in qe.select_questions(self.gold, "smoke")]
        self.assertEqual(picked, [
            "nat-002", "nat-005", "nat-007", "nat-008", "nat-010", "nat-014", "nat-019",
            "nat-020", "nat-023", "nat-029", "nat-030", "nat-034", "nat-038", "nat-041",
            "nat-042", "nat-047", "nat-058", "nat-060", "nat-062"])

    def test_all_subset_keeps_gold_order(self):
        self.assertEqual(len(qe.select_questions(self.gold, "all")), len(self.gold))

    def test_ids_override_subset_and_keep_their_order(self):
        picked = qe.select_questions(self.gold, "smoke", ids="nat-060, nat-001")
        self.assertEqual([q["id"] for q in picked], ["nat-060", "nat-001"])

    def test_unknown_id_is_an_error(self):
        with self.assertRaises(SystemExit):
            qe.select_questions(self.gold, "smoke", ids="nat-999")

    def test_argument_defaults(self):
        args = qe.parse_args(["quick_eval.py", "--tag", "t"])
        self.assertEqual(args.workers, 1)
        self.assertEqual(args.subset, "smoke")
        self.assertEqual(args.judge_model, "qwen2.5:14b-instruct")
        self.assertEqual(args.base_url, "http://localhost:8000")
        self.assertEqual(args.gold.name, "gold_natural.json")
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), \
                unittest.mock.patch("sys.stderr", io.StringIO()):
            qe.parse_args(["quick_eval.py"])          # --tag is required


class RetryTests(unittest.TestCase):

    def run_policy(self, outcomes):
        calls, sleeps = [], []

        def ask_one(q):
            calls.append(q["id"])
            return outcomes.pop(0)
        outcome, attempts = qe.ask_with_policy(question(), ask_one, sleep=sleeps.append)
        return outcome, attempts, calls, sleeps

    def test_no_retry_after_read_timeout(self):
        outcome, attempts, calls, sleeps = self.run_policy([failed("ReadTimeout: timed out"), ok()])
        self.assertEqual((attempts, len(calls), sleeps), (1, 1, []))
        self.assertEqual(outcome["error"], "ReadTimeout: timed out")

    def test_retry_once_after_a_connection_error(self):
        outcome, attempts, calls, sleeps = self.run_policy(
            [failed("ConnectError: [WinError 10061] refused"), ok()])
        self.assertEqual((attempts, len(calls), sleeps), (2, 2, [5.0]))
        self.assertEqual(outcome["error"], "")

    def test_connection_error_twice_is_recorded_not_retried_again(self):
        outcome, attempts, calls, _ = self.run_policy(
            [failed("ConnectTimeout: x"), failed("ConnectError: y"), ok()])
        self.assertEqual((attempts, len(calls)), (2, 2))
        self.assertEqual(outcome["error"], "ConnectError: y")

    def test_no_retry_after_an_http_error(self):
        _, attempts, calls, _ = self.run_policy([failed("Internal Server Error", status=500), ok()])
        self.assertEqual((attempts, len(calls)), (1, 1))

    def test_answers_keep_question_order_with_several_workers(self):
        qs = [question(f"nat-{n:03d}") for n in range(1, 6)]
        records = quiet(qe.collect_answers, qs, lambda q: ok(), workers=3)
        self.assertEqual([r["id"] for r in records], [q["id"] for q in qs])


class JudgeTests(unittest.TestCase):

    def test_parse_plain_fenced_and_wrapped_json(self):
        self.assertEqual(qe.parse_judge('{"grade":"Y","reason":"same amount"}'),
                         {"grade": "Y", "reason": "same amount"})
        self.assertEqual(qe.parse_judge('```json\n{"grade": "p", "reason": "partial"}\n```')["grade"], "P")
        self.assertEqual(qe.parse_judge('Here: {"grade":"N","reason":"wrong"} done')["grade"], "N")

    def test_parse_rejects_malformed_or_out_of_range(self):
        for text in ("Y", "", '{"grade":"Maybe","reason":"x"}', "{grade: Y}", "[1]"):
            self.assertIsNone(qe.parse_judge(text), text)

    def test_malformed_output_retries_once_then_err(self):
        judge = judge_returning("not json", "still not json", '{"grade":"Y","reason":"x"}')
        record = qe.answer_record(question(), ok(), 1)
        self.assertEqual(qe.grade_record(record, judge)["grade"], "ERR")
        self.assertEqual(len(judge.calls), 2)

    def test_malformed_then_valid_uses_the_second_output(self):
        judge = judge_returning("oops", '{"grade":"P","reason":"incomplete"}')
        record = qe.answer_record(question(), ok(), 1)
        self.assertEqual(qe.grade_record(record, judge), {"grade": "P", "reason": "incomplete"})
        self.assertEqual(len(judge.calls), 2)

    def test_judge_prompt_carries_question_evidence_and_answer(self):
        judge = judge_returning()
        record = qe.answer_record(question(), ok("उत्तर पाच पट"), 1)
        qe.grade_record(record, judge)
        self.assertIn("QUESTION:\nq?", judge.calls[0])
        self.assertIn("EVIDENCE (the passage of the document that answers the question):\nपाच पट",
                      judge.calls[0])
        self.assertIn("SYSTEM ANSWER:\nउत्तर पाच पट", judge.calls[0])
        self.assertIn('{"grade":"Y" or "P" or "N","reason":"<one short sentence>"}', judge.calls[0])

    def test_make_judge_payload(self):
        sent = {}

        class Res:
            def raise_for_status(self):
                pass

            def json(self):
                return {"message": {"content": '{"grade":"Y","reason":"ok"}'}}

        def post(url, json=None, headers=None, timeout=None):
            sent.update(url=url, json=json, headers=headers)
            return Res()

        judge = qe.make_judge("https://llm.example/", "qwen2.5:14b-instruct", post=post)
        self.assertEqual(judge("p"), '{"grade":"Y","reason":"ok"}')
        self.assertEqual(sent["url"], "https://llm.example/api/chat")
        self.assertEqual(sent["json"], {"model": "qwen2.5:14b-instruct",
                                        "messages": [{"role": "user", "content": "p"}],
                                        "stream": False,
                                        "options": {"temperature": 0, "num_ctx": 8192}})


class GradingTests(unittest.TestCase):

    def test_trap_graded_by_refusal_without_judge(self):
        judge = judge_returning()
        trap = question("nat-058", tier="negative_trap", expected_file=None)
        refused = qe.answer_record(trap, ok(REFUSAL_EN, cite=None), 1)
        answered = qe.answer_record(trap, ok("Yes, by 4 percent."), 1)
        self.assertEqual(qe.grade_record(refused, judge)["grade"], "Y")
        self.assertEqual(qe.grade_record(answered, judge)["grade"], "N")
        self.assertEqual(judge.calls, [])
        self.assertIsNone(refused["citation_file_hit"])

    def test_infra_records_are_never_judged_and_excluded_from_rates(self):
        judge = judge_returning('{"grade":"N","reason":"wrong"}')
        records = [
            qe.answer_record(question("nat-001"), ok(), 1),
            qe.answer_record(question("nat-002"), ok("⚠️ LLM error: boom"), 1),
            qe.answer_record(question("nat-003", language="mr"),
                             failed("ReadTimeout: timed out"), 1),
        ]
        quiet(qe.grade_all, records, judge)
        self.assertEqual([r["grade"] for r in records], ["N", "INFRA", "INFRA"])
        self.assertEqual(len(judge.calls), 1)
        s = qe.summarise(records)
        self.assertEqual(s["counts"]["INFRA"], 2)
        self.assertEqual(s["graded"], 1)
        self.assertEqual(s["strict"], {"k": 0, "n": 1, "rate": 0.0})
        self.assertEqual(s["marathi"]["graded"], 0)
        self.assertIsNone(s["marathi"]["strict"]["rate"])
        self.assertEqual(s["median_sec"], 2.0)       # infra latencies excluded

    def test_strict_lenient_and_language_breakdown(self):
        grades = [("Y", "mr"), ("P", "mr"), ("N", "en"), ("Y", "en"), ("ERR", "en")]
        records = []
        for n, (g, lang) in enumerate(grades, 1):
            r = qe.answer_record(question(f"nat-{n:03d}", language=lang), ok(), 1)
            r.update(grade=g, reason="")
            records.append(r)
        s = qe.summarise(records)
        self.assertEqual(s["graded"], 4)
        self.assertEqual(s["strict"], {"k": 2, "n": 4, "rate": 0.5})
        self.assertEqual(s["lenient"], {"k": 3, "n": 4, "rate": 0.75})
        self.assertEqual(s["counts"]["ERR"], 1)
        self.assertEqual(s["marathi"]["lenient"], {"k": 2, "n": 2, "rate": 1.0})
        self.assertEqual(s["english"]["strict"], {"k": 1, "n": 2, "rate": 0.5})


class MainTests(unittest.TestCase):

    def run_main(self, ask_one, judge=None, extra=(), preflight=(True, "ok")):
        self.tmp = tempfile.TemporaryDirectory()
        out = Path(self.tmp.name)

        class Client:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        argv = ["quick_eval.py", "--tag", "t", "--ids", "nat-002,nat-058",
                "--out-dir", str(out), *extra]
        with redirect_stdout(io.StringIO()) as stdout:
            code = qe.main(argv, client_factory=Client, preflight=lambda url: preflight,
                           read_url=lambda: "http://llm", ask_one=ask_one,
                           judge=judge or judge_returning(), sleep=lambda s: None)
        return code, out, stdout.getvalue()

    def tearDown(self):
        if hasattr(self, "tmp"):
            self.tmp.cleanup()

    def test_stale_url_exits_4_and_writes_nothing(self):
        for marker in ("tunnel viewing-unretired-jazz.ngrok-free.app is offline",
                       "ERR_NGROK_3200"):
            with self.subTest(marker=marker):
                code, out, text = self.run_main(lambda q: ok(f"⚠️ LLM error: {marker}"))
                self.assertEqual(code, 4)
                self.assertIn("BACKEND USES A STALE OLLAMA_BASE_URL: restart the backend and "
                              "clear OLLAMA_BASE_URL environment variables", text)
                self.assertEqual(list(out.iterdir()), [])
                self.tmp.cleanup()

    def test_preflight_failure_exits_2(self):
        code, out, text = self.run_main(lambda q: self.fail("must not ask"),
                                        preflight=(False, "HTTP 502"))
        self.assertEqual(code, 2)
        self.assertIn("HTTP 502", text)
        self.assertEqual(list(out.iterdir()), [])

    def test_outputs_contain_full_answers_and_the_note(self):
        long_answer = "Section 20 requires twenty-five lakhs. " * 20
        code, out, text = self.run_main(
            lambda q: ok(REFUSAL_EN, cite=None) if q["id"] == "nat-058" else ok(long_answer))
        self.assertEqual(code, 0)
        self.assertIn("judge grades are automatic and approximate; verify by hand before "
                      "reporting", text)
        [json_path] = list(out.glob("quick_eval_t_*.json"))
        [md_path] = list(out.glob("quick_eval_t_*.md"))
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual([r["grade"] for r in payload["results"]], ["Y", "Y"])
        self.assertEqual(payload["results"][0]["answer"], long_answer)
        self.assertIn(long_answer.strip(), md_path.read_text(encoding="utf-8"))
        self.assertIn("| id | language | tier | grade | seconds | refused | citation_file_hit "
                      "| reason |", md_path.read_text(encoding="utf-8"))


class CompareTests(unittest.TestCase):

    def test_improved_regressed_unchanged_and_deltas(self):
        previous = {"tag": "baseline", "summary": {"strict": {"rate": 0.25}, "lenient": {"rate": 0.5}},
                    "results": [{"id": "nat-001", "grade": "N"}, {"id": "nat-002", "grade": "Y"},
                                {"id": "nat-003", "grade": "P"}, {"id": "nat-004", "grade": "INFRA"}]}
        records = []
        for qid, g in (("nat-001", "Y"), ("nat-002", "P"), ("nat-003", "P"), ("nat-004", "N")):
            r = qe.answer_record(question(qid), ok(), 1)
            r.update(grade=g, reason="")
            records.append(r)
        c = qe.compare(previous, records, qe.summarise(records))

        self.assertEqual(c["changes"]["improved"], [{"id": "nat-001", "before": "N", "after": "Y"}])
        self.assertEqual(c["changes"]["regressed"], [{"id": "nat-002", "before": "Y", "after": "P"}])
        self.assertEqual([e["id"] for e in c["changes"]["unchanged"]], ["nat-003"])
        self.assertEqual([e["id"] for e in c["changes"]["not_comparable"]], ["nat-004"])
        self.assertEqual(c["strict_delta"], 0.0)          # 1/4 now vs 0.25 before
        self.assertEqual(c["lenient_delta"], 0.25)        # 3/4 now vs 0.5 before
        text = "\n".join(qe.compare_lines(c))
        self.assertIn("- improved (1): nat-001 N->Y", text)
        self.assertIn("- regressed (1): nat-002 Y->P", text)
        self.assertIn("strict accuracy delta: +0.0 pts; lenient accuracy delta: +25.0 pts", text)

    def test_compare_flag_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            prev = Path(tmp) / "prev.json"
            prev.write_text(json.dumps({"tag": "baseline",
                                        "summary": {"strict": {"rate": 0.0}, "lenient": {"rate": 0.0}},
                                        "results": [{"id": "nat-002", "grade": "N"},
                                                    {"id": "nat-058", "grade": "Y"}]}),
                            encoding="utf-8")

            class Client:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

            with redirect_stdout(io.StringIO()) as stdout:
                code = qe.main(["quick_eval.py", "--tag", "fix", "--ids", "nat-002,nat-058",
                                "--compare", str(prev), "--out-dir", tmp],
                               client_factory=Client, preflight=lambda url: (True, "ok"),
                               read_url=lambda: "http://llm",
                               ask_one=lambda q: ok("Yes, 4 percent."),
                               judge=judge_returning(), sleep=lambda s: None)
            self.assertEqual(code, 0)
            out = stdout.getvalue()
            self.assertIn("- improved (1): nat-002 N->Y", out)
            self.assertIn("- regressed (1): nat-058 Y->N", out)


if __name__ == "__main__":
    unittest.main()
