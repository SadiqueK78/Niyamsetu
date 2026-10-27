#!/usr/bin/env python
# backend/tests/test_harness_natural.py
# ─────────────────────────────────────────────────────────
# Offline unit tests for the natural-mode auto-checks in
# tests/run_harness.py. Every API response here is a fake dict shaped like
# POST /api/query/chat's; nothing touches a network, a backend or an LLM.
#
# Run:
#   python -m unittest tests.test_harness_natural -q
# ─────────────────────────────────────────────────────────

import ast
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))

import run_harness as rh   # noqa: E402


def response(answer="", citations=(), status=200, error=""):
    """A fake outcome, shaped like ask()'s return value."""
    return {"http_status": status, "error": error, "answer": answer,
            "citations": [dict(c) for c in citations], "language": "",
            "success": not error, "server_elapsed_sec": 1.0,
            "client_elapsed_sec": 1.5}


def cite(file, page):
    # As build_citations() returns it: page is a 1-based display string.
    return {"file": file, "page": page, "preview": "...", "relevance_score": 1.0}


def question(qid="nat-001", tier="detail", language="en",
             expected_file="GR5.pdf", expected_page=1):
    return {"id": qid, "tier": tier, "language": language, "question": "q?",
            "expected_file": expected_file, "expected_page": expected_page}


def run(questions, ask_one):
    """run_natural with no real sleeping and the progress lines swallowed."""
    sleeps = []
    with redirect_stdout(io.StringIO()):
        records = rh.run_natural(questions, ask_one, sleep=sleeps.append)
    return records, sleeps


class PageConventionTests(unittest.TestCase):
    """The API page is already 1-based; gold expected_page is 1-based."""

    def test_display_page_equal_to_expected_page_hits(self):
        checks = rh.citation_checks(rh.cited_pairs([cite("GR5.pdf", "2")]), "GR5.pdf", 2)
        self.assertTrue(checks["citation_page_hit"])

    def test_no_offset_is_applied(self):
        # The 0-based chunk page of page 2 is 1; seeing "1" in a citation
        # means page 1, so it must NOT match an expected page of 2.
        checks = rh.citation_checks(rh.cited_pairs([cite("GR5.pdf", "1")]), "GR5.pdf", 2)
        self.assertFalse(checks["citation_page_hit"])
        self.assertTrue(checks["citation_file_hit"])

    def test_right_page_of_the_wrong_file_misses(self):
        checks = rh.citation_checks(rh.cited_pairs([cite("GR4.pdf", "1")]), "GR5.pdf", 1)
        self.assertFalse(checks["citation_page_hit"])
        self.assertFalse(checks["citation_file_hit"])

    def test_unknown_page_never_matches(self):
        checks = rh.citation_checks(rh.cited_pairs([cite("GR5.pdf", "?")]), "GR5.pdf", 1)
        self.assertFalse(checks["citation_page_hit"])

    def test_top_citation(self):
        pairs = rh.cited_pairs([cite("GR4.pdf", "1"), cite("GR5.pdf", "1")])
        checks = rh.citation_checks(pairs, "GR5.pdf", 1)
        self.assertFalse(checks["top_citation_is_expected"])
        self.assertTrue(checks["citation_page_hit"])

    def test_pairs_are_stored_exactly_as_returned(self):
        self.assertEqual(rh.cited_pairs([cite("letter 1.pdf", "2"), cite("GR1.pdf", "?")]),
                         [["letter 1.pdf", "2"], ["GR1.pdf", "?"]])

    def test_a_trap_has_no_citation_hits(self):
        checks = rh.citation_checks(rh.cited_pairs([cite("GR5.pdf", "1")]), None, None)
        self.assertFalse(any(checks.values()))


class LanguageMatchTests(unittest.TestCase):

    def test_marathi_answer_for_a_marathi_question(self):
        self.assertTrue(rh.language_match("निवासस्थानात जूनअखेरपर्यंत राहता येईल.", "mr"))

    def test_english_answer_for_a_marathi_question(self):
        self.assertFalse(rh.language_match("You may stay until the end of June.", "mr"))

    def test_marathi_threshold_is_inclusive_at_half(self):
        self.assertTrue(rh.language_match("कखab", "mr"))      # share 0.5
        self.assertFalse(rh.language_match("कखabc", "mr"))    # share 0.4

    def test_english_threshold_is_exclusive_at_a_fifth(self):
        self.assertFalse(rh.language_match("abcdक", "en"))    # share 0.2
        self.assertTrue(rh.language_match("abcdefghiक", "en"))  # share 0.1

    def test_english_answer_quoting_a_marathi_name_still_matches(self):
        answer = "The decision was signed by the under secretary (उर्मिला सावंत) on behalf of the department."
        self.assertTrue(rh.language_match(answer, "en"))

    def test_no_letters_matches_neither(self):
        for lang in ("mr", "en"):
            self.assertFalse(rh.language_match("", lang))
            self.assertFalse(rh.language_match("3,01,71,099/-", lang))

    def test_unknown_language_never_matches(self):
        self.assertFalse(rh.language_match("hello", "hi"))


class RefusalTests(unittest.TestCase):

    def test_both_fixed_messages_are_detected(self):
        for message in rh.REFUSAL_MESSAGES:
            self.assertTrue(rh.is_refusal(message))

    def test_message_inside_a_longer_answer(self):
        self.assertTrue(rh.is_refusal(
            'उत्तर: "हे माहिती दिलेल्या दस्तऐवजात आढळली नाही."\n\nस्रोत: —'))

    def test_line_break_and_missing_full_stop(self):
        self.assertTrue(rh.is_refusal("This information was not found\nin the uploaded documents"))

    def test_an_ordinary_answer_is_not_a_refusal(self):
        self.assertFalse(rh.is_refusal("The licence fee is five times the current rate."))

    def test_system_failures_are_not_refusals(self):
        self.assertFalse(rh.is_refusal(
            "⚠️ Vector store is not ready. Please ask admin to embed the GR documents first."))
        self.assertFalse(rh.is_refusal("⚠️ LLM timed out after 120s"))

    def test_copies_match_core_rag(self):
        # The harness copies the strings to avoid importing the backend;
        # read them out of core/rag.py's source so the copies cannot drift.
        tree = ast.parse((TESTS_DIR.parent / "core" / "rag.py").read_text(encoding="utf-8"))
        found = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.startswith("NOT_FOUND_"):
                        found[target.id] = node.value.value
        self.assertEqual(set(found.values()), set(rh.REFUSAL_MESSAGES))

    def test_trap_and_answerable_flags(self):
        refusal = rh.REFUSAL_MESSAGES[1]
        trap = dict(question(tier="negative_trap", expected_file=None, expected_page=None),
                    answer=refusal, citations=[])
        normal = dict(question(), answer=refusal, citations=[])
        self.assertEqual(rh.auto_checks(trap)["correct_refusal"], True)
        self.assertNotIn("false_refusal", rh.auto_checks(trap))
        self.assertEqual(rh.auto_checks(normal)["false_refusal"], True)
        self.assertNotIn("correct_refusal", rh.auto_checks(normal))


class RunTests(unittest.TestCase):

    def test_a_failing_question_does_not_stop_the_run(self):
        calls = []

        def ask_one(q):
            calls.append(q["id"])
            if q["id"] == "nat-002":
                return response(status=None, error="ReadTimeout: timed out")
            return response("The rate is five times the fee.", [cite("GR5.pdf", "1")])

        qs = [question("nat-001"), question("nat-002"), question("nat-003")]
        records, sleeps = run(qs, ask_one)

        self.assertEqual([r["id"] for r in records], ["nat-001", "nat-002", "nat-003"])
        self.assertEqual(calls, ["nat-001", "nat-002", "nat-002", "nat-003"])  # one retry
        self.assertEqual(sleeps, [rh.RETRY_DELAY])
        failed = records[1]
        self.assertEqual(failed["error"], "ReadTimeout: timed out")
        self.assertEqual(failed["attempts"], 2)
        self.assertFalse(failed["citation_file_hit"])
        self.assertTrue(records[2]["citation_page_hit"])

    def test_an_exception_from_ask_is_recorded_not_raised(self):
        def ask_one(q):
            raise ValueError("bad JSON")
        records, _ = run([question()], ask_one)
        self.assertIn("ValueError: bad JSON", records[0]["error"])

    def test_a_retry_that_succeeds_keeps_the_answer(self):
        attempts = iter([response(status=500, error="Internal Server Error"),
                         response("ok", [cite("GR5.pdf", "1")])])
        records, sleeps = run([question()], lambda q: next(attempts))
        self.assertEqual(records[0]["error"], "")
        self.assertEqual(records[0]["http_status"], 200)
        self.assertEqual(records[0]["attempts"], 2)
        self.assertEqual(len(sleeps), 1)

    def test_summary_and_grading_sheet(self):
        refusal = rh.REFUSAL_MESSAGES[0]
        answers = {
            "nat-001": response("The rate is five times the fee.", [cite("GR5.pdf", "1")]),
            "nat-002": response(status=None, error="ConnectError: refused"),
            "nat-003": response(refusal, []),
        }
        qs = [question("nat-001"), question("nat-002"),
              question("nat-003", tier="negative_trap", language="mr",
                       expected_file=None, expected_page=None)]
        records, _ = run(qs, lambda q: answers[q["id"]])
        s = rh.summarise_natural(records)

        self.assertEqual((s["citation_file_hit"]["k"], s["citation_file_hit"]["n"]), (1, 2))
        self.assertEqual((s["correct_refusal"]["k"], s["correct_refusal"]["n"]), (1, 1))
        self.assertEqual((s["false_refusal"]["k"], s["false_refusal"]["n"]), (0, 2))
        self.assertEqual(s["errors"], 1)
        self.assertEqual(s["max_sec"], 1.5)

        meta = {"run_at": "now", "base_url": "http://x", "question_file": "g.json", "total": 3}
        sheet = rh.grading_markdown(meta, records, s)
        self.assertIn("| Correct (Y/P/N) | Notes |", sheet)
        self.assertIn("50.0% (1/2)", sheet)
        self.assertIn("ERROR: ConnectError: refused", sheet)
        self.assertEqual(sum(1 for l in sheet.splitlines() if l.startswith("| nat-")), 3)


LLM_ERROR = "⚠️ LLM error: Server disconnected. Is Ollama running with qwen2.5:7b?"


def infra():
    """The answer core/rag.py returns when the LLM call raises."""
    return response("⚠️ LLM error: boom. Is Ollama running with qwen2.5:7b?")


def good():
    return response("The rate is five times the fee.", [cite("GR5.pdf", "1")])


class FakeClock:
    """A monotonic clock that only moves when the fake sleep is called."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeHTTPResponse:
    def __init__(self, status=200, body=None, text=None):
        self.status_code = status
        self._body, self._text = body, text

    def json(self):
        if self._text is not None:
            import json
            return json.loads(self._text)
        return self._body


class InfraClassifierTests(unittest.TestCase):

    def test_llm_error_answer(self):
        self.assertTrue(rh.is_infra_failure(response(LLM_ERROR)))

    def test_llm_error_without_variation_selector_or_with_leading_space(self):
        self.assertTrue(rh.is_infra_failure(response("⚠ LLM error: x")))
        self.assertTrue(rh.is_infra_failure(response("  ⚠️  LLM error: x")))

    def test_llm_error_only_counts_at_the_start(self):
        self.assertFalse(rh.is_infra_failure(response("The LLM error log is not a GR topic.")))

    def test_ngrok_error_in_answer_or_error_text(self):
        self.assertTrue(rh.is_infra_failure(response("tunnel said ERR_NGROK_3200")))
        self.assertTrue(rh.is_infra_failure(
            response(status=502, error="<html>ngrok ERR_NGROK_8012 upstream</html>")))

    def test_winerror(self):
        self.assertTrue(rh.is_infra_failure(response(
            status=None, error="ConnectError: [WinError 10061] No connection could be made")))

    def test_server_disconnected(self):
        self.assertTrue(rh.is_infra_failure(response(
            status=None, error="RemoteProtocolError: Server disconnected without sending a response.")))

    def test_read_timeout(self):
        self.assertTrue(rh.is_infra_failure(response(status=None, error="ReadTimeout: timed out")))
        self.assertFalse(rh.is_infra_failure(response(status=None, error="ConnectTimeout: timed out")))

    def test_answers_and_refusals_are_not_infra(self):
        self.assertFalse(rh.is_infra_failure(good()))
        self.assertFalse(rh.is_infra_failure(response(rh.REFUSAL_MESSAGES[0])))
        # Not on the list: a slow LLM is reported but is not an infra failure.
        self.assertFalse(rh.is_infra_failure(response("⚠️ LLM timed out after 120s")))
        self.assertFalse(rh.is_infra_failure(response(status=500, error="Internal Server Error")))

    def test_every_record_carries_the_flag(self):
        records, _ = run([question("a"), question("b")],
                         lambda q: infra() if q["id"] == "a" else good())
        self.assertEqual([r["infra_failure"] for r in records], [True, False])


class PreflightTests(unittest.TestCase):

    def fake_get(self, result):
        self.calls = []

        def get(url, headers=None, timeout=None):
            self.calls.append((url, headers, timeout))
            if isinstance(result, BaseException):
                raise result
            return result
        return get

    def test_qwen_listed_passes_and_sends_the_ngrok_header(self):
        get = self.fake_get(FakeHTTPResponse(body={"models": [{"name": "nomic-embed"},
                                                              {"name": "qwen2.5:7b"}]}))
        ok, _ = rh.preflight("https://x.ngrok-free.app/", get=get)
        self.assertTrue(ok)
        self.assertEqual(self.calls, [("https://x.ngrok-free.app/api/tags",
                                       {"ngrok-skip-browser-warning": "1"}, 15.0)])

    def test_model_field_is_accepted_as_well_as_name(self):
        get = self.fake_get(FakeHTTPResponse(body={"models": [{"model": "qwen2.5:14b"}]}))
        self.assertTrue(rh.preflight("http://h", get=get)[0])

    def test_failures(self):
        cases = [
            FakeHTTPResponse(body={"models": [{"name": "llama3:8b"}]}),     # wrong model
            FakeHTTPResponse(body={"models": []}),                           # none
            FakeHTTPResponse(body=[{"name": "qwen2.5:7b"}]),                 # not an object
            FakeHTTPResponse(status=502, body={"models": [{"name": "qwen2.5:7b"}]}),
            FakeHTTPResponse(text="<html>ERR_NGROK_3200</html>"),            # not JSON
            OSError("[WinError 10061] refused"),                             # transport
        ]
        for case in cases:
            with self.subTest(case=case):
                ok, reason = rh.preflight("http://h", get=self.fake_get(case))
                self.assertFalse(ok)
                self.assertTrue(reason)


class CircuitBreakerTests(unittest.TestCase):

    def run_breaker(self, questions, ask_one, checks):
        """run_natural with the breaker on, a fake clock and a scripted preflight."""
        clock = FakeClock()
        script = iter(checks)
        polls = []

        def check():
            polls.append(clock.now)
            return next(script)

        with redirect_stdout(io.StringIO()):
            result = rh.run_natural(questions, ask_one, sleep=clock.sleep,
                                    clock=clock, preflight_check=check)
        return result, polls, clock

    def test_three_infra_failures_trip_then_recovery_retries_and_continues(self):
        answers = {"q1": [infra()], "q2": [infra()], "q3": [infra(), good()], "q4": [good()]}
        qs = [question(i) for i in ("q1", "q2", "q3", "q4")]
        records, polls, clock = self.run_breaker(
            qs, lambda q: answers[q["id"]].pop(0),
            checks=[(False, "down"), (False, "down"), (True, "ok")])

        self.assertEqual([r["id"] for r in records], ["q1", "q2", "q3", "q4"])
        self.assertEqual(polls, [0.0, 20.0, 40.0])        # every 20 seconds
        self.assertEqual(clock.sleeps, [20.0, 20.0])      # no retry_delay: no HTTP error
        q3 = records[2]
        self.assertTrue(q3["breaker_retry"])
        self.assertFalse(q3["infra_failure"])
        self.assertEqual(q3["attempts"], 2)
        self.assertTrue(records[0]["infra_failure"] and records[1]["infra_failure"])
        self.assertNotIn("breaker_retry", records[3])

    def test_never_recovering_aborts_with_partial_results(self):
        written = []
        clock = FakeClock()
        polls = []

        def check():
            polls.append(clock.now)
            return False, "down"

        qs = [question(i) for i in ("q1", "q2", "q3", "q4")]
        asked = []

        def ask_one(q):
            asked.append(q["id"])
            return infra()

        with redirect_stdout(io.StringIO()), self.assertRaises(rh.RunAborted) as caught:
            rh.run_natural(qs, ask_one, sleep=clock.sleep, clock=clock,
                           preflight_check=check, on_record=lambda r: written.append(len(r)))

        self.assertEqual([r["id"] for r in caught.exception.records], ["q1", "q2", "q3"])
        self.assertEqual(asked, ["q1", "q2", "q3"])          # q4 never asked
        self.assertEqual(written[-1], 3)                      # partial results written
        self.assertEqual(polls[0], 0.0)
        self.assertEqual(polls[-1], 600.0)                    # up to 10 minutes
        self.assertEqual(len(polls), 31)

    def test_non_consecutive_failures_do_not_trip(self):
        seq = {"q1": infra(), "q2": infra(), "q3": good(), "q4": infra(), "q5": infra()}
        qs = [question(i) for i in seq]
        records, polls, _ = self.run_breaker(qs, lambda q: seq[q["id"]], checks=[])
        self.assertEqual(len(records), 5)
        self.assertEqual(polls, [])

    def test_streak_restarts_after_recovery(self):
        # Trip on q3, recover, q3's retry fails again: that counts as 1, so
        # q4 and q5 failing trips the breaker a second time at q5.
        seq = {"q1": [infra()], "q2": [infra()], "q3": [infra(), infra()],
               "q4": [infra()], "q5": [infra(), good()]}
        qs = [question(i) for i in seq]
        records, polls, _ = self.run_breaker(
            qs, lambda q: seq[q["id"]].pop(0), checks=[(True, "ok"), (True, "ok")])
        self.assertEqual(len(polls), 2)
        self.assertTrue(records[2]["breaker_retry"] and records[2]["infra_failure"])
        self.assertTrue(records[4]["breaker_retry"])
        self.assertFalse(records[4]["infra_failure"])

    def test_breaker_is_off_without_a_preflight(self):
        records, _ = run([question(i) for i in ("q1", "q2", "q3", "q4")], lambda q: infra())
        self.assertEqual(len(records), 4)

    def test_read_timeouts_trip_the_breaker(self):
        # Three questions, each timing out on both of its ordinary attempts
        # (6 calls); the breaker's retry of q3 is the 7th call and succeeds.
        calls = []

        def ask_one(q):
            calls.append(q["id"])
            if len(calls) <= 6:
                return response(status=None, error="ReadTimeout: timed out")
            return good()

        qs = [question(i) for i in ("q1", "q2", "q3")]
        records, polls, clock = self.run_breaker(qs, ask_one, checks=[(True, "ok")])
        self.assertEqual(len(polls), 1)
        self.assertEqual(clock.sleeps.count(rh.RETRY_DELAY), 3)   # ordinary retry still runs
        self.assertFalse(records[2]["infra_failure"])


class SummaryExclusionTests(unittest.TestCase):

    def test_infra_records_leave_answer_metrics_but_not_citation_metrics(self):
        trap = question("t1", tier="negative_trap", language="mr",
                        expected_file=None, expected_page=None)
        seq = {
            "a1": good(),                                                  # en answer, hit
            "a2": dict(infra(), citations=[cite("GR5.pdf", "1")]),         # infra, but cited
            "t1": infra(),                                                 # infra trap
        }
        records, _ = run([question("a1"), question("a2"), trap], lambda q: seq[q["id"]])
        s = rh.summarise_natural(records)

        self.assertEqual(s["infra_excluded"], 2)
        self.assertEqual((s["citation_file_hit"]["k"], s["citation_file_hit"]["n"]), (2, 2))
        self.assertEqual((s["language_match"]["k"], s["language_match"]["n"]), (1, 1))
        self.assertEqual(s["correct_refusal"]["n"], 0)
        self.assertEqual(s["false_refusal"]["n"], 1)
        self.assertEqual(s["median_sec"], 1.5)
        lines = "\n".join(rh.format_summary(s))
        self.assertIn("| excluded (infra_failure) | 2 |", lines)


class MainNaturalExitTests(unittest.TestCase):
    """main_natural's exit codes, with every network touchpoint replaced."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.saved = {n: getattr(rh, n) for n in
                      ("read_ollama_base_url", "preflight", "login", "ask_natural", "time")}

    def tearDown(self):
        for name, value in self.saved.items():
            setattr(rh, name, value)
        self.tmp.cleanup()

    def args(self):
        import argparse
        gold = self.out / "gold.json"
        import json
        gold.write_text(json.dumps({"questions": [question(f"q{i}") for i in range(1, 5)]}),
                        encoding="utf-8")
        out_dir = self.out / "results"
        return argparse.Namespace(questions=gold, scoped=False, validate=False, limit=None,
                                  timeout=None, out_dir=out_dir, base_url="http://backend",
                                  username="u", password="p", tag="natural")

    def test_preflight_failure_exits_2_and_writes_nothing(self):
        rh.read_ollama_base_url = lambda: "http://llm"
        rh.preflight = lambda url, get=None: (False, "HTTP 502")
        rh.login = lambda *a, **k: self.fail("must not log in after a failed preflight")
        args = self.args()
        out = io.StringIO()
        with redirect_stdout(out):
            code = rh.main_natural(args)
        self.assertEqual(code, 2)
        self.assertIn("PREFLIGHT FAILED: LLM endpoint not reachable, fix Colab and rerun",
                      out.getvalue())
        self.assertFalse(args.out_dir.exists() and any(args.out_dir.iterdir()))

    def test_unreadable_settings_count_as_a_preflight_failure(self):
        def boom():
            raise ValueError("bad .env")
        rh.read_ollama_base_url = boom
        with redirect_stdout(io.StringIO()):
            self.assertEqual(rh.main_natural(self.args()), 2)

    def test_abort_exits_3_with_aborted_true(self):
        import json
        import types
        clock = FakeClock()
        # Real time.time for elapsed stamps; fake sleep and clock for the breaker.
        rh.time = types.SimpleNamespace(time=self.saved["time"].time,
                                        sleep=clock.sleep, monotonic=clock)
        checks = iter([(True, "ok")])   # the initial preflight passes, then never again
        rh.read_ollama_base_url = lambda: "http://llm"
        rh.preflight = lambda url, get=None: next(checks, (False, "down"))
        rh.login = lambda *a, **k: "token"
        rh.ask_natural = lambda *a, **k: infra()
        args = self.args()
        with redirect_stdout(io.StringIO()):
            code = rh.main_natural(args)
        self.assertEqual(code, 3)
        [result] = list(args.out_dir.glob("harness_natural_*[0-9].json"))
        payload = json.loads(result.read_text(encoding="utf-8"))
        self.assertTrue(payload["aborted"])
        self.assertEqual(payload["completed"], 3)
        self.assertTrue(all(r["infra_failure"] for r in payload["results"]))
        self.assertTrue(list(args.out_dir.glob("harness_natural_*_grading.md")))


if __name__ == "__main__":
    unittest.main()
