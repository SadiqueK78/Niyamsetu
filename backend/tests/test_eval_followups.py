#!/usr/bin/env python
# backend/tests/test_eval_followups.py
# ─────────────────────────────────────────────────────────
# Offline unit tests for tests/eval_followups.py. A fake HTTP client
# stands in for the backend: it records every request body and answers
# from a script. Nothing touches the network or an LLM.
#
# Run:
#   python -m unittest tests.test_eval_followups -q
# ─────────────────────────────────────────────────────────

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))

import eval_followups as ef   # noqa: E402


class FakeResponse:
    def __init__(self, status=200, body=None, text=""):
        self.status_code = status
        self._body = body or {}
        self.text = text

    def json(self):
        return self._body


class FakeClient:
    """
    A stand-in for httpx.Client. Login returns a token; each chat request
    is answered by reply(body) and recorded in self.requests.
    """

    def __init__(self, reply):
        self.reply = reply
        self.requests = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json=None, headers=None, timeout=None):
        if url.endswith("/api/auth/login"):
            return FakeResponse(body={"access_token": "token"})
        self.requests.append({"url": url, "body": json, "headers": headers, "timeout": timeout})
        return self.reply(json)


def answer(text, *cites, status=200):
    return FakeResponse(status=status, body={
        "success": True, "answer": text, "language": "english", "elapsed_sec": 1.0,
        "citations": [{"file": f, "page": p, "preview": ""} for f, p in cites]})


CONV = {"id": "fu-001", "language": "en", "turn1_question": "What is the canteen decision?",
        "turn1_expected_file": "GR3.pdf", "turn2_question": "How much was approved for it?",
        "expected_file": "GR3.pdf", "expected_page": 2,
        "evidence_snippet": "...", "review_status": "draft"}


def conv(cid, **over):
    c = dict(CONV, id=cid)
    c.update(over)
    return c


def scripted(with_history, without_history, turn1=None):
    """reply(): turn 1, turn 2 with history, turn 2 without, told apart by body."""
    turn1 = turn1 or answer("It approves kitchen modernisation.", ("GR3.pdf", "1"))

    def reply(body):
        if body["query"] == CONV["turn1_question"]:
            return turn1
        return with_history if body["history"] else without_history
    return reply


def run_one(reply, conversation=CONV):
    client = FakeClient(reply)
    send = lambda q, h: ef.ask_with_history(client, "http://b", {"Authorization": "x"}, q, h, 300.0)
    return ef.run_conversation(conversation, send), client


class HistoryTests(unittest.TestCase):

    def test_three_requests_in_order_with_the_chat_page_payload(self):
        record, client = run_one(scripted(answer("Rs 3 crore", ("GR3.pdf", "2")),
                                          answer("Unclear", ("GR4.pdf", "1"))))
        bodies = [r["body"] for r in client.requests]
        self.assertEqual(len(bodies), 3)
        self.assertEqual(bodies[0], {"query": CONV["turn1_question"], "history": []})
        self.assertEqual(bodies[1], {"query": CONV["turn2_question"], "history": [
            {"role": "user", "content": CONV["turn1_question"]},
            {"role": "assistant", "content": "It approves kitchen modernisation."},
        ]})
        self.assertEqual(bodies[2], {"query": CONV["turn2_question"], "history": []})
        self.assertEqual(record["history_sent"], bodies[1]["history"])
        self.assertTrue(all(r["timeout"] == 300.0 for r in client.requests))
        self.assertTrue(all(r["url"] == "http://b/api/query/chat" for r in client.requests))

    def test_failed_turn1_sends_no_history_like_the_chat_page(self):
        record, client = run_one(scripted(answer("x"), answer("y"),
                                          turn1=FakeResponse(status=500, text="boom")))
        self.assertEqual(record["history_sent"], [])
        self.assertEqual(client.requests[1]["body"]["history"], [])

    def test_build_history_keeps_the_answer_verbatim(self):
        h = ef.build_history("q?", {"error": "", "answer": "  long answer \n"})
        self.assertEqual(h, [{"role": "user", "content": "q?"},
                             {"role": "assistant", "content": "  long answer \n"}])

    def test_a_raising_client_is_recorded_not_raised(self):
        def reply(body):
            raise ValueError("bad JSON")
        record, _ = run_one(reply)
        self.assertIn("ValueError: bad JSON", record["turn2_with_history"]["error"])


class MetricTests(unittest.TestCase):

    def records(self, outcomes):
        """outcomes: list of (with_history_response, without_history_response)."""
        out = []
        for n, (w, b) in enumerate(outcomes, 1):
            record, _ = run_one(scripted(w, b), conv(f"fu-{n:03d}"))
            out.append(record)
        return out

    def test_hits_and_improvement(self):
        recs = self.records([
            (answer("a", ("GR3.pdf", "2")), answer("b", ("GR4.pdf", "1"))),   # helps
            (answer("a", ("GR3.pdf", "1")), answer("b", ("GR3.pdf", "1"))),   # file only, both
            (answer("a", ("GR1.pdf", "2")), answer("b", ("GR3.pdf", "2"))),   # hurts
            (answer("a", ("GR3.pdf", "2")), answer("b", ("GR3.pdf", "2"))),   # both hit
        ])
        s = ef.summarise(recs)
        file_hit, page_hit = s["citation_file_hit"], s["citation_page_hit"]
        self.assertEqual(file_hit["with_history"], {"k": 3, "n": 4, "rate": 0.75})
        self.assertEqual(file_hit["without_history"], {"k": 3, "n": 4, "rate": 0.75})
        self.assertEqual(file_hit["improvement"], {"k": 0, "n": 4, "rate": 0.0})
        self.assertEqual(page_hit["with_history"]["k"], 2)
        self.assertEqual(page_hit["without_history"]["k"], 2)
        self.assertEqual(s["excluded_paired"], 0)

    def test_page_uses_the_api_display_page_with_no_offset(self):
        recs = self.records([(answer("a", ("GR3.pdf", "1")), answer("b", ("GR3.pdf", "3")))])
        self.assertFalse(recs[0]["turn2_with_history"]["citation_page_hit"])
        self.assertFalse(recs[0]["turn2_without_history"]["citation_page_hit"])


class InfraTests(unittest.TestCase):

    def test_markers(self):
        for text in ("⚠️ LLM error: x", "tunnel ERR_NGROK_3200", "[WinError 10061]",
                     "Server disconnected without sending a response."):
            self.assertTrue(ef.is_infra_failure({"answer": text, "error": ""}), text)
        self.assertTrue(ef.is_infra_failure(
            {"answer": "", "error": "RemoteProtocolError: Server disconnected"}))
        self.assertFalse(ef.is_infra_failure({"answer": "Rs 3 crore", "error": ""}))
        self.assertFalse(ef.is_infra_failure({"answer": "", "error": "Internal Server Error"}))

    def test_infra_failures_are_excluded_from_rates(self):
        infra = answer("⚠️ LLM error: Server disconnected")
        recs = []
        for n, (w, b, t1) in enumerate([
            (answer("a", ("GR3.pdf", "2")), answer("b", ("GR3.pdf", "2")), None),
            (infra, answer("b", ("GR4.pdf", "1")), None),                 # with-history infra
            (answer("a", ("GR3.pdf", "2")), infra, None),                 # baseline infra
            (answer("a", ("GR3.pdf", "2")), answer("b", ("GR3.pdf", "2")), infra),  # turn 1 infra
        ], 1):
            record, _ = run_one(scripted(w, b, turn1=t1), conv(f"fu-{n:03d}"))
            recs.append(record)
        s = ef.summarise(recs)
        self.assertEqual(s["citation_file_hit"]["with_history"]["n"], 2)       # fu-001, fu-003
        self.assertEqual(s["citation_file_hit"]["without_history"]["n"], 3)    # all but fu-003
        self.assertEqual(s["citation_file_hit"]["improvement"]["n"], 1)        # fu-001 only
        self.assertEqual((s["excluded_with_history"], s["excluded_without_history"]), (2, 1))
        self.assertEqual(s["infra_failures"], 3)
        self.assertIn("Excluded as infrastructure failures: 2 (with history), 1 (without "
                      "history), 3 (paired)", "\n".join(ef.summary_lines(s)))


class OutputTests(unittest.TestCase):

    def test_main_writes_json_and_grading_sheet(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gold = tmp / "gold.json"
            gold.write_text(json.dumps({"conversations": [conv("fu-001"), conv("fu-002")]}),
                            encoding="utf-8")
            reply = scripted(answer("Rs 3,01,71,099 | approved", ("GR3.pdf", "2")),
                             answer("Not found", ("GR4.pdf", "1")))
            saved = ef.read_query_rewrite
            ef.read_query_rewrite = lambda: "false"
            try:
                with redirect_stdout(io.StringIO()) as out:
                    code = ef.main(["eval_followups.py", "--gold", str(gold), "--tag", "rewrite_off",
                                    "--out-dir", str(tmp / "out")],
                                   client_factory=lambda: FakeClient(reply))
            finally:
                ef.read_query_rewrite = saved

            self.assertEqual(code, 0)
            self.assertIn("QUERY_REWRITE = false", out.getvalue())
            [json_path] = list((tmp / "out").glob("followups_rewrite_off_*.json"))
            [md_path] = list((tmp / "out").glob("followups_rewrite_off_*.md"))
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["query_rewrite"], "false")
            self.assertEqual(payload["completed"], 2)
            self.assertEqual(payload["summary"]["citation_file_hit"]["with_history"]["k"], 2)
            self.assertEqual(payload["summary"]["citation_file_hit"]["without_history"]["k"], 0)
            r = payload["results"][0]
            for key in ("turn1", "turn2_with_history", "turn2_without_history"):
                self.assertEqual(set(r[key]) >= {"answer", "citations", "seconds"}, True)

            sheet = md_path.read_text(encoding="utf-8")
            self.assertIn("- QUERY_REWRITE: false", sheet)
            self.assertIn("| id | turn2 question | expected file and page | cited files with "
                          "history | cited files without history | answer with history "
                          "(first 400 chars) | Correct (Y/P/N) |", sheet)
            row = next(l for l in sheet.splitlines() if l.startswith("| fu-001"))
            self.assertIn("GR3.pdf p2", row)
            self.assertIn("GR4.pdf p1", row)
            self.assertIn("3,01,71,099 \\| approved", row)   # pipes escaped
            self.assertTrue(row.endswith("|  |"))              # empty Correct column

    def test_gold_file_is_well_formed(self):
        gold = json.loads((TESTS_DIR / "gold_followups.json").read_text(encoding="utf-8"))
        convs = gold["conversations"]
        self.assertEqual([c["id"] for c in convs], [f"fu-{n:03d}" for n in range(1, 11)])
        files = ["GR1.pdf", "GR3.pdf", "GR4.pdf", "aadesh 2.pdf", "letter 3.pdf"]
        self.assertEqual([c["expected_file"] for c in convs], files * 2)
        self.assertEqual([c["language"] for c in convs], ["mr"] * 5 + ["en"] * 5)
        for c in convs:
            self.assertEqual(c["turn1_expected_file"], c["expected_file"])
            self.assertLessEqual(len(c["evidence_snippet"]), 200)
            self.assertEqual(c["review_status"], "draft")
            self.assertIsInstance(c["expected_page"], int)


if __name__ == "__main__":
    unittest.main()
