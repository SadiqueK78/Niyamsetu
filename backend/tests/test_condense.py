#!/usr/bin/env python
# backend/tests/test_condense.py
# ─────────────────────────────────────────────────────────
# Offline unit tests for follow-up condensation: core/condense.py and the
# HISTORY_REWRITE branch of core.rag.query().
#
# A fake LLM replaces core.condense._generate, and query() runs with fake
# retrieval and a fake chat model. Nothing touches Ollama, FAISS or the
# network.
#
# Run:
#   python -m unittest tests.test_condense -q
# ─────────────────────────────────────────────────────────

import asyncio
import io
import sys
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document   # noqa: E402

import core.condense as condense                 # noqa: E402
import core.rag as rag                           # noqa: E402
from config import settings                      # noqa: E402

FOLLOW_UP = "त्यासाठी किती खर्च मंजूर झाला?"
HISTORY = [
    {"role": "user", "content": "Q1 first question"},
    {"role": "assistant", "content": "A1 first answer"},
    {"role": "user", "content": "Q2 canteen kitchen decision?"},
    {"role": "assistant", "content": "A2 " + "x" * 400},
    {"role": "user", "content": "Q3 which buildings?"},
    {"role": "assistant", "content": "A3 main and annexe"},
]


class FakeLLM:
    """Stands in for condense._generate: records prompts, replies in order."""

    def __init__(self, *outputs, raises=None, delay=0.0):
        self.outputs = list(outputs)
        self.raises = raises
        self.delay = delay
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if self.delay:
            time.sleep(self.delay)
        if self.raises:
            raise self.raises
        return self.outputs.pop(0) if self.outputs else ""


class CondenseTestCase(unittest.TestCase):

    def setUp(self):
        self._saved = (condense._generate, settings.HISTORY_REWRITE_TURNS,
                       settings.HISTORY_REWRITE_TIMEOUT)
        settings.HISTORY_REWRITE_TURNS = 2
        settings.HISTORY_REWRITE_TIMEOUT = 60

    def tearDown(self):
        (condense._generate, settings.HISTORY_REWRITE_TURNS,
         settings.HISTORY_REWRITE_TIMEOUT) = self._saved

    def condense_with(self, llm, question=FOLLOW_UP, history=HISTORY):
        condense._generate = llm
        return condense.condense_query(question, history, "marathi")


class HistoryFormattingTests(CondenseTestCase):

    def test_empty_history_returns_the_question_without_a_call(self):
        llm = FakeLLM("should not be used")
        self.assertEqual(self.condense_with(llm, history=[]), FOLLOW_UP)
        self.assertEqual(self.condense_with(llm, history=None), FOLLOW_UP)
        self.assertEqual(llm.prompts, [])

    def test_window_is_the_last_two_exchanges(self):
        text = condense.format_history(HISTORY, 2)
        self.assertNotIn("Q1", text)
        self.assertNotIn("A1", text)
        self.assertTrue(text.startswith("User: Q2 canteen kitchen decision?\nAssistant: A2 "))
        self.assertTrue(text.endswith("User: Q3 which buildings?\nAssistant: A3 main and annexe"))
        self.assertEqual(len(text.splitlines()), 4)

    def test_window_follows_the_setting(self):
        settings.HISTORY_REWRITE_TURNS = 1
        llm = FakeLLM("standalone?")
        self.condense_with(llm)
        self.assertIn("CONVERSATION:\nUser: Q3 which buildings?\nAssistant: A3 main and annexe\n\n",
                      llm.prompts[0])
        self.assertNotIn("Q2", llm.prompts[0])

    def test_assistant_turns_are_cut_to_300_characters(self):
        text = condense.format_history(HISTORY, 2)
        a2 = next(line for line in text.splitlines() if line.startswith("Assistant: A2"))
        self.assertEqual(len(a2), len("Assistant: ") + 300)
        # User turns are not cut.
        long_user = [{"role": "user", "content": "u" * 500}, {"role": "assistant", "content": "ok"}]
        self.assertIn("u" * 500, condense.format_history(long_user, 2))

    def test_exact_prompt_text(self):
        llm = FakeLLM("standalone?")
        history = [{"role": "user", "content": "What is the canteen decision?"},
                   {"role": "assistant", "content": "It approves kitchen modernisation."}]
        self.condense_with(llm, question="How much was approved for it?", history=history)
        self.assertEqual(llm.prompts[0], (
            "Given the conversation below and the follow-up question, rewrite the follow-up as "
            "one standalone question that can be understood without the conversation. Keep the "
            "same language as the follow-up question. Include the specific document subject, "
            "names and numbers from the conversation that the follow-up refers to. If the "
            "follow-up is already standalone, return it unchanged. Return only the question."
            "\n\nCONVERSATION:\nUser: What is the canteen decision?\nAssistant: It approves "
            "kitchen modernisation.\n\nFOLLOW-UP: How much was approved for it?\n\n"
            "STANDALONE QUESTION:"))


class OutputHandlingTests(CondenseTestCase):

    def test_first_non_empty_line_with_quotes_stripped(self):
        out = self.condense_with(FakeLLM(
            '\n\n  "मंत्रालय उपाहारगृह स्वयंपाकगृह आधुनिकीकरणासाठी किती खर्च मंजूर झाला?"  \n'
            "Explanation: the follow-up refers to the canteen."))
        self.assertEqual(out, "मंत्रालय उपाहारगृह स्वयंपाकगृह आधुनिकीकरणासाठी किती खर्च मंजूर झाला?")

    def test_curly_and_single_quotes(self):
        self.assertEqual(condense.extract_question("“Who signed the canteen order?”"),
                         "Who signed the canteen order?")
        self.assertEqual(condense.extract_question("'Who signed it?'"), "Who signed it?")

    def test_exception_falls_back(self):
        self.assertEqual(self.condense_with(FakeLLM(raises=RuntimeError("LLM down"))), FOLLOW_UP)

    def test_timeout_falls_back(self):
        settings.HISTORY_REWRITE_TIMEOUT = 0.05
        started = time.monotonic()
        self.assertEqual(self.condense_with(FakeLLM("late answer", delay=1.0)), FOLLOW_UP)
        self.assertLess(time.monotonic() - started, 0.9)   # did not wait for the call

    def test_empty_output_falls_back(self):
        for output in ("", "   \n\n", '""'):
            with self.subTest(output=output):
                self.assertEqual(self.condense_with(FakeLLM(output)), FOLLOW_UP)

    def test_over_long_output_falls_back(self):
        self.assertEqual(self.condense_with(FakeLLM("q" * 401)), FOLLOW_UP)
        self.assertEqual(self.condense_with(FakeLLM("q" * 400)), "q" * 400)

    def test_generate_uses_the_chat_helper_with_a_100_token_ceiling(self):
        seen = {}

        class Response:
            content = "standalone?"

        class Copy:
            def invoke(self, messages, **kwargs):
                seen["prompt"] = messages[0].content
                seen["kwargs"] = kwargs
                return Response()

        class Helper:
            def model_copy(self, update):
                seen["update"] = update
                return Copy()

        saved = rag.get_llm
        rag.get_llm = lambda: Helper()
        try:
            self.assertEqual(self._saved[0]("PROMPT"), "standalone?")
        finally:
            rag.get_llm = saved
        self.assertEqual(seen, {"update": {"num_predict": 100}, "prompt": "PROMPT",
                                "kwargs": {"stream": False}})


class QueryIntegrationTests(unittest.TestCase):
    """core.rag.query() with fake retrieval and a fake chat model."""

    FLAGS = ("META_FASTPATH", "ROUTE_ENABLED", "HEADER_INJECT", "GRAPH_EXPAND",
             "QUERY_REWRITE", "ANSWER_VERIFY", "HISTORY_REWRITE")

    def setUp(self):
        self._saved_flags = {f: getattr(settings, f) for f in self.FLAGS}
        for f in self.FLAGS:
            setattr(settings, f, False)
        self._saved = (rag.hybrid_search, rag.get_llm, rag.expand_query, condense._generate,
                       settings.HISTORY_REWRITE_TIMEOUT)
        settings.HISTORY_REWRITE_TIMEOUT = 60
        self.retrieved, self.prompts = [], []

        def fake_search(query, top_k=None):
            self.retrieved.append(query)
            return [Document(page_content="अंदाजित खर्च रुपये 3,01,71,099/-",
                             metadata={"source_file": "GR3.pdf", "page": 1})]

        test = self

        class Chat:
            def model_copy(self, update):
                # rag.get_chat_llm() copies the client with num_predict.
                return self

            def invoke(self, messages, **kwargs):
                test.prompts.append(messages[0].content)

                class R:
                    content = "Rs 3,01,71,099"
                return R()

        rag.hybrid_search = fake_search
        rag.get_llm = lambda: Chat()

    def tearDown(self):
        for f, v in self._saved_flags.items():
            setattr(settings, f, v)
        (rag.hybrid_search, rag.get_llm, rag.expand_query, condense._generate,
         settings.HISTORY_REWRITE_TIMEOUT) = self._saved

    def run_query(self, question=FOLLOW_UP, history=HISTORY):
        out = io.StringIO()
        with redirect_stdout(out):
            result = asyncio.run(rag.query(user_query=question, chat_history=history,
                                           language="marathi"))
        return result, out.getvalue()

    def test_flag_off_leaves_query_unchanged(self):
        llm = FakeLLM("SHOULD NOT BE USED")
        condense._generate = llm
        result, log = self.run_query()
        self.assertEqual(self.retrieved, [FOLLOW_UP])
        self.assertEqual(llm.prompts, [])
        self.assertNotIn("CONDENSE", log)
        self.assertEqual(result["query"], FOLLOW_UP)
        self.assertTrue(result["success"])

    def test_flag_on_with_empty_history_makes_no_call(self):
        settings.HISTORY_REWRITE = True
        llm = FakeLLM("SHOULD NOT BE USED")
        condense._generate = llm
        _, log = self.run_query(history=[])
        self.assertEqual(self.retrieved, [FOLLOW_UP])
        self.assertEqual(llm.prompts, [])
        self.assertNotIn("CONDENSE", log)

    def test_flag_on_retrieves_condensed_but_prompts_with_the_original(self):
        condensed = "मंत्रालय उपाहारगृह स्वयंपाकगृह आधुनिकीकरणासाठी किती खर्च मंजूर झाला?"
        condense._generate = FakeLLM(condensed)

        settings.HISTORY_REWRITE = False
        off_result, _ = self.run_query()
        off_prompt = self.prompts[-1]

        settings.HISTORY_REWRITE = True
        on_result, log = self.run_query()

        self.assertEqual(self.retrieved, [FOLLOW_UP, condensed])
        # The prompt, history formatting and returned question are untouched.
        self.assertEqual(self.prompts[-1], off_prompt)
        self.assertIn(FOLLOW_UP, self.prompts[-1])
        self.assertNotIn(condensed, self.prompts[-1])
        self.assertEqual(on_result["query"], FOLLOW_UP)
        self.assertEqual(on_result["citations"], off_result["citations"])
        line = next(l for l in log.splitlines() if "CONDENSE done" in l)
        self.assertTrue(line.endswith(f"] CONDENSE done | {condensed[:120]}"))

    def test_query_rewrite_expands_the_condensed_text(self):
        settings.HISTORY_REWRITE = True
        settings.QUERY_REWRITE = True
        condense._generate = FakeLLM("standalone canteen cost?")
        expanded_from = []

        async def fake_expand(text):
            expanded_from.append(text)
            return text + " ALT"
        rag.expand_query = fake_expand
        self.run_query()
        self.assertEqual(expanded_from, ["standalone canteen cost?"])
        self.assertEqual(self.retrieved, ["standalone canteen cost? ALT"])

    def test_condense_failure_retrieves_on_the_original(self):
        settings.HISTORY_REWRITE = True
        condense._generate = FakeLLM(raises=RuntimeError("down"))
        _, log = self.run_query()
        self.assertEqual(self.retrieved, [FOLLOW_UP])
        self.assertIn(f"CONDENSE done | {FOLLOW_UP}", log)


if __name__ == "__main__":
    unittest.main()
