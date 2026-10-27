#!/usr/bin/env python
# backend/tests/test_reranker.py
# ─────────────────────────────────────────────────────────
# Offline unit tests for the cross-encoder reranker (core/reranker.py),
# its integration in core.rag.query(), the chat-generation output cap
# (get_chat_llm, LLM_NUM_PREDICT), the token-usage diagnostics, and the
# reranked configuration in tests/score_retrieval.py.
#
# A fake cross-encoder replaces the model loader, so neither torch nor
# sentence-transformers is imported and nothing is downloaded. Retrieval
# and the chat model are fakes too; nothing touches the network.
#
# Run:
#   python -m unittest tests.test_reranker -q
# ─────────────────────────────────────────────────────────

import asyncio
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.append(str(TESTS_DIR.parent))
sys.path.insert(0, str(TESTS_DIR))

from langchain_core.documents import Document   # noqa: E402

import core.condense as condense                 # noqa: E402
import core.rag as rag                           # noqa: E402
import core.reranker as reranker                 # noqa: E402
from config import settings                      # noqa: E402


def chunk(n, page=0, source=None):
    return Document(page_content=f"chunk {n}",
                    metadata={"source_file": source or f"F{n}.pdf", "page": page,
                              "relevance_score": 1.0 / (n + 1)})


class FakeCrossEncoder:
    """Scores 'chunk N' by a table keyed on N; records every predict call."""

    def __init__(self, scores=None, raises=None):
        self.scores = scores or {}
        self.raises = raises
        self.calls = []

    def predict(self, pairs, batch_size=None):
        self.calls.append({"pairs": list(pairs), "batch_size": batch_size})
        if self.raises:
            raise self.raises
        return [self.scores.get(int(text.split()[-1]), 0.0) for _, text in pairs]


class RerankerTestCase(unittest.TestCase):
    """Installs a fake model loader; restores the loader and the singleton."""

    def setUp(self):
        self._saved_loader = reranker._load_cross_encoder
        self.loaded = []
        reranker.reset_model()

    def tearDown(self):
        reranker._load_cross_encoder = self._saved_loader
        reranker.reset_model()

    def install(self, model):
        def load(name):
            self.loaded.append(name)
            if isinstance(model, BaseException):
                raise model
            return model
        reranker._load_cross_encoder = load
        return model

    def rerank(self, *args):
        out = io.StringIO()
        with redirect_stdout(out):
            result = reranker.rerank(*args)
        return result, out.getvalue()


class RerankTests(RerankerTestCase):

    def test_orders_by_score_descending(self):
        model = self.install(FakeCrossEncoder({0: 0.1, 1: 0.9, 2: 0.5, 3: 0.7}))
        chunks = [chunk(n) for n in range(4)]
        result, log = self.rerank("question", chunks, 4)
        self.assertEqual([d.page_content for d in result], ["chunk 1", "chunk 3", "chunk 2", "chunk 0"])
        self.assertEqual(model.calls[0]["pairs"][0], ("question", "chunk 0"))
        self.assertEqual(model.calls[0]["batch_size"], 8)
        self.assertIn("RERANK done | 4 candidates -> 4 in ", log)

    def test_top_n_cut(self):
        self.install(FakeCrossEncoder({n: float(n) for n in range(20)}))
        result, log = self.rerank("q", [chunk(n) for n in range(20)], 5)
        self.assertEqual([d.page_content for d in result],
                         ["chunk 19", "chunk 18", "chunk 17", "chunk 16", "chunk 15"])
        self.assertIn("RERANK done | 20 candidates -> 5 in ", log)

    def test_metadata_and_objects_untouched(self):
        self.install(FakeCrossEncoder({0: 0.2, 1: 0.8}))
        chunks = [chunk(0, page=0, source="GR5.pdf"), chunk(1, page=3, source="GR3.pdf")]
        before = [dict(d.metadata) for d in chunks]
        result, _ = self.rerank("q", chunks, 2)
        self.assertIs(result[0], chunks[1])
        self.assertEqual([dict(d.metadata) for d in chunks], before)
        # Page stays 0-based; the citation shows page + 1.
        cites = rag.build_citations(result)
        self.assertEqual([(c["file"], c["page"]) for c in cites], [("GR3.pdf", "4"), ("GR5.pdf", "1")])

    def test_scoring_exception_keeps_original_order(self):
        self.install(FakeCrossEncoder(raises=RuntimeError("CUDA out of memory")))
        chunks = [chunk(n) for n in range(6)]
        result, log = self.rerank("q", chunks, 3)
        self.assertEqual(result, chunks[:3])
        self.assertIn("WARNING | reranker failed, keeping original order", log)
        self.assertNotIn("RERANK done", log)

    def test_load_failure_keeps_original_order(self):
        self.install(ImportError("No module named 'sentence_transformers'"))
        chunks = [chunk(n) for n in range(4)]
        before = dict(reranker.stats)
        result, log = self.rerank("q", chunks, 2)
        self.assertEqual(result, chunks[:2])
        self.assertIn("reranker failed, keeping original order", log)
        self.assertEqual(reranker.stats["failures"], before["failures"] + 1)

    def test_model_loads_once(self):
        self.install(FakeCrossEncoder())
        self.rerank("q", [chunk(0)], 1)
        self.rerank("q", [chunk(1)], 1)
        self.assertEqual(self.loaded, [settings.RERANK_MODEL])

    def test_empty_input(self):
        self.install(FakeCrossEncoder())
        self.assertEqual(self.rerank("q", [], 5)[0], [])


class QueryPipelineTests(RerankerTestCase):
    """core.rag.query() with fake retrieval, fake chat model, fake cross-encoder."""

    FLAGS = ("META_FASTPATH", "ROUTE_ENABLED", "HEADER_INJECT", "GRAPH_EXPAND",
             "QUERY_REWRITE", "ANSWER_VERIFY", "HISTORY_REWRITE", "RERANK_ENABLED")

    def setUp(self):
        super().setUp()
        self._flags = {f: getattr(settings, f) for f in self.FLAGS}
        for f in self.FLAGS:
            setattr(settings, f, False)
        self._saved = (rag.hybrid_search, rag.get_llm, condense._generate,
                       settings.RERANK_CANDIDATES, settings.RERANK_TOP_N, settings.TOP_K,
                       settings.LLM_NUM_PREDICT)
        settings.RERANK_CANDIDATES, settings.RERANK_TOP_N, settings.TOP_K = 20, 5, 12
        self.searches, self.prompts, self.copies = [], [], []

        def fake_search(query, top_k=None):
            self.searches.append((query, top_k))
            return [chunk(n, page=n % 3) for n in range(top_k)]

        test = self

        class Chat:
            def model_copy(self, update):
                test.copies.append(update)
                return self

            def invoke(self, messages, **kwargs):
                test.prompts.append(messages[0].content)

                class R:
                    content = "answer"
                    response_metadata = {"prompt_eval_count": 900, "eval_count": 40}
                return R()

        rag.hybrid_search = fake_search
        rag.get_llm = lambda: Chat()

    def tearDown(self):
        for f, v in self._flags.items():
            setattr(settings, f, v)
        (rag.hybrid_search, rag.get_llm, condense._generate,
         settings.RERANK_CANDIDATES, settings.RERANK_TOP_N, settings.TOP_K,
         settings.LLM_NUM_PREDICT) = self._saved
        super().tearDown()

    def run_query(self, question="Who signed the canteen order?", history=None):
        out = io.StringIO()
        with redirect_stdout(out):
            result = asyncio.run(rag.query(user_query=question, chat_history=history or [],
                                           language="english"))
        return result, out.getvalue()

    def test_flag_off_leaves_the_pipeline_untouched(self):
        saved_rerank = rag.rerank
        rag.rerank = lambda *a, **k: self.fail("rerank must not run with the flag off")
        try:
            result, log = self.run_query()
        finally:
            rag.rerank = saved_rerank
        self.assertEqual(self.searches, [("Who signed the canteen order?", 12)])
        self.assertEqual(len(result["citations"]), 12)
        self.assertEqual(self.prompts[0].count("[Source "), 12)
        self.assertNotIn("RERANK", log)
        self.assertIn("RETRIEVAL start | top_k=12 ", log)

    def test_flag_on_sends_exactly_top_n_to_prompt_and_citations(self):
        settings.RERANK_ENABLED = True
        model = self.install(FakeCrossEncoder({n: float(n) for n in range(20)}))
        result, log = self.run_query()
        self.assertEqual(self.searches[0][1], 20)                 # RERANK_CANDIDATES
        self.assertEqual(len(model.calls[0]["pairs"]), 20)
        self.assertEqual(self.prompts[0].count("[Source "), 5)    # RERANK_TOP_N
        self.assertEqual([c["file"] for c in result["citations"]],
                         ["F19.pdf", "F18.pdf", "F17.pdf", "F16.pdf", "F15.pdf"])
        self.assertIn("[Source 1: F19.pdf", self.prompts[0])
        self.assertIn("RERANK done | 20 candidates -> 5 in ", log)

    def test_rerank_query_is_the_original_question_not_the_condensed_one(self):
        settings.RERANK_ENABLED = True
        settings.HISTORY_REWRITE = True
        condense._generate = lambda prompt: "Who signed the Mantralaya canteen kitchen order?"
        model = self.install(FakeCrossEncoder())
        history = [{"role": "user", "content": "What is the canteen decision?"},
                   {"role": "assistant", "content": "Kitchen modernisation."}]
        self.run_query("Who signed it?", history)
        self.assertEqual(self.searches[0][0], "Who signed the Mantralaya canteen kitchen order?")
        self.assertEqual({q for q, _ in model.calls[0]["pairs"]}, {"Who signed it?"})

    def test_reranker_failure_still_answers_with_top_n(self):
        settings.RERANK_ENABLED = True
        self.install(ImportError("no sentence_transformers"))
        result, log = self.run_query()
        self.assertTrue(result["success"])
        self.assertEqual([c["file"] for c in result["citations"]],
                         [f"F{n}.pdf" for n in range(5)])
        self.assertIn("reranker failed, keeping original order", log)

    def test_chat_generation_uses_the_num_predict_copy(self):
        settings.LLM_NUM_PREDICT = 800
        self.run_query()
        self.assertEqual(self.copies, [{"num_predict": 800}])

    def test_token_usage_is_logged(self):
        _, log = self.run_query()
        self.assertIn("INFO | llm prompt_tokens=900 completion_tokens=40", log)
        self.assertNotIn("prompt may be truncated", log)


class ChatLlmTests(unittest.TestCase):
    """get_chat_llm() against the real get_llm() — builds clients only, no calls."""

    def setUp(self):
        self._saved = settings.LLM_NUM_PREDICT

    def tearDown(self):
        settings.LLM_NUM_PREDICT = self._saved

    def test_num_predict_copy_leaves_get_llm_unchanged(self):
        settings.LLM_NUM_PREDICT = 800
        base = rag.get_llm()
        chat = rag.get_chat_llm()
        self.assertEqual(chat.num_predict, 800)
        self.assertIsNone(base.num_predict)
        self.assertIsNone(rag.get_llm().num_predict)
        for field in ("model", "base_url", "temperature", "num_ctx", "keep_alive",
                      "client_kwargs"):
            self.assertEqual(getattr(chat, field), getattr(base, field), field)

    def test_zero_means_no_ceiling(self):
        settings.LLM_NUM_PREDICT = 0
        self.assertIsNone(rag.get_chat_llm().num_predict)


class TokenUsageTests(unittest.TestCase):

    def log(self, response):
        out = io.StringIO()
        with redirect_stdout(out):
            rag.log_token_usage(response)
        return out.getvalue()

    def response(self, **kw):
        class R:
            pass
        r = R()
        for k, v in kw.items():
            setattr(r, k, v)
        return r

    def test_warning_near_the_context_limit(self):
        text = self.log(self.response(response_metadata={
            "prompt_eval_count": settings.LLM_NUM_CTX - 64, "eval_count": 120}))
        self.assertIn(f"WARNING | prompt may be truncated: prompt_tokens="
                      f"{settings.LLM_NUM_CTX - 64} num_ctx={settings.LLM_NUM_CTX}", text)
        self.assertIn("INFO | llm prompt_tokens=", text)

    def test_no_warning_below_the_margin(self):
        text = self.log(self.response(response_metadata={
            "prompt_eval_count": settings.LLM_NUM_CTX - 65, "eval_count": 1}))
        self.assertNotIn("WARNING", text)

    def test_usage_metadata_fallback_and_missing_counts(self):
        text = self.log(self.response(response_metadata={},
                                      usage_metadata={"input_tokens": 10, "output_tokens": 2}))
        self.assertIn("INFO | llm prompt_tokens=10 completion_tokens=2", text)
        self.assertEqual(self.log(self.response(response_metadata={})), "")
        self.assertEqual(self.log(object()), "")

    def test_log_lines_have_no_emoji(self):
        text = self.log(self.response(response_metadata={
            "prompt_eval_count": 99999, "eval_count": 1}))
        self.assertTrue(text.isascii())


class ScorerConfigTests(RerankerTestCase):

    def test_reranked_configuration(self):
        import score_retrieval as sr
        seen = {}

        def fake_search(question, top_k=None):
            seen.update(top_k=top_k, script=settings.SCRIPT_AWARE_FUSION,
                        rerank=settings.RERANK_ENABLED)
            return [chunk(n) for n in range(top_k)]

        saved = (sr.hybrid_search, settings.SCRIPT_AWARE_FUSION, settings.RERANK_ENABLED)
        sr.hybrid_search = fake_search
        settings.SCRIPT_AWARE_FUSION = settings.RERANK_ENABLED = False
        self.install(FakeCrossEncoder({n: float(n) for n in range(30)}))
        try:
            with redirect_stdout(io.StringIO()):
                docs = sr.run_hybrid_script_aware_rerank("q", 12)
        finally:
            sr.hybrid_search = saved[0]
            flags = (settings.SCRIPT_AWARE_FUSION, settings.RERANK_ENABLED)
            settings.SCRIPT_AWARE_FUSION, settings.RERANK_ENABLED = saved[1], saved[2]

        self.assertEqual(seen, {"top_k": settings.RERANK_CANDIDATES, "script": True, "rerank": True})
        self.assertEqual(flags, (False, False))          # restored after the run
        self.assertEqual(len(docs), settings.RERANK_TOP_N)
        self.assertIn(sr.RERANK_CONFIG, [n for n, _ in sr.NATURAL_CONFIGURATIONS])
        self.assertNotIn(sr.RERANK_CONFIG, [n for n, _ in sr.CONFIGURATIONS])

    def test_summarise_rerank_scores_short_lists_over_what_exists(self):
        import score_retrieval as sr
        rows = [{"rank": 1, "files": 3, "page_hit_5": True},
                {"rank": 4, "files": 5, "page_hit_5": False},
                {"rank": None, "files": 5, "page_hit_5": False}]
        s = sr.summarise_rerank(rows)
        self.assertEqual([s[f"hit@{k}_count"] for k in (1, 3, 5, 12)], [1, 1, 2, 2])
        self.assertEqual(s["page_hit@5_count"], 1)
        self.assertEqual(s["mrr"], round((1 + 0.25) / 3, 3))
        table = "\n".join(sr.rerank_metric_table("Overall", s))
        self.assertIn("| hit@1 | hit@3 | hit@5 | hit@12 | MRR | page_hit@5 |", table)
        self.assertIn("66.7% (2/3)", table)


if __name__ == "__main__":
    unittest.main()
