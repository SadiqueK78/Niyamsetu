#!/usr/bin/env python
# backend/tests/test_script_aware_fusion.py
# ─────────────────────────────────────────────────────────
# Unit tests for script-aware fusion in core.hybrid_search:
# is_latin_query() and the SCRIPT_AWARE_FUSION branch of _fuse().
#
# No network and no FAISS index. The store and the lexical index are
# synthetic stand-ins that hand back the rankings a test states, so the
# fused output can be checked against RRF computed directly from those
# rankings.
#
# stdout is captured around every search: the dense-failure warning
# prints an emoji, which raises on a cp1252 console, and the latin branch
# logs a line these tests assert on.
#
# Run:
#   python -m unittest tests.test_script_aware_fusion -q
# ─────────────────────────────────────────────────────────

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document   # noqa: E402

import core.hybrid_search as hs                 # noqa: E402
from config import settings                     # noqa: E402

ENGLISH = "Which department relieves the officer sent on deputation?"
MARATHI = "प्रतिनियुक्तीवर जाणाऱ्या अधिकाऱ्याला कोणता विभाग कार्यमुक्त करणार?"


class FakeIndex:
    """Stands in for LexicalIndex; counts rank() calls so a test can
    assert BM25 never ran."""

    def __init__(self, documents, ranking):
        self.documents = documents
        self._ranking = ranking
        self.calls = 0

    def __len__(self):
        return len(self.documents)

    def rank(self, query):
        self.calls += 1
        return [(pos, 1.0 / (n + 1)) for n, pos in enumerate(self._ranking)]


class FakeStore:
    """Stands in for FAISS: dense is a list of positions, or an exception
    instance to raise."""

    def __init__(self, documents, dense):
        self.documents = documents
        self.dense = dense

    def similarity_search(self, query, k=None):
        if isinstance(self.dense, BaseException):
            raise self.dense
        return [self.documents[pos] for pos in self.dense[:k]]


class IsLatinQueryTests(unittest.TestCase):

    def test_pure_english(self):
        self.assertTrue(hs.is_latin_query("What is the licence fee rate?"))

    def test_pure_marathi(self):
        self.assertFalse(hs.is_latin_query(MARATHI))

    def test_english_with_one_marathi_word(self):
        # शासन contributes 3 letters (its vowel signs are marks, not
        # letters) against 32 Latin ones: a share of 0.086.
        self.assertTrue(hs.is_latin_query("What does the शासन decision say about leave?"))

    def test_digits_only(self):
        self.assertFalse(hs.is_latin_query("2026 12 08"))

    def test_punctuation_only(self):
        self.assertFalse(hs.is_latin_query("?!.,-/"))

    def test_empty_string(self):
        self.assertFalse(hs.is_latin_query(""))
        self.assertFalse(hs.is_latin_query(None))

    def test_marathi_with_an_english_acronym(self):
        self.assertFalse(hs.is_latin_query("AICTE मान्यता असलेल्या संस्थेचे शुल्क कोण ठरवते?"))

    def test_the_threshold_is_exclusive(self):
        # Exactly one Devanagari letter in five is a share of 0.2, which
        # is not below the threshold.
        self.assertFalse(hs.is_latin_query("abcdक"))
        self.assertTrue(hs.is_latin_query("abcdeफ" + "ghij"))


class ScriptAwareFusionTestCase(unittest.TestCase):
    """Patches the store, the lexical index and the flag, and restores them."""

    def setUp(self):
        self.corpus = [
            Document(page_content=f"chunk {n}", metadata={"source_file": f"{n}.pdf", "page": 0})
            for n in "ABCDEF"
        ]
        self._saved = (hs.load_store, hs.get_lexical_index, settings.SCRIPT_AWARE_FUSION)

    def tearDown(self):
        hs.load_store, hs.get_lexical_index, settings.SCRIPT_AWARE_FUSION = self._saved

    def install(self, dense, lexical, flag):
        index = FakeIndex(self.corpus, lexical)
        store = FakeStore(self.corpus, dense)
        hs.load_store = lambda: store
        hs.get_lexical_index = lambda db=None: index
        settings.SCRIPT_AWARE_FUSION = flag
        return index

    def search(self, query, top_k):
        out = io.StringIO()
        with redirect_stdout(out):
            docs = hs.hybrid_search(query, top_k=top_k)
        return ([(d.metadata["source_file"], d.metadata["relevance_score"]) for d in docs],
                out.getvalue())

    def expected(self, rankings, top_k):
        """RRF and min-max relevance computed straight from the rankings,
        independently of _fuse()."""
        fused = hs.reciprocal_rank_fusion([r for r in rankings if r])[:top_k]
        scores = [score for _, score in fused]
        low, span = min(scores), max(scores) - min(scores)
        return [(self.corpus[pos].metadata["source_file"],
                 1.0 if span == 0 else (score - low) / span) for pos, score in fused]


class FlagOffTests(ScriptAwareFusionTestCase):

    def test_english_query_is_fused_exactly_as_before(self):
        dense, lexical = [0, 1, 2, 3], [3, 4, 0, 5]
        index = self.install(dense, lexical, flag=False)
        results, log = self.search(ENGLISH, top_k=4)

        self.assertEqual(results, self.expected([dense, lexical], 4))
        self.assertEqual(index.calls, 1)
        self.assertNotIn("latin-script", log)


class FlagOnTests(ScriptAwareFusionTestCase):

    def test_english_query_returns_the_dense_order_and_never_runs_bm25(self):
        dense, lexical = [2, 0, 5, 1, 3], [4, 3, 1]
        index = self.install(dense, lexical, flag=True)
        results, log = self.search(ENGLISH, top_k=4)

        self.assertEqual([f for f, _ in results], ["C.pdf", "A.pdf", "F.pdf", "B.pdf"])
        # Same relevance computation as the fused path, over the dense list alone.
        self.assertEqual(results, self.expected([dense], 4))
        self.assertEqual(index.calls, 0)
        self.assertIn("hybrid_search: latin-script query, BM25 skipped", log)

    def test_explained_reports_no_lexical_ranks_on_the_latin_branch(self):
        index = self.install([1, 0], [0, 2], flag=True)
        with redirect_stdout(io.StringIO()):
            results, info = hs.hybrid_search_explained(ENGLISH, top_k=4)

        self.assertTrue(info["dense_ok"])
        self.assertEqual([r["dense_rank"] for r in results], [1, 2])
        self.assertTrue(all(r["lexical_rank"] is None for r in results))
        self.assertEqual(index.calls, 0)

    def test_marathi_query_gets_the_rrf_result(self):
        dense, lexical = [0, 1, 2, 3], [3, 4, 0, 5]
        index = self.install(dense, lexical, flag=True)
        results, log = self.search(MARATHI, top_k=4)

        self.assertEqual(results, self.expected([dense, lexical], 4))
        self.assertEqual(index.calls, 1)
        self.assertNotIn("latin-script", log)

    def test_english_query_with_failing_dense_falls_back_to_bm25(self):
        lexical = [4, 2, 0]
        index = self.install(RuntimeError("embedding endpoint down"), lexical, flag=True)
        results, log = self.search(ENGLISH, top_k=4)

        self.assertEqual(results, self.expected([lexical], 4))
        self.assertEqual([f for f, _ in results], ["E.pdf", "C.pdf", "A.pdf"])
        self.assertEqual(index.calls, 1)
        self.assertIn("dense side failed, using lexical only", log)
        self.assertNotIn("latin-script", log)

    def test_the_fallback_matches_the_flag_off_fallback(self):
        lexical = [4, 2, 0]
        self.install(RuntimeError("down"), lexical, flag=False)
        off, _ = self.search(ENGLISH, top_k=4)
        self.install(RuntimeError("down"), lexical, flag=True)
        on, _ = self.search(ENGLISH, top_k=4)

        self.assertEqual(on, off)


if __name__ == "__main__":
    unittest.main()
