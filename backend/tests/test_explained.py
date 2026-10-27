#!/usr/bin/env python
# backend/tests/test_explained.py
# ─────────────────────────────────────────────────────────
# Unit tests for core.hybrid_search.hybrid_search_explained().
#
# No network and no FAISS index. The store, the lexical index and the
# dense side are all replaced with synthetic stand-ins, which is what
# makes the interesting cases testable at all: a dense side that RAISES is
# the behaviour that matters most here, and waiting for a real embedding
# endpoint to be down is not a test strategy.
#
# ── WHAT IS NOT COVERED HERE ─────────────────────────────
# The dense ranking itself. Producing one requires embedding the query,
# which needs the embedding model, so nothing in this file asserts that
# FAISS returns sensible neighbours — only that whatever it returns is
# reported at the right rank, and that its failure degrades correctly.
# The real dense path is exercised by tests/test_hybrid_search.py, which
# replays cached dense rankings from disk.
#
# Run:
#   python -m pytest tests/test_explained.py -q
#   python tests/test_explained.py
# ─────────────────────────────────────────────────────────

import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document   # noqa: E402

import core.hybrid_search as hs                 # noqa: E402


def chunk(source_file, text, page=0, section_type=None):
    metadata = {"source_file": source_file, "page": page}
    if section_type is not None:
        metadata["section_type"] = section_type
    return Document(page_content=text, metadata=metadata)


class FakeIndex:
    """
    Stands in for LexicalIndex.

    rank() returns the positions it was told to, so a test states the BM25
    ranking directly instead of trying to coax BM25 into producing one.
    """

    def __init__(self, documents, ranking=None):
        self.documents = documents
        self._ranking = ranking if ranking is not None else list(range(len(documents)))

    def __len__(self):
        return len(self.documents)

    def rank(self, query):
        # (position, score) pairs, which is the shape the real index gives.
        return [(pos, 1.0 / (n + 1)) for n, pos in enumerate(self._ranking)]


class FakeStore:
    """
    Stands in for the FAISS store.

    dense is a list of positions into the corpus, or an exception instance
    to raise — the two cases this file exists to separate.
    """

    def __init__(self, documents, dense):
        self.documents = documents
        self.dense = dense

    def similarity_search(self, query, k=None):
        if isinstance(self.dense, BaseException):
            raise self.dense
        return [self.documents[pos] for pos in self.dense[:k]]


class ExplainedTestCase(unittest.TestCase):
    """Patches the module's three collaborators and restores them."""

    def setUp(self):
        self.corpus = [
            chunk("A.pdf", "alpha text about leave policy", page=0, section_type="__header__"),
            chunk("B.pdf", "bravo text about transfers",    page=1, section_type="शासन निर्णय"),
            chunk("C.pdf", "charlie text about libraries",  page=2, section_type="संदर्भ"),
            chunk("D.pdf", "delta text about seniority",    page=3),
        ]
        self._saved = (hs.load_store, hs.get_lexical_index)

    def tearDown(self):
        hs.load_store, hs.get_lexical_index = self._saved

    def install(self, dense, lexical_ranking):
        """Wires a dense ranking (or exception) and a BM25 ranking in."""
        index = FakeIndex(self.corpus, lexical_ranking)
        store = FakeStore(self.corpus, dense)
        hs.load_store = lambda: store
        hs.get_lexical_index = lambda db=None: index
        return index


class RankReportingTests(ExplainedTestCase):

    def test_both_sides_ranks_are_reported(self):
        # Dense likes A then B; BM25 likes B then A. Both chunks should
        # therefore carry a rank from each side.
        self.install(dense=[0, 1], lexical_ranking=[1, 0])
        results, info = hs.hybrid_search_explained("query", top_k=4)

        by_file = {r["source_file"]: r for r in results}
        self.assertEqual(by_file["A.pdf"]["dense_rank"], 1)
        self.assertEqual(by_file["A.pdf"]["lexical_rank"], 2)
        self.assertEqual(by_file["B.pdf"]["dense_rank"], 2)
        self.assertEqual(by_file["B.pdf"]["lexical_rank"], 1)
        self.assertTrue(info["dense_ok"])

    def test_a_chunk_only_the_dense_side_found_has_no_lexical_rank(self):
        self.install(dense=[0, 3], lexical_ranking=[1])
        results, _ = hs.hybrid_search_explained("query", top_k=4)

        by_file = {r["source_file"]: r for r in results}
        self.assertEqual(by_file["D.pdf"]["dense_rank"], 2)
        self.assertIsNone(by_file["D.pdf"]["lexical_rank"])

    def test_a_chunk_only_bm25_found_has_no_dense_rank(self):
        self.install(dense=[0], lexical_ranking=[0, 2])
        results, _ = hs.hybrid_search_explained("query", top_k=4)

        by_file = {r["source_file"]: r for r in results}
        self.assertIsNone(by_file["C.pdf"]["dense_rank"])
        self.assertEqual(by_file["C.pdf"]["lexical_rank"], 2)

    def test_fused_rank_is_one_based_and_contiguous(self):
        self.install(dense=[0, 1, 2], lexical_ranking=[2, 1, 0])
        results, _ = hs.hybrid_search_explained("query", top_k=3)

        self.assertEqual([r["fused_rank"] for r in results], [1, 2, 3])

    def test_agreed_top_hit_is_ranked_first(self):
        # Both sides put C first, so fusion must too — this is the
        # property the per-side ranks are there to make visible.
        self.install(dense=[2, 0, 1], lexical_ranking=[2, 1, 0])
        results, _ = hs.hybrid_search_explained("query", top_k=4)

        self.assertEqual(results[0]["source_file"], "C.pdf")
        self.assertEqual(results[0]["dense_rank"], 1)
        self.assertEqual(results[0]["lexical_rank"], 1)

    def test_top_k_is_respected(self):
        self.install(dense=[0, 1, 2, 3], lexical_ranking=[0, 1, 2, 3])
        results, _ = hs.hybrid_search_explained("query", top_k=2)

        self.assertEqual(len(results), 2)


class ResultShapeTests(ExplainedTestCase):

    def test_every_documented_field_is_present(self):
        self.install(dense=[0], lexical_ranking=[0])
        results, info = hs.hybrid_search_explained("query", top_k=1)

        self.assertEqual(
            set(results[0]),
            {"source_file", "page", "section_type", "preview",
             "dense_rank", "lexical_rank", "fused_rank", "rrf_score", "relevance"},
        )
        self.assertEqual(set(info), {"dense_ok", "pool_depth"})

    def test_metadata_is_carried_through(self):
        self.install(dense=[0], lexical_ranking=[0])
        results, _ = hs.hybrid_search_explained("query", top_k=1)

        self.assertEqual(results[0]["source_file"], "A.pdf")
        self.assertEqual(results[0]["page"], 0)
        self.assertEqual(results[0]["section_type"], "__header__")

    def test_preview_is_capped_at_200_characters(self):
        self.corpus[0] = chunk("A.pdf", "x" * 500)
        self.install(dense=[0], lexical_ranking=[0])
        results, _ = hs.hybrid_search_explained("query", top_k=1)

        self.assertEqual(len(results[0]["preview"]), 200)

    def test_relevance_is_normalized_across_the_returned_set(self):
        self.install(dense=[0, 1, 2, 3], lexical_ranking=[0, 1, 2, 3])
        results, _ = hs.hybrid_search_explained("query", top_k=4)

        self.assertAlmostEqual(results[0]["relevance"], 1.0)
        self.assertAlmostEqual(results[-1]["relevance"], 0.0)
        # Monotonically non-increasing, since fusion already sorted them.
        values = [r["relevance"] for r in results]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_a_single_result_gets_full_relevance(self):
        # No spread to interpolate over, so the one hit is the top hit.
        self.install(dense=[0], lexical_ranking=[0])
        results, _ = hs.hybrid_search_explained("query", top_k=1)

        self.assertAlmostEqual(results[0]["relevance"], 1.0)

    def test_the_docstore_documents_are_not_mutated(self):
        # hybrid_search() writes relevance_score onto the shared Documents;
        # this must not, since it builds its own dicts.
        self.install(dense=[0, 1], lexical_ranking=[1, 0])
        hs.hybrid_search_explained("query", top_k=2)

        for doc in self.corpus:
            self.assertNotIn("relevance_score", doc.metadata)


class DenseFailureTests(ExplainedTestCase):
    """The case this file exists for: distinguishing "the dense side found
    nothing" from "the dense side is broken"."""

    def test_a_raising_dense_side_degrades_to_lexical_only(self):
        self.install(dense=RuntimeError("embedding endpoint down"),
                     lexical_ranking=[2, 0, 1])
        results, info = hs.hybrid_search_explained("query", top_k=4)

        self.assertFalse(info["dense_ok"])
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["source_file"], "C.pdf")
        # Nothing can carry a dense rank when the dense side never ran.
        self.assertTrue(all(r["dense_rank"] is None for r in results))
        self.assertTrue(all(r["lexical_rank"] is not None for r in results))

    def test_the_lexical_order_survives_a_dense_failure(self):
        self.install(dense=OSError("connection refused"),
                     lexical_ranking=[3, 1, 0, 2])
        results, info = hs.hybrid_search_explained("query", top_k=4)

        self.assertFalse(info["dense_ok"])
        self.assertEqual([r["source_file"] for r in results],
                         ["D.pdf", "B.pdf", "A.pdf", "C.pdf"])

    def test_an_empty_dense_ranking_is_not_a_failure(self):
        # FAISS returning nothing is a legitimate answer, and it must not
        # be reported as the endpoint being down.
        self.install(dense=[], lexical_ranking=[0, 1])
        results, info = hs.hybrid_search_explained("query", top_k=4)

        self.assertTrue(info["dense_ok"])
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["dense_rank"] is None for r in results))

    def test_hybrid_search_degrades_the_same_way(self):
        # The two functions share _fuse(), so this pins the shared
        # behaviour rather than restating it.
        self.install(dense=RuntimeError("down"), lexical_ranking=[2, 0])
        docs = hs.hybrid_search("query", top_k=4)

        self.assertEqual([d.metadata["source_file"] for d in docs],
                         ["C.pdf", "A.pdf"])


class NoStoreTests(ExplainedTestCase):

    def test_no_store_returns_empty_and_dense_not_ok(self):
        hs.load_store = lambda: None
        results, info = hs.hybrid_search_explained("query")

        self.assertEqual(results, [])
        self.assertEqual(info, {"dense_ok": False, "pool_depth": 0})

    def test_an_empty_corpus_returns_empty(self):
        hs.load_store = lambda: FakeStore([], [])
        hs.get_lexical_index = lambda db=None: FakeIndex([], [])
        results, info = hs.hybrid_search_explained("query")

        self.assertEqual(results, [])
        self.assertFalse(info["dense_ok"])

    def test_neither_side_producing_anything_returns_empty(self):
        self.install(dense=[], lexical_ranking=[])
        results, info = hs.hybrid_search_explained("query")

        self.assertEqual(results, [])
        self.assertEqual(info["pool_depth"], 0)


class AgreementWithHybridSearchTests(ExplainedTestCase):
    """Both functions must order results identically, since they share
    _fuse(). This is the guarantee the extraction was for."""

    def test_the_two_functions_return_the_same_order(self):
        self.install(dense=[0, 2, 1], lexical_ranking=[1, 2, 3])
        docs = hs.hybrid_search("query", top_k=4)
        results, _ = hs.hybrid_search_explained("query", top_k=4)

        self.assertEqual(
            [d.metadata["source_file"] for d in docs],
            [r["source_file"] for r in results],
        )

    def test_the_two_functions_agree_on_relevance(self):
        self.install(dense=[0, 2, 1], lexical_ranking=[1, 2, 3])
        docs = hs.hybrid_search("query", top_k=4)
        results, _ = hs.hybrid_search_explained("query", top_k=4)

        for doc, row in zip(docs, results):
            self.assertAlmostEqual(doc.metadata["relevance_score"], row["relevance"])


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    unittest.main(verbosity=2)
