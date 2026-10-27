#!/usr/bin/env python
# backend/tests/test_summary_v2.py
# ─────────────────────────────────────────────────────────
# Offline unit tests for the map-reduce summarizer (core/summary_v2.py),
# its faithfulness check (core/summary_check.py) and the flag in the
# summary route (api/summary.py).
#
# A fake LLM and a fake chunk store replace the module's collaborators;
# nothing touches Ollama, FAISS, MongoDB or the network.
#
# Run:
#   python -m unittest tests.test_summary_v2 -q
# ─────────────────────────────────────────────────────────

import asyncio
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document   # noqa: E402

import core.summary_v2 as sv                    # noqa: E402
from config import settings                     # noqa: E402
from core.summary_check import check_faithfulness, number_tokens   # noqa: E402

MARATHI_TEXT = "शासन निर्णय सामान्य प्रशासन विभाग अनुज्ञप्ती शुल्क पाच पट"
ENGLISH_TEXT = "The Act provides for public libraries in the State of Maharashtra"


def chunk(text, page=0, source="GR5.pdf"):
    return Document(page_content=text, metadata={"source_file": source, "page": page})


class FakeLLM:
    """Records every prompt; answers MAP prompts from map_outputs in order
    and REDUCE prompts from reduce_outputs in order."""

    def __init__(self, map_outputs=None, reduce_outputs=None):
        self.map_outputs = list(map_outputs or [])
        self.reduce_outputs = list(reduce_outputs or [])
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if prompt.startswith("You are extracting facts"):
            return self.map_outputs.pop(0) if self.map_outputs else "- a fact"
        return self.reduce_outputs.pop(0) if self.reduce_outputs else "Subject\n- a summary"

    @property
    def map_prompts(self):
        return [p for p in self.prompts if p.startswith("You are extracting facts")]

    @property
    def reduce_prompts(self):
        return [p for p in self.prompts if p.startswith("Combine the bullet points")]


class SummaryTestCase(unittest.TestCase):
    """Installs a fake store, fake metadata and a fake LLM; restores all."""

    def setUp(self):
        self._saved = (sv._store_chunks, sv._load_meta, sv._llm,
                       settings.SUMMARY_BATCH_CHARS, settings.SUMMARY_MAX_BATCHES)
        settings.SUMMARY_BATCH_CHARS = 5000
        settings.SUMMARY_MAX_BATCHES = 12
        self.meta = {"GR5.pdf": {"gr_number": "GAD-42011/1/2026", "department": "सामान्य प्रशासन विभाग",
                                 "doc_date": "2026-08-14", "subject": "निवासस्थान"}}
        sv._load_meta = lambda: self.meta

    def tearDown(self):
        (sv._store_chunks, sv._load_meta, sv._llm,
         settings.SUMMARY_BATCH_CHARS, settings.SUMMARY_MAX_BATCHES) = self._saved

    def install(self, chunks, llm):
        sv._store_chunks = lambda filename: [c for c in chunks
                                             if c.metadata["source_file"] == filename]
        sv._llm = llm
        return llm

    def summarize(self, filename="GR5.pdf", language=None):
        with redirect_stdout(io.StringIO()):
            return sv.summarize_document(filename, language)


class BatchingTests(SummaryTestCase):

    def test_consecutive_texts_fill_a_batch_up_to_the_limit(self):
        batches = sv.build_batches(["a" * 40, "b" * 40, "c" * 40], batch_chars=100, max_batches=12)
        self.assertEqual(batches, ["a" * 40 + "\n\n" + "b" * 40, "c" * 40])
        self.assertTrue(all(len(b) <= 100 for b in batches))

    def test_an_oversized_text_is_its_own_batch_not_cut(self):
        batches = sv.build_batches(["a" * 10, "b" * 300, "c" * 10], batch_chars=100, max_batches=12)
        self.assertEqual(batches, ["a" * 10, "b" * 300, "c" * 10])

    def test_excess_batches_merge_into_the_last(self):
        texts = [str(i) * 90 for i in range(6)]
        batches = sv.build_batches(texts, batch_chars=100, max_batches=3)
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[:2], [texts[0], texts[1]])
        self.assertEqual(batches[2], "\n\n".join(texts[2:]))
        # Nothing is dropped.
        self.assertEqual("".join(batches).replace("\n", ""), "".join(texts))

    def test_pipeline_uses_the_settings_and_one_map_call_per_batch(self):
        settings.SUMMARY_BATCH_CHARS = 100
        settings.SUMMARY_MAX_BATCHES = 2
        llm = self.install([chunk("x" * 90, page=p) for p in range(5)], FakeLLM())
        result = self.summarize()
        self.assertEqual(result["batches"], 2)
        self.assertEqual(len(llm.map_prompts), 2)
        self.assertTrue(llm.map_prompts[0].startswith(
            "You are extracting facts from part 1 of 2 of a Maharashtra Government document."))
        self.assertTrue(llm.map_prompts[1].startswith("You are extracting facts from part 2 of 2"))

    def test_chunks_are_ordered_by_page_then_position(self):
        chunks = [chunk("third", page=1), chunk("first", page=0), chunk("fourth", page=1),
                  chunk("second", page=0)]
        llm = self.install(chunks, FakeLLM())
        self.summarize()
        batch = llm.map_prompts[0].split("TEXT:\n", 1)[1]
        self.assertEqual(batch.split("\n\n"), ["first", "second", "third", "fourth"])

    def test_map_prompt_is_exact(self):
        llm = self.install([chunk(ENGLISH_TEXT)], FakeLLM())
        self.summarize()
        self.assertEqual(llm.map_prompts[0], (
            "You are extracting facts from part 1 of 1 of a Maharashtra Government document. "
            "Write at most 6 short bullet points, in English, listing only facts that are "
            "explicitly stated in the text below: who issued or decided what, amounts, dates, "
            "numbers, named persons and posts, references to other orders. Do not add anything "
            "that is not in the text. Copy numbers exactly as written. If the part contains no "
            "facts, write: NONE.\n\nTEXT:\n" + ENGLISH_TEXT))

    def test_no_chunks_raises(self):
        self.install([], FakeLLM())
        with self.assertRaises(ValueError):
            self.summarize("missing.pdf")


class LanguageTests(SummaryTestCase):

    def test_detection(self):
        self.assertEqual(sv.detect_language(MARATHI_TEXT), "marathi")
        self.assertEqual(sv.detect_language(ENGLISH_TEXT), "english")
        self.assertEqual(sv.detect_language("1234 ..."), "english")   # no letters
        # Exactly half Devanagari letters is Marathi.
        self.assertEqual(sv.detect_language("कखab"), "marathi")
        self.assertEqual(sv.detect_language("कखabc"), "english")

    def test_marathi_document_gets_marathi_prompts_and_headings(self):
        llm = self.install([chunk(MARATHI_TEXT)], FakeLLM())
        result = self.summarize()
        self.assertEqual(result["language"], "marathi")
        self.assertIn("in Marathi,", llm.map_prompts[0])
        self.assertEqual(llm.reduce_prompts[0], (
            "Combine the bullet points below into one summary of the document in Marathi. "
            "Use exactly these headings: विषय; मुख्य निर्णय; रक्कम व आकडे; संदर्भ आदेश; "
            "आवश्यक कार्यवाही. Under each heading write short bullet points; write 'नमूद नाही' "
            "if the points contain nothing for that heading. Use only facts present in the bullet "
            "points. Copy numbers exactly. Do not add a date, order number or department; they "
            "are added separately.\n\nBULLET POINTS:\n- a fact"))

    def test_override_changes_the_output_language_only(self):
        llm = self.install([chunk(MARATHI_TEXT)], FakeLLM())
        result = self.summarize(language="en")
        self.assertEqual(result["language"], "english")
        self.assertIn("in English,", llm.map_prompts[0])
        self.assertIn("Subject; Key decisions; Amounts and numbers; Referenced orders; "
                      "Action required", llm.reduce_prompts[0])
        # The source text itself is untouched.
        self.assertTrue(llm.map_prompts[0].endswith(MARATHI_TEXT))

    def test_unknown_language_is_rejected(self):
        self.install([chunk(ENGLISH_TEXT)], FakeLLM())
        with self.assertRaises(ValueError):
            self.summarize(language="hindi")


class NoneFilteringTests(SummaryTestCase):

    def test_none_variants(self):
        for text in ("NONE", " none. ", "- NONE", "\"NONE\"", "NONE.\n"):
            self.assertTrue(sv.is_none_output(text), text)
        for text in ("NONE of the amounts are stated", "- 5 posts", ""):
            self.assertFalse(sv.is_none_output(text) and text, text)

    def test_none_outputs_are_dropped_from_the_reduce_prompt(self):
        settings.SUMMARY_BATCH_CHARS = 10
        llm = self.install([chunk("a" * 10, page=0), chunk("b" * 10, page=1),
                            chunk("c" * 10, page=2)],
                           FakeLLM(map_outputs=["- fact one", "NONE", "- fact three"]))
        self.summarize()
        self.assertTrue(llm.reduce_prompts[0].endswith("BULLET POINTS:\n- fact one\n- fact three"))

    def test_all_none_skips_the_reduce_call(self):
        llm = self.install([chunk(ENGLISH_TEXT)], FakeLLM(map_outputs=["NONE"]))
        result = self.summarize()
        self.assertEqual(llm.reduce_prompts, [])
        self.assertEqual(result["summary"].count("- Not stated"), 5)
        self.assertEqual(result["faithfulness"]["score"], 1.0)


class FaithfulnessTests(unittest.TestCase):

    def test_devanagari_digits_are_normalised(self):
        self.assertEqual(number_tokens("रु.३०,०००/- दि.२५/०८/२०२६"), ["30000", "25", "08", "2026"])
        check = check_faithfulness("निधी रु.३०,०००", "निधी रु.30,000/- इतका")
        self.assertEqual(check, {"score": 1.0, "checked": 1, "unsupported": []})

    def test_amounts_with_indian_grouping_are_one_token(self):
        self.assertEqual(number_tokens("रुपये 3,01,71,099/-"), ["30171099"])
        check = check_faithfulness("Total Rs. 30171099", "एकूण रुपये 3,01,71,099/-")
        self.assertEqual(check["unsupported"], [])

    def test_dates_keep_their_dots(self):
        self.assertEqual(number_tokens("दि.17.12.2016"), ["17.12.2016"])
        check = check_faithfulness("dated 17.12.2016", "17 12 2016")
        self.assertEqual(check["unsupported"], ["17.12.2016"])

    def test_single_digit_tokens_are_ignored(self):
        check = check_faithfulness("1. Subject\n2. Decisions\n3 years", "nothing numeric")
        self.assertEqual(check, {"score": 1.0, "checked": 0, "unsupported": []})

    def test_unsupported_tokens_are_reported_once_each(self):
        source = "अंदाजित खर्च रुपये 64,96,111/- व 2,36,74,988/-"
        summary = "Civil 6496111, electrical 23674988, total 30171099 and again 30171099"
        check = check_faithfulness(summary, source)
        self.assertEqual(check["checked"], 3)
        self.assertEqual(check["unsupported"], ["30171099"])
        self.assertAlmostEqual(check["score"], 2 / 3)

    def test_a_token_must_match_whole_not_as_a_substring(self):
        check = check_faithfulness("5 posts and 30 days", "a period of 300 days")
        self.assertEqual(check["unsupported"], ["30"])

    def test_empty_summary_scores_one(self):
        self.assertEqual(check_faithfulness("", "123")["score"], 1.0)


class RepairTests(SummaryTestCase):

    SOURCE = "Total estimated cost Rs 3,01,71,099 approved for the canteen kitchen"

    def test_no_repair_when_every_number_is_supported(self):
        llm = self.install([chunk(self.SOURCE)], FakeLLM(reduce_outputs=["Amounts\n- 30171099"]))
        result = self.summarize()
        self.assertEqual(len(llm.reduce_prompts), 1)
        self.assertEqual(result["faithfulness"]["unsupported"], [])

    def test_repair_runs_once_and_its_better_result_is_kept(self):
        llm = self.install([chunk(self.SOURCE)], FakeLLM(
            reduce_outputs=["Amounts\n- 31000000", "Amounts\n- 30171099"]))
        result = self.summarize()
        self.assertEqual(len(llm.reduce_prompts), 2)
        self.assertTrue(llm.reduce_prompts[1].endswith(
            "\n\nThese figures are not in the source and must be removed or corrected: 31000000."))
        self.assertTrue(llm.reduce_prompts[1].startswith(llm.reduce_prompts[0]))
        self.assertEqual(result["summary"], "Amounts\n- 30171099")
        self.assertEqual(result["faithfulness"]["score"], 1.0)

    def test_repair_is_attempted_at_most_once(self):
        llm = self.install([chunk(self.SOURCE)], FakeLLM(
            reduce_outputs=["Amounts\n- 31000000", "Amounts\n- 32000000", "never used"]))
        result = self.summarize()
        self.assertEqual(len(llm.reduce_prompts), 2)
        self.assertEqual(len(llm.prompts), 3)   # one MAP, REDUCE, one repair
        # A tie keeps the first summary.
        self.assertEqual(result["summary"], "Amounts\n- 31000000")
        self.assertEqual(result["faithfulness"]["unsupported"], ["31000000"])

    def test_worse_repair_is_discarded(self):
        llm = self.install([chunk(self.SOURCE)], FakeLLM(reduce_outputs=[
            "Amounts\n- 30171099 and 99", "Amounts\n- 11 and 22 and 33"]))
        result = self.summarize()
        self.assertEqual(result["summary"], "Amounts\n- 30171099 and 99")
        self.assertEqual(result["faithfulness"]["score"], 0.5)


class HeaderTests(SummaryTestCase):

    def test_header_comes_from_metadata(self):
        llm = self.install([chunk(MARATHI_TEXT)], FakeLLM())
        result = self.summarize()
        self.assertEqual(result["header"], {"department": "सामान्य प्रशासन विभाग",
                                            "number": "GAD-42011/1/2026",
                                            "date": "2026-08-14"})
        # The header is never asked of the LLM.
        self.assertFalse(any("GAD-42011" in p for p in llm.prompts))

    def test_missing_fields_get_the_not_stated_word(self):
        self.meta["GR5.pdf"] = {"gr_number": "GAD-42011/1/2026", "department": "", "doc_date": None}
        self.install([chunk(MARATHI_TEXT)], FakeLLM())
        self.assertEqual(self.summarize()["header"], {"department": "नमूद नाही",
                                                      "number": "GAD-42011/1/2026",
                                                      "date": "नमूद नाही"})

    def test_file_absent_from_metadata(self):
        self.install([chunk(ENGLISH_TEXT, source="GR1.1.pdf")], FakeLLM())
        self.assertEqual(self.summarize("GR1.1.pdf")["header"],
                         {"department": "Not stated", "number": "Not stated", "date": "Not stated"})

    def test_result_keys(self):
        self.install([chunk(ENGLISH_TEXT)], FakeLLM())
        result = self.summarize()
        self.assertEqual(set(result), {"header", "summary", "faithfulness", "seconds",
                                       "batches", "language"})
        self.assertIsInstance(result["seconds"], float)


class RouteFlagTests(unittest.TestCase):
    """api/summary.py with process_gr and process_gr_v2 replaced by fakes."""

    @classmethod
    def setUpClass(cls):
        import api.summary as route
        cls.route = route

    def setUp(self):
        self._saved = (self.route.process_gr, self.route.process_gr_v2,
                       settings.SUMMARY_V2_ENABLED, dict(self.route._summary_state))
        self.calls = []

        async def old(*args, **kwargs):
            self.calls.append(("old", args, kwargs))
            return {"success": True, "message": "ok", "metadata": {"gr_number": "1"},
                    "summary": "old summary", "json_path": "j", "txt_path": "t"}

        async def new(*args, **kwargs):
            self.calls.append(("new", args, kwargs))
            return {"success": True, "message": "ok", "metadata": {"gr_number": "1"},
                    "summary": "new summary", "json_path": "j", "txt_path": "t",
                    "header": {"number": "1"}, "faithfulness": {"score": 1.0}}

        self.route.process_gr, self.route.process_gr_v2 = old, new

    def tearDown(self):
        (self.route.process_gr, self.route.process_gr_v2,
         settings.SUMMARY_V2_ENABLED, state) = self._saved
        self.route._summary_state.clear()
        self.route._summary_state.update(state)

    def test_flag_off_calls_the_old_pipeline_with_the_old_arguments(self):
        settings.SUMMARY_V2_ENABLED = False
        asyncio.run(self.route._run_summary_job("C:/grdocs/GR5.pdf", "GR5.pdf"))
        [(which, args, kwargs)] = self.calls
        self.assertEqual(which, "old")
        self.assertEqual(args, ())
        self.assertEqual(set(kwargs), {"pdf_path", "progress_callback"})
        self.assertEqual(self.route._summary_state["result"],
                         {"metadata": {"gr_number": "1"}, "summary": "old summary",
                          "txt_path": "t"})

    def test_flag_off_batch_job_calls_the_old_pipeline(self):
        settings.SUMMARY_V2_ENABLED = False
        with redirect_stdout(io.StringIO()):
            asyncio.run(self.route._run_batch_summary_job(["GR5.pdf"]))
        [(which, args, kwargs)] = self.calls
        self.assertEqual((which, kwargs), ("old", {}))
        self.assertEqual(args, (str(settings.GRDOCS_PATH / "GR5.pdf"),))

    def test_flag_on_returns_old_fields_plus_header_and_faithfulness(self):
        settings.SUMMARY_V2_ENABLED = True
        asyncio.run(self.route._run_summary_job("C:/grdocs/GR5.pdf", "GR5.pdf"))
        self.assertEqual([c[0] for c in self.calls], ["new"])
        self.assertEqual(self.route._summary_state["result"],
                         {"metadata": {"gr_number": "1"}, "summary": "new summary",
                          "txt_path": "t", "header": {"number": "1"},
                          "faithfulness": {"score": 1.0}})

    def test_flag_on_batch_job_calls_the_new_pipeline(self):
        settings.SUMMARY_V2_ENABLED = True
        with redirect_stdout(io.StringIO()):
            asyncio.run(self.route._run_batch_summary_job(["GR5.pdf"]))
        self.assertEqual([c[0] for c in self.calls], ["new"])


if __name__ == "__main__":
    unittest.main()
