#!/usr/bin/env python
# backend/tests/test_meta.py
# ─────────────────────────────────────────────────────────
# Unit tests for core/gr_meta.py and the metadata fast path in
# core/rag.py.
#
# Synthetic Documents only: no network, no FAISS index, no MongoDB, and
# no model of any kind. The fast path exists precisely so that header
# questions are answered without a model, so testing it must not need one
# either — the tests below would fail if any of it reached for an LLM.
#
# Run:
#   python -m pytest tests/test_meta.py -q
#   python tests/test_meta.py
# ─────────────────────────────────────────────────────────

import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document   # noqa: E402

from core.gr_meta import (                      # noqa: E402
    extract_department,
    extract_metadata,
    header_date,
)
from core.rag import (                          # noqa: E402
    _names_known_stem,
    _strip_known_stems,
    build_meta_answer,
    detect_meta_intents,
    resolve_meta_target,
)


def chunk(source_file, text, section_type=None, page=0, section_index=0,
          gr_number="NUM-1", subject="subject line"):
    """One synthetic chunk, shaped like the ones core.vectorstore writes."""
    stem = source_file.rsplit(".", 1)[0]
    metadata = {
        "source_file":   source_file,
        "page":          page,
        "section_index": section_index,
        "doc_identity":  f"{stem} | {gr_number} | {subject}",
    }
    if section_type is not None:
        metadata["section_type"] = section_type
    return Document(page_content=text, metadata=metadata)


def letterhead(department=None, number="NUM-1", date_text="14.08.2026",
               state_prefix=True):
    """
    A letterhead in the layout this corpus uses:

        महाराष्ट्र शासन
        <department>
        शासन निर्णय क्रमांक <number>
        दिनांक : <date>
    """
    lines = []
    if state_prefix:
        lines.append("महाराष्ट्र शासन")
    if department:
        lines.append(department)
    lines.append(f"शासन निर्णय क्रमांक {number}")
    lines.append(f"दिनांक : {date_text}")
    return "\n".join(lines)


class DepartmentRuleTests(unittest.TestCase):
    """The department is read positionally — the first short line built
    around विभाग that is not some other kind of letterhead line."""

    def test_a_department_line_is_found(self):
        department, raw = extract_department(
            letterhead(department="सामान्य प्रशासन विभाग"))
        self.assertEqual(department, "सामान्य प्रशासन विभाग")
        self.assertEqual(raw, "सामान्य प्रशासन विभाग")

    def test_the_state_prefix_is_removed_when_on_the_same_line(self):
        # OCR frequently runs the boilerplate line and the department
        # together, and the prefix is the same in every document.
        department, raw = extract_department(
            "महाराष्ट्र शासन सामान्य प्रशासन विभाग\nशासन निर्णय क्रमांक X")
        self.assertEqual(department, "सामान्य प्रशासन विभाग")
        # The raw line keeps the prefix, so what was matched stays visible.
        self.assertEqual(raw, "महाराष्ट्र शासन सामान्य प्रशासन विभाग")

    def test_punctuation_and_whitespace_are_stripped(self):
        department, _ = extract_department("महाराष्ट्र शासन\n|| वित्त विभाग, ||\n")
        self.assertEqual(department, "वित्त विभाग")

    def test_a_line_naming_a_gr_is_rejected(self):
        # The number line can mention विभाग and is not the department.
        department, _ = extract_department(
            "महाराष्ट्र शासन\nशासन निर्णय वित्त विभाग क्रमांक 123")
        self.assertIsNone(department)

    def test_a_line_carrying_a_number_is_rejected(self):
        department, _ = extract_department("महाराष्ट्र शासन\nवित्त विभाग क्रमांक 123")
        self.assertIsNone(department)

    def test_a_reference_line_is_rejected(self):
        department, _ = extract_department("संदर्भ वित्त विभाग यांचे पत्र")
        self.assertIsNone(department)
        department, _ = extract_department("वाचा वित्त विभाग यांचे पत्र")
        self.assertIsNone(department)

    def test_a_long_line_is_rejected(self):
        # A sentence that mentions a department is not a department name.
        long_line = "अ" * 40 + " विभाग " + "ब" * 45
        self.assertGreater(len(long_line), 80)
        department, _ = extract_department(f"महाराष्ट्र शासन\n{long_line}")
        self.assertIsNone(department)

    def test_a_letterhead_with_no_department_line_gives_none(self):
        department, raw = extract_department(letterhead(department=None))
        self.assertIsNone(department)
        self.assertIsNone(raw)

    def test_the_first_qualifying_line_wins(self):
        text = "महाराष्ट्र शासन\nवित्त विभाग\nसामान्य प्रशासन विभाग"
        department, _ = extract_department(text)
        self.assertEqual(department, "वित्त विभाग")


class HeaderDateTests(unittest.TestCase):

    def test_the_date_after_the_label_is_taken(self):
        self.assertEqual(header_date("दिनांक : 14.08.2026"),
                         ("2026-08-14", "14.08.2026"))

    def test_the_label_beats_an_earlier_reference_date(self):
        # A letterhead lists the dates of the orders it references above
        # its own, so position alone picks the wrong one.
        text = "संदर्भ: दि. 01.01.2020\nमहाराष्ट्र शासन\nदिनांक : 14 ऑगस्ट, 2026"
        self.assertEqual(header_date(text)[0], "2026-08-14")

    def test_the_first_date_is_used_when_there_is_no_label(self):
        self.assertEqual(header_date("जारी 03.03.2021 रोजी")[0], "2021-03-03")

    def test_devanagari_digits_are_returned_raw_and_normalized(self):
        iso, raw = header_date("दिनांक : १४.०८.२०२६")
        self.assertEqual(iso, "2026-08-14")
        # The raw form is sliced out of the ORIGINAL text, so it keeps the
        # script the document used while doc_date stays sortable.
        self.assertEqual(raw, "१४.०८.२०२६")

    def test_a_marathi_month_name_is_resolved(self):
        iso, raw = header_date("दिनांक : 14 ऑगस्ट, 2026")
        self.assertEqual(iso, "2026-08-14")
        self.assertEqual(raw, "14 ऑगस्ट, 2026")

    def test_no_date_gives_a_pair_of_nones(self):
        self.assertEqual(header_date("महाराष्ट्र शासन"), (None, None))


class ExtractMetadataTests(unittest.TestCase):

    def test_all_fields_come_out_of_one_header_chunk(self):
        docs = [chunk("GR1.pdf",
                      letterhead(department="उच्च व तंत्र शिक्षण विभाग",
                                 number="मराग्रं-2025/58"),
                      section_type="__header__",
                      gr_number="मराग्रं-2025/58",
                      subject="राज्य ग्रंथालय परिषद")]
        entry = extract_metadata(docs)["GR1.pdf"]

        self.assertEqual(entry["source_file"], "GR1.pdf")
        self.assertEqual(entry["stem"], "GR1")
        self.assertEqual(entry["gr_number"], "मराग्रं-2025/58")
        self.assertEqual(entry["subject"], "राज्य ग्रंथालय परिषद")
        self.assertEqual(entry["doc_date"], "2026-08-14")
        self.assertEqual(entry["doc_date_raw"], "14.08.2026")
        self.assertEqual(entry["department"], "उच्च व तंत्र शिक्षण विभाग")
        self.assertEqual(entry["header_page"], 0)
        self.assertFalse(entry["header_fallback"])

    def test_header_chunks_are_joined_in_reading_order(self):
        # The letterhead can be split across chunks; the date lives in the
        # later one and must still be found.
        docs = [
            chunk("A.pdf", "दिनांक : 02.02.2020", section_type="__header__",
                  page=0, section_index=1),
            chunk("A.pdf", "महाराष्ट्र शासन\nवित्त विभाग", section_type="__header__",
                  page=0, section_index=0),
        ]
        entry = extract_metadata(docs)["A.pdf"]
        self.assertEqual(entry["department"], "वित्त विभाग")
        self.assertEqual(entry["doc_date"], "2020-02-02")

    def test_header_fallback_uses_the_smallest_page(self):
        # No chunk is marked __header__, so the opening page stands in.
        docs = [
            chunk("B.pdf", "later page text", page=5, section_index=1),
            chunk("B.pdf", "महाराष्ट्र शासन\nवित्त विभाग\nदिनांक : 02.02.2020",
                  page=3, section_index=0),
        ]
        entry = extract_metadata(docs)["B.pdf"]

        self.assertTrue(entry["header_fallback"])
        self.assertEqual(entry["header_page"], 3)
        self.assertEqual(entry["department"], "वित्त विभाग")
        self.assertEqual(entry["doc_date"], "2020-02-02")

    def test_the_fallback_ignores_text_from_later_pages(self):
        # A date on a later page must not become the document's own date.
        docs = [
            chunk("B.pdf", "दिनांक : 09.09.1999", page=7, section_index=1),
            chunk("B.pdf", "महाराष्ट्र शासन\nदिनांक : 02.02.2020", page=3, section_index=0),
        ]
        entry = extract_metadata(docs)["B.pdf"]
        self.assertEqual(entry["doc_date"], "2020-02-02")

    def test_missing_identity_parts_become_empty_strings(self):
        docs = [Document(page_content="x", metadata={
            "source_file": "C.pdf", "page": 0, "doc_identity": "C"})]
        entry = extract_metadata(docs)["C.pdf"]

        self.assertEqual(entry["stem"], "C")
        self.assertEqual(entry["gr_number"], "")
        self.assertEqual(entry["subject"], "")

    def test_header_chunk_text_is_capped(self):
        docs = [chunk("D.pdf", "अ" * 900, section_type="__header__")]
        entry = extract_metadata(docs)["D.pdf"]
        self.assertEqual(len(entry["header_chunk_text"]), 400)

    def test_several_files_are_keyed_separately(self):
        docs = [
            chunk("A.pdf", letterhead(department="वित्त विभाग"), section_type="__header__"),
            chunk("B.pdf", letterhead(department="मराठी भाषा विभाग"), section_type="__header__"),
        ]
        meta = extract_metadata(docs)
        self.assertEqual(set(meta), {"A.pdf", "B.pdf"})
        self.assertEqual(meta["B.pdf"]["department"], "मराठी भाषा विभाग")

    def test_empty_input(self):
        self.assertEqual(extract_metadata([]), {})


class DetectMetaIntentsTests(unittest.TestCase):
    """Guards first, intents second. Every guard has to reject on its own,
    because a fast path that answers a question it misread is worse than
    no fast path at all."""

    def test_a_date_question_is_detected(self):
        self.assertEqual(detect_meta_intents("शासन निर्णयाचा दिनांक काय आहे?"),
                         ["date"])

    def test_a_department_question_is_detected(self):
        self.assertEqual(detect_meta_intents("या शासन आदेशाचा विभाग कोणता?"),
                         ["department"])

    def test_a_number_question_is_detected(self):
        self.assertEqual(detect_meta_intents("शासन निर्णय क्रमांक काय आहे?"),
                         ["gr_number"])

    def test_several_intents_come_back_in_the_fixed_order(self):
        self.assertEqual(
            detect_meta_intents("शासन निर्णयाचा क्रमांक, विभाग व दिनांक?"),
            ["date", "department", "gr_number"],
        )

    def test_english_triggers_work(self):
        self.assertEqual(detect_meta_intents("What is the date of this GR?"),
                         ["date"])

    def test_alternate_date_words(self):
        for word in ("तारीख", "कधी"):
            with self.subTest(word=word):
                self.assertEqual(
                    detect_meta_intents(f"शासन निर्णयाची {word} काय?"), ["date"])

    # ── guards ──────────────────────────────────────────────

    def test_a_question_over_eighty_characters_is_rejected(self):
        question = "शासन निर्णयाचा दिनांक " + "अ" * 80
        self.assertGreater(len(question), 80)
        self.assertEqual(detect_meta_intents(question), [])

    def test_an_ascii_digit_rejects(self):
        # A question carrying numbers is almost always about the
        # document's CONTENTS, where the metadata table says nothing.
        self.assertEqual(detect_meta_intents("शासन निर्णय 1518 चा दिनांक?"), [])

    def test_devanagari_digits_reject_too(self):
        # They are folded to ASCII before the guard runs.
        self.assertEqual(detect_meta_intents("शासन निर्णय १५१८ चा दिनांक?"), [])

    def test_a_relationship_word_rejects(self):
        for word in ("संदर्भ", "वाचा", "संबंध"):
            with self.subTest(word=word):
                self.assertEqual(
                    detect_meta_intents(f"शासन निर्णयाचा {word} दिनांक काय?"), [])

    def test_no_document_type_root_rejects(self):
        # Without a document type named, there is nothing to resolve.
        self.assertEqual(detect_meta_intents("दिनांक काय आहे?"), [])

    def test_a_document_type_root_with_no_intent_word_gives_nothing(self):
        self.assertEqual(detect_meta_intents("शासन निर्णयात किती सदस्य आहेत?"), [])

    def test_each_document_type_root_qualifies(self):
        for root in ("शासन", "परिपत्रक", "आदेश"):
            with self.subTest(root=root):
                self.assertEqual(
                    detect_meta_intents(f"{root} दिनांक काय?"), ["date"])

    def test_empty_and_none(self):
        self.assertEqual(detect_meta_intents(""), [])
        self.assertEqual(detect_meta_intents(None), [])


class KnownStemDigitExemptionTests(unittest.TestCase):
    """Every stem in this corpus carries a digit — GR1, aadesh 1, letter 2
    — so naming the document a question is about is the one legitimate way
    a header question contains a number. known_stems discounts those
    digits before the guard runs, and nothing else."""

    STEMS = ["GR1", "GR1.5"]

    def test_a_question_naming_a_document_is_accepted(self):
        self.assertEqual(
            detect_meta_intents("GR1.pdf शासन निर्णयाचा दिनांक काय आहे?",
                                known_stems=self.STEMS),
            ["date"],
        )

    def test_the_same_question_is_still_rejected_without_the_stems(self):
        # The exemption is what changed the outcome, not the question.
        self.assertEqual(
            detect_meta_intents("GR1.pdf शासन निर्णयाचा दिनांक काय आहे?"),
            [],
        )

    def test_a_bare_year_is_still_rejected(self):
        # No stem to discount, so the digit guard fires as it always did.
        self.assertEqual(
            detect_meta_intents("2026 मधील शासन निर्णयाचा दिनांक?",
                                known_stems=self.STEMS),
            [],
        )

    def test_a_number_beside_a_named_stem_is_still_rejected(self):
        # Removing GR1 leaves 1518 behind, which is a question about the
        # document's contents, so the guard must still reject it.
        self.assertEqual(
            detect_meta_intents("GR1 मधील 1518 शासन निर्णयाचा दिनांक?",
                                known_stems=self.STEMS),
            [],
        )

    def test_the_longest_stem_is_removed_first(self):
        # Removing "GR1" first would leave ".5", and the stray digit would
        # reject a question that names its document perfectly well.
        self.assertEqual(
            detect_meta_intents("GR1.5 शासन निर्णयाचा दिनांक काय आहे?",
                                known_stems=self.STEMS),
            ["date"],
        )

    def test_the_longest_stem_wins_whatever_order_it_is_given_in(self):
        for stems in (["GR1", "GR1.5"], ["GR1.5", "GR1"]):
            with self.subTest(stems=stems):
                self.assertEqual(
                    detect_meta_intents("GR1.5 शासन आदेशाचा दिनांक?",
                                        known_stems=stems),
                    ["date"],
                )

    def test_stem_matching_is_case_insensitive(self):
        self.assertEqual(
            detect_meta_intents("gr1.5 शासन निर्णयाचा दिनांक?",
                                known_stems=["GR1.5"]),
            ["date"],
        )
        self.assertEqual(
            detect_meta_intents("GR1.5 शासन निर्णयाचा दिनांक?",
                                known_stems=["gr1.5"]),
            ["date"],
        )

    def test_a_devanagari_digit_stem_is_matched_after_folding(self):
        # Question and stem are both folded before comparison, so a stem
        # written in one script matches a question written in the other.
        self.assertEqual(
            detect_meta_intents("GR१ शासन निर्णयाचा दिनांक?",
                                known_stems=["GR1"]),
            ["date"],
        )

    def test_every_occurrence_is_removed(self):
        self.assertEqual(
            detect_meta_intents("GR1 व GR1 शासन आदेशाचा दिनांक?",
                                known_stems=["GR1"]),
            ["date"],
        )

    def test_no_stems_behaves_exactly_as_before(self):
        for stems in (None, [], ["", None]):
            with self.subTest(stems=stems):
                self.assertEqual(
                    detect_meta_intents("शासन निर्णयाचा दिनांक काय आहे?",
                                        known_stems=stems),
                    ["date"],
                )
                self.assertEqual(
                    detect_meta_intents("शासन निर्णय 1518 चा दिनांक?",
                                        known_stems=stems),
                    [],
                )

    def test_the_other_guards_are_unaffected_by_the_exemption(self):
        # Each of these must still reject even with its stem discounted.
        rejected = [
            # over the length limit, measured on the ORIGINAL question
            "GR1 शासन निर्णयाचा दिनांक " + "अ" * 80,
            # relationship words
            "GR1 शासन निर्णयाच्या संदर्भातील दिनांक काय?",
            "GR1 शासन परिपत्रकात वाचा दिनांक काय?",
            "GR1 शासन निर्णयाचा या प्रकरणाशी संबंध काय आहे?",
            # no intent word
            "GR1 शासन निर्णयात किती सदस्य आहेत?",
        ]
        for question in rejected:
            with self.subTest(question=question[:40]):
                self.assertEqual(
                    detect_meta_intents(question, known_stems=self.STEMS), [])

    def test_the_document_type_guard_still_rejects(self):
        # REPLACED ASSERTION. This used to pin
        #   detect_meta_intents("letter 1 दिनांक काय आहे?",
        #                       known_stems=["letter 1"]) == []
        # on the grounds that "letter 1" contains no document-type root.
        # A known stem now satisfies that guard on its own, so the same
        # call returns ["date"] — see
        # KnownStemSatisfiesDocumentTypeTests.test_the_letter_stem_fires,
        # which asserts the new behaviour.
        #
        # What the guard still rejects is a question that identifies NO
        # document: no document-type root and no stem it was told about.
        # Deliberately digit-free, so the DIGIT guard cannot be what
        # rejects it — "letter 1 ..." with a stem list that does not
        # contain "letter 1" would fail on the stray 1 long before the
        # document-type guard was reached, and would prove nothing.
        self.assertEqual(
            detect_meta_intents("दिनांक काय आहे?", known_stems=["letter 1"]),
            [],
        )
        # The same question with a document type named is accepted, which
        # is what shows the rejection came from that guard and not from
        # the digit in the stem.
        self.assertEqual(
            detect_meta_intents("letter 1 शासन आदेशाचा दिनांक काय आहे?",
                                known_stems=["letter 1"]),
            ["date"],
        )

    def test_the_length_guard_is_measured_on_the_original_question(self):
        # Long only because of the stem. Stripping first would let it
        # through, and the guard is asking how much the user wrote.
        question = "GR1.5" * 17 + " शासन निर्णयाचा दिनांक?"
        self.assertGreater(len(question), 80)
        self.assertEqual(
            detect_meta_intents(question, known_stems=self.STEMS), [])


class StripKnownStemsTests(unittest.TestCase):
    """The removal itself, so a failure points at the stripping rather
    than at the guard that consumes it."""

    def test_longest_first_leaves_no_stray_digit(self):
        self.assertNotIn("5", _strip_known_stems("GR1.5", ["GR1", "GR1.5"]))

    def test_shortest_only_would_have_left_one(self):
        # Pins WHY the ordering matters: removing just the short stem
        # leaves behind the digit the guard would then reject.
        self.assertIn("5", _strip_known_stems("GR1.5", ["GR1"]))

    def test_unrelated_digits_survive(self):
        self.assertIn("2026", _strip_known_stems("GR1 2026", ["GR1"]))

    def test_no_stems_returns_the_input_unchanged(self):
        self.assertEqual(_strip_known_stems("GR1.5", None), "GR1.5")
        self.assertEqual(_strip_known_stems("GR1.5", []), "GR1.5")

    def test_blank_stems_are_skipped(self):
        self.assertEqual(_strip_known_stems("GR1", ["", None]), "GR1")

    def test_a_stem_that_does_not_appear_changes_nothing(self):
        self.assertEqual(_strip_known_stems("GR4", ["GR1"]), "GR4")

    def test_removal_does_not_weld_neighbours_together(self):
        # A stem is replaced by a space, so the words either side cannot
        # fuse into a word nobody wrote.
        self.assertEqual(_strip_known_stems("abcGR1def", ["GR1"]), "abc def")

    def test_stems_with_spaces_are_removed(self):
        # "aadesh 1" and "letter 2" are real stems in this corpus.
        self.assertNotIn("1", _strip_known_stems("aadesh 1", ["aadesh 1"]))


class KnownStemSatisfiesDocumentTypeTests(unittest.TestCase):
    """
    Naming a document is naming a document.

    The document-type guard exists to reject a question that identifies
    nothing — a bare "कधी?" has no document to read a header from. But it
    was written as a vocabulary test against शासन / परिपत्रक / आदेश / GR,
    so "aadesh 5 चा दिनांक काय आहे?" failed it: the question names exactly
    one document and happens to use none of those words. Those questions
    went to retrieval and the model instead, which is the path that
    answered them with the wrong document's date.

    A known stem now satisfies the guard on its own. Nothing else about
    the guards changed, and the tests below pin each of them still firing.
    """

    # Real stems from this corpus. None of them contains a document-type
    # root, which is the whole point — "GR1" would pass the old guard
    # through the letters "GR" alone and prove nothing.
    STEMS = ["aadesh 5", "letter 1", "paripatrika 2"]

    def test_the_reported_question_now_fires(self):
        self.assertEqual(
            detect_meta_intents("aadesh 5 चा दिनांक काय आहे?",
                                known_stems=self.STEMS),
            ["date"],
        )

    def test_the_same_question_without_stems_is_still_rejected(self):
        # The stem list is what changed the outcome, not the question.
        # With no stems the digit guard is what rejects it, since the "5"
        # is no longer exempt — either way the fast path does not fire.
        self.assertEqual(
            detect_meta_intents("aadesh 5 चा दिनांक काय आहे?"),
            [],
        )
        self.assertEqual(
            detect_meta_intents("aadesh 5 चा दिनांक काय आहे?", known_stems=[]),
            [],
        )

    def test_the_letter_stem_fires(self):
        # This replaces the first assertion of
        # KnownStemDigitExemptionTests.test_the_document_type_guard_still_rejects,
        # which pinned the old rejection of exactly this question.
        self.assertEqual(
            detect_meta_intents("letter 1 चा दिनांक काय आहे?",
                                known_stems=self.STEMS),
            ["date"],
        )

    def test_a_blocking_root_still_falls_through(self):
        # संदर्भ / वाचा / संबंध mean the question is about relationships
        # between documents, which the header table cannot answer. Naming
        # a document does not change that.
        for question in (
            "aadesh 5 च्या संदर्भातील दिनांक काय?",
            "aadesh 5 मध्ये वाचा दिनांक काय?",
            "aadesh 5 चा संबंध काय आहे?",
        ):
            with self.subTest(question=question[:36]):
                self.assertEqual(
                    detect_meta_intents(question, known_stems=self.STEMS), [])

    def test_a_contents_question_naming_a_stem_still_gives_nothing(self):
        # No intent word, so there is no header field being asked for.
        # "रक्कम" is a contents question: the amount is in the body.
        self.assertEqual(
            detect_meta_intents("aadesh 5 ची रक्कम किती आहे?",
                                known_stems=self.STEMS),
            [],
        )

    def test_a_digit_outside_the_stem_still_rejects(self):
        # The stem's own digit is exempt; a second number is not, because
        # a question carrying numbers is almost always about the body.
        self.assertEqual(
            detect_meta_intents("aadesh 5 आणि 7 चा दिनांक",
                                known_stems=self.STEMS),
            [],
        )

    def test_the_length_guard_still_applies(self):
        question = "aadesh 5 चा दिनांक " + "अ" * 80
        self.assertGreater(len(question), 80)
        self.assertEqual(
            detect_meta_intents(question, known_stems=self.STEMS), [])

    def test_a_question_naming_nothing_is_still_rejected(self):
        # The guard's original job, unchanged: no document type, no stem.
        self.assertEqual(
            detect_meta_intents("दिनांक काय आहे?", known_stems=self.STEMS), [])

    def test_all_three_intents_still_resolve_through_a_stem(self):
        self.assertEqual(
            detect_meta_intents("aadesh 5 चा क्रमांक, विभाग व दिनांक?",
                                known_stems=self.STEMS),
            ["date", "department", "gr_number"],
        )

    def test_stem_matching_is_case_insensitive(self):
        self.assertEqual(
            detect_meta_intents("AADESH 5 चा दिनांक काय आहे?",
                                known_stems=["aadesh 5"]),
            ["date"],
        )

    def test_a_devanagari_digit_stem_matches_after_folding(self):
        # Both sides are folded, so the scripts need not agree.
        self.assertEqual(
            detect_meta_intents("aadesh ५ चा दिनांक काय आहे?",
                                known_stems=["aadesh 5"]),
            ["date"],
        )

    def test_a_document_type_root_alone_still_works(self):
        # The old path is untouched: no stems supplied, root present.
        self.assertEqual(
            detect_meta_intents("शासन निर्णयाचा दिनांक काय आहे?"), ["date"])


class TwoStemsInOneQuestionTests(unittest.TestCase):
    """
    What happens when a question names two documents.

    Unchanged by this fix, and worth pinning because the two halves of the
    fast path answer differently on purpose: detect_meta_intents does not
    disambiguate — it only decides whether a header field is being asked
    for — while resolve_meta_target refuses to choose between candidates
    and returns None, so try_meta_fastpath falls through to retrieval.
    """

    STEMS = ["aadesh 5", "letter 1"]
    META = {
        "aadesh 5.pdf": {"stem": "aadesh 5", "gr_number": "असव-1226"},
        "letter 1.pdf": {"stem": "letter 1", "gr_number": "उसप्र-1126"},
    }

    QUESTION = "aadesh 5 व letter 1 चा दिनांक काय आहे?"

    def test_intents_are_still_detected(self):
        # One stem is enough to satisfy the guard, and a second changes
        # nothing about which field is being asked for.
        self.assertEqual(
            detect_meta_intents(self.QUESTION, known_stems=self.STEMS),
            ["date"],
        )

    def test_the_target_is_ambiguous_so_nothing_resolves(self):
        self.assertIsNone(
            resolve_meta_target(self.QUESTION, None, self.META))

    def test_an_explicit_focus_file_still_decides(self):
        # The caller naming the document is the one thing that beats
        # ambiguity, and that behaviour is unchanged.
        self.assertEqual(
            resolve_meta_target(self.QUESTION, "letter 1.pdf", self.META),
            "letter 1.pdf",
        )


class NamesKnownStemTests(unittest.TestCase):
    """The presence test itself, so a failure points here rather than at
    the guard that consumes it."""

    def test_a_stem_is_found(self):
        self.assertTrue(_names_known_stem("aadesh 5 चा दिनांक", ["aadesh 5"]))

    def test_an_absent_stem_is_not_found(self):
        self.assertFalse(_names_known_stem("GR9 चा दिनांक", ["aadesh 5"]))

    def test_no_stems_is_false(self):
        self.assertFalse(_names_known_stem("aadesh 5", None))
        self.assertFalse(_names_known_stem("aadesh 5", []))

    def test_blank_stems_are_skipped(self):
        self.assertFalse(_names_known_stem("aadesh 5", ["", None]))

    def test_matching_is_case_insensitive(self):
        self.assertTrue(_names_known_stem("AADESH 5", ["aadesh 5"]))
        self.assertTrue(_names_known_stem("aadesh 5", ["AADESH 5"]))

    def test_a_longer_stem_is_also_found(self):
        # Ordering is longest-first, which cannot change a boolean answer
        # — this pins that both still match.
        for stems in (["GR1", "GR1.5"], ["GR1.5", "GR1"]):
            with self.subTest(stems=stems):
                self.assertTrue(_names_known_stem("GR1.5 चा दिनांक", stems))


class ResolveMetaTargetTests(unittest.TestCase):

    META = {
        "GR1.pdf":   {"stem": "GR1",   "gr_number": "मराग्रं-2025/प्र.क्र.58/ई-1093601"},
        "GR1.5.pdf": {"stem": "GR1.5", "gr_number": "मराग्रं-2025/प्र.क्र.58/ई-9999999"},
        "GR4.pdf":   {"stem": "GR4",   "gr_number": "GAD-42011/1/2026"},
    }

    def test_focus_file_wins(self):
        # The caller knows which document the user is looking at, and no
        # amount of string matching beats that.
        self.assertEqual(
            resolve_meta_target("कोणताही प्रश्न", "GR4.pdf", self.META),
            "GR4.pdf",
        )

    def test_an_unknown_focus_file_is_ignored(self):
        self.assertEqual(
            resolve_meta_target("GR4 चा दिनांक?", "missing.pdf", self.META),
            "GR4.pdf",
        )

    def test_a_stem_is_matched_as_a_whole_token(self):
        self.assertEqual(resolve_meta_target("GR4 चा दिनांक?", None, self.META),
                         "GR4.pdf")

    def test_gr1_does_not_match_a_question_about_gr1_point_5(self):
        # The dot-digit rule. Without it every GR1.5 question would also
        # resolve to GR1, and the fast path would answer from the wrong
        # document with full confidence.
        self.assertEqual(resolve_meta_target("GR1.5 चा दिनांक?", None, self.META),
                         "GR1.5.pdf")

    def test_a_stem_followed_by_a_letter_does_not_match(self):
        self.assertIsNone(resolve_meta_target("GR4X चा दिनांक?", None, self.META))

    def test_a_stem_preceded_by_a_letter_does_not_match(self):
        self.assertIsNone(resolve_meta_target("XGR4 चा दिनांक?", None, self.META))

    def test_a_shared_digit_run_in_the_gr_number_resolves(self):
        self.assertEqual(
            resolve_meta_target("1093601 चा दिनांक?", None, self.META),
            "GR1.pdf",
        )

    def test_a_short_digit_run_does_not_resolve(self):
        # Fewer than six digits is a year or a clause number, which
        # collide across unrelated documents.
        self.assertIsNone(resolve_meta_target("2025 चा दिनांक?", None, self.META))

    def test_two_candidates_give_none(self):
        meta = {
            "A.pdf": {"stem": "GR", "gr_number": "X"},
            "B.pdf": {"stem": "GR", "gr_number": "Y"},
        }
        self.assertIsNone(resolve_meta_target("GR चा दिनांक?", None, meta))

    def test_nothing_matching_gives_none(self):
        self.assertIsNone(resolve_meta_target("काहीही", None, self.META))

    def test_an_empty_table_gives_none(self):
        self.assertIsNone(resolve_meta_target("GR4", "GR4.pdf", {}))


class BuildMetaAnswerTests(unittest.TestCase):

    ENTRY = {
        "source_file":  "GR1.pdf",
        "header_page":  0,
        "doc_date":     "2026-08-14",
        "doc_date_raw": "14 ऑगस्ट, 2026",
        "department":   "उच्च व तंत्र शिक्षण विभाग",
        "gr_number":    "मराग्रं-2025/प्र.क्र.58",
    }

    def test_one_intent(self):
        answer = build_meta_answer(self.ENTRY, ["date"])
        self.assertEqual(
            answer,
            "दिनांक (Date): 14 ऑगस्ट, 2026\nSource: GR1.pdf, page 1",
        )

    def test_all_three_intents_in_the_fixed_order(self):
        answer = build_meta_answer(
            self.ENTRY, ["gr_number", "department", "date"])
        lines = answer.split("\n")

        self.assertTrue(lines[0].startswith("दिनांक (Date): "))
        self.assertTrue(lines[1].startswith("विभाग (Department): "))
        self.assertTrue(lines[2].startswith("शासन निर्णय क्रमांक (GR number): "))
        self.assertEqual(lines[3], "Source: GR1.pdf, page 1")

    def test_the_raw_date_is_preferred(self):
        # The answer reads back the date the document prints, rather than
        # a normalised form the reader would have to trust.
        self.assertIn("14 ऑगस्ट, 2026", build_meta_answer(self.ENTRY, ["date"]))

    def test_the_normalized_date_is_the_fallback(self):
        entry = dict(self.ENTRY, doc_date_raw=None)
        self.assertIn("2026-08-14", build_meta_answer(entry, ["date"]))

    def test_a_missing_requested_field_gives_none(self):
        # Half an answer presented as a whole one is worse than falling
        # through to the normal pipeline.
        entry = dict(self.ENTRY, department=None)
        self.assertIsNone(build_meta_answer(entry, ["date", "department"]))

    def test_a_missing_field_that_was_not_requested_is_harmless(self):
        entry = dict(self.ENTRY, department=None)
        self.assertIsNotNone(build_meta_answer(entry, ["date"]))

    def test_a_missing_date_in_both_forms_gives_none(self):
        entry = dict(self.ENTRY, doc_date=None, doc_date_raw=None)
        self.assertIsNone(build_meta_answer(entry, ["date"]))

    def test_no_intents_gives_none(self):
        self.assertIsNone(build_meta_answer(self.ENTRY, []))

    def test_no_entry_gives_none(self):
        self.assertIsNone(build_meta_answer(None, ["date"]))
        self.assertIsNone(build_meta_answer({}, ["date"]))


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    unittest.main(verbosity=2)
