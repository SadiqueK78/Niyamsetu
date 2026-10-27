#!/usr/bin/env python
# backend/tests/test_graph.py
# ─────────────────────────────────────────────────────────
# Unit tests for core/gr_graph.py's pure extraction layer.
#
# Everything here runs on synthetic LangChain Documents built in the test
# itself. No network, no FAISS index, no MongoDB, no PDFs — which is the
# whole point of build_graph_from_docs() being pure: the extraction rules
# can be pinned down on hand-written chunks where the right answer is
# known, rather than inferred from whatever the live corpus happens to
# contain today.
#
# Run:
#   python -m pytest tests/test_graph.py -q
#   python tests/test_graph.py            # unittest runner
# ─────────────────────────────────────────────────────────

import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document   # noqa: E402

import core.rag as rag                          # noqa: E402
from config import settings                     # noqa: E402

from core.gr_graph import (                      # noqa: E402
    build_graph_from_docs,
    find_dates,
    fold_digits,
    normalize_number,
    RELATION_SUPERSEDES,
    RELATION_AMENDS,
    RELATION_REFERENCES,
)


# A GR reference number of the kind this corpus uses, and a long digit run
# of the kind a strong-key match keys on.
NUMBER     = "मराग्रं-2025/प्र.क्र.58"
STRONG_KEY = "1093601"


def chunk(source_file, text, section_type=None, page=0, section_index=0,
          gr_number="NUM", subject="subject line"):
    """
    One synthetic chunk, shaped like the ones core.vectorstore writes.

    doc_identity is "stem | number | subject", which is the format
    _document_identity() produces and the format the node fields are
    split out of.
    """
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


def header(source_file, date_text, gr_number="NUM", **kwargs):
    """A letterhead chunk carrying a दिनांक line and the document's number."""
    return chunk(
        source_file,
        f"महाराष्ट्र शासन\nशासन निर्णय क्रमांक {gr_number}\nदिनांक : {date_text}",
        section_type="__header__",
        gr_number=gr_number,
        **kwargs,
    )


def references(source_file, body, gr_number="NUM", **kwargs):
    """A संदर्भ chunk holding the numbered reference list."""
    return chunk(source_file, body, section_type="संदर्भ",
                 page=0, section_index=1, gr_number=gr_number, **kwargs)


def body_chunk(source_file, body, gr_number="NUM", **kwargs):
    """A body chunk, where relation wording lives."""
    return chunk(source_file, body, section_type="शासन निर्णय",
                 page=1, section_index=2, gr_number=gr_number, **kwargs)


def edges_from(edges, source):
    return [e for e in edges if e["source"] == source]


class DateNormalizationTests(unittest.TestCase):
    """Dates are the only field two documents can be joined on when their
    numbers disagree, so every supported spelling has to land on the same
    ISO string."""

    def test_dotted_form(self):
        self.assertEqual(find_dates("दिनांक : 14.08.2026"), ["2026-08-14"])

    def test_slashed_form(self):
        self.assertEqual(find_dates("दिनांक 14/08/2026"), ["2026-08-14"])

    def test_devanagari_digits_fold(self):
        self.assertEqual(fold_digits("१४.०८.२०२६"), "14.08.2026")
        self.assertEqual(find_dates("१४.०८.२०२६"), ["2026-08-14"])

    def test_marathi_month_name_form(self):
        self.assertEqual(find_dates("14 ऑगस्ट, 2026"), ["2026-08-14"])

    def test_marathi_month_name_without_comma(self):
        self.assertEqual(find_dates("14 ऑगस्ट 2026"), ["2026-08-14"])

    def test_every_marathi_month_name(self):
        names = ["जानेवारी", "फेब्रुवारी", "मार्च", "एप्रिल", "मे", "जून",
                 "जुलै", "ऑगस्ट", "सप्टेंबर", "ऑक्टोबर", "नोव्हेंबर", "डिसेंबर"]
        for number, name in enumerate(names, 1):
            with self.subTest(month=name):
                self.assertEqual(
                    find_dates(f"5 {name}, 2026"),
                    [f"2026-{number:02d}-05"],
                )

    def test_ocr_damaged_month_name_still_resolves(self):
        # ऑगस्त for ऑगस्ट — one wrong consonant, which is what extraction
        # from a legacy Marathi font produces.
        self.assertEqual(find_dates("14 ऑगस्त, 2026"), ["2026-08-14"])

    def test_english_month_names(self):
        self.assertEqual(find_dates("3 March 2019"), ["2019-03-03"])
        self.assertEqual(find_dates("3 Mar, 2019"), ["2019-03-03"])

    def test_unrecognisable_month_token_is_rejected(self):
        self.assertEqual(find_dates("14  झझझझ, 2026"), [])

    def test_impossible_dates_are_rejected(self):
        self.assertEqual(find_dates("31.02.2026"), [])
        self.assertEqual(find_dates("45.13.2026"), [])

    def test_dates_come_back_in_reading_order(self):
        self.assertEqual(
            find_dates("पहिली 01.01.2020 नंतर 2 मार्च, 2021"),
            ["2020-01-01", "2021-03-02"],
        )

    def test_date_word_wins_over_an_earlier_date(self):
        # The letterhead quotes a referenced GR's date above its own, so
        # position alone picks the wrong one; दिनांक disambiguates.
        docs = [chunk(
            "A.pdf",
            "संदर्भ: शासन निर्णय दि. 01.01.2020\nमहाराष्ट्र शासन\nदिनांक : 14 ऑगस्ट, 2026",
            section_type="__header__",
        )]
        nodes, _ = build_graph_from_docs(docs)
        self.assertEqual(nodes[0]["doc_date"], "2026-08-14")

    def test_no_header_chunk_means_no_date(self):
        nodes, _ = build_graph_from_docs([chunk("A.pdf", "दिनांक : 14.08.2026")])
        self.assertIsNone(nodes[0]["doc_date"])


class NodeFieldTests(unittest.TestCase):
    """The node is what the graph page shows, so its fields have to survive
    a letterhead that could only be read in part."""

    def test_identity_splits_into_three_fields(self):
        nodes, _ = build_graph_from_docs(
            [header("GR1.pdf", "14.08.2026", gr_number=NUMBER)]
        )
        node = nodes[0]
        self.assertEqual(node["id"], "GR1.pdf")
        self.assertEqual(node["label"], "GR1")
        self.assertEqual(node["gr_number"], NUMBER)
        self.assertEqual(node["subject"], "subject line")
        self.assertTrue(node["resolved"])

    def test_missing_identity_parts_become_empty_strings(self):
        docs = [Document(page_content="x", metadata={
            "source_file": "B.pdf", "page": 0, "doc_identity": "B"})]
        nodes, _ = build_graph_from_docs(docs)
        self.assertEqual(nodes[0]["label"], "B")
        self.assertEqual(nodes[0]["gr_number"], "")
        self.assertEqual(nodes[0]["subject"], "")

    def test_filename_field_is_kept(self):
        # The stored shape has always carried filename; removing it would
        # break remove_from_graph(), which matches nodes on it.
        nodes, _ = build_graph_from_docs([header("GR1.pdf", "14.08.2026")])
        self.assertEqual(nodes[0]["filename"], "GR1.pdf")


class NumberNormalizationTests(unittest.TestCase):

    def test_punctuation_and_spaces_are_dropped(self):
        self.assertEqual(
            normalize_number("मराग्रं 2025 / प्र. क्र. 58"),
            normalize_number("मराग्रं-2025/प्र.क्र.58"),
        )

    def test_devanagari_digits_are_folded(self):
        self.assertIn("1026", normalize_number("१०२६/प्र.क्र.२०"))


class StrongKeyMatchTests(unittest.TestCase):

    def test_shared_long_digit_run_links_the_documents(self):
        docs = [
            header("NEW.pdf", "01.01.2026", gr_number="NEW-0000001"),
            references("NEW.pdf",
                       f"1) शासन निर्णय क्र. {STRONG_KEY} दि. 02.02.2020",
                       gr_number="NEW-0000001"),
            header("OLD.pdf", "02.02.2020", gr_number=STRONG_KEY),
        ]
        _, edges = build_graph_from_docs(docs)
        linked = edges_from(edges, "NEW.pdf")

        self.assertEqual(len(linked), 1)
        self.assertEqual(linked[0]["target"], "OLD.pdf")
        self.assertEqual(linked[0]["match_reason"], "strong_key")
        self.assertEqual(linked[0]["ref_index"], 1)
        self.assertFalse(linked[0]["ambiguous"])

    def test_short_digit_runs_do_not_link(self):
        # Four digits is a year or a clause number; those collide across
        # unrelated documents constantly, so they must not be strong keys.
        docs = [
            header("A.pdf", "01.01.2026", gr_number="A-1111111"),
            references("A.pdf", "1) शासन निर्णय क्र. 2025 मधील तरतूद",
                       gr_number="A-1111111"),
            header("B.pdf", "03.03.2019", gr_number="2025"),
        ]
        _, edges = build_graph_from_docs(docs)
        self.assertEqual(
            [e["target"] for e in edges_from(edges, "A.pdf") if e["match_reason"]],
            [],
        )


class SelfExclusionTests(unittest.TestCase):

    def test_a_document_never_links_to_itself(self):
        # Its own number appears in its own letterhead and again in its
        # body, which is exactly the shape that would produce a self-edge.
        docs = [
            header("A.pdf", "01.01.2026", gr_number=STRONG_KEY),
            references("A.pdf",
                       f"1) शासन निर्णय क्र. {STRONG_KEY} दि. 01.01.2026",
                       gr_number=STRONG_KEY),
        ]
        _, edges = build_graph_from_docs(docs)
        self.assertEqual([e for e in edges if e["source"] == e["target"]], [])

    def test_a_letterhead_does_not_become_an_external_node(self):
        # The letterhead is part of the reference text, so a document
        # stating its own number must not acquire a phantom neighbour.
        docs = [header("A.pdf", "01.01.2026", gr_number=NUMBER)]
        nodes, edges = build_graph_from_docs(docs)
        self.assertEqual([n for n in nodes if not n["resolved"]], [])
        self.assertEqual(edges, [])


class DateDisambiguationTests(unittest.TestCase):
    """Two GRs can carry the same reference number and differ only by date,
    which is the case the date check exists for."""

    def setUp(self):
        self.docs = [
            header("CITE.pdf", "01.01.2026", gr_number="CITE-2222222"),
            references("CITE.pdf",
                       f"1) शासन निर्णय क्र. {NUMBER} दि. 02.02.2020",
                       gr_number="CITE-2222222"),
            header("OLD.pdf", "02.02.2020", gr_number=NUMBER),
            header("NEW.pdf", "03.03.2021", gr_number=NUMBER),
        ]

    def test_resolves_to_the_document_with_the_cited_date(self):
        _, edges = build_graph_from_docs(self.docs)
        linked = edges_from(edges, "CITE.pdf")

        self.assertEqual(len(linked), 1)
        self.assertEqual(linked[0]["target"], "OLD.pdf")
        self.assertEqual(linked[0]["match_reason"], "date_and_number")
        self.assertFalse(linked[0]["ambiguous"])

    def test_both_are_linked_and_flagged_when_dates_also_agree(self):
        docs = [
            header("CITE.pdf", "01.01.2026", gr_number="CITE-2222222"),
            references("CITE.pdf",
                       f"1) शासन निर्णय क्र. {NUMBER} दि. 02.02.2020",
                       gr_number="CITE-2222222"),
            header("T1.pdf", "02.02.2020", gr_number=NUMBER),
            header("T2.pdf", "02.02.2020", gr_number=NUMBER),
        ]
        _, edges = build_graph_from_docs(docs)
        linked = edges_from(edges, "CITE.pdf")

        self.assertEqual({e["target"] for e in linked}, {"T1.pdf", "T2.pdf"})
        self.assertTrue(all(e["ambiguous"] for e in linked))


class OcrNoisyNumberTests(unittest.TestCase):

    def test_date_and_number_match_survives_a_damaged_number(self):
        # The cited number is misrendered, so no strong key is shared and
        # an exact comparison fails. The date is the independent check that
        # lets a fuzzy number match be trusted.
        noisy = "मरग्र-2025/प्रक्र.5८"
        docs = [
            header("CITE.pdf", "01.01.2026", gr_number="CITE-3333333"),
            references("CITE.pdf",
                       f"1) शासन निर्णय क्र. {noisy} दि. 02.02.2020",
                       gr_number="CITE-3333333"),
            header("OLD.pdf", "02.02.2020", gr_number=NUMBER),
        ]
        _, edges = build_graph_from_docs(docs)
        linked = edges_from(edges, "CITE.pdf")

        self.assertEqual(len(linked), 1)
        self.assertEqual(linked[0]["target"], "OLD.pdf")
        self.assertEqual(linked[0]["match_reason"], "date_and_number")

    def test_a_different_date_blocks_a_fuzzy_number_match(self):
        noisy = "मरग्र-2025/प्रक्र.5८"
        docs = [
            header("CITE.pdf", "01.01.2026", gr_number="CITE-3333333"),
            references("CITE.pdf",
                       f"1) शासन निर्णय क्र. {noisy} दि. 09.09.1999",
                       gr_number="CITE-3333333"),
            header("OLD.pdf", "02.02.2020", gr_number=NUMBER),
        ]
        _, edges = build_graph_from_docs(docs)
        self.assertEqual(
            [e for e in edges_from(edges, "CITE.pdf") if e["match_reason"]],
            [],
        )


class ExternalNodeTests(unittest.TestCase):

    def setUp(self):
        self.docs = [
            header("A.pdf", "01.01.2026", gr_number="A-4444444"),
            references("A.pdf",
                       "1) केंद्र शासन पत्र क्र. XYZ-0909 दि. 09.09.1999",
                       gr_number="A-4444444"),
        ]
        self.nodes, self.edges = build_graph_from_docs(self.docs)
        self.external = [n for n in self.nodes if not n["resolved"]]

    def test_one_unresolved_node_is_created(self):
        self.assertEqual(len(self.external), 1)
        self.assertTrue(self.external[0]["id"].startswith("ext:"))
        self.assertFalse(self.external[0]["resolved"])

    def test_the_edge_points_at_it(self):
        linked = edges_from(self.edges, "A.pdf")
        self.assertEqual(len(linked), 1)
        self.assertEqual(linked[0]["target"], self.external[0]["id"])
        self.assertEqual(linked[0]["match_reason"], "")

    def test_label_is_the_item_text(self):
        self.assertIn("केंद्र शासन पत्र", self.external[0]["label"])
        self.assertLessEqual(len(self.external[0]["label"]), 80)

    def test_id_is_bounded(self):
        # "ext:" plus at most 40 characters of normalised text.
        self.assertLessEqual(len(self.external[0]["id"]), 4 + 40)

    def test_real_documents_stay_resolved(self):
        self.assertTrue([n for n in self.nodes if n["id"] == "A.pdf"][0]["resolved"])


class RelationTypingTests(unittest.TestCase):
    """The reference list says WHICH document is cited; only the body says
    what is being done to it."""

    def graph_with_body(self, body):
        docs = [
            header("NEW.pdf", "01.01.2026", gr_number="NEW-5555555"),
            references("NEW.pdf",
                       f"1) शासन निर्णय क्र. {STRONG_KEY} दि. 02.02.2020",
                       gr_number="NEW-5555555"),
            body_chunk("NEW.pdf", body, gr_number="NEW-5555555"),
            header("OLD.pdf", "02.02.2020", gr_number=STRONG_KEY),
        ]
        _, edges = build_graph_from_docs(docs)
        return edges_from(edges, "NEW.pdf")[0]

    def test_radda_root_is_supersedes(self):
        edge = self.graph_with_body(
            "संदर्भ क्रमांक 1 येथील शासन निर्णय रद्द करण्यात येत आहे."
        )
        self.assertEqual(edge["relation"], RELATION_SUPERSEDES)
        self.assertTrue(edge["evidence"])

    def test_adhikramit_root_is_supersedes(self):
        edge = self.graph_with_body(
            "संदर्भ क्र. 1 मधील आदेश अधिक्रमित करण्यात येत आहे."
        )
        self.assertEqual(edge["relation"], RELATION_SUPERSEDES)

    def test_sudhar_root_is_amends(self):
        edge = self.graph_with_body(
            "संदर्भ क्रमांक 1 मध्ये सुधारणा करण्यात येत आहे."
        )
        self.assertEqual(edge["relation"], RELATION_AMENDS)

    def test_shuddhipatrak_root_is_amends(self):
        edge = self.graph_with_body(
            "सं. क्र. 1 अन्वये शुद्धीपत्रक निर्गमित करण्यात येते."
        )
        self.assertEqual(edge["relation"], RELATION_AMENDS)

    def test_a_plain_mention_is_references(self):
        edge = self.graph_with_body(
            "संदर्भ क्रमांक 1 येथील तरतुदीनुसार कार्यवाही करावी."
        )
        self.assertEqual(edge["relation"], RELATION_REFERENCES)

    def test_no_mention_at_all_is_references_with_no_evidence(self):
        edge = self.graph_with_body("या आदेशाची अंमलबजावणी तातडीने करावी.")
        self.assertEqual(edge["relation"], RELATION_REFERENCES)
        self.assertEqual(edge["evidence"], "")

    def test_supersession_wins_over_amendment(self):
        # A document that both cancels and amends has cancelled; reporting
        # the amendment would leave the superseded GR looking live.
        edge = self.graph_with_body(
            "संदर्भ क्रमांक 1 मधील तरतुदीत सुधारणा करून तो रद्द करण्यात येत आहे."
        )
        self.assertEqual(edge["relation"], RELATION_SUPERSEDES)

    def test_evidence_is_capped_at_200_characters(self):
        filler = "या आदेशाची अंमलबजावणी तातडीने करण्यात यावी. " * 10
        edge = self.graph_with_body(
            filler + "संदर्भ क्रमांक 1 येथील निर्णय रद्द करण्यात येत आहे. " + filler
        )
        self.assertEqual(edge["relation"], RELATION_SUPERSEDES)
        self.assertLessEqual(len(edge["evidence"]), 200)


class EdgeShapeTests(unittest.TestCase):

    def test_edge_carries_exactly_the_expected_keys(self):
        docs = [
            header("A.pdf", "01.01.2026", gr_number="A-6666666"),
            references("A.pdf",
                       f"1) शासन निर्णय क्र. {STRONG_KEY} दि. 02.02.2020",
                       gr_number="A-6666666"),
            header("B.pdf", "02.02.2020", gr_number=STRONG_KEY),
        ]
        _, edges = build_graph_from_docs(docs)
        self.assertEqual(
            set(edges[0]),
            {"source", "target", "relation", "ref_index",
             "match_reason", "ambiguous", "evidence"},
        )

    def test_node_carries_the_expected_keys(self):
        # department and header_page were added for the graph page's
        # colouring and side panel. Every key that was here before is
        # still here — this assertion is exact on purpose, so a field
        # being dropped fails rather than going unnoticed.
        nodes, _ = build_graph_from_docs([header("A.pdf", "01.01.2026")])
        self.assertEqual(
            set(nodes[0]),
            {"id", "label", "filename", "gr_number",
             "doc_date", "subject", "resolved",
             "department", "header_page"},
        )


class ReferenceItemTests(unittest.TestCase):

    def test_numbered_items_keep_their_own_index(self):
        docs = [
            header("A.pdf", "01.01.2026", gr_number="A-7777777"),
            references(
                "A.pdf",
                "1) पहिला संदर्भ क्र. 1111111\n"
                "2. दुसरा संदर्भ क्र. 2222221\n"
                "(3) तिसरा संदर्भ क्र. 3333331",
                gr_number="A-7777777",
            ),
            header("P.pdf", "02.02.2020", gr_number="1111111"),
            header("Q.pdf", "03.03.2021", gr_number="2222221"),
            header("R.pdf", "04.04.2022", gr_number="3333331"),
        ]
        _, edges = build_graph_from_docs(docs)
        resolved = {e["target"]: e["ref_index"]
                    for e in edges_from(edges, "A.pdf") if e["match_reason"]}
        self.assertEqual(resolved, {"P.pdf": 1, "Q.pdf": 2, "R.pdf": 3})

    def test_an_unnumbered_block_is_one_item_at_index_one(self):
        docs = [
            header("A.pdf", "01.01.2026", gr_number="A-8888888"),
            references("A.pdf", f"शासन निर्णय क्र. {STRONG_KEY} दि. 02.02.2020",
                       gr_number="A-8888888"),
            header("B.pdf", "02.02.2020", gr_number=STRONG_KEY),
        ]
        _, edges = build_graph_from_docs(docs)
        linked = edges_from(edges, "A.pdf")
        self.assertEqual(len(linked), 1)
        self.assertEqual(linked[0]["ref_index"], 1)

    def test_vacha_marker_also_opens_a_reference_list(self):
        docs = [
            header("A.pdf", "01.01.2026", gr_number="A-9999991"),
            chunk("A.pdf", f"1) शासन निर्णय क्र. {STRONG_KEY} दि. 02.02.2020",
                  section_type="वाचा", section_index=1, gr_number="A-9999991"),
            header("B.pdf", "02.02.2020", gr_number=STRONG_KEY),
        ]
        _, edges = build_graph_from_docs(docs)
        self.assertEqual(
            [e["target"] for e in edges_from(edges, "A.pdf") if e["match_reason"]],
            ["B.pdf"],
        )

    def test_an_ocr_damaged_marker_still_opens_one(self):
        docs = [
            header("A.pdf", "01.01.2026", gr_number="A-9999992"),
            chunk("A.pdf", f"1) शासन निर्णय क्र. {STRONG_KEY} दि. 02.02.2020",
                  section_type="सदर्भ", section_index=1, gr_number="A-9999992"),
            header("B.pdf", "02.02.2020", gr_number=STRONG_KEY),
        ]
        _, edges = build_graph_from_docs(docs)
        self.assertEqual(
            [e["target"] for e in edges_from(edges, "A.pdf") if e["match_reason"]],
            ["B.pdf"],
        )


class ExternalItemFilterTests(unittest.TestCase):
    """An external node is a claim that a document cites something the
    corpus does not hold. These are the four tests an unresolved item has
    to survive before that claim is made — the live corpus otherwise turns
    numbered body paragraphs and a digital-signature blob into
    "referenced documents"."""

    def graph_with_item(self, item_text, ref_index=1):
        """
        One citing document whose single reference item is item_text, with
        nothing in the corpus for it to resolve to.

        The citing file's own number and date are deliberately unrelated to
        the item, so the item reaches the external path rather than being
        recognised as the letterhead describing itself.
        """
        body = f"{ref_index}) {item_text}"
        docs = [
            header("A.pdf", "01.01.2026", gr_number="A-1234561"),
            references("A.pdf", body, gr_number="A-1234561"),
        ]
        stats = {}
        nodes, edges = build_graph_from_docs(docs, stats=stats)
        external = [n for n in nodes if not n["resolved"]]
        return external, edges_from(edges, "A.pdf"), stats

    # ── kept ────────────────────────────────────────────────

    def test_a_real_looking_citation_is_kept(self):
        external, edges, stats = self.graph_with_item(
            "केंद्र शासन पत्र क्र. XYZ-0909 दि. 09.09.1999"
        )
        self.assertEqual(len(external), 1)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["target"], external[0]["id"])
        self.assertEqual(stats["external_dropped"], 0)

    def test_a_date_alone_is_enough(self):
        external, _, _ = self.graph_with_item(
            "दि. 09.09.1999 रोजीच्या मान्यतेनुसार पुढील कार्यवाही"
        )
        self.assertEqual(len(external), 1)

    def test_a_document_type_word_alone_is_enough(self):
        external, _, _ = self.graph_with_item(
            "महाराष्ट्र सार्वजनिक ग्रंथालये अधिनियम मधील तरतूद"
        )
        self.assertEqual(len(external), 1)

    # ── (a) length ──────────────────────────────────────────

    def test_too_short_is_dropped(self):
        external, edges, stats = self.graph_with_item("शासन आदेश")
        self.assertEqual(external, [])
        self.assertEqual(edges, [])
        self.assertEqual(stats["external_dropped"], 1)

    def test_short_is_dropped_even_carrying_a_date(self):
        # A bare date is not a citation: it identifies no document.
        external, _, stats = self.graph_with_item("दि.09.09.1999")
        self.assertEqual(external, [])
        self.assertEqual(stats["external_dropped"], 1)

    def test_exactly_twenty_characters_is_kept(self):
        item = "शासन आदेश दि.09.1999"
        self.assertEqual(len(" ".join(item.split())), 20)
        external, _, stats = self.graph_with_item(item)
        self.assertEqual(len(external), 1)
        self.assertEqual(stats["external_dropped"], 0)

    # ── (b) date or document-type word ──────────────────────

    def test_prose_with_neither_date_nor_type_word_is_dropped(self):
        external, edges, stats = self.graph_with_item(
            "उपरोक्त बाबींचा विचार करून पुढील कार्यवाही करावी"
        )
        self.assertEqual(external, [])
        self.assertEqual(edges, [])
        self.assertEqual(stats["external_dropped"], 1)

    def test_each_document_type_root_qualifies(self):
        roots = ["शासन", "परिपत्रक", "आदेश", "शुद्धीपत्रक",
                 "पत्र", "अधिनियम", "नियम"]
        for root in roots:
            with self.subTest(root=root):
                # Padded past the length floor, and carrying no date, so
                # the root is the only thing that can keep it.
                external, _, _ = self.graph_with_item(
                    f"{root} क्रमांक ABC मधील तरतुदीनुसार"
                )
                self.assertEqual(len(external), 1)

    # ── (c) opaque token ────────────────────────────────────

    def test_an_item_that_is_mostly_a_hash_is_dropped(self):
        # 40 hex characters inside a 50-character item: one run, more than
        # half. The type word and the length would otherwise keep it, so
        # this isolates rule (c).
        item = "शासन आदेश 5420AC65E55BC2F86EC37C6F25108B6605BC20E7"
        external, edges, stats = self.graph_with_item(item)
        self.assertEqual(external, [])
        self.assertEqual(edges, [])
        self.assertEqual(stats["external_dropped"], 1)

    def test_a_long_run_that_is_under_half_is_kept(self):
        # Same 40-character run, but the item around it is long enough that
        # the run no longer dominates — so this is a real citation that
        # happens to quote a long identifier.
        item = ("शासन निर्णय क्रमांक 5420AC65E55BC2F86EC37C6F25108B6605BC20E7 "
                "अन्वये सामान्य प्रशासन विभागाने कळविले आहे की सदर प्रकरणी")
        external, _, stats = self.graph_with_item(item)
        self.assertEqual(len(external), 1)
        self.assertEqual(stats["external_dropped"], 0)

    def test_an_ordinary_reference_number_is_not_opaque(self):
        # Reference numbers are broken up by punctuation, so no single run
        # of letters-and-digits gets anywhere near 16 characters.
        external, _, stats = self.graph_with_item(
            "शासन निर्णय क्र. मराग्रं-2025/प्र.क्र.58 दि. 02.02.2020"
        )
        self.assertEqual(len(external), 1)
        self.assertEqual(stats["external_dropped"], 0)

    # ── (d) item index ──────────────────────────────────────

    def test_index_above_fifteen_is_dropped(self):
        external, edges, stats = self.graph_with_item(
            "केंद्र शासन पत्र क्र. XYZ-0909 दि. 09.09.1999", ref_index=16
        )
        self.assertEqual(external, [])
        self.assertEqual(edges, [])
        self.assertEqual(stats["external_dropped"], 1)

    def test_index_fifteen_is_kept(self):
        external, _, stats = self.graph_with_item(
            "केंद्र शासन पत्र क्र. XYZ-0909 दि. 09.09.1999", ref_index=15
        )
        self.assertEqual(len(external), 1)
        self.assertEqual(stats["external_dropped"], 0)

    # ── the filter never touches a resolved item ────────────

    def test_a_resolved_item_is_kept_however_it_looks(self):
        # Index 40, three characters of prose, no date, no type word: every
        # rule would reject it as an external claim. It resolves to a real
        # document, so none of them apply.
        docs = [
            header("A.pdf", "01.01.2026", gr_number="A-1234562"),
            references("A.pdf", f"40) {STRONG_KEY}", gr_number="A-1234562"),
            header("B.pdf", "02.02.2020", gr_number=STRONG_KEY),
        ]
        stats = {}
        _, edges = build_graph_from_docs(docs, stats=stats)
        linked = [e for e in edges_from(edges, "A.pdf") if e["match_reason"]]

        self.assertEqual(len(linked), 1)
        self.assertEqual(linked[0]["target"], "B.pdf")
        self.assertEqual(linked[0]["ref_index"], 40)
        self.assertEqual(stats["external_dropped"], 0)


class BuildStatsTests(unittest.TestCase):
    """The counters ride on an out-parameter so that adding them did not
    change the (nodes, edges) contract every other caller depends on."""

    def test_stats_are_optional(self):
        nodes, edges = build_graph_from_docs([header("A.pdf", "01.01.2026")])
        self.assertEqual(len(nodes), 1)
        self.assertEqual(edges, [])

    def test_counters_are_filled_in(self):
        docs = [
            header("A.pdf", "01.01.2026", gr_number="A-1234563"),
            references(
                "A.pdf",
                # kept, dropped (too short), dropped (prose only)
                "1) केंद्र शासन पत्र क्र. XYZ-0909 दि. 09.09.1999\n"
                "2) शासन आदेश\n"
                "3) उपरोक्त बाबींचा विचार करून कार्यवाही करावी",
                gr_number="A-1234563",
            ),
        ]
        stats = {}
        nodes, _ = build_graph_from_docs(docs, stats=stats)

        self.assertEqual(stats["resolved_nodes"], 1)
        self.assertEqual(stats["external_nodes"], 1)
        self.assertEqual(stats["external_dropped"], 2)
        self.assertEqual(len(nodes), 2)

    def test_counters_on_an_empty_corpus(self):
        stats = {}
        build_graph_from_docs([], stats=stats)
        self.assertEqual(stats["external_dropped"], 0)
        self.assertEqual(stats["resolved_nodes"], 0)
        self.assertEqual(stats["external_nodes"], 0)


class UniqueDateFuzzyNumberTests(unittest.TestCase):
    """The third resolution path. OCR can destroy a cited number past the
    point where any similarity score should be trusted on its own — but a
    date that belongs to exactly one document in the corpus identifies it
    without help, and then the number only has to not contradict."""

    # Damaged past the date_and_number threshold of 0.6, still above the
    # 0.4 this path accepts.
    NOISY = "मग्र-2025/प्रक्र"

    def graph(self, citing_date, targets, item_number=None):
        """
        One citing document whose single reference names NOISY and a date,
        plus one header per entry in targets.

        Args:
            citing_date : the citing file's own दिनांक
            targets     : [(source_file, date, gr_number), ...]
        """
        number = item_number if item_number is not None else self.NOISY
        docs = [
            header("CITE.pdf", citing_date, gr_number="CITE-1111111"),
            references("CITE.pdf",
                       f"1) शासन निर्णय क्र. {number} दि. 02.02.2020",
                       gr_number="CITE-1111111"),
        ]
        for fname, date_text, gr_number in targets:
            docs.append(header(fname, date_text, gr_number=gr_number))

        _, edges = build_graph_from_docs(docs)
        return edges_from(edges, "CITE.pdf")

    def test_a_unique_date_resolves(self):
        linked = self.graph("01.01.2026", [("OLD.pdf", "02.02.2020", NUMBER)])
        resolved = [e for e in linked if e["match_reason"]]

        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["target"], "OLD.pdf")
        self.assertEqual(resolved[0]["match_reason"], "date_unique_fuzzy_number")
        self.assertFalse(resolved[0]["ambiguous"])

    def test_two_candidates_sharing_the_date_do_not_link(self):
        # Uniqueness is the whole justification for the loose threshold.
        # Without it this would be a coin flip between two documents.
        linked = self.graph("01.01.2026", [
            ("T1.pdf", "02.02.2020", NUMBER),
            ("T2.pdf", "02.02.2020", NUMBER),
        ])
        self.assertEqual([e for e in linked if e["match_reason"]], [])

    def test_the_citing_files_own_date_does_not_link(self):
        # The citing file is the only document on that date, so without
        # the self-exclusion it would be the unique match for its own
        # reference.
        linked = self.graph("02.02.2020", [])
        self.assertEqual([e for e in linked if e["match_reason"]], [])

    def test_a_number_below_the_floor_does_not_link(self):
        # The date is unique, but the number agrees with nothing, so there
        # is no confirmation to accept.
        linked = self.graph("01.01.2026",
                            [("OLD.pdf", "02.02.2020", NUMBER)],
                            item_number="ZZZZZZZZ-0000/QQQQ")
        self.assertEqual([e for e in linked if e["match_reason"]], [])

    def test_strong_key_keeps_priority(self):
        docs = [
            header("CITE.pdf", "01.01.2026", gr_number="CITE-1111111"),
            references("CITE.pdf",
                       f"1) शासन निर्णय क्र. {STRONG_KEY} {self.NOISY} दि. 02.02.2020",
                       gr_number="CITE-1111111"),
            # KEY.pdf shares the strong key; DATE.pdf would win the
            # unique-date path if it ran.
            header("KEY.pdf", "09.09.1999", gr_number=STRONG_KEY),
            header("DATE.pdf", "02.02.2020", gr_number=NUMBER),
        ]
        _, edges = build_graph_from_docs(docs)
        resolved = [e for e in edges_from(edges, "CITE.pdf") if e["match_reason"]]

        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["target"], "KEY.pdf")
        self.assertEqual(resolved[0]["match_reason"], "strong_key")

    def test_date_and_number_keeps_priority(self):
        # An undamaged number clears the stricter threshold, so the stricter
        # reason is the one reported.
        linked = self.graph("01.01.2026",
                            [("OLD.pdf", "02.02.2020", NUMBER)],
                            item_number=NUMBER)
        resolved = [e for e in linked if e["match_reason"]]

        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["match_reason"], "date_and_number")


class WebAndOpaqueFilterTests(unittest.TestCase):
    """Two more reasons an unresolved item is not a citation."""

    def graph_with_item(self, item_text, ref_index=1):
        docs = [
            header("A.pdf", "01.01.2026", gr_number="A-1234561"),
            references("A.pdf", f"{ref_index}) {item_text}",
                       gr_number="A-1234561"),
        ]
        stats = {}
        nodes, edges = build_graph_from_docs(docs, stats=stats)
        external = [n for n in nodes if not n["resolved"]]
        return external, edges_from(edges, "A.pdf"), stats

    def test_portal_boilerplate_is_dropped(self):
        # Every GR closes with this sentence. It is the same text in every
        # file and cites nothing.
        external, edges, stats = self.graph_with_item(
            "सदर शासन आदेश महाराष्ट्र शासनाच्या www.maharashtra.gov.in "
            "या संकेतस्थळावर उपलब्ध करण्यात आला आहे"
        )
        self.assertEqual(external, [])
        self.assertEqual(edges, [])
        self.assertEqual(stats["external_dropped"], 1)

    def test_the_gov_domain_alone_is_enough_to_drop(self):
        external, _, stats = self.graph_with_item(
            "शासन आदेश maharashtra.gov वर उपलब्ध आहे या ठिकाणी पहावे"
        )
        self.assertEqual(external, [])
        self.assertEqual(stats["external_dropped"], 1)

    def test_an_ocr_split_signature_blob_is_dropped(self):
        # The longest-run test misses this: OCR broke the blob into several
        # medium runs, so no single one exceeds half. Their combined share
        # does, which is what the second opaque test measures.
        external, edges, stats = self.graph_with_item(
            "5.4.20=AC65E55BC2F86EC37C6F 25 108B6605BC20E7 1B4742DD2 "
            "OD7E8F5E14CDA6A2A7F serialNumb B3C122EDC26AC3532130D69CD06C9664 शासन"
        )
        self.assertEqual(external, [])
        self.assertEqual(edges, [])
        self.assertEqual(stats["external_dropped"], 1)

    def test_a_genuine_citation_survives_both(self):
        external, _, stats = self.graph_with_item(
            "केंद्र शासन पत्र क्र. XYZ-0909 दि. 09.09.1999"
        )
        self.assertEqual(len(external), 1)
        self.assertEqual(stats["external_dropped"], 0)

    def test_a_long_reference_number_is_not_opaque_in_total(self):
        # Reference numbers are punctuated, so their letter-digit runs stay
        # short and their combined share stays under half.
        external, _, stats = self.graph_with_item(
            "शासन निर्णय क्र. मराग्रं-2025/प्र.क्र.58/ई-1093601 दि. 02.02.2020 "
            "अन्वये सामान्य प्रशासन विभागाने कळविले आहे"
        )
        self.assertEqual(len(external), 1)
        self.assertEqual(stats["external_dropped"], 0)


class NodeMetadataFieldTests(unittest.TestCase):
    """department and header_page ride on real-file nodes so the graph page
    can colour by department without a second request."""

    def test_real_nodes_carry_the_added_fields(self):
        docs = [
            chunk("A.pdf",
                  "महाराष्ट्र शासन\nवित्त विभाग\nशासन निर्णय क्रमांक A-1\nदिनांक : 01.01.2026",
                  section_type="__header__", page=2),
        ]
        nodes, _ = build_graph_from_docs(docs)
        node = nodes[0]

        self.assertEqual(node["department"], "वित्त विभाग")
        self.assertEqual(node["header_page"], 2)

    def test_every_existing_node_key_is_still_present(self):
        nodes, _ = build_graph_from_docs([header("A.pdf", "01.01.2026")])
        self.assertEqual(
            set(nodes[0]),
            {"id", "label", "filename", "gr_number", "doc_date", "subject",
             "resolved", "department", "header_page"},
        )

    def test_a_letterhead_without_a_department_gives_none(self):
        docs = [chunk("A.pdf", "महाराष्ट्र शासन\nशासन निर्णय क्रमांक A-1\nदिनांक : 01.01.2026",
                      section_type="__header__")]
        nodes, _ = build_graph_from_docs(docs)
        self.assertIsNone(nodes[0]["department"])

    def test_external_nodes_are_unchanged(self):
        docs = [
            header("A.pdf", "01.01.2026", gr_number="A-1234561"),
            references("A.pdf", "1) केंद्र शासन पत्र क्र. XYZ-0909 दि. 09.09.1999",
                       gr_number="A-1234561"),
        ]
        nodes, _ = build_graph_from_docs(docs)
        external = [n for n in nodes if not n["resolved"]][0]
        self.assertEqual(
            set(external),
            {"id", "label", "filename", "gr_number", "doc_date", "subject", "resolved"},
        )


class GraphSeedTests(unittest.TestCase):
    """
    Which retrieved files are allowed to contribute graph neighbours.

    Following edges from every retrieved file meant a document sitting at
    rank nine — there because it shares a word with the question, not
    because the question is about it — pulled its own neighbours into the
    context, where they looked exactly as authoritative as a superseding
    order. Seeds are capped at the top of the ranking instead.

    Synthetic snapshot and synthetic corpus: no index, no network, no
    model. The chunk order IS the rank order, because retrieval returns
    best first.
    """

    def setUp(self):
        # F1 .. F4 retrieved in that order; N1 is linked to F1, N3 to F3.
        self.corpus = {
            name: [
                Document(page_content=f"{name} header",
                         metadata={"source_file": name, "page": 0,
                                   "section_index": 0, "section_type": "__header__"}),
                Document(page_content=f"{name} body",
                         metadata={"source_file": name, "page": 1,
                                   "section_index": 1}),
            ]
            for name in ("F1.pdf", "F2.pdf", "F3.pdf", "F4.pdf",
                         "N1.pdf", "N3.pdf")
        }

        self.snapshot = {
            "nodes": [{"id": n, "resolved": True} for n in self.corpus],
            "edges": [
                {"source": "N1.pdf", "target": "F1.pdf", "relation": "supersedes"},
                {"source": "N3.pdf", "target": "F3.pdf", "relation": "supersedes"},
            ],
        }

        flat = [d for docs in self.corpus.values() for d in docs]

        class FakeIndex:
            documents = flat

        self._saved = (rag.load_graph_snapshot, rag.get_lexical_index,
                       rag.lexical_search)
        rag.load_graph_snapshot = lambda: self.snapshot
        rag.get_lexical_index = lambda *a, **k: FakeIndex()
        # Nothing question-relevant, so _neighbor_chunks falls back to each
        # file's own opening chunks — which keeps this test about seeding.
        rag.lexical_search = lambda q, top_k=None: []

        self._settings = {
            name: getattr(settings, name)
            for name in ("GRAPH_SEED_DOCS", "GRAPH_MAX_NEIGHBOR_DOCS",
                         "GRAPH_CHUNKS_PER_NEIGHBOR")
        }
        settings.GRAPH_MAX_NEIGHBOR_DOCS = 2
        settings.GRAPH_CHUNKS_PER_NEIGHBOR = 1

    def tearDown(self):
        (rag.load_graph_snapshot, rag.get_lexical_index,
         rag.lexical_search) = self._saved
        for name, value in self._settings.items():
            setattr(settings, name, value)

    def retrieved(self, *names):
        """One chunk per named file, in the order given — i.e. rank order."""
        return [self.corpus[name][1] for name in names]

    def added_files(self, chunks):
        """The distinct files graph expansion appended."""
        out = rag.expand_with_graph(chunks, "question")
        return [
            d.metadata.get("source_file") for d in out
            if d.metadata.get("graph_expanded")
        ]

    # ── seeding ─────────────────────────────────────────────

    def test_a_neighbour_of_the_first_file_is_added(self):
        settings.GRAPH_SEED_DOCS = 2
        self.assertEqual(
            self.added_files(self.retrieved("F1.pdf", "F2.pdf", "F3.pdf")),
            ["N1.pdf"],
        )

    def test_a_neighbour_of_the_third_file_is_not_added(self):
        # F3 is retrieved, but outside the seed window, so N3 stays out.
        settings.GRAPH_SEED_DOCS = 2
        added = self.added_files(self.retrieved("F1.pdf", "F2.pdf", "F3.pdf"))
        self.assertNotIn("N3.pdf", added)

    def test_widening_the_window_to_three_admits_it(self):
        # Same corpus, same snapshot, same question — only the setting
        # changed, which is what makes this a test of the setting.
        settings.GRAPH_SEED_DOCS = 3
        added = self.added_files(self.retrieved("F1.pdf", "F2.pdf", "F3.pdf"))
        self.assertIn("N3.pdf", added)
        self.assertIn("N1.pdf", added)

    def test_zero_returns_the_input_unchanged(self):
        settings.GRAPH_SEED_DOCS = 0
        chunks = self.retrieved("F1.pdf", "F2.pdf", "F3.pdf")
        out = rag.expand_with_graph(chunks, "question")

        self.assertEqual(out, chunks)
        self.assertEqual([d.metadata.get("graph_expanded") for d in out],
                         [None, None, None])

    def test_a_negative_value_also_returns_the_input_unchanged(self):
        settings.GRAPH_SEED_DOCS = -1
        chunks = self.retrieved("F1.pdf", "F2.pdf")
        self.assertEqual(rag.expand_with_graph(chunks, "question"), chunks)

    def test_one_seed_follows_only_the_top_file(self):
        settings.GRAPH_SEED_DOCS = 1
        added = self.added_files(self.retrieved("F1.pdf", "F3.pdf"))
        self.assertEqual(added, ["N1.pdf"])

    # ── order of first appearance, not alphabetical ─────────

    def test_first_appearance_decides_the_seeds_not_the_alphabet(self):
        # F3 first, F1 third. Alphabetically F1 would be the first seed;
        # by rank it is outside the window, so N1 must NOT appear and N3
        # must.
        settings.GRAPH_SEED_DOCS = 2
        added = self.added_files(self.retrieved("F3.pdf", "F2.pdf", "F1.pdf"))

        self.assertIn("N3.pdf", added)
        self.assertNotIn("N1.pdf", added)

    def test_repeated_chunks_of_one_file_do_not_consume_two_seed_slots(self):
        # Seeds are DISTINCT files. Two chunks of F1 then one of F3 must
        # still leave F3 inside a window of two.
        settings.GRAPH_SEED_DOCS = 2
        chunks = [self.corpus["F1.pdf"][0], self.corpus["F1.pdf"][1],
                  self.corpus["F3.pdf"][1]]
        added = self.added_files(chunks)

        self.assertIn("N1.pdf", added)
        self.assertIn("N3.pdf", added)

    # ── everything else is unchanged ────────────────────────

    def test_a_file_already_retrieved_is_never_re_added(self):
        # N1 is linked to F1 AND already present at rank three. Narrowing
        # the seeds must not let it be appended as its own neighbour.
        settings.GRAPH_SEED_DOCS = 2
        added = self.added_files(
            self.retrieved("F1.pdf", "F2.pdf", "N1.pdf"))
        self.assertEqual(added, [])

    def test_external_nodes_are_still_excluded(self):
        self.snapshot["nodes"].append({"id": "ext:z", "resolved": False})
        self.snapshot["edges"].append(
            {"source": "F1.pdf", "target": "ext:z", "relation": "supersedes"})
        settings.GRAPH_SEED_DOCS = 2
        added = self.added_files(self.retrieved("F1.pdf", "F2.pdf"))

        self.assertNotIn("ext:z", added)
        self.assertEqual(added, ["N1.pdf"])

    def test_the_neighbour_cap_still_applies(self):
        # Two neighbours of the top file, but only one slot.
        self.snapshot["edges"].append(
            {"source": "N3.pdf", "target": "F1.pdf", "relation": "amends"})
        settings.GRAPH_SEED_DOCS = 2
        settings.GRAPH_MAX_NEIGHBOR_DOCS = 1
        added = self.added_files(self.retrieved("F1.pdf", "F2.pdf"))

        self.assertEqual(len(set(added)), 1)

    def test_the_chunks_per_neighbour_cap_still_applies(self):
        settings.GRAPH_SEED_DOCS = 2
        settings.GRAPH_CHUNKS_PER_NEIGHBOR = 2
        added = self.added_files(self.retrieved("F1.pdf", "F2.pdf"))

        self.assertEqual(added, ["N1.pdf", "N1.pdf"])

    def test_the_tagging_is_unchanged(self):
        settings.GRAPH_SEED_DOCS = 2
        out = rag.expand_with_graph(self.retrieved("F1.pdf", "F2.pdf"), "question")
        appended = [d for d in out if d.metadata.get("graph_expanded")]

        self.assertEqual(len(appended), 1)
        self.assertTrue(appended[0].metadata["graph_expanded"])
        self.assertEqual(appended[0].metadata["graph_relation"], "supersedes")

    def test_the_docstore_documents_are_not_mutated(self):
        settings.GRAPH_SEED_DOCS = 2
        rag.expand_with_graph(self.retrieved("F1.pdf", "F2.pdf"), "question")

        for docs in self.corpus.values():
            for doc in docs:
                self.assertNotIn("graph_expanded", doc.metadata)

    def test_no_snapshot_returns_the_input_unchanged(self):
        rag.load_graph_snapshot = lambda: None
        settings.GRAPH_SEED_DOCS = 2
        chunks = self.retrieved("F1.pdf", "F2.pdf")
        self.assertEqual(rag.expand_with_graph(chunks, "question"), chunks)

    def test_empty_input_returns_empty(self):
        settings.GRAPH_SEED_DOCS = 2
        self.assertEqual(rag.expand_with_graph([], "question"), [])


class DegenerateInputTests(unittest.TestCase):

    def test_empty_input(self):
        self.assertEqual(build_graph_from_docs([]), ([], []))

    def test_chunks_without_a_source_file_are_ignored(self):
        docs = [Document(page_content="orphan", metadata={"page": 0})]
        self.assertEqual(build_graph_from_docs(docs), ([], []))

    def test_missing_section_index_does_not_raise(self):
        docs = [
            Document(page_content="दिनांक : 01.01.2026", metadata={
                "source_file": "A.pdf", "page": 0,
                "section_type": "__header__", "doc_identity": "A | NUM | s"}),
            Document(page_content="body", metadata={
                "source_file": "A.pdf", "page": None,
                "doc_identity": "A | NUM | s"}),
        ]
        nodes, _ = build_graph_from_docs(docs)
        self.assertEqual(nodes[0]["doc_date"], "2026-01-01")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    unittest.main(verbosity=2)
