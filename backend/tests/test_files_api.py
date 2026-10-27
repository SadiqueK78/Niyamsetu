#!/usr/bin/env python
# backend/tests/test_files_api.py
# ─────────────────────────────────────────────────────────
# Unit tests for GET /api/files/{filename}.
#
# This is the only route that turns user input into a filesystem path, so
# most of this file is about what it must REFUSE. A temporary grdocs
# directory holds one small valid PDF, and a decoy sits outside it that no
# traversal attempt may reach.
#
# No network, no model, no MongoDB. The auth dependency is overridden
# rather than satisfied with a real token: what is being tested is the
# path handling, and a test that needed a login would be testing the login.
# One case deliberately does NOT override it, to pin that the route is
# protected at all.
#
# Run:
#   python -m pytest tests/test_files_api.py -q
#   python tests/test_files_api.py
# ─────────────────────────────────────────────────────────

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient    # noqa: E402

import api.files as files_api                # noqa: E402
from api.files import router as files_router  # noqa: E402
from auth.router import get_current_user     # noqa: E402
from config import settings                  # noqa: E402


# The smallest thing that is genuinely a PDF: a real header, one object,
# and a trailer. Enough that a reader would accept it, small enough to
# write inline rather than shipping a fixture file.
MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
    b"trailer<</Root 1 0 R>>\n"
    b"%%EOF\n"
)

DOCUMENT = "GR1.pdf"

# Written OUTSIDE grdocs. Every traversal attempt below aims at this file,
# so a passing test means the route could not reach something that really
# is there — not merely that it 404'd on a name with no file behind it.
DECOY = "secret.pdf"


def _build_app(protected: bool):
    """
    A FastAPI app carrying only the files router.

    Args:
        protected : when False, the auth dependency is overridden with a
                    stub user, which is how the path-handling tests get
                    past the door. When True it is left alone, so the
                    request meets the real dependency.
    """
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(files_router)
    if not protected:
        app.dependency_overrides[get_current_user] = lambda: {"username": "tester"}
    return app


class FilesApiTestCase(unittest.TestCase):
    """Points GRDOCS_PATH at a temporary directory and restores it."""

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        root = Path(self._temp.name)

        self.grdocs = root / "grdocs"
        self.grdocs.mkdir()
        (self.grdocs / DOCUMENT).write_bytes(MINIMAL_PDF)

        # The target of every traversal attempt, one level above grdocs.
        (root / DECOY).write_bytes(MINIMAL_PDF)
        self.outside = root / DECOY

        # settings.GRDOCS_PATH is a property derived from DATA_DIR, so the
        # directory is redirected by moving DATA_DIR rather than by
        # assigning to the property.
        self._saved_data_dir = settings.DATA_DIR
        settings.DATA_DIR = str(root)

        self.client = TestClient(_build_app(protected=False))

    def tearDown(self):
        settings.DATA_DIR = self._saved_data_dir
        self._temp.cleanup()

    def assertRejected(self, response):
        """Every refusal is the same 404 carrying no filesystem path."""
        self.assertEqual(response.status_code, 404)
        body = response.text
        self.assertNotIn(self._temp.name, body)
        self.assertNotIn("grdocs", body)


class ServingTests(FilesApiTestCase):

    def test_an_existing_document_is_served_as_pdf(self):
        res = self.client.get(f"/api/files/{DOCUMENT}")

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["content-type"], "application/pdf")
        self.assertEqual(res.content, MINIMAL_PDF)

    def test_it_is_served_inline_so_the_viewer_can_embed_it(self):
        res = self.client.get(f"/api/files/{DOCUMENT}")
        self.assertIn("inline", res.headers.get("content-disposition", ""))

    def test_the_extension_check_is_case_insensitive(self):
        (self.grdocs / "Mixed.PDF").write_bytes(MINIMAL_PDF)
        res = self.client.get("/api/files/Mixed.PDF")
        self.assertEqual(res.status_code, 200)

    def test_a_filename_with_spaces_is_served(self):
        # Real names in this corpus look like "aadesh 1.pdf".
        (self.grdocs / "aadesh 1.pdf").write_bytes(MINIMAL_PDF)
        res = self.client.get("/api/files/aadesh 1.pdf")
        self.assertEqual(res.status_code, 200)

    def test_a_missing_document_is_a_plain_404(self):
        self.assertRejected(self.client.get("/api/files/nope.pdf"))


class RejectionTests(FilesApiTestCase):
    """Each case names a real file outside grdocs, so success would be a
    genuine escape rather than a lucky miss."""

    def test_the_decoy_really_exists(self):
        # Without this the traversal tests below could pass simply because
        # nothing was there to find.
        self.assertTrue(self.outside.is_file())

    def test_a_parent_reference_is_rejected(self):
        self.assertRejected(self.client.get(f"/api/files/../{DECOY}"))

    def test_a_url_encoded_parent_reference_is_rejected(self):
        self.assertRejected(self.client.get(f"/api/files/..%2F{DECOY}"))

    def test_a_double_encoded_parent_reference_is_rejected(self):
        self.assertRejected(self.client.get(f"/api/files/..%252F{DECOY}"))

    def test_a_forward_slash_is_rejected(self):
        self.assertRejected(self.client.get("/api/files/a/b.pdf"))

    def test_a_backslash_is_rejected(self):
        self.assertRejected(self.client.get("/api/files/a%5Cb.pdf"))

    def test_an_absolute_path_is_rejected(self):
        self.assertRejected(self.client.get(f"/api/files/{self.outside}"))

    def test_a_null_byte_is_rejected(self):
        self.assertRejected(self.client.get("/api/files/GR1.pdf%00.txt"))

    def test_a_name_without_the_pdf_extension_is_rejected(self):
        (self.grdocs / "notes.txt").write_bytes(b"plain text")
        self.assertRejected(self.client.get("/api/files/notes.txt"))

    def test_an_extensionless_name_is_rejected(self):
        (self.grdocs / "GR1").write_bytes(MINIMAL_PDF)
        self.assertRejected(self.client.get("/api/files/GR1"))

    def test_a_directory_is_rejected(self):
        (self.grdocs / "folder.pdf").mkdir()
        self.assertRejected(self.client.get("/api/files/folder.pdf"))

    def test_the_route_does_not_list_the_directory(self):
        # No collection endpoint on this router: a caller must already
        # know the name, which it gets from a citation.
        for path in ("/api/files/", "/api/files"):
            with self.subTest(path=path):
                self.assertNotEqual(self.client.get(path).status_code, 200)


class PathHelperTests(FilesApiTestCase):
    """The containment rule, checked directly — a route test cannot
    express the sibling-prefix case, because the URL never reaches it."""

    def test_a_valid_name_resolves_inside_grdocs(self):
        path = files_api._safe_grdocs_path(DOCUMENT)
        self.assertIsNotNone(path)
        self.assertEqual(path.name, DOCUMENT)
        self.assertTrue(path.is_relative_to(self.grdocs.resolve()))

    def test_a_sibling_directory_sharing_a_prefix_is_not_inside(self):
        # "grdocs_old" starts with the same text as "grdocs". A containment
        # test written as a string prefix would wrongly accept this.
        sibling = Path(self._temp.name) / "grdocs_old"
        sibling.mkdir()
        (sibling / DOCUMENT).write_bytes(MINIMAL_PDF)

        self.assertIsNone(files_api._safe_grdocs_path(f"../grdocs_old/{DOCUMENT}"))

    def test_every_forbidden_sequence_is_refused(self):
        for name in (f"../{DECOY}", f"..\\{DECOY}", "a/b.pdf",
                     "a\\b.pdf", "GR1.pdf\x00.txt"):
            with self.subTest(name=name):
                self.assertIsNone(files_api._safe_grdocs_path(name))

    def test_empty_and_none(self):
        self.assertIsNone(files_api._safe_grdocs_path(""))
        self.assertIsNone(files_api._safe_grdocs_path(None))


class AuthTests(FilesApiTestCase):

    def test_an_unauthenticated_request_is_refused(self):
        # The real dependency, not the override — and compared against
        # another protected route so this asserts "same as the rest of the
        # API" rather than a status this test invented.
        from api.graph import router as graph_router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(files_router)
        app.include_router(graph_router)
        client = TestClient(app)

        files_status = client.get(f"/api/files/{DOCUMENT}").status_code
        graph_status = client.get("/api/graph/").status_code

        self.assertEqual(files_status, graph_status)
        self.assertNotEqual(files_status, 200)

    def test_an_authenticated_request_succeeds_on_the_same_app(self):
        # The pair above and below together show the refusal came from
        # auth and not from the path handling.
        self.assertEqual(
            self.client.get(f"/api/files/{DOCUMENT}").status_code, 200)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    unittest.main(verbosity=2)
