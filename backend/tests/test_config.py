# backend/tests/test_config.py
# ─────────────────────────────────────────────────────────
# Unit tests for config.py — verifies defaults and computed
# paths work correctly, without needing a real .env file.
# Required fields (MONGODB_URL, JWT_SECRET) are faked via
# os.environ so this stays safe for CI's blank runner.
# ─────────────────────────────────────────────────────────

import sys
import os
import unittest
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSettingsDefaults(unittest.TestCase):

    def setUp(self):
        # Provide the two required fields that have no default,
        # so Settings() doesn't throw a validation error in CI.
        os.environ["MONGODB_URL"] = "mongodb://localhost:27017"
        os.environ["JWT_SECRET"]  = "test-secret-for-ci"

        # Import here (not at module level) so each test gets
        # a fresh Settings instance built from current os.environ
        from config import Settings
        self.Settings = Settings

    def tearDown(self):
        os.environ.pop("MONGODB_URL", None)
        os.environ.pop("JWT_SECRET", None)

    def test_database_name_default(self):
        s = self.Settings(_env_file=None)
        self.assertEqual(s.DATABASE_NAME, "niyamsetu")

    def test_jwt_expire_hours_default(self):
        s = self.Settings(_env_file=None)
        self.assertEqual(s.JWT_EXPIRE_HOURS, 24)

    def test_ollama_base_url_default(self):
        s = self.Settings(_env_file=None)
        self.assertEqual(s.OLLAMA_BASE_URL, "http://localhost:11434")

    def test_embedding_model_default(self):
        # bge-m3, not nomic-embed-text: multilingual and 1024-dimensional
        # against English-first and 768, which matters on a Marathi corpus.
        s = self.Settings(_env_file=None)
        self.assertEqual(s.EMBEDDING_MODEL, "bge-m3")

    def test_llm_model_default(self):
        # qwen2.5:7b, not phi4-mini. Note it is qwen2.5 and NOT
        # qwen2.5-coder, which is code-specialised.
        s = self.Settings(_env_file=None)
        self.assertEqual(s.LLM_MODEL, "qwen2.5:7b")

    def test_rag_defaults(self):
        s = self.Settings(_env_file=None)
        self.assertEqual(s.CHUNK_SIZE, 800)
        self.assertEqual(s.CHUNK_OVERLAP, 150)
        self.assertEqual(s.TOP_K, 12)
        self.assertEqual(s.CONTEXT_WINDOW, 6)

    def test_required_fields_are_read_correctly(self):
        s = self.Settings(_env_file=None)
        self.assertEqual(s.MONGODB_URL, "mongodb://localhost:27017")
        self.assertEqual(s.JWT_SECRET, "test-secret-for-ci")


class TestComputedPaths(unittest.TestCase):

    def setUp(self):
        os.environ["MONGODB_URL"] = "mongodb://localhost:27017"
        os.environ["JWT_SECRET"]  = "test-secret-for-ci"
        os.environ["DATA_DIR"]    = "./test_data"

        from config import Settings
        self.settings_instance = Settings(_env_file=None)

    def tearDown(self):
        os.environ.pop("MONGODB_URL", None)
        os.environ.pop("JWT_SECRET", None)
        os.environ.pop("DATA_DIR", None)

    def test_grdocs_path_derived_correctly(self):
        expected = Path("./test_data") / "grdocs"
        self.assertEqual(self.settings_instance.GRDOCS_PATH, expected)

    def test_vectorstore_path_derived_correctly(self):
        expected = Path("./test_data") / "vectorstore"
        self.assertEqual(self.settings_instance.VECTORSTORE_PATH, expected)

    def test_summaries_path_derived_correctly(self):
        expected = Path("./test_data") / "summaries"
        self.assertEqual(self.settings_instance.SUMMARIES_PATH, expected)

    def test_paths_are_path_objects_not_strings(self):
        self.assertIsInstance(self.settings_instance.GRDOCS_PATH, Path)


if __name__ == "__main__":
    unittest.main()