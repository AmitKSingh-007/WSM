"""Automated regression tests for the WebMiningSearch engine."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parent / "src"))

from advanced_query import evaluate
from indexer import build_index
from query_engine import phrase_search, proximity_search, wildcard_search
from stemmer import stem
from tolerant_retrieval import correct_query, levenshtein_distance, soundex, soundex_matches


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = build_index(Path(__file__).parent / "corpus")

    def test_corpus_size(self) -> None:
        self.assertEqual(len(self.index.documents), 25)

    def test_porter_stemmer(self) -> None:
        self.assertEqual(stem("relational"), "relat")
        self.assertEqual(stem("formalize"), "formal")

    def test_boolean_precedence(self) -> None:
        result = evaluate("data OR database AND network", self.index)
        self.assertIn("doc09_proximity_search", result)

    def test_parentheses(self) -> None:
        result = evaluate("(data OR database) AND NOT network", self.index)
        self.assertIn("doc20_databases", result)

    def test_phrase(self) -> None:
        self.assertIn("doc09_proximity_search", phrase_search("data mining", self.index))

    def test_proximity(self) -> None:
        self.assertIn("doc09_proximity_search", proximity_search("data", 3, "mining", self.index))

    def test_permuterm_wildcards(self) -> None:
        self.assertIn("computer", wildcard_search("comput*", self.index)[0])
        self.assertIn("computer", wildcard_search("comp*er", self.index)[0])

    def test_edit_distance(self) -> None:
        self.assertEqual(levenshtein_distance("retrival", "retrieval"), 1)

    def test_full_query_correction(self) -> None:
        corrected, changes = correct_query("retrival", self.index)
        self.assertEqual(corrected, "retrieval")
        self.assertEqual(changes["retrival"], "retrieval")

    def test_soundex(self) -> None:
        self.assertEqual(soundex("Robert"), "R163")
        self.assertIn("search", soundex_matches("sarch", self.index))


if __name__ == "__main__":
    unittest.main(verbosity=2)
