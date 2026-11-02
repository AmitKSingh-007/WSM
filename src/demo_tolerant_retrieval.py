"""Demonstrate spelling, n-gram, Soundex, and synonym helpers."""

from pathlib import Path

from indexer import build_index
from tolerant_retrieval import (
    expand_synonyms,
    levenshtein_distance,
    ngram_similarity,
    soundex,
    soundex_matches,
    spelling_suggestions,
)


CORPUS_PATH = Path(__file__).parent.parent / "corpus"


def main() -> None:
    index = build_index(CORPUS_PATH)
    print("Suggestions for 'retrival':", spelling_suggestions("retrival", index))
    print("Edit distance (retrival, retrieval):", levenshtein_distance("retrival", "retrieval"))
    print("Bigram similarity (retrival, retrieval):", f"{ngram_similarity('retrival', 'retrieval'):.2f}")
    print("Soundex for 'sarch':", soundex("sarch"))
    print("Same-Soundex vocabulary terms:", soundex_matches("sarch", index))
    print("Synonym expansion for 'automobile':", expand_synonyms("automobile"))


if __name__ == "__main__":
    main()
