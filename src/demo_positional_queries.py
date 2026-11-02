"""Demonstrate phrase and proximity queries using the positional index."""

from pathlib import Path

from indexer import build_index
from query_engine import phrase_search, proximity_search


CORPUS_PATH = Path(__file__).parent.parent / "corpus"


def main() -> None:
    index = build_index(CORPUS_PATH)

    print('"information retrieval" ->', phrase_search("information retrieval", index))
    print('"web mining"           ->', phrase_search("web mining", index))
    print("data /3 mining          ->", proximity_search("data", 3, "mining", index))


if __name__ == "__main__":
    main()
