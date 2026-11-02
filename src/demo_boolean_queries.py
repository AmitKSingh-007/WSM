"""Run a few Boolean queries against the corpus."""

from pathlib import Path

from indexer import build_index
from query_engine import boolean_search


CORPUS_PATH = Path(__file__).parent.parent / "corpus"


def main() -> None:
    index = build_index(CORPUS_PATH)
    example_queries = [
        "retrieval",
        "search AND retrieval",
        "crawler OR ranking",
        "search AND NOT mining",
    ]

    for query in example_queries:
        print(f"{query:<24} -> {boolean_search(query, index)}")


if __name__ == "__main__":
    main()
