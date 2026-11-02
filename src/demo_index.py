"""Build the corpus index and inspect a few postings lists."""

from pathlib import Path

from indexer import build_index


CORPUS_PATH = Path(__file__).parent.parent / "corpus"


def main() -> None:
    index = build_index(CORPUS_PATH)

    print(f"Indexed {len(index.term_frequency)} documents.")
    print(f"Vocabulary size: {len(index.inverted)} terms.\n")

    for term in ["retrieval", "search", "mining", "information"]:
        print(f"{term!r}")
        print("  inverted postings:", index.inverted.get(term, []))
        print("  document frequency:", index.document_frequency.get(term, 0))
        print("  positions:", index.positional.get(term, {}))


if __name__ == "__main__":
    main()
