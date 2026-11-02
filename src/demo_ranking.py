"""Rank documents for a multi-word search query."""

from pathlib import Path

from indexer import build_index
from ranker import rank_documents


CORPUS_PATH = Path(__file__).parent.parent / "corpus"


def main() -> None:
    query = "search retrieval"
    index = build_index(CORPUS_PATH)
    print(f"Query: {query!r}\n")
    for rank, (document_id, score) in enumerate(rank_documents(query, index), start=1):
        print(f"{rank}. {document_id:<28} cosine score = {score:.3f}")


if __name__ == "__main__":
    main()
