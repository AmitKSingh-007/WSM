"""Show how a wildcard query expands into vocabulary terms and documents."""

from pathlib import Path

from indexer import build_index
from query_engine import wildcard_search


CORPUS_PATH = Path(__file__).parent.parent / "corpus"


def main() -> None:
    index = build_index(CORPUS_PATH)
    terms, documents = wildcard_search("comput*", index)
    print("comput* expands to:", terms)
    print("matching documents:", documents)


if __name__ == "__main__":
    main()
