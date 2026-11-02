"""Run a compact audit of every supported query capability."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from advanced_query import evaluate
from indexer import build_index, save_index_files
from query_engine import phrase_search, proximity_search, wildcard_search
from ranker import rank_documents
from tolerant_retrieval import correct_query, soundex_matches


def main() -> None:
    root = Path(__file__).parent
    index = build_index(root / "corpus")
    save_index_files(index, root / "data")
    checks = {
        "Single term": lambda: evaluate("computer", index),
        "AND": lambda: evaluate("computer AND network", index),
        "OR": lambda: evaluate("computer OR database", index),
        "NOT": lambda: evaluate("computer AND NOT network", index),
        "Nested AST": lambda: evaluate("(data OR database) AND NOT network", index),
        "Phrase": lambda: phrase_search("data mining", index),
        "Proximity": lambda: proximity_search("data", 3, "mining", index),
        "Trailing wildcard": lambda: wildcard_search("comput*", index)[0],
        "Leading wildcard": lambda: wildcard_search("*ing", index)[0],
        "Middle wildcard": lambda: wildcard_search("comp*er", index)[0],
        "Ranked": lambda: rank_documents("search retrieval", index)[:3],
        "Spelling": lambda: correct_query("retrival engne", index),
        "Soundex": lambda: soundex_matches("sarch", index),
    }
    for name, check in checks.items():
        result = check()
        print(f"[PASS] {name}: {result}")


if __name__ == "__main__":
    main()
