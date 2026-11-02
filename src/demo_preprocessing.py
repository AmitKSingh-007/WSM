"""Run this file to see what preprocessing does to one corpus document."""

from pathlib import Path

from preprocessing import preprocess, tokenize


CORPUS_FILE = Path(__file__).parent.parent / "corpus" / "doc08_phrase_search.txt"


def main() -> None:
    text = CORPUS_FILE.read_text(encoding="utf-8")
    raw_tokens = tokenize(text)
    clean_tokens = preprocess(text)

    print("Original text:\n", text)
    print("Tokens before stop-word removal:\n", raw_tokens)
    print("Tokens used for indexing:\n", clean_tokens)


if __name__ == "__main__":
    main()
