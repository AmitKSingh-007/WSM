"""Build inverted and positional indexes from the corpus."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from math import isqrt
from pathlib import Path

from permuterm import build_permuterm_index
from preprocessing import preprocess, tokenize


@dataclass
class SearchIndex:
    """The data structures produced after reading every corpus document."""

    inverted: dict[str, list[str]]
    positional: dict[str, dict[str, list[int]]]
    term_frequency: dict[str, dict[str, int]]
    document_frequency: dict[str, int]
    stemmed_inverted: dict[str, list[str]]
    stemmed_term_frequency: dict[str, dict[str, int]]
    raw_vocabulary: list[str]
    skip_pointers: dict[str, dict[str, str]]
    documents: dict[str, str]
    permuterm_index: dict[str, set[str]]


def _build_skip_pointers(postings: list[str]) -> dict[str, str]:
    """Create sqrt(L)-spaced forward links for one sorted posting list."""
    step = isqrt(len(postings))
    if step < 2:
        return {}
    return {
        postings[position]: postings[position + step]
        for position in range(0, len(postings) - step, step)
    }


def build_index(corpus_path: Path) -> SearchIndex:
    """Read ``*.txt`` files and build searchable index structures.

    The inverted index excludes stop words because they add little value to a
    normal word search. The positional index uses every normalized token, so
    phrases retain their true word positions.
    """
    inverted_sets: dict[str, set[str]] = defaultdict(set)
    positional: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    term_frequency: dict[str, dict[str, int]] = {}
    stemmed_sets: dict[str, set[str]] = defaultdict(set)
    stemmed_term_frequency: dict[str, dict[str, int]] = {}
    documents: dict[str, str] = {}

    for document_path in sorted(corpus_path.glob("*.txt")):
        document_id = document_path.stem
        text = document_path.read_text(encoding="utf-8")
        documents[document_id] = text

        # Ordinary word search uses cleaned terms.
        clean_tokens = preprocess(text)
        counts = Counter(clean_tokens)
        term_frequency[document_id] = dict(counts)
        for term in counts:
            inverted_sets[term].add(document_id)

        stemmed_counts = Counter(preprocess(text, apply_stemming=True))
        stemmed_term_frequency[document_id] = dict(stemmed_counts)
        for term in stemmed_counts:
            stemmed_sets[term].add(document_id)

        # Phrase/proximity search must know the original normalized positions.
        for position, term in enumerate(tokenize(text)):
            positional[term][document_id].append(position)

    inverted = {term: sorted(documents) for term, documents in inverted_sets.items()}
    document_frequency = {term: len(documents) for term, documents in inverted.items()}
    stemmed_inverted = {
        term: sorted(document_ids) for term, document_ids in stemmed_sets.items()
    }
    skip_pointers = {
        term: _build_skip_pointers(postings) for term, postings in inverted.items()
    }

    return SearchIndex(
        inverted=inverted,
        positional={term: dict(documents) for term, documents in positional.items()},
        term_frequency=term_frequency,
        document_frequency=document_frequency,
        stemmed_inverted=stemmed_inverted,
        stemmed_term_frequency=stemmed_term_frequency,
        raw_vocabulary=sorted(inverted),
        skip_pointers=skip_pointers,
        documents=documents,
        permuterm_index=build_permuterm_index(sorted(inverted)),
    )


def save_index_files(index: SearchIndex, data_path: Path) -> None:
    """Serialize the project indexes to readable JSON files."""
    data_path.mkdir(parents=True, exist_ok=True)
    dictionary = {
        term: {"document_frequency": index.document_frequency[term]}
        for term in index.raw_vocabulary
    }
    inverted = {
        term: {
            "postings": postings,
            "skip_pointers": index.skip_pointers[term],
        }
        for term, postings in index.inverted.items()
    }
    files = {
        "dictionary.json": dictionary,
        "documents.json": index.documents,
        "inverted_index.json": inverted,
        "positional_index.json": index.positional,
        "raw_vocabulary.json": index.raw_vocabulary,
    }
    for name, contents in files.items():
        (data_path / name).write_text(
            json.dumps(contents, indent=2, sort_keys=True), encoding="utf-8"
        )
