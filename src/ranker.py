"""TF-IDF ranking with cosine similarity."""

from collections import Counter
from math import log, sqrt

from indexer import SearchIndex
from preprocessing import preprocess


def _tf_weight(count: int) -> float:
    """Use logarithmic term-frequency weighting to avoid overvaluing repeats."""
    return 1.0 + log(count) if count else 0.0


def _idf(term: str, index: SearchIndex) -> float:
    """Return a smoothed inverse-document-frequency score for one term."""
    document_count = len(index.term_frequency)
    frequency = index.document_frequency.get(term, 0)
    return log((document_count + 1) / (frequency + 1)) + 1.0


def rank_documents(query: str, index: SearchIndex) -> list[tuple[str, float]]:
    """Rank documents matching *query* by TF-IDF cosine similarity."""
    query_counts = Counter(preprocess(query))
    if not query_counts:
        return []

    query_weights = {
        term: _tf_weight(count) * _idf(term, index)
        for term, count in query_counts.items()
        if term in index.inverted
    }
    if not query_weights:
        return []

    query_norm = sqrt(sum(weight * weight for weight in query_weights.values()))
    candidate_documents = set().union(
        *(set(index.inverted[term]) for term in query_weights)
    )
    ranked = []

    for document_id in candidate_documents:
        document_terms = index.term_frequency[document_id]
        dot_product = sum(
            query_weight * _tf_weight(document_terms.get(term, 0)) * _idf(term, index)
            for term, query_weight in query_weights.items()
        )
        document_norm = sqrt(
            sum(
                (_tf_weight(count) * _idf(term, index)) ** 2
                for term, count in document_terms.items()
            )
        )
        score = dot_product / (query_norm * document_norm)
        ranked.append((document_id, score))

    return sorted(ranked, key=lambda result: (-result[1], result[0]))
