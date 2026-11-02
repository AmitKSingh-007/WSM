"""Basic Boolean query processing using the inverted index."""

import re

from indexer import SearchIndex
from permuterm import expand_wildcard
from preprocessing import preprocess, tokenize


def _postings_for_term(term: str, index: SearchIndex) -> set[str]:
    """Return the documents for one query term after normalizing it."""
    tokens = preprocess(term)
    if len(tokens) != 1:
        raise ValueError("Each Boolean operand must be one searchable word.")
    return set(index.inverted.get(tokens[0], []))


def boolean_search(query: str, index: SearchIndex) -> list[str]:
    """Evaluate a simple single-term Boolean query.

    Supported examples: ``search``, ``search AND retrieval``,
    ``crawler OR ranking``, and ``search AND NOT mining``.
    """
    query = query.strip()
    all_documents = set(index.term_frequency)

    # NOT on its own means every corpus document except the matching ones.
    not_only = re.fullmatch(r"NOT\s+(.+)", query, flags=re.IGNORECASE)
    if not_only:
        return sorted(all_documents - _postings_for_term(not_only.group(1), index))

    # This first version intentionally supports one binary operation at a time.
    match = re.fullmatch(
        r"(.+?)\s+(AND|OR)\s+(NOT\s+)?(.+)", query, flags=re.IGNORECASE
    )
    if not match:
        return sorted(_postings_for_term(query, index))

    left_term, operator, negation, right_term = match.groups()
    left_documents = _postings_for_term(left_term, index)
    right_documents = _postings_for_term(right_term, index)

    if negation:
        right_documents = all_documents - right_documents

    if operator.upper() == "AND":
        return sorted(left_documents & right_documents)
    return sorted(left_documents | right_documents)


def phrase_search(phrase: str, index: SearchIndex) -> list[str]:
    """Return documents containing every word in *phrase* consecutively."""
    terms = tokenize(phrase)
    if not terms:
        return []

    # A phrase can only occur in documents containing every phrase term.
    candidate_documents = set(index.positional.get(terms[0], {}))
    for term in terms[1:]:
        candidate_documents &= set(index.positional.get(term, {}))

    matches = []
    for document_id in candidate_documents:
        first_positions = index.positional[terms[0]][document_id]
        other_position_sets = [
            set(index.positional[term][document_id]) for term in terms[1:]
        ]

        # If the first word is at p, later phrase words must be at p+1, p+2, ...
        if any(
            all(start + offset in positions for offset, positions in enumerate(other_position_sets, 1))
            for start in first_positions
        ):
            matches.append(document_id)

    return sorted(matches)


def proximity_search(
    left_term: str, max_gap: int, right_term: str, index: SearchIndex
) -> list[str]:
    """Find documents where *right_term* follows *left_term* within *max_gap* words.

    For example, ``data /3 mining`` permits zero to three words between
    ``data`` and ``mining``.
    """
    left_tokens = tokenize(left_term)
    right_tokens = tokenize(right_term)
    if len(left_tokens) != 1 or len(right_tokens) != 1:
        raise ValueError("A proximity query needs one word on each side.")
    if max_gap < 0:
        raise ValueError("The maximum gap cannot be negative.")

    left, right = left_tokens[0], right_tokens[0]
    candidate_documents = set(index.positional.get(left, {})) & set(
        index.positional.get(right, {})
    )
    matches = []
    for document_id in candidate_documents:
        left_positions = index.positional[left][document_id]
        right_positions = index.positional[right][document_id]
        if any(0 < right_pos - left_pos <= max_gap + 1
               for left_pos in left_positions for right_pos in right_positions):
            matches.append(document_id)

    return sorted(matches)


def wildcard_search(pattern: str, index: SearchIndex) -> tuple[list[str], list[str]]:
    """Return vocabulary terms and documents matching a ``*`` wildcard pattern.

    ``comput*`` matches words beginning with ``comput``; the function also
    supports an asterisk in the middle, such as ``c*r``.
    """
    normalized_pattern = pattern.lower().strip()
    if "*" not in normalized_pattern:
        raise ValueError("A wildcard query must contain an asterisk (*).")
    if not re.fullmatch(r"[a-z*]+", normalized_pattern):
        raise ValueError("A wildcard query may contain only letters and *.")

    matching_terms = expand_wildcard(normalized_pattern, index.permuterm_index)
    matching_documents = set()
    for term in matching_terms:
        matching_documents.update(index.inverted[term])

    return matching_terms, sorted(matching_documents)
