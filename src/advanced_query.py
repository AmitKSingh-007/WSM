"""AST evaluation and skip-pointer-accelerated Boolean retrieval."""

from math import isqrt

from parser import QueryNode, parse_query
from indexer import SearchIndex
from preprocessing import preprocess
from query_engine import phrase_search, proximity_search, wildcard_search


def _term_postings(term: str, index: SearchIndex) -> list[str]:
    tokens = preprocess(term, apply_stemming=True)
    if len(tokens) != 1:
        return []
    return index.stemmed_inverted.get(tokens[0], [])


def intersect_with_skips(left: list[str], right: list[str], index: SearchIndex) -> set[str]:
    """Intersect ordered postings, following a skip link when it cannot overshoot."""
    left_positions = {document: position for position, document in enumerate(left)}
    right_positions = {document: position for position, document in enumerate(right)}
    def skips(postings: list[str]) -> dict[str, str]:
        step = isqrt(len(postings))
        return {postings[p]: postings[p + step] for p in range(0, len(postings) - step, step)} if step >= 2 else {}
    left_skip, right_skip = skips(left), skips(right)
    result, left_pos, right_pos = set(), 0, 0
    while left_pos < len(left) and right_pos < len(right):
        left_doc, right_doc = left[left_pos], right[right_pos]
        if left_doc == right_doc:
            result.add(left_doc); left_pos += 1; right_pos += 1
        elif left_doc < right_doc:
            jump = left_skip.get(left_doc)
            left_pos = left_positions[jump] if jump and jump <= right_doc else left_pos + 1
        else:
            jump = right_skip.get(right_doc)
            right_pos = right_positions[jump] if jump and jump <= left_doc else right_pos + 1
    return result


def evaluate(query: str, index: SearchIndex) -> list[str]:
    """Parse and evaluate Boolean, phrase, proximity, and wildcard expressions."""
    universe = set(index.documents)

    def visit(node: QueryNode) -> set[str]:
        if node.kind == "TERM":
            return set(_term_postings(node.value or "", index))
        if node.kind == "WILDCARD":
            return set(wildcard_search(node.value or "", index)[1])
        if node.kind == "PHRASE":
            return set(phrase_search(node.value or "", index))
        if node.kind == "PROXIMITY":
            left, gap, right = (node.value or "").split("|")
            return set(proximity_search(left, int(gap), right, index))
        if node.kind == "NOT":
            return universe - visit(node.left)  # type: ignore[arg-type]
        left, right = visit(node.left), visit(node.right)  # type: ignore[arg-type]
        if node.kind == "AND":
            return intersect_with_skips(sorted(left), sorted(right), index)
        return left | right

    return sorted(visit(parse_query(query)))
