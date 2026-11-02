"""Permuterm index for efficient leading, trailing, and middle wildcards."""


def build_permuterm_index(vocabulary: list[str]) -> dict[str, set[str]]:
    """Map every rotation of ``term$`` to the original vocabulary term."""
    index: dict[str, set[str]] = {}
    for term in vocabulary:
        marked = term + "$"
        for offset in range(len(marked)):
            rotation = marked[offset:] + marked[:offset]
            index.setdefault(rotation, set()).add(term)
    return index


def expand_wildcard(pattern: str, index: dict[str, set[str]]) -> list[str]:
    """Expand one-asterisk patterns using a prefix lookup over rotations."""
    pattern = pattern.lower().strip()
    if pattern.count("*") != 1:
        raise ValueError("Use exactly one asterisk in a wildcard query.")
    prefix, suffix = pattern.split("*")
    rotated_prefix = suffix + "$" + prefix
    matches: set[str] = set()
    for rotation, terms in index.items():
        if rotation.startswith(rotated_prefix):
            matches.update(terms)
    return sorted(matches)
