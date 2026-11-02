"""Helpful retrieval features for misspellings, similar sounds, and synonyms."""

from indexer import SearchIndex
from preprocessing import tokenize


SYNONYMS = {
    "automobile": {"car"},
    "car": {"automobile"},
    "engine": {"search"},
    "search": {"engine"},
    "web": {"internet"},
    "internet": {"web"},
}


def levenshtein_distance(left: str, right: str) -> int:
    """Return the minimum insertions, deletions, and substitutions needed."""
    if len(left) < len(right):
        left, right = right, left

    previous_row = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current_row = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            insert = current_row[-1] + 1
            delete = previous_row[right_index] + 1
            substitute = previous_row[right_index - 1] + (
                left_character != right_character
            )
            current_row.append(min(insert, delete, substitute))
        previous_row = current_row
    return previous_row[-1]


def character_ngrams(word: str, size: int = 2) -> set[str]:
    """Create padded character n-grams, e.g. search -> {^s, se, ..., h$}."""
    padded_word = f"^{word.lower()}$"
    return {
        padded_word[position : position + size]
        for position in range(len(padded_word) - size + 1)
    }


def ngram_similarity(left: str, right: str, size: int = 2) -> float:
    """Return Jaccard similarity for character n-gram sets."""
    left_ngrams = character_ngrams(left, size)
    right_ngrams = character_ngrams(right, size)
    return len(left_ngrams & right_ngrams) / len(left_ngrams | right_ngrams)


def spelling_suggestions(word: str, index: SearchIndex, limit: int = 3) -> list[str]:
    """Suggest close vocabulary terms, preferring low edit distance."""
    word_tokens = tokenize(word)
    if len(word_tokens) != 1:
        raise ValueError("Enter one word for a spelling suggestion.")
    query_word = word_tokens[0]

    candidates = sorted(
        index.inverted,
        key=lambda term: (
            levenshtein_distance(query_word, term),
            -ngram_similarity(query_word, term),
            term,
        ),
    )
    return candidates[:limit]


def correct_query(query: str, index: SearchIndex) -> tuple[str, dict[str, str]]:
    """Return a corrected multi-word query and its accepted word replacements."""
    replacements: dict[str, str] = {}
    corrected = []
    for word in tokenize(query):
        if word in index.raw_vocabulary:
            corrected.append(word)
            continue
        suggestion = spelling_suggestions(word, index, limit=1)[0]
        if levenshtein_distance(word, suggestion) <= 2:
            replacements[word] = suggestion
            corrected.append(suggestion)
        else:
            corrected.append(word)
    return " ".join(corrected), replacements


def soundex(word: str) -> str:
    """Encode a word's approximate English pronunciation as a Soundex code."""
    letters = "".join(tokenize(word))
    if not letters:
        return ""

    groups = {
        **dict.fromkeys("bfpv", "1"),
        **dict.fromkeys("cgjkqsxz", "2"),
        **dict.fromkeys("dt", "3"),
        "l": "4",
        **dict.fromkeys("mn", "5"),
        "r": "6",
    }
    encoded_digits = []
    previous_digit = groups.get(letters[0], "")
    for character in letters[1:]:
        digit = groups.get(character, "")
        if digit and digit != previous_digit:
            encoded_digits.append(digit)
        previous_digit = digit

    return (letters[0].upper() + "".join(encoded_digits) + "000")[:4]


def soundex_matches(word: str, index: SearchIndex) -> list[str]:
    """Find vocabulary terms with the same Soundex code as *word*."""
    code = soundex(word)
    return sorted(term for term in index.inverted if soundex(term) == code)


def expand_synonyms(word: str) -> list[str]:
    """Return the normalized word plus its small, transparent synonym set."""
    tokens = tokenize(word)
    if len(tokens) != 1:
        raise ValueError("Synonym expansion needs one word.")
    normalized_word = tokens[0]
    return sorted({normalized_word, *SYNONYMS.get(normalized_word, set())})
