"""Small, transparent text-preprocessing functions for our search engine."""

import re

from stemmer import stem


# These frequent grammatical words rarely help distinguish one document from another.
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "their",
    "to", "uses", "when", "while", "with",
}


def tokenize(text: str) -> list[str]:
    """Return lowercase alphabetic word tokens from *text*.

    `re.findall` keeps words such as "tf-idf" as the two useful tokens
    "tf" and "idf", while discarding punctuation.
    """
    return re.findall(r"[a-z]+", text.lower())


def remove_stop_words(tokens: list[str]) -> list[str]:
    """Remove high-frequency words that have little search value."""
    return [token for token in tokens if token not in STOP_WORDS]


def preprocess(text: str, apply_stemming: bool = False) -> list[str]:
    """Apply normalization, tokenization, stop-word removal, and optional stemming."""
    tokens = remove_stop_words(tokenize(text))
    return [stem(token) for token in tokens] if apply_stemming else tokens
