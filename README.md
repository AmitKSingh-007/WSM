# WebMiningSearch

A Python information-retrieval project that searches a small collection of web-mining documents.

## Project layout

- `corpus/` — the 15 plain-text documents that form our searchable collection.
- `src/` — Python code for preprocessing, indexing, queries, ranking, and the GUI.
- `tests/` — checks that protect the search features as we add them.

## Learning path

1. Read the corpus and understand what is being indexed.
2. Convert text into consistent tokens.
3. Build inverted and positional indexes.
4. Add Boolean, phrase, proximity, and wildcard queries.
5. Rank results with TF-IDF and cosine similarity.
6. Add spelling and other tolerant-retrieval features.
7. Connect everything to a Tkinter interface.

The first version uses only Python's standard library, so it can run without installing packages.
