"""Tkinter interface for the WebMiningSearch assignment."""

import re
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from indexer import build_index
from query_engine import boolean_search, phrase_search, proximity_search, wildcard_search
from ranker import rank_documents
from tolerant_retrieval import expand_synonyms, soundex, soundex_matches, spelling_suggestions
from preprocessing import tokenize


PROJECT_PATH = Path(__file__).parent.parent
CORPUS_PATH = PROJECT_PATH / "corpus"
MODES = (
    "Ranked search (TF-IDF)",
    "Boolean",
    "Phrase",
    "Proximity",
    "Wildcard",
    "Spelling suggestion",
    "Soundex",
    "Synonym expansion",
)


class SearchApp(tk.Tk):
    """A small desktop interface over the project's existing search modules."""

    def __init__(self) -> None:
        super().__init__()
        self.title("WebMiningSearch")
        self.minsize(820, 570)
        self.index = build_index(CORPUS_PATH)

        self.mode = tk.StringVar(value=MODES[0])
        self.query = tk.StringVar()
        self.status = tk.StringVar(value=f"Index ready: {len(self.index.documents)} documents.")
        self.active_terms: list[str] = []
        self._configure_style()
        self._build_interface()

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Search.TButton", font=("Segoe UI", 10, "bold"))

    def _build_interface(self) -> None:
        container = ttk.Frame(self, padding=18)
        container.grid(sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(3, weight=1)

        ttk.Label(
            container,
            text="WebMiningSearch",
            style="Title.TLabel",
        ).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(
            container,
            text="Choose a mode, enter a query, and inspect the result list.",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 14))

        ttk.Label(container, text="Search mode").grid(row=2, column=0, sticky="w")
        mode_box = ttk.Combobox(
            container, textvariable=self.mode, values=MODES, state="readonly", width=25
        )
        mode_box.grid(row=2, column=1, sticky="ew", padx=(10, 10))
        mode_box.bind("<<ComboboxSelected>>", lambda _event: self._show_hint())

        self.search_box = ttk.Entry(container, textvariable=self.query)
        self.search_box.grid(row=2, column=2, sticky="ew", padx=(0, 10))
        container.columnconfigure(2, weight=2)
        self.search_box.bind("<Return>", lambda _event: self.run_search())
        ttk.Button(container, text="Search", style="Search.TButton", command=self.run_search).grid(
            row=2, column=3, sticky="e"
        )
        ttk.Button(container, text="Syntax help", command=self._show_syntax_help).grid(
            row=2, column=4, padx=(8, 0)
        )
        ttk.Button(container, text="Clear", command=self._clear).grid(
            row=2, column=5, padx=(8, 0)
        )

        self.results = tk.Text(container, height=22, wrap="word", state="disabled")
        self.results.grid(row=3, column=0, columnspan=4, sticky="nsew", pady=(16, 8))
        ttk.Label(container, textvariable=self.status).grid(
            row=4, column=0, columnspan=4, sticky="w"
        )
        self._show_hint()
        self.search_box.focus_set()

    def _show_syntax_help(self) -> None:
        messagebox.showinfo(
            "Search syntax guide",
            "Ranked: search retrieval\n"
            "Boolean: (data OR database) AND NOT network\n"
            "Phrase: information retrieval\n"
            "Proximity: data /3 mining\n"
            "Wildcard: comput*, *ing, comp*er\n"
            "Spelling: retrival\n"
            "Soundex: sarch\n"
            "Synonyms: automobile",
        )

    def _clear(self) -> None:
        self.query.set("")
        self.active_terms = []
        self._set_results("Results", ["Enter a query and select Search."])
        self._show_hint()
        self.search_box.focus_set()

    def _show_hint(self) -> None:
        hints = {
            "Ranked search (TF-IDF)": "Example: search retrieval",
            "Boolean": "Examples: search AND retrieval, search AND NOT mining",
            "Phrase": "Example: information retrieval",
            "Proximity": "Format: data /3 mining  (up to 3 words between)",
            "Wildcard": "Example: comput*",
            "Spelling suggestion": "Example: retrival",
            "Soundex": "Example: sarch",
            "Synonym expansion": "Example: automobile",
        }
        self.status.set(hints[self.mode.get()])

    def _set_results(self, heading: str, lines: list[str], highlights: list[str] | None = None) -> None:
        self.results.configure(state="normal")
        self.results.delete("1.0", tk.END)
        self.results.tag_configure("heading", font=("Segoe UI", 12, "bold"))
        self.results.tag_configure("highlight", background="#ffe86b", foreground="#1b1b1b")
        self.results.insert(tk.END, heading + "\n", "heading")
        self.results.insert(tk.END, "=" * len(heading) + "\n\n")
        content = "\n".join(lines) if lines else "No matches found."
        self._insert_highlighted(content, highlights or [])
        self.results.configure(state="disabled")

    def _insert_highlighted(self, content: str, terms: list[str]) -> None:
        terms = sorted({term for term in terms if term}, key=len, reverse=True)
        if not terms:
            self.results.insert(tk.END, content)
            return
        pattern = re.compile("(" + "|".join(re.escape(term) for term in terms) + ")", re.IGNORECASE)
        last_end = 0
        for match in pattern.finditer(content):
            self.results.insert(tk.END, content[last_end:match.start()])
            self.results.insert(tk.END, match.group(0), "highlight")
            last_end = match.end()
        self.results.insert(tk.END, content[last_end:])

    @staticmethod
    def _snippet(text: str, terms: list[str]) -> str:
        clean_text = " ".join(text.split())
        match = next((re.search(re.escape(term), clean_text, re.IGNORECASE) for term in terms if term), None)
        if not match:
            return clean_text[:280] + ("..." if len(clean_text) > 280 else "")
        start = max(0, match.start() - 100)
        end = min(len(clean_text), match.end() + 180)
        return ("..." if start else "") + clean_text[start:end] + ("..." if end < len(clean_text) else "")

    def _document_lines(self, document_ids: list[str], terms: list[str]) -> list[str]:
        lines = []
        for document_id in document_ids:
            document_path = CORPUS_PATH / f"{document_id}.txt"
            preview = self._snippet(document_path.read_text(encoding="utf-8"), terms)
            lines.append(f"• {document_id}\n  {preview}\n")
        return lines

    def run_search(self) -> None:
        query = self.query.get().strip()
        if not query:
            messagebox.showinfo("Enter a query", "Type a search query first.")
            return

        try:
            mode = self.mode.get()
            self.active_terms = tokenize(query)
            if mode == "Ranked search (TF-IDF)":
                rankings = rank_documents(query, self.index)
                lines = [
                    f"{position}. {document_id}  —  cosine score: {score:.3f}\n  "
                    f"{self._snippet(self.index.documents[document_id], self.active_terms)}\n"
                    for position, (document_id, score) in enumerate(rankings, start=1)
                ]
                self._set_results(f"Ranked results for: {query}", lines, self.active_terms)
            elif mode == "Boolean":
                self._set_results(f"Boolean results for: {query}", self._document_lines(boolean_search(query, self.index), self.active_terms), self.active_terms)
            elif mode == "Phrase":
                self._set_results(f"Phrase results for: \"{query}\"", self._document_lines(phrase_search(query, self.index), self.active_terms), self.active_terms)
            elif mode == "Proximity":
                match = re.fullmatch(r"([a-zA-Z]+)\s*/\s*(\d+)\s+([a-zA-Z]+)", query)
                if not match:
                    raise ValueError("Use this format: data /3 mining")
                left, distance, right = match.groups()
                documents = proximity_search(left, int(distance), right, self.index)
                self._set_results(f"Proximity results for: {query}", self._document_lines(documents, self.active_terms), self.active_terms)
            elif mode == "Wildcard":
                terms, documents = wildcard_search(query, self.index)
                lines = [f"Matched vocabulary: {', '.join(terms) or 'none'}", ""]
                lines.extend(self._document_lines(documents, terms))
                self._set_results(f"Wildcard results for: {query}", lines, terms)
            elif mode == "Spelling suggestion":
                suggestions = spelling_suggestions(query, self.index)
                self._set_results(f"Suggestions for: {query}", [f"• {term}" for term in suggestions], suggestions)
            elif mode == "Soundex":
                code = soundex(query)
                terms = soundex_matches(query, self.index)
                self._set_results(f"Soundex results for: {query}", [f"Soundex code: {code}", "", *[f"• {term}" for term in terms]], terms)
            else:
                terms = expand_synonyms(query)
                documents = sorted(set().union(*(set(self.index.inverted.get(term, [])) for term in terms)))
                lines = [f"Expanded terms: {', '.join(terms)}", ""]
                lines.extend(self._document_lines(documents, terms))
                self._set_results(f"Synonym expansion for: {query}", lines, terms)
            self.status.set("Search complete.")
        except ValueError as error:
            messagebox.showerror("Query problem", str(error))


def main() -> None:
    """Start the interface."""
    SearchApp().mainloop()
