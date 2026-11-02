"""Recursive-descent parser for Boolean expressions with standard precedence."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class QueryNode:
    kind: str
    value: str | None = None
    left: "QueryNode | None" = None
    right: "QueryNode | None" = None


def lex(query: str) -> list[str]:
    """Split a query while keeping quoted phrases, parentheses, and /k tokens."""
    tokens = re.findall(r'"[^"\n]+"|\(|\)|/\d+|[A-Za-z*]+', query)
    if not tokens:
        raise ValueError("Enter a query.")
    return tokens


class BooleanParser:
    """Parse NOT > AND > OR expressions into an abstract syntax tree."""

    def __init__(self, query: str) -> None:
        self.tokens = lex(query)
        self.position = 0

    def parse(self) -> QueryNode:
        node = self._parse_or()
        if self._peek() is not None:
            raise ValueError(f"Unexpected token: {self._peek()}")
        return node

    def _peek(self) -> str | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def _take(self) -> str:
        token = self._peek()
        if token is None:
            raise ValueError("The query ended unexpectedly.")
        self.position += 1
        return token

    def _parse_or(self) -> QueryNode:
        node = self._parse_and()
        while (self._peek() or "").upper() == "OR":
            self._take()
            node = QueryNode("OR", left=node, right=self._parse_and())
        return node

    def _parse_and(self) -> QueryNode:
        node = self._parse_not()
        while (self._peek() or "").upper() == "AND":
            self._take()
            node = QueryNode("AND", left=node, right=self._parse_not())
        return node

    def _parse_not(self) -> QueryNode:
        if (self._peek() or "").upper() == "NOT":
            self._take()
            return QueryNode("NOT", left=self._parse_not())
        return self._parse_primary()

    def _parse_primary(self) -> QueryNode:
        token = self._take()
        if token == "(":
            node = self._parse_or()
            if self._take() != ")":
                raise ValueError("A closing parenthesis is required.")
            return node
        if token == ")":
            raise ValueError("Unexpected closing parenthesis.")
        if token.startswith('"'):
            return QueryNode("PHRASE", value=token[1:-1])
        if token.upper() in {"AND", "OR", "NOT"} or token.startswith("/"):
            raise ValueError(f"Expected a term, not {token}.")

        if (self._peek() or "").startswith("/"):
            distance = self._take()[1:]
            right = self._take()
            if right in {"(", ")"} or right.upper() in {"AND", "OR", "NOT"}:
                raise ValueError("A proximity query needs a word after /k.")
            return QueryNode("PROXIMITY", value=f"{token}|{distance}|{right}")
        return QueryNode("WILDCARD" if "*" in token else "TERM", value=token)


def parse_query(query: str) -> QueryNode:
    return BooleanParser(query).parse()
