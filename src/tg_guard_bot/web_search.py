from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

import httpx


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class DuckDuckGoHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[SearchResult] = []
        self._in_title = False
        self._in_snippet = False
        self._current_title: list[str] = []
        self._current_snippet: list[str] = []
        self._current_url = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key: value or "" for key, value in attrs}
        classes = set(attr_map.get("class", "").split())
        if tag == "a" and "result__a" in classes:
            self._flush_current()
            self._in_title = True
            self._current_title = []
            self._current_snippet = []
            self._current_url = clean_duckduckgo_url(attr_map.get("href", ""))
        elif "result__snippet" in classes:
            self._in_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            self._in_title = False
        elif self._in_snippet and tag in {"a", "div"}:
            self._in_snippet = False
            self._flush_current()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._current_title.append(data)
        elif self._in_snippet:
            self._current_snippet.append(data)

    def close(self) -> None:
        super().close()
        self._flush_current()

    def _flush_current(self) -> None:
        title = normalize_space(" ".join(self._current_title))
        snippet = normalize_space(" ".join(self._current_snippet))
        if title and self._current_url:
            result = SearchResult(title=title, url=self._current_url, snippet=snippet)
            if result not in self.results:
                self.results.append(result)
        self._current_title = []
        self._current_snippet = []
        self._current_url = ""


async def search_web(query: str, limit: int = 4) -> list[SearchResult]:
    async with httpx.AsyncClient(
        timeout=8,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 tg-guard-bot/0.1"},
    ) as client:
        response = await client.get("https://duckduckgo.com/html/", params={"q": query})
        response.raise_for_status()

    parser = DuckDuckGoHtmlParser()
    parser.feed(response.text)
    parser.close()
    return parser.results[:limit]


def format_search_results(results: list[SearchResult]) -> str:
    if not results:
        return ""
    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result.title}\nURL: {result.url}\nФрагмент: {result.snippet}")
    return "\n\n".join(lines)


def clean_duckduckgo_url(value: str) -> str:
    value = unescape(value)
    if value.startswith("//"):
        value = "https:" + value
    parsed = urlparse(value)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    return value


def normalize_space(value: str) -> str:
    return " ".join(unescape(value).split())
