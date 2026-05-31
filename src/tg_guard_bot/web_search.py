from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

import httpx


SEARCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
}
SEARCH_ENDPOINTS = (
    "https://html.duckduckgo.com/html/",
    "https://lite.duckduckgo.com/lite/",
)


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    excerpt: str = ""


class ReadableTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "header"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "header"}:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = normalize_space(data)
        if len(text) >= 40:
            self.parts.append(text)

    def text(self, limit: int = 1800) -> str:
        return normalize_space(" ".join(self.parts))[:limit]


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
        if tag == "a" and ("result__a" in classes or "result-link" in classes):
            self._flush_current()
            self._in_title = True
            self._current_title = []
            self._current_snippet = []
            self._current_url = clean_duckduckgo_url(attr_map.get("href", ""))
        elif "result__snippet" in classes or "result-snippet" in classes:
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
        headers=SEARCH_HEADERS,
    ) as client:
        return await fetch_search_results(client, query, limit)


async def search_web_deep(query: str, limit: int = 4, excerpt_chars: int = 1800) -> list[SearchResult]:
    async with httpx.AsyncClient(
        timeout=10,
        follow_redirects=True,
        headers=SEARCH_HEADERS,
    ) as client:
        results = await fetch_search_results(client, query, limit)

        enriched: list[SearchResult] = []
        for result in results:
            excerpt = ""
            if is_fetchable_url(result.url):
                try:
                    excerpt = await fetch_page_excerpt(client, result.url, excerpt_chars)
                except (httpx.HTTPError, UnicodeDecodeError):
                    excerpt = ""
            enriched.append(
                SearchResult(
                    title=result.title,
                    url=result.url,
                    snippet=result.snippet,
                    excerpt=excerpt,
                )
            )
        return enriched


async def fetch_search_results(
    client: httpx.AsyncClient,
    query: str,
    limit: int,
) -> list[SearchResult]:
    seen: set[str] = set()
    merged: list[SearchResult] = []
    for endpoint in SEARCH_ENDPOINTS:
        response = await client.get(endpoint, params={"q": query})
        response.raise_for_status()
        parser = DuckDuckGoHtmlParser()
        parser.feed(response.text)
        parser.close()
        for result in parser.results:
            if result.url in seen:
                continue
            seen.add(result.url)
            merged.append(result)
            if len(merged) >= limit:
                return merged
    return merged


async def fetch_page_excerpt(client: httpx.AsyncClient, url: str, limit: int) -> str:
    response = await client.get(url)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type and "text/plain" not in content_type:
        return ""
    parser = ReadableTextParser()
    parser.feed(response.text[:300_000])
    parser.close()
    return parser.text(limit)


def format_search_results(results: list[SearchResult]) -> str:
    if not results:
        return ""
    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        excerpt = result.excerpt or result.snippet
        lines.append(
            f"{index}. {result.title}\n"
            f"URL: {result.url}\n"
            f"Сниппет: {result.snippet}\n"
            f"Выдержка со страницы: {excerpt}"
        )
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


def is_fetchable_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def normalize_space(value: str) -> str:
    return " ".join(unescape(value).split())
