from tg_guard_bot.web_search import (
    DuckDuckGoHtmlParser,
    ReadableTextParser,
    clean_duckduckgo_url,
    is_fetchable_url,
)


def test_duckduckgo_html_parser_extracts_results() -> None:
    parser = DuckDuckGoHtmlParser()
    parser.feed(
        """
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa">
            Example title
        </a>
        <a class="result__snippet">Useful snippet here</a>
        """
    )
    parser.close()

    assert len(parser.results) == 1
    assert parser.results[0].title == "Example title"
    assert parser.results[0].url == "https://example.com/a"
    assert parser.results[0].snippet == "Useful snippet here"


def test_duckduckgo_lite_parser_extracts_results() -> None:
    parser = DuckDuckGoHtmlParser()
    parser.feed(
        """
        <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Flite" class='result-link'>
            Lite title
        </a>
        <td class='result-snippet'>Lite snippet here</td>
        """
    )
    parser.close()

    assert len(parser.results) == 1
    assert parser.results[0].title == "Lite title"
    assert parser.results[0].url == "https://example.com/lite"
    assert parser.results[0].snippet == "Lite snippet here"


def test_clean_duckduckgo_url_keeps_regular_url() -> None:
    assert clean_duckduckgo_url("https://example.com") == "https://example.com"


def test_readable_text_parser_skips_script_and_short_noise() -> None:
    parser = ReadableTextParser()
    parser.feed(
        """
        <script>this should not appear even if it is very long</script>
        <p>tiny</p>
        <main>Useful readable paragraph with enough words to survive parser filtering.</main>
        """
    )
    parser.close()

    assert parser.text() == "Useful readable paragraph with enough words to survive parser filtering."


def test_is_fetchable_url_accepts_only_http_urls() -> None:
    assert is_fetchable_url("https://example.com")
    assert is_fetchable_url("http://example.com")
    assert not is_fetchable_url("mailto:test@example.com")
