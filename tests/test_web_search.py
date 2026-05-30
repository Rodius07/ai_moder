from tg_guard_bot.web_search import DuckDuckGoHtmlParser, clean_duckduckgo_url


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


def test_clean_duckduckgo_url_keeps_regular_url() -> None:
    assert clean_duckduckgo_url("https://example.com") == "https://example.com"
