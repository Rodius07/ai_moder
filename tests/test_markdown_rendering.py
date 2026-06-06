from tg_guard_bot.bot import normalize_telegram_markdown, strip_markdown


def test_normalizes_common_ai_markdown_for_telegram() -> None:
    source = "### Вот что сейчас в топе:\n* **Voice Engine** — описание\n* **Eleven V3** — модель"

    rendered = normalize_telegram_markdown(source)

    assert rendered == "Вот что сейчас в топе:\n- *Voice Engine* — описание\n- *Eleven V3* — модель"


def test_plain_fallback_does_not_show_markdown_markers() -> None:
    source = "*Жирный* и `код`, [ссылка](https://example.com)"

    assert strip_markdown(source) == "Жирный и код, ссылка: https://example.com"
