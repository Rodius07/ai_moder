import base64

from tg_guard_bot.image_generation import decode_data_url


def test_decode_data_url_returns_bytes_and_filename() -> None:
    payload = base64.b64encode(b"png-bytes").decode()

    image_bytes, filename = decode_data_url(f"data:image/png;base64,{payload}")

    assert image_bytes == b"png-bytes"
    assert filename == "image.png"
