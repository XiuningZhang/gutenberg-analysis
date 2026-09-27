"""Unit tests for word extraction and counting functions."""

from bookstats.counts import extract_words


def test_extract_words_normalizes_text():
    """Test that extract_words normalizes text to lowercase and removes punctuation."""
    assert extract_words("Hello, HELLO! World?") == [
        "hello",
        "hello",
        "world",
    ]
