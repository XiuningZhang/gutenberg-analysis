"""bookstats: A reproducible analysis package for Gutenberg word frequencies."""

from bookstats.counts import (
    count_words,
    extract_words,
    process_book_file,
    strip_gutenberg_headers,
)

__all__ = [
    "strip_gutenberg_headers",
    "extract_words",
    "count_words",
    "process_book_file",
]