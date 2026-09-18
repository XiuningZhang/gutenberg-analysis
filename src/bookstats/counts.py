"""Word extraction and frequency counting functions."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import polars as pl


def strip_gutenberg_headers(text: str) -> str:
    """Strip Project Gutenberg header and footer licenses from text.

    Parameters
    ----------
    text : str
        Raw text content of a Project Gutenberg book.

    Returns
    -------
    str
        Text content with license headers and footers removed.
    """
    start_match = re.search(
        r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", text
    )
    if start_match:
        text = text[start_match.end() :]
    end_match = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK", text)
    if end_match:
        text = text[: end_match.start()]
    return text


def extract_words(text: str) -> list[str]:
    """Extract and normalize lowercase words from text.

    Parameters
    ----------
    text : str
        Input text to extract words from.

    Returns
    -------
    list of str
        List of lowercased word tokens with punctuation removed.
    """
    cleaned = strip_gutenberg_headers(text)
    return re.findall(r"\b[a-zA-Z]+\b", cleaned.lower())


def count_words(words: list[str]) -> pl.DataFrame:
    """Count occurrences of each word and sort by frequency descending.

    Parameters
    ----------
    words : list of str
        List of normalized words.

    Returns
    -------
    polars.DataFrame
        DataFrame with columns 'word' and 'count', ordered from most
        frequent to least frequent.
    """
    if not words:
        return pl.DataFrame(
            {"word": [], "count": []},
            schema={"word": pl.String, "count": pl.UInt32},
        )
    df = pl.DataFrame({"word": words})
    return (
        df.group_by("word").agg(pl.len().alias("count")).sort("count", descending=True)
    )


def process_book_file(input_path: Path | str, output_path: Path | str) -> pl.DataFrame:
    """Process a single book text file and save word counts to CSV.

    Parameters
    ----------
    input_path : Path or str
        Path to the raw text input file.
    output_path : Path or str
        Destination path for the intermediate count CSV.

    Returns
    -------
    polars.DataFrame
        DataFrame of word counts that was written to disk.
    """
    in_p = Path(input_path)
    out_p = Path(output_path)
    text = in_p.read_text(encoding="utf-8")
    words = extract_words(text)
    counts = count_words(words)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    counts.write_csv(out_p)
    return counts


def main() -> None:
    """Command-line interface for word counting."""
    parser = argparse.ArgumentParser(
        description="Count word frequencies in Project Gutenberg books."
    )
    parser.add_argument(
        "input",
        help="Input text file path.",
    )
    parser.add_argument(
        "output",
        help="Output CSV file path.",
    )

    args = parser.parse_args()
    process_book_file(args.input, args.output)
    print(f"Processed {args.input} -> {args.output}")


if __name__ == "__main__":
    main()