"""Visualize the Gutenberg book analysis results."""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from bookstats.zipf import compute_zipf_fit

    return Path, alt, compute_zipf_fit, mo, pl


@app.cell
def _(mo):
    mo.md(r"""
	# Book Word Frequency Analysis

	An interactive visualization of word frequency distributions across Project Gutenberg books.
	""")
    return  # noqa: PLR1711 -- Marimo cell boundary.


@app.cell
def _(Path, pl):
    # Load processed book counts
    processed_path = Path("data/processed/book-counts.csv")
    if processed_path.exists():
        counts_df = pl.read_csv(processed_path)
    else:
        # Fallback to intermediate counts if processed is not yet generated
        intermediate_dir = Path("data/intermediate")
        csv_files = list(intermediate_dir.glob("*.csv"))
        if csv_files:
            frames = []
            for p in csv_files:
                d = pl.read_csv(p).with_columns(pl.lit(p.stem).alias("book"))
                frames.append(d.select(["book", "word", "count"]))
            counts_df = pl.concat(frames)
        else:
            counts_df = pl.DataFrame(
                {"book": [], "word": [], "count": []},
                schema={"book": pl.String, "word": pl.String, "count": pl.UInt32},
            )
    return (counts_df,)


@app.cell
def _(counts_df, mo):
    books = sorted(counts_df["book"].unique().to_list()) if len(counts_df) > 0 else ["None"]
    book_selector = mo.ui.dropdown(
        options=books,
        value=books[0] if books else "None",
        label="Select a book:",
    )
    book_selector  # noqa: B018 -- Display the dropdown as the cell output.
    return (book_selector,)


@app.cell
def _(alt, book_selector, compute_zipf_fit, counts_df, mo, pl):
    mo.stop(book_selector.value in (None, "None"), mo.md("No books available."))

    selected_df = counts_df.filter(pl.col("book") == book_selector.value)
    mo.stop(selected_df.is_empty(), mo.md("No data for this book."))
    fit = compute_zipf_fit(selected_df)
    chart_data = alt.Data(values=fit.data.to_dicts())

    points = (
        alt.Chart(chart_data)
        .mark_circle(opacity=0.4, size=15, color="#2563eb")
        .encode(
            x=alt.X("rank:Q", scale=alt.Scale(type="log"), title="Rank (log scale)"),
            y=alt.Y("count:Q", scale=alt.Scale(type="log"), title="Frequency (log scale)"),
            tooltip=["word:N", "rank:Q", "count:Q"],
        )
    )

    line = (
        alt.Chart(chart_data)
        .mark_line(color="#dc2626", strokeWidth=2)
        .encode(
            x=alt.X("rank:Q", scale=alt.Scale(type="log")),
            y=alt.Y("fitted_count:Q", scale=alt.Scale(type="log")),
            order=alt.Order("rank:Q"),
        )
    )
    chart = (points + line).properties(
        title=f"Descriptive Zipf Fit: {book_selector.value}",
        width=600,
        height=450,
    )

    mo.vstack(
        [
            mo.md(
                f"""
			**Slope:** {fit.slope:.4f}  
			**Intercept:** {fit.intercept:.4f}  
			**R²:** {fit.r_squared:.4f}

			Blue points: observed counts. Red line: descriptive fit.
			"""
            ),
            mo.ui.altair_chart(chart),
        ]
    )
    return  # noqa: PLR1711 -- Marimo cell boundary.


if __name__ == "__main__":
    app.run()
