"""Descriptive Zipf's law analysis and log-log linear fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import linregress


@dataclass(frozen=True)
class ZipfFitResult:
    """Results from a descriptive log-log linear fit.

    Attributes
    ----------
    slope : float
        Slope of the descriptive log-log linear fit.
    intercept : float
        Intercept of the descriptive log-log linear fit.
    r_squared : float
        Coefficient of determination (R^2).
    data : polars.DataFrame
        DataFrame with rank, count, log_rank, log_count, fitted values.
    """

    slope: float
    intercept: float
    r_squared: float
    data: pl.DataFrame


def compute_zipf_fit(counts_df: pl.DataFrame) -> ZipfFitResult:
    """Compute ranks, log transforms, and descriptive log-log linear fit.

    Parameters
    ----------
    counts_df : polars.DataFrame
        DataFrame containing at least 'count' (and optionally 'word').

    Returns
    -------
    ZipfFitResult
        Fit metrics and transformed DataFrame.
    """
    if len(counts_df) == 0:
        empty_df = pl.DataFrame(
            schema={
                "rank": pl.UInt32,
                "count": pl.UInt32,
                "log_rank": pl.Float64,
                "log_count": pl.Float64,
                "fitted_log_count": pl.Float64,
                "fitted_count": pl.Float64,
            }
        )
        return ZipfFitResult(slope=0.0, intercept=0.0, r_squared=0.0, data=empty_df)

    sorted_df = counts_df.sort("count", descending=True)
    ranks = np.arange(1, len(sorted_df) + 1, dtype=np.float64)
    counts = sorted_df["count"].to_numpy().astype(np.float64)

    log_ranks = np.log(ranks)
    log_counts = np.log(counts)

    if len(ranks) > 1:
        res = linregress(log_ranks, log_counts)
        slope = float(res.slope)
        intercept = float(res.intercept)
        r_squared = float(res.rvalue**2)
    else:
        slope = 0.0
        intercept = float(log_counts[0]) if len(log_counts) > 0 else 0.0
        r_squared = 1.0

    fitted_log_counts = slope * log_ranks + intercept
    fitted_counts = np.exp(fitted_log_counts)

    result_df = sorted_df.with_columns(
        [
            pl.Series("rank", ranks.astype(np.uint32)),
            pl.Series("log_rank", log_ranks),
            pl.Series("log_count", log_counts),
            pl.Series("fitted_log_count", fitted_log_counts),
            pl.Series("fitted_count", fitted_counts),
        ]
    )

    return ZipfFitResult(slope=slope, intercept=intercept, r_squared=r_squared, data=result_df)
