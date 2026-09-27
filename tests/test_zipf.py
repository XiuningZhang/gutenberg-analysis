"""Unit tests for Zipf's law fitting functions."""

import polars as pl
import pytest

from bookstats.zipf import compute_zipf_fit


def test_compute_zipf_fit_linear_decay():
    """Test that compute_zipf_fit correctly computes slope and R^2 for synthetic Zipf data."""
    # Construct synthetic data with an exact 1/rank Zipf decay
    ranks = list(range(1, 11))
    counts = [int(1000 / r) for r in ranks]
    df = pl.DataFrame({"word": [f"w{i}" for i in ranks], "count": counts})

    fit = compute_zipf_fit(df)

    # Slope should be approximately -1.0 with high R^2
    assert pytest.approx(fit.slope, rel=0.1) == -1.0
    assert fit.r_squared > 0.95
    assert "rank" in fit.data.columns
    assert "log_rank" in fit.data.columns
    assert "log_count" in fit.data.columns
    assert "fitted_log_count" in fit.data.columns
    assert "fitted_count" in fit.data.columns


def test_compute_zipf_fit_empty():
    """Test that compute_zipf_fit handles an empty DataFrame gracefully."""
    df = pl.DataFrame({"word": [], "count": []}, schema={"word": pl.String, "count": pl.UInt32})
    fit = compute_zipf_fit(df)
    assert fit.slope == 0.0
    assert fit.intercept == 0.0
    assert fit.r_squared == 0.0
    assert len(fit.data) == 0
