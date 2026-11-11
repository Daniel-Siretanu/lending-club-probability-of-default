"""Tests for probdefaultmodel.woe, checked against Methodology Sections 3 and 4."""
import numpy as np
import pandas as pd
import pytest

from probdefaultmodel.woe import MISSING, fit_categorical, fit_numeric, information_value, woe_table


def _loans_from_counts(counts):
    """Build (bin, y) data from {bin: (goods, bads)}."""
    bins, y = [], []
    for label, (good, bad) in counts.items():
        bins += [label] * (good + bad)
        y += [0] * good + [1] * bad
    return pd.Series(bins), pd.Series(y)


def test_woe_and_iv_match_methodology_worked_example():
    # Methodology Section 3.1: 85,000 goods and 15,000 bads in four DTI bins
    counts = {"0-10": (25_000, 3_000), "10-18": (25_000, 4_000),
              "18-25": (20_000, 4_000), "25+": (15_000, 4_000)}
    bins, y = _loans_from_counts(counts)
    table = woe_table(bins, y, order=list(counts))

    assert table["woe"].round(2).tolist() == [0.39, 0.10, -0.13, -0.41]
    assert information_value(table) == pytest.approx(0.0802, abs=1e-4)
    # shares of goods and of bads each sum to 1
    assert table["dist_good"].sum() == pytest.approx(1)
    assert table["dist_bad"].sum() == pytest.approx(1)


def test_woe_relation_to_odds():
    # odds of good in a bin = overall odds x exp(WoE)   (Methodology Section 3.2)
    counts = {"a": (900, 100), "b": (700, 300)}
    bins, y = _loans_from_counts(counts)
    table = woe_table(bins, y)
    overall_odds = 1600 / 400
    for label, (good, bad) in counts.items():
        assert good / bad == pytest.approx(overall_odds * np.exp(table.loc[label, "woe"]))


def test_bin_without_bads_raises():
    bins, y = _loans_from_counts({"a": (100, 10), "b": (50, 0)})
    with pytest.raises(ValueError, match="no good or no bad"):
        woe_table(bins, y)


def test_iv_is_zero_when_feature_is_uninformative():
    bins, y = _loans_from_counts({"a": (800, 200), "b": (400, 100)})
    assert information_value(woe_table(bins, y)) == pytest.approx(0, abs=1e-12)


@pytest.fixture
def synthetic():
    """Default probability rising with x, plus some missing values."""
    rng = np.random.default_rng(0)
    n = 50_000
    x = rng.uniform(0, 40, n)
    y = rng.random(n) < 0.05 + 0.006 * x
    x = pd.Series(x)
    x[rng.random(n) < 0.03] = np.nan
    return x, pd.Series(y.astype(int))


def test_numeric_bins_are_monotonic_and_large_enough(synthetic):
    x, y = synthetic
    binning = fit_numeric(x, y, "x", min_share=0.05)
    regular = binning.table.drop(index=MISSING, errors="ignore")

    rates = regular["default_rate"].to_numpy()
    assert np.all(np.diff(rates) >= 0), "default rate should rise with x"
    assert (regular["share_loans"] >= 0.05).all()
    assert len(regular) >= 3
    assert binning.iv > 0


def test_missing_values_get_their_own_bin(synthetic):
    x, y = synthetic
    binning = fit_numeric(x, y, "x")
    assert MISSING in binning.table.index
    assert binning.transform(pd.Series([np.nan])).iloc[0] == pytest.approx(binning.woe[MISSING])


def test_transform_uses_train_woe_on_new_data(synthetic):
    x, y = synthetic
    binning = fit_numeric(x, y, "x")
    new = pd.Series([-5.0, 0.5, 20.0, 39.0, 1_000.0])        # includes values outside the train range
    woe = binning.transform(new)
    assert woe.notna().all()
    assert woe.iloc[0] == woe.iloc[1]                         # both fall in the lowest bin
    assert woe.iloc[-1] == woe.iloc[-2]                       # both fall in the highest bin
    assert woe.iloc[0] > woe.iloc[-1]                         # low x = fewer defaults = higher WoE


def test_categorical_groups_small_categories_and_handles_unseen():
    rng = np.random.default_rng(1)
    cats = rng.choice(["A", "B", "C", "D", "rare1", "rare2"], size=20_000,
                      p=[0.4, 0.3, 0.2, 0.08, 0.01, 0.01])
    p = pd.Series(cats).map({"A": 0.05, "B": 0.10, "C": 0.20, "D": 0.30, "rare1": 0.25, "rare2": 0.06})
    y = pd.Series((rng.random(20_000) < p).astype(int))
    binning = fit_categorical(pd.Series(cats), y, "cat", min_share=0.05)

    assert (binning.table["share_loans"] >= 0.05).all()
    assert binning.category_to_bin["rare1"] != "rare1"        # merged into a bigger group
    unseen = binning.transform(pd.Series(["never seen"]))
    assert unseen.notna().all()
