"""Weight of Evidence (WoE) binning and Information Value (IV).

Implements Methodology Sections 3 and 4 (docs/methodology.md):

* Numeric features are split into quantile bins, which are then merged until
  (1) the default rate moves in one direction only (monotonic WoE), and
  (2) every bin holds at least ``min_share`` of loans and both good and bad loans.
* Categorical features are coarse-classed: categories are sorted by default rate
  and adjacent ones merged until every group satisfies condition (2).
* Missing values get their own bin when there are enough of them; otherwise they
  join the bin with the closest default rate.
* WoE is computed exactly, ``ln(share of goods / share of bads)``. A bin with no
  good or no bad loans raises an error instead of being silently adjusted.

Everything is fitted on the train set only; ``FeatureBinning.transform`` then
applies the stored bins and WoE values to any other data (test set, new applicants).

Target convention: ``y = 1`` for a bad loan (default), ``y = 0`` for a good loan.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MISSING = "Missing"


# --------------------------------------------------------------------------- WoE / IV
def woe_table(bins: pd.Series, y: pd.Series, order: list[str] | None = None) -> pd.DataFrame:
    """WoE table for already-binned data (Methodology Sections 3.1 and 4.1).

    Parameters
    ----------
    bins : bin label of each loan.
    y : target, 1 = bad, 0 = good.
    order : optional display order of the bins.

    Returns one row per bin with counts, shares, default rate, WoE and the bin's
    contribution to IV. The total IV is ``table["iv"].sum()``.
    """
    y = pd.Series(np.asarray(y, dtype=float), index=bins.index)
    grouped = pd.DataFrame({"bin": bins, "bad": y}).groupby("bin", observed=True)["bad"]
    table = pd.DataFrame({"loans": grouped.size(), "bad": grouped.sum()})
    table["good"] = table["loans"] - table["bad"]
    if order is not None:
        table = table.reindex([b for b in order if b in table.index])

    empty = table[(table["good"] == 0) | (table["bad"] == 0)]
    if len(empty):
        raise ValueError(
            "WoE is undefined for bins with no good or no bad loans: "
            f"{list(empty.index)}. Merge these bins first (Methodology Section 3.1)."
        )

    n_good, n_bad = table["good"].sum(), table["bad"].sum()
    table["share_loans"] = table["loans"] / table["loans"].sum()
    table["default_rate"] = table["bad"] / table["loans"]
    table["dist_good"] = table["good"] / n_good          # D^G_k
    table["dist_bad"] = table["bad"] / n_bad             # D^B_k
    table["woe"] = np.log(table["dist_good"] / table["dist_bad"])
    table["iv"] = (table["dist_good"] - table["dist_bad"]) * table["woe"]
    table.index.name = "bin"
    return table[["loans", "share_loans", "good", "bad", "default_rate",
                  "dist_good", "dist_bad", "woe", "iv"]]


def information_value(table: pd.DataFrame) -> float:
    """Information Value of a feature from its WoE table (Methodology Section 4.1)."""
    return float(table["iv"].sum())


# --------------------------------------------------------------------------- binning result
@dataclass
class FeatureBinning:
    """Bins and WoE values for one feature, learned on the train set."""

    name: str
    kind: str                                   # "numeric" or "categorical"
    labels: list[str]                           # bin labels in display order
    table: pd.DataFrame                         # WoE table on the train set
    edges: list[float] = field(default_factory=list)          # numeric: inner bin edges
    category_to_bin: dict[str, str] = field(default_factory=dict)  # categorical mapping
    missing_bin: str = ""                       # where missing values go
    fallback_bin: str = ""                      # where unseen categories go

    @property
    def iv(self) -> float:
        return information_value(self.table)

    @property
    def woe(self) -> dict[str, float]:
        return self.table["woe"].to_dict()

    def bin_of(self, x: pd.Series) -> pd.Series:
        """Bin label for each value of ``x``."""
        if self.kind == "numeric":
            values = pd.to_numeric(x, errors="coerce")
            idx = np.searchsorted(np.asarray(self.edges, dtype=float), values.to_numpy(), side="left")
            regular = [lab for lab in self.labels if lab != MISSING]
            out = pd.Series(np.asarray(regular, dtype=object)[np.minimum(idx, len(regular) - 1)], index=x.index)
            out[values.isna()] = self.missing_bin
            return out
        cats = x.astype("string").fillna(MISSING).astype(object)
        return cats.map(self.category_to_bin).fillna(self.fallback_bin)

    def transform(self, x: pd.Series) -> pd.Series:
        """Replace each value of ``x`` by the WoE of its bin (as learned on train)."""
        out = self.bin_of(x).map(self.woe)
        if out.isna().any():
            raise ValueError(f"{self.name}: some values could not be assigned a WoE")
        return out.astype(float).rename(self.name)

    def to_dict(self) -> dict:
        """Plain-Python description, for saving to JSON."""
        return {
            "name": self.name, "kind": self.kind, "labels": self.labels,
            "edges": [float(e) for e in self.edges], "category_to_bin": self.category_to_bin,
            "missing_bin": self.missing_bin, "fallback_bin": self.fallback_bin,
            "woe": {k: float(v) for k, v in self.woe.items()}, "iv": self.iv,
        }


# --------------------------------------------------------------------------- helpers
def _fmt(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}"


def _numeric_labels(edges: list[float]) -> list[str]:
    """Labels for right-closed bins (lo, hi] defined by inner edges."""
    if not edges:
        return ["all values"]
    labels = [f"≤ {_fmt(edges[0])}"]
    labels += [f"{_fmt(lo)} – {_fmt(hi)}" for lo, hi in zip(edges[:-1], edges[1:])]
    labels.append(f"> {_fmt(edges[-1])}")
    return labels


def _numeric_stats(x: np.ndarray, y: np.ndarray, edges: list[float]) -> tuple[np.ndarray, np.ndarray]:
    idx = np.searchsorted(np.asarray(edges, dtype=float), x, side="left")
    n = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    bad = np.bincount(idx, weights=y, minlength=len(edges) + 1)
    return n, bad


def _too_small_or_pure(n: np.ndarray, bad: np.ndarray, min_count: float) -> np.ndarray:
    return (n < min_count) | (bad == 0) | (bad == n)


def _neighbour_to_merge(i: int, rate: np.ndarray) -> int:
    """Index of the neighbour of bin i with the closest default rate."""
    if i == 0:
        return 1
    if i == len(rate) - 1:
        return i - 1
    return i - 1 if abs(rate[i] - rate[i - 1]) <= abs(rate[i] - rate[i + 1]) else i + 1


def _group_label(names: list[str]) -> str:
    names = sorted(names)
    return ", ".join(names) if len(names) <= 4 else f"{', '.join(names[:3])} + {len(names) - 3} more"


# --------------------------------------------------------------------------- fitting
def fit_numeric(x: pd.Series, y: pd.Series, name: str | None = None, *, min_share: float = 0.05,
                n_initial: int = 20, min_missing_share: float = 0.005) -> FeatureBinning:
    """Monotonic binning of a numeric feature (Methodology Sections 3.1 and 3.4)."""
    name = name or str(x.name)
    x = pd.to_numeric(x, errors="coerce")
    y = pd.Series(np.asarray(y, dtype=float), index=x.index)
    missing = x.isna()
    xv, yv = x[~missing].to_numpy(dtype=float), y[~missing].to_numpy()
    n_total = len(x)
    min_count = min_share * n_total

    # 1. start from quantile bins
    edges = list(np.unique(np.quantile(xv, np.linspace(0, 1, n_initial + 1)[1:-1])))
    if edges and edges[-1] >= xv.max():          # an edge at the maximum would leave an empty top bin
        edges = edges[:-1]

    # 2. direction of the relationship: does the default rate rise or fall with x?
    rho = pd.Series(xv).corr(pd.Series(yv), method="spearman")
    direction = 1.0 if (np.isnan(rho) or rho >= 0) else -1.0

    # 3. merge adjacent bins until the default rate is monotonic in that direction
    while edges:
        n, bad = _numeric_stats(xv, yv, edges)
        rate = np.divide(bad, n, out=np.zeros_like(bad), where=n > 0)
        violations = [i for i in range(len(rate) - 1)
                      if n[i] == 0 or n[i + 1] == 0 or (rate[i + 1] - rate[i]) * direction < 0]
        if not violations:
            break
        del edges[violations[0]]                   # merge bin i with bin i+1

    # 4. merge bins that are too small or contain only goods or only bads
    #    (merging neighbours keeps the default rate monotonic)
    while edges:
        n, bad = _numeric_stats(xv, yv, edges)
        problem = _too_small_or_pure(n, bad, min_count)
        if not problem.any():
            break
        rate = bad / n
        i = int(np.argmin(np.where(problem, n, np.inf)))
        j = _neighbour_to_merge(i, rate)
        del edges[min(i, j)]

    labels = _numeric_labels(edges)
    bins = pd.Series(np.asarray(labels, dtype=object)[np.searchsorted(edges, x.fillna(0).to_numpy(), side="left")],
                     index=x.index)

    # 5. missing values: own bin if there are enough of them, else join the closest bin
    n_miss, bad_miss = float(missing.sum()), float(y[missing].sum())
    regular = woe_table(bins[~missing], y[~missing], order=labels)
    if n_miss >= max(min_missing_share * n_total, 1) and 0 < bad_miss < n_miss:
        missing_bin = MISSING
        labels = labels + [MISSING]
    else:
        target_rate = bad_miss / n_miss if n_miss else y.mean()
        missing_bin = str((regular["default_rate"] - target_rate).abs().idxmin())
    bins[missing] = missing_bin

    table = woe_table(bins, y, order=labels)
    return FeatureBinning(name=name, kind="numeric", labels=labels, table=table, edges=[float(e) for e in edges],
                          missing_bin=missing_bin,
                          fallback_bin=str(table["woe"].abs().idxmin()))


def fit_categorical(x: pd.Series, y: pd.Series, name: str | None = None, *,
                    min_share: float = 0.05) -> FeatureBinning:
    """Coarse classing of a categorical feature (Methodology Section 3.1).

    Categories are sorted by default rate; the smallest group that is too small,
    or has only goods or only bads, is merged with its neighbour of closest default
    rate, until every group is large enough and contains both goods and bads.
    """
    name = name or str(x.name)
    cats = x.astype("string").fillna(MISSING).astype(object)
    y = pd.Series(np.asarray(y, dtype=float), index=x.index)
    stats = pd.DataFrame({"c": cats, "y": y}).groupby("c")["y"].agg(["size", "sum"])
    stats = stats.assign(rate=stats["sum"] / stats["size"]).sort_values("rate")

    groups = [[c] for c in stats.index]
    n = stats["size"].to_numpy(dtype=float).copy()
    bad = stats["sum"].to_numpy(dtype=float).copy()
    min_count = min_share * len(x)
    while len(groups) > 1:
        problem = _too_small_or_pure(n, bad, min_count)
        if not problem.any():
            break
        i = int(np.argmin(np.where(problem, n, np.inf)))
        j = _neighbour_to_merge(i, bad / n)
        a, b = sorted((i, j))
        groups[a] = groups[a] + groups[b]
        n[a], bad[a] = n[a] + n[b], bad[a] + bad[b]
        del groups[b]
        n, bad = np.delete(n, b), np.delete(bad, b)

    labels = [_group_label(g) for g in groups]
    category_to_bin = {c: lab for g, lab in zip(groups, labels) for c in g}
    table = woe_table(cats.map(category_to_bin), y, order=labels)
    fallback = category_to_bin.get(MISSING, str(table["woe"].abs().idxmin()))
    return FeatureBinning(name=name, kind="categorical", labels=labels, table=table,
                          category_to_bin=category_to_bin, missing_bin=category_to_bin.get(MISSING, fallback),
                          fallback_bin=fallback)


def fit_binning(x: pd.Series, y: pd.Series, name: str | None = None, **kwargs) -> FeatureBinning:
    """Fit numeric or categorical binning depending on the column's type."""
    if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
        return fit_numeric(x, y, name, **kwargs)
    kwargs.pop("n_initial", None)
    kwargs.pop("min_missing_share", None)
    return fit_categorical(x, y, name, **kwargs)
