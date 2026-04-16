from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.stats import pearsonr


class Fenwick:
    def __init__(self, n: int):
        self.n = n
        self.tree = np.zeros(n + 1, dtype=np.int64)

    def add(self, i: int, delta: int = 1) -> None:
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def sum(self, i: int) -> int:
        s = 0
        while i > 0:
            s += int(self.tree[i])
            i -= i & -i
        return s


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Concordance index for continuous labels with ties in y_true excluded and ties in y_pred counted as 0.5.

    Uses an O(n log n) algorithm via a Fenwick tree after sorting by y_true.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = y_true.shape[0]
    if n < 2:
        return float("nan")

    order = np.argsort(y_true, kind="mergesort")
    y_true = y_true[order]
    y_pred = y_pred[order]

    uniq_pred = np.unique(y_pred)
    ranks = np.searchsorted(uniq_pred, y_pred) + 1  # 1-based
    bit = Fenwick(len(uniq_pred))

    concordant = 0.0
    comparable = 0

    total_prev = 0
    start = 0
    while start < n:
        end = start
        while end < n and y_true[end] == y_true[start]:
            end += 1

        # Evaluate current group against all previous (strictly smaller y_true)
        for r in ranks[start:end]:
            less = bit.sum(int(r) - 1)
            leq = bit.sum(int(r))
            equal = leq - less
            concordant += float(less) + 0.5 * float(equal)
            comparable += total_prev

        # Add current group to BIT
        for r in ranks[start:end]:
            bit.add(int(r), 1)
            total_prev += 1

        start = end

    return float(concordant / comparable) if comparable > 0 else float("nan")


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.shape[0] < 2:
        return float("nan")
    r, _ = pearsonr(y_true, y_pred)
    return float(r)


def bootstrap_pearson_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> Tuple[float, Optional[tuple[float, float]]]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = y_true.shape[0]
    if n < 2:
        return float("nan"), None

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    rs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        yt = y_true[idx[i]]
        yp = y_pred[idx[i]]
        if np.std(yt) == 0 or np.std(yp) == 0:
            rs[i] = np.nan
        else:
            rs[i] = pearson_r(yt, yp)

    rs = rs[np.isfinite(rs)]
    if rs.size == 0:
        return pearson_r(y_true, y_pred), None
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return pearson_r(y_true, y_pred), (float(lo), float(hi))

