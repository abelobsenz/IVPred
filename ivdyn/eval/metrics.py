"""Evaluation metrics helpers."""

from __future__ import annotations

import math
import numpy as np


def sample_skew_kurtosis(x: np.ndarray) -> tuple[float, float]:
    """Return (skew, excess_kurtosis) for a 1D series."""
    a = np.asarray(x, dtype=float).reshape(-1)
    a = a[np.isfinite(a)]
    n = int(a.size)
    if n < 4:
        return float("nan"), float("nan")
    m = float(a.mean())
    v = float(np.mean((a - m) ** 2))
    if v <= 1e-18:
        return 0.0, 0.0
    s = float(np.mean((a - m) ** 3)) / (v ** 1.5)
    k = float(np.mean((a - m) ** 4)) / (v ** 2) - 3.0
    return s, k


def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    *,
    benchmark_sharpe_ann: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012).

    Returns probability that true Sharpe exceeds benchmark.
    Uses a normal approximation with skew/kurtosis adjustment.
    """
    r = np.asarray(returns, dtype=float).reshape(-1)
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n < 10:
        return float("nan")
    mu = float(r.mean())
    sigma = float(r.std(ddof=0))
    if sigma <= 1e-18:
        return 0.0
    sr_sample = (mu / sigma) * float(np.sqrt(periods_per_year))
    skew, ex_kurt = sample_skew_kurtosis(r)
    # Variance of SR estimate with non-normality correction.
    denom = 1.0 - skew * sr_sample + ((ex_kurt + 2.0) / 4.0) * (sr_sample**2)
    if not np.isfinite(denom) or denom <= 1e-12:
        denom = 1.0
    sr_std = float(np.sqrt(max(1.0 / (n - 1), 0.0) * denom))
    if sr_std <= 1e-18:
        return 0.0
    z = (sr_sample - float(benchmark_sharpe_ann)) / sr_std
    # Standard normal CDF.
    return float(0.5 * (1.0 + math.erf(z / np.sqrt(2.0))))


def _residual(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    p = np.asarray(pred, dtype=float)
    t = np.asarray(target, dtype=float)
    err = p - t
    return err[np.isfinite(err)]


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    err = _residual(pred, target)
    if err.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(err * err)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    err = _residual(pred, target)
    if err.size == 0:
        return float("nan")
    return float(np.mean(np.abs(err)))


def r2(pred: np.ndarray, target: np.ndarray) -> float:
    y = np.asarray(target)
    yhat = np.asarray(pred)
    denom = np.sum((y - y.mean()) ** 2)
    if denom <= 1e-12:
        return float("nan")
    num = np.sum((y - yhat) ** 2)
    return float(1.0 - num / denom)


def directional_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    p = np.asarray(pred)
    t = np.asarray(target)
    if p.size == 0:
        return float("nan")
    return float(np.mean(np.sign(p) == np.sign(t)))


def brier_score(prob: np.ndarray, target: np.ndarray) -> float:
    p = np.clip(np.asarray(prob), 1e-6, 1.0 - 1e-6)
    y = np.asarray(target)
    return float(np.mean((p - y) ** 2))
