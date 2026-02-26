"""Black-Scholes helpers used for diagnostics and fallback IV estimation."""

from __future__ import annotations

import math

import numpy as np


def _norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2.0)))


def bs_price(
    spot: np.ndarray,
    strike: np.ndarray,
    tau: np.ndarray,
    vol: np.ndarray,
    cp_sign: np.ndarray,
    rate: float = 0.0,
    dividend: float = 0.0,
) -> np.ndarray:
    """Vectorized Black-Scholes price for calls (cp_sign=1) and puts (cp_sign=-1)."""
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    tau = np.asarray(tau, dtype=float)
    vol = np.asarray(vol, dtype=float)
    cp_sign = np.asarray(cp_sign, dtype=float)

    eps = 1e-12
    tau = np.maximum(tau, eps)
    vol = np.maximum(vol, 1e-6)

    sqrt_tau = np.sqrt(tau)
    d1 = (
        np.log(np.maximum(spot, eps) / np.maximum(strike, eps))
        + (rate - dividend + 0.5 * vol * vol) * tau
    ) / (vol * sqrt_tau)
    d2 = d1 - vol * sqrt_tau

    nd1 = _norm_cdf(cp_sign * d1)
    nd2 = _norm_cdf(cp_sign * d2)

    disc_q = np.exp(-dividend * tau)
    disc_r = np.exp(-rate * tau)

    return cp_sign * (spot * disc_q * nd1 - strike * disc_r * nd2)


def bs_delta(
    spot: np.ndarray,
    strike: np.ndarray,
    tau: np.ndarray,
    vol: np.ndarray,
    cp_sign: np.ndarray,
    rate: float = 0.0,
    dividend: float = 0.0,
) -> np.ndarray:
    """Vectorized Black-Scholes delta for calls (cp_sign=1) and puts (cp_sign=-1)."""
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    tau = np.asarray(tau, dtype=float)
    vol = np.asarray(vol, dtype=float)
    cp_sign = np.asarray(cp_sign, dtype=float)

    eps = 1e-12
    tau = np.maximum(tau, eps)
    vol = np.maximum(vol, 1e-6)

    sqrt_tau = np.sqrt(tau)
    d1 = (
        np.log(np.maximum(spot, eps) / np.maximum(strike, eps))
        + (rate - dividend + 0.5 * vol * vol) * tau
    ) / (vol * sqrt_tau)

    disc_q = np.exp(-dividend * tau)
    return disc_q * cp_sign * _norm_cdf(cp_sign * d1)


def implied_vol_bisection(
    price: float,
    spot: float,
    strike: float,
    tau: float,
    cp_sign: int,
    rate: float = 0.0,
    dividend: float = 0.0,
    vol_low: float = 1e-4,
    vol_high: float = 5.0,
    max_iter: int = 80,
) -> float:
    """Robust scalar implied-vol solver used only as fallback for missing IVs."""
    if price <= 0 or spot <= 0 or strike <= 0 or tau <= 0:
        return float("nan")

    low = vol_low
    high = vol_high

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        model = float(
            bs_price(
                np.array([spot]),
                np.array([strike]),
                np.array([tau]),
                np.array([mid]),
                np.array([cp_sign]),
                rate=rate,
                dividend=dividend,
            )[0]
        )
        if model > price:
            high = mid
        else:
            low = mid

    return 0.5 * (low + high)
