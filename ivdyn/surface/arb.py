"""No-arbitrage diagnostics and penalties for IV surfaces."""

from __future__ import annotations

import numpy as np

from ivdyn.finance import bs_price


def calendar_violations(iv_surface: np.ndarray, tenor_days: np.ndarray) -> np.ndarray:
    """Return per-sample calendar-violation rate."""
    tau = np.asarray(tenor_days, dtype=float) / 365.0
    w = iv_surface * iv_surface * tau.reshape(1, 1, -1)
    diff = w[:, :, :-1] - w[:, :, 1:]
    viol = np.maximum(diff, 0.0)
    denom = max(1, diff.shape[1] * diff.shape[2])
    return (viol > 1e-10).sum(axis=(1, 2)) / denom


def butterfly_violations(iv_surface: np.ndarray, x_grid: np.ndarray, tenor_days: np.ndarray) -> np.ndarray:
    """Approximate butterfly violations via convexity of call prices in strike."""
    batch, nx, nt = iv_surface.shape
    strike = np.exp(x_grid)
    cp = np.ones(nx)
    rates = np.zeros(nx)
    divs = np.zeros(nx)
    viol_rate = np.zeros(batch)

    for b in range(batch):
        local_viol = 0
        local_total = 0
        for j, dte in enumerate(tenor_days):
            tau = max(float(dte) / 365.0, 1e-5)
            vol = np.clip(iv_surface[b, :, j], 1e-4, 4.0)
            call = bs_price(
                spot=np.ones(nx),
                strike=strike,
                tau=np.full(nx, tau),
                vol=vol,
                cp_sign=cp,
                rate=float(rates[0]),
                dividend=float(divs[0]),
            )
            second = call[:-2] - 2.0 * call[1:-1] + call[2:]
            local_viol += int((second < -1e-8).sum())
            local_total += second.size
        viol_rate[b] = local_viol / max(1, local_total)
    return viol_rate


def repair_calendar_monotonic(iv_surface: np.ndarray, tenor_days: np.ndarray) -> np.ndarray:
    """Project total variance onto non-decreasing tenor profile."""
    tau = np.asarray(tenor_days, dtype=float) / 365.0
    out = np.clip(iv_surface, 1e-4, 4.0).copy()
    w = out * out * tau.reshape(1, -1)
    w_proj = np.maximum.accumulate(w, axis=1)
    out = np.sqrt(np.maximum(w_proj / tau.reshape(1, -1), 1e-8))
    return np.clip(out, 1e-4, 4.0)


def calendar_penalty_and_grad(iv_surface: np.ndarray, tenor_days: np.ndarray) -> tuple[float, np.ndarray]:
    """Differentiable calendar penalty and gradient wrt IV surface."""
    tau = np.asarray(tenor_days, dtype=float) / 365.0
    tau_grid = tau.reshape(1, 1, -1)

    w = iv_surface * iv_surface * tau_grid
    diff = w[:, :, :-1] - w[:, :, 1:]
    viol = np.maximum(diff, 0.0)

    n = max(1, viol.size)
    loss = float(np.sum(viol * viol) / n)

    grad_w = np.zeros_like(w)
    scale = 2.0 / n
    pos = diff > 0
    grad_term = scale * viol * pos
    grad_w[:, :, :-1] += grad_term
    grad_w[:, :, 1:] -= grad_term

    grad_iv = grad_w * (2.0 * iv_surface * tau_grid)
    return loss, grad_iv


def summarize_noarb(iv_surface: np.ndarray, x_grid: np.ndarray, tenor_days: np.ndarray) -> dict[str, float]:
    batch = iv_surface.shape[0]
    cal = calendar_violations(iv_surface, tenor_days)
    bfly = butterfly_violations(iv_surface, x_grid, tenor_days)
    return {
        "calendar_violation_rate_mean": float(np.mean(cal)),
        "calendar_violation_rate_p95": float(np.quantile(cal, 0.95)),
        "butterfly_violation_rate_mean": float(np.mean(bfly)),
        "butterfly_violation_rate_p95": float(np.quantile(bfly, 0.95)),
        "samples": int(batch),
    }
