"""Surface construction from raw option chains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ivdyn.finance import implied_vol_bisection
from ivdyn.surface.arb import repair_calendar_monotonic


@dataclass(slots=True)
class SurfaceConfig:
    x_grid: np.ndarray
    tenor_days: np.ndarray
    max_neighbors: int = 18
    max_dte_distance: int = 25
    max_relative_spread: float = 1.25


def _fill_curve(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = y.copy()
    mask = np.isfinite(out)
    if mask.sum() == 0:
        return np.full_like(out, np.nan)
    if mask.sum() == 1:
        return np.full_like(out, float(out[mask][0]))
    out[~mask] = np.interp(x[~mask], x[mask], out[mask])
    return out


def fill_surface(surface: np.ndarray, x_grid: np.ndarray, tenor_days: np.ndarray) -> np.ndarray:
    out = surface.copy()

    for j in range(out.shape[1]):
        out[:, j] = _fill_curve(x_grid, out[:, j])

    for i in range(out.shape[0]):
        out[i, :] = _fill_curve(tenor_days.astype(float), out[i, :])

    global_median = float(np.nanmedian(out)) if np.isfinite(out).any() else 0.25
    out = np.where(np.isfinite(out), out, global_median)
    return out


def enrich_chain(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    bid = pd.to_numeric(out.get("bid"), errors="coerce")
    ask = pd.to_numeric(out.get("ask"), errors="coerce")
    out["spread"] = (ask - bid).clip(lower=0.0)
    rel_spread = out["spread"] / pd.to_numeric(out.get("mid"), errors="coerce").clip(lower=1e-6)
    rel_spread = pd.to_numeric(rel_spread, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if rel_spread.isna().all():
        # Flatfile day aggs typically lack bid/ask; use a conservative volume-based proxy.
        vol = pd.to_numeric(out.get("volume"), errors="coerce").fillna(0.0).clip(lower=0.0)
        rel_spread = (0.30 / np.sqrt(1.0 + vol)).clip(lower=0.01, upper=1.0)
    else:
        rel_spread = rel_spread.fillna(float(rel_spread.dropna().median()))
    out["rel_spread"] = rel_spread.clip(lower=0.0, upper=5.0)

    fwd_raw = pd.to_numeric(out.get("forward_price"), errors="coerce")
    spot_raw = pd.to_numeric(out.get("underlying_close"), errors="coerce")
    fwd = np.where(np.isfinite(fwd_raw) & (fwd_raw > 0.0), fwd_raw, spot_raw)
    out["forward_price"] = pd.to_numeric(fwd, errors="coerce")
    out["x"] = np.log(out["strike"] / out["forward_price"].clip(lower=1e-6))
    out["tau"] = out["dte"] / 365.0
    out["cp_sign"] = np.where(out["call_put"] == "C", 1, -1)

    out["volume"] = out["volume"].fillna(0.0)
    out["open_interest"] = out["open_interest"].fillna(0.0)
    out["liquidity"] = (
        1.0
        + np.log1p(out["volume"].clip(lower=0.0))
        + 0.35 * np.log1p(out["open_interest"].clip(lower=0.0))
    )

    # Fallback IV filling for sparse rows.
    iv_missing = out["iv"].isna()
    if iv_missing.any():
        grouped = out.groupby(["dte", "call_put"])["iv"].transform(lambda s: s.interpolate(limit_direction="both"))
        out.loc[iv_missing, "iv"] = grouped.loc[iv_missing]

    iv_missing = out["iv"].isna()
    if iv_missing.any():
        atm_guess = float(out["iv"].median()) if out["iv"].notna().any() else 0.2
        unresolved = out.loc[iv_missing].copy()
        solved = []
        for _, row in unresolved.iterrows():
            try:
                iv = implied_vol_bisection(
                    price=float(row["mid"]),
                    spot=float(row["underlying_close"]),
                    strike=float(row["strike"]),
                    tau=float(row["tau"]),
                    cp_sign=int(row["cp_sign"]),
                )
                solved.append(iv)
            except Exception:
                solved.append(np.nan)
        out.loc[iv_missing, "iv"] = solved
        out["iv"] = out["iv"].fillna(atm_guess)

    out["iv"] = out["iv"].clip(lower=1e-4, upper=4.0)
    return out


def _grid_weighted_average(
    df: pd.DataFrame,
    x_target: float,
    dte_target: float,
    col: str,
    max_neighbors: int,
) -> float:
    if df.empty:
        return float("nan")

    d_x = (df["x"].to_numpy() - x_target) ** 2
    d_t = ((df["dte"].to_numpy() - dte_target) / max(1.0, dte_target)) ** 2
    dist = np.sqrt(d_x + 0.08 * d_t)

    order = np.argsort(dist)
    k = min(max_neighbors, len(order))
    idx = order[:k]

    vals = df[col].to_numpy()[idx]
    liq = df["liquidity"].to_numpy()[idx]

    w = (liq + 0.2) / (dist[idx] + 5e-3)
    w = np.clip(w, 1e-4, None)

    if not np.isfinite(vals).any():
        return float("nan")
    vals = np.where(np.isfinite(vals), vals, np.nanmedian(vals[np.isfinite(vals)]))
    return float(np.sum(w * vals) / np.sum(w))


def build_daily_surface(df: pd.DataFrame, cfg: SurfaceConfig) -> dict[str, np.ndarray]:
    chain = enrich_chain(df)

    chain = chain[(chain["rel_spread"] <= cfg.max_relative_spread) | chain["rel_spread"].isna()]
    if chain.empty:
        nx = cfg.x_grid.size
        nt = cfg.tenor_days.size
        nan_grid = np.full((nx, nt), np.nan, dtype=float)
        return {
            "iv_surface": nan_grid,
            "w_surface": nan_grid,
            "price_surface": nan_grid,
            "liq_surface": nan_grid,
        }

    nx = cfg.x_grid.size
    nt = cfg.tenor_days.size

    iv_grid = np.full((nx, nt), np.nan, dtype=float)
    price_grid = np.full((nx, nt), np.nan, dtype=float)
    liq_grid = np.full((nx, nt), np.nan, dtype=float)

    for j, t in enumerate(cfg.tenor_days.astype(float)):
        td = np.abs(chain["dte"] - t)
        mask_t = td <= min(cfg.max_dte_distance, max(5.0, 0.45 * t))
        sub = chain.loc[mask_t]
        if sub.empty:
            nearest = np.argsort(td.to_numpy())[: min(len(chain), 150)]
            sub = chain.iloc[nearest]

        for i, x in enumerate(cfg.x_grid.astype(float)):
            iv_grid[i, j] = _grid_weighted_average(sub, x, t, "iv", cfg.max_neighbors)
            price_grid[i, j] = _grid_weighted_average(sub, x, t, "mid", cfg.max_neighbors)
            liq_grid[i, j] = _grid_weighted_average(sub, x, t, "liquidity", cfg.max_neighbors)

    fwd_ref = float(np.nanmedian(pd.to_numeric(chain.get("forward_price"), errors="coerce")))
    if not np.isfinite(fwd_ref) or fwd_ref <= 0:
        fwd_ref = float(np.nanmedian(pd.to_numeric(chain.get("underlying_close"), errors="coerce")))
    if not np.isfinite(fwd_ref) or fwd_ref <= 0:
        fwd_ref = 1.0
    price_grid = price_grid / fwd_ref

    iv_grid = fill_surface(iv_grid, cfg.x_grid, cfg.tenor_days)
    price_grid = fill_surface(price_grid, cfg.x_grid, cfg.tenor_days)
    liq_grid = fill_surface(liq_grid, cfg.x_grid, cfg.tenor_days)

    tau = (cfg.tenor_days.astype(np.float32) / 365.0).reshape(1, -1)
    w_grid = np.clip(iv_grid, 1e-4, None) ** 2 * np.clip(tau, 1e-6, None)
    # Static calendar repair in total-variance space to keep downstream training stable.
    w_grid = repair_calendar_monotonic(w_grid, cfg.tenor_days)
    iv_grid = np.sqrt(np.clip(w_grid / np.clip(tau, 1e-6, None), 1e-8, None))

    return {
        "iv_surface": iv_grid.astype(np.float32),
        "w_surface": w_grid.astype(np.float32),
        "price_surface": price_grid.astype(np.float32),
        "liq_surface": liq_grid.astype(np.float32),
    }
