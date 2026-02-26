"""Build multi-asset surface datasets from WRDS contract panels."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ivdyn.data.earnings_extract import daily_earnings_features, earnings_before_expiry_flags, load_earnings_calendar
from ivdyn.surface import butterfly_violations, calendar_violations
from ivdyn.surface.arb import repair_calendar_monotonic
from ivdyn.surface.build import fill_surface


@dataclass(slots=True)
class SurfaceDatasetBuildConfig:
    contracts_path: Path
    out_dir: Path
    earnings_path: Path | None = None
    symbols: tuple[str, ...] = ()
    start_date: str | None = None
    end_date: str | None = None
    x_grid: tuple[float, ...] = (
        -0.35,
        -0.30,
        -0.25,
        -0.20,
        -0.16,
        -0.12,
        -0.09,
        -0.06,
        -0.04,
        -0.02,
        0.0,
        0.02,
        0.04,
        0.06,
        0.09,
        0.12,
        0.16,
        0.20,
        0.25,
        0.30,
        0.35,
    )
    tenor_days: tuple[int, ...] = (7, 14, 30, 60, 90, 180)
    max_contracts_per_day: int = 1200
    min_contracts_per_day: int = 80
    max_neighbors: int = 20
    max_dte_distance: int = 30
    max_rel_spread: float = 2.0
    random_seed: int = 7
    surface_variable: str = "total_variance"
    num_workers: int = 0
    surface_pca_factors: int = 3


WRDS_CONTRACT_FEATURE_COLUMNS = [
    "x",
    "tau",
    "cp_sign",
    "rel_spread",
    "vega",
    "log_volume",
    "log_open_interest",
    "liquidity",
    "abs_x",
]


def _parse_date_opt(v: str | None) -> date | None:
    if not v:
        return None
    return date.fromisoformat(v)


def _seed_for_asset_day(base_seed: int, asset_id: int, asof: date) -> int:
    return (int(base_seed) * 1_000_003 + int(asset_id) * 97_409 + int(asof.strftime("%Y%m%d"))) % (2**32 - 1)


def _resolve_num_workers(requested: int, n_tasks: int) -> int:
    if n_tasks <= 1:
        return 1
    if requested == 1:
        return 1
    if requested <= 0:
        cpu = os.cpu_count() or 1
        return max(1, min(cpu - 1, n_tasks))
    return max(1, min(requested, n_tasks))


def _weighted_grid_value(
    df: pd.DataFrame,
    *,
    x_target: float,
    dte_target: float,
    col: str,
    max_neighbors: int,
) -> float:
    if df.empty:
        return float("nan")

    d_x = (df["x"].to_numpy(dtype=float) - x_target) ** 2
    d_t = ((df["dte"].to_numpy(dtype=float) - dte_target) / max(1.0, dte_target)) ** 2
    dist = np.sqrt(d_x + 0.08 * d_t)
    order = np.argsort(dist)
    idx = order[: min(max_neighbors, len(order))]
    if len(idx) == 0:
        return float("nan")

    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)[idx]
    liq = pd.to_numeric(df.get("liquidity"), errors="coerce").fillna(0.0).to_numpy(dtype=float)[idx]
    spread = pd.to_numeric(df.get("rel_spread"), errors="coerce").fillna(1.0).to_numpy(dtype=float)[idx]
    vega = np.abs(pd.to_numeric(df.get("vega"), errors="coerce").fillna(0.0).to_numpy(dtype=float)[idx])

    liq_w = np.clip(liq + 0.2, 1e-4, None)
    spread_w = 1.0 / np.clip(spread, 0.01, 5.0)
    vega_w = np.clip(vega, 0.05, None)
    w = (liq_w * spread_w * vega_w) / np.clip(dist[idx], 2e-3, None)

    finite = np.isfinite(vals)
    if not np.any(finite):
        return float("nan")
    vals = np.where(finite, vals, np.nanmedian(vals[finite]))
    return float(np.sum(w * vals) / np.sum(w))


def _build_day_surface(day: pd.DataFrame, *, x_grid: np.ndarray, tenor_days: np.ndarray, cfg: SurfaceDatasetBuildConfig) -> dict[str, np.ndarray | float]:
    d = day.copy()
    d = d[np.isfinite(d["x"]) & np.isfinite(d["dte"]) & np.isfinite(d["iv"]) & (d["iv"] > 0.0)]
    d = d[np.isfinite(d["rel_spread"]) & (d["rel_spread"] <= float(cfg.max_rel_spread))]
    if d.empty:
        nx = len(x_grid)
        nt = len(tenor_days)
        nan_grid = np.full((nx, nt), np.nan, dtype=np.float32)
        return {
            "iv_surface": nan_grid,
            "w_surface": nan_grid,
            "liq_surface": nan_grid,
            "spread_surface": nan_grid,
            "vega_surface": nan_grid,
            "coverage_ratio": 0.0,
            "avg_rel_spread": float("nan"),
            "avg_liquidity": float("nan"),
        }

    nx = len(x_grid)
    nt = len(tenor_days)
    iv_grid = np.full((nx, nt), np.nan, dtype=float)
    liq_grid = np.full((nx, nt), np.nan, dtype=float)
    spread_grid = np.full((nx, nt), np.nan, dtype=float)
    vega_grid = np.full((nx, nt), np.nan, dtype=float)

    for j, t in enumerate(tenor_days.astype(float)):
        td = np.abs(d["dte"].to_numpy(dtype=float) - t)
        mask_t = td <= min(float(cfg.max_dte_distance), max(5.0, 0.45 * t))
        sub = d.loc[mask_t]
        if sub.empty:
            nearest = np.argsort(td)[: min(len(d), 200)]
            sub = d.iloc[nearest]

        for i, x in enumerate(x_grid.astype(float)):
            iv_grid[i, j] = _weighted_grid_value(
                sub,
                x_target=x,
                dte_target=t,
                col="iv",
                max_neighbors=int(cfg.max_neighbors),
            )
            liq_grid[i, j] = _weighted_grid_value(
                sub,
                x_target=x,
                dte_target=t,
                col="liquidity",
                max_neighbors=int(cfg.max_neighbors),
            )
            spread_grid[i, j] = _weighted_grid_value(
                sub,
                x_target=x,
                dte_target=t,
                col="rel_spread",
                max_neighbors=int(cfg.max_neighbors),
            )
            vega_grid[i, j] = _weighted_grid_value(
                sub,
                x_target=x,
                dte_target=t,
                col="vega",
                max_neighbors=int(cfg.max_neighbors),
            )

    coverage_ratio = float(np.mean(np.isfinite(iv_grid)))
    iv_grid = fill_surface(iv_grid, x_grid, tenor_days)
    liq_grid = fill_surface(liq_grid, x_grid, tenor_days)
    spread_grid = fill_surface(spread_grid, x_grid, tenor_days)
    vega_grid = fill_surface(np.abs(vega_grid), x_grid, tenor_days)

    tau = (tenor_days.astype(np.float32) / 365.0).reshape(1, -1)
    w_grid = np.clip(iv_grid, 1e-4, None) ** 2 * np.clip(tau, 1e-6, None)
    w_grid = repair_calendar_monotonic(w_grid, tenor_days)
    iv_grid = np.sqrt(np.clip(w_grid / np.clip(tau, 1e-6, None), 1e-8, None))

    return {
        "iv_surface": iv_grid.astype(np.float32),
        "w_surface": w_grid.astype(np.float32),
        "liq_surface": liq_grid.astype(np.float32),
        "spread_surface": spread_grid.astype(np.float32),
        "vega_surface": vega_grid.astype(np.float32),
        "coverage_ratio": coverage_ratio,
        "avg_rel_spread": float(pd.to_numeric(d["rel_spread"], errors="coerce").mean()),
        "avg_liquidity": float(pd.to_numeric(d["liquidity"], errors="coerce").mean()),
    }


def _surface_summary_features(iv_surface: np.ndarray, x_grid: np.ndarray, tenor_days: np.ndarray) -> pd.DataFrame:
    ix_atm = int(np.argmin(np.abs(x_grid - 0.0)))
    ix_put = int(np.argmin(np.abs(x_grid + 0.10)))
    ix_call = int(np.argmin(np.abs(x_grid - 0.10)))

    it_14 = int(np.argmin(np.abs(tenor_days - 14)))
    it_30 = int(np.argmin(np.abs(tenor_days - 30)))
    it_60 = int(np.argmin(np.abs(tenor_days - 60)))
    it_90 = int(np.argmin(np.abs(tenor_days - 90)))

    atm_14 = iv_surface[:, ix_atm, it_14]
    atm_30 = iv_surface[:, ix_atm, it_30]
    atm_60 = iv_surface[:, ix_atm, it_60]
    atm_90 = iv_surface[:, ix_atm, it_90]
    skew_30 = iv_surface[:, ix_put, it_30] - iv_surface[:, ix_call, it_30]
    fly_30 = iv_surface[:, ix_put, it_30] - 2.0 * atm_30 + iv_surface[:, ix_call, it_30]

    return pd.DataFrame(
        {
            "atm_iv_14": atm_14,
            "atm_iv_30": atm_30,
            "atm_iv_60": atm_60,
            "atm_iv_90": atm_90,
            "skew_30": skew_30,
            "fly_30": fly_30,
            "term_14_60": atm_60 - atm_14,
            "term_30_90": atm_90 - atm_30,
            "iv_level": np.mean(iv_surface, axis=(1, 2)),
        }
    )


def _surface_pca_features(iv_surface: np.ndarray, n_factors: int) -> pd.DataFrame:
    k_req = max(int(n_factors), 0)
    n_days = int(iv_surface.shape[0])
    if k_req <= 0 or n_days <= 0:
        return pd.DataFrame(index=np.arange(n_days))

    flat = iv_surface.reshape(n_days, -1).astype(np.float64)
    col_med = np.nanmedian(flat, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    valid = np.isfinite(flat)
    flat = np.where(valid, flat, col_med[None, :])
    flat = np.where(np.isfinite(flat), flat, 0.0)
    centered = flat - np.mean(flat, axis=0, keepdims=True)

    try:
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
    except Exception:
        u = np.zeros((n_days, 1), dtype=np.float64)
        s = np.zeros(1, dtype=np.float64)
    k = max(1, min(k_req, int(u.shape[1])))
    factors = u[:, :k] * s[:k]
    out = {f"surface_pca_{i + 1}": factors[:, i].astype(np.float32) for i in range(k)}
    return pd.DataFrame(out, index=np.arange(n_days))


def _build_context(
    *,
    dates: list[date],
    spots: np.ndarray,
    iv_surface: np.ndarray,
    x_grid: np.ndarray,
    tenor_days: np.ndarray,
    quality: pd.DataFrame,
    earnings: pd.DataFrame | None,
    surface_pca_factors: int = 3,
) -> tuple[np.ndarray, list[str]]:
    idx = pd.Index(dates)
    spot = pd.Series(spots.astype(float), index=idx)
    ret = spot.pct_change()

    ctx = pd.DataFrame(index=idx)
    ctx["ret_1"] = ret
    ctx["ret_5"] = spot.pct_change(5)
    ctx["rv_5"] = ret.rolling(5).std() * np.sqrt(252)
    ctx["rv_20"] = ret.rolling(20).std() * np.sqrt(252)
    ctx["rv_ratio"] = ctx["rv_5"] / ctx["rv_20"].replace(0.0, np.nan)
    ctx["momentum_5_20"] = spot.pct_change(5) - spot.pct_change(20)
    ctx["drawdown_20"] = spot / spot.rolling(20).max() - 1.0

    surf = _surface_summary_features(iv_surface, x_grid, tenor_days)
    surf.index = idx
    ctx = pd.concat([ctx, surf], axis=1)
    pca = _surface_pca_features(iv_surface, n_factors=surface_pca_factors)
    pca.index = idx
    ctx = pd.concat([ctx, pca], axis=1)

    q = quality.set_index("date")
    ctx["calendar_viol_rate"] = q["calendar_violation_rate"].reindex(idx).values
    ctx["butterfly_viol_rate"] = q["butterfly_violation_rate"].reindex(idx).values
    ctx["surface_coverage"] = q["surface_coverage"].reindex(idx).values
    ctx["avg_rel_spread"] = q["avg_rel_spread"].reindex(idx).values
    ctx["avg_liquidity"] = q["avg_liquidity"].reindex(idx).values

    if earnings is not None and not earnings.empty:
        e = earnings.copy()
        e.index = idx
        ctx = pd.concat([ctx, e], axis=1)
        if "days_to_next_earnings" in e.columns:
            flags = earnings_before_expiry_flags(
                e["days_to_next_earnings"].to_numpy(dtype=np.float32),
                tenor_days,
            )
            for j, t in enumerate(tenor_days):
                ctx[f"earnings_before_expiry_{int(t)}d"] = flags[:, j]

    ctx = ctx.ffill().fillna(0.0)
    names = ctx.columns.tolist()
    return ctx.to_numpy(dtype=np.float32), names


def _sample_contracts(
    day: pd.DataFrame,
    *,
    max_contracts: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if day.empty:
        return pd.DataFrame()
    c = day.copy()
    c = c[c["mid"] > 0.0]
    if c.empty:
        return pd.DataFrame()

    c["log_volume"] = np.log1p(pd.to_numeric(c["volume"], errors="coerce").fillna(0.0).clip(lower=0.0))
    c["log_open_interest"] = np.log1p(pd.to_numeric(c["open_interest"], errors="coerce").fillna(0.0).clip(lower=0.0))
    c["abs_x"] = np.abs(pd.to_numeric(c["x"], errors="coerce"))
    c["price_norm"] = pd.to_numeric(c["mid"], errors="coerce") / np.clip(
        pd.to_numeric(c["forward_price"], errors="coerce"),
        1e-6,
        None,
    )
    rel_spread = pd.to_numeric(c["rel_spread"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    c["fill_label"] = (
        (pd.to_numeric(c["volume"], errors="coerce").fillna(0.0) > 0)
        & (rel_spread < rel_spread.quantile(0.65))
    ).astype(int)

    for col in WRDS_CONTRACT_FEATURE_COLUMNS:
        if col not in c.columns:
            c[col] = 0.0
        c[col] = pd.to_numeric(c[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    keep_cols = [
        "date",
        "ticker",
        "optionid",
        "call_put",
        "dte",
        "strike",
        "mid",
        "underlying_close",
        "forward_price",
        "spread",
        "price_norm",
        "fill_label",
        *WRDS_CONTRACT_FEATURE_COLUMNS,
    ]
    keep = c[keep_cols].copy()

    if len(keep) > int(max_contracts):
        probs = pd.to_numeric(keep["liquidity"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        probs = np.clip(probs, 1e-8, None)
        probs = probs / probs.sum()
        pick = rng.choice(np.arange(len(keep)), size=int(max_contracts), replace=False, p=probs)
        keep = keep.iloc[pick]

    keep.reset_index(drop=True, inplace=True)
    return keep


def _build_asset_slice(
    *,
    asset_id: int,
    ticker: str,
    sub: pd.DataFrame,
    x_grid: np.ndarray,
    tenor_days: np.ndarray,
    cfg: SurfaceDatasetBuildConfig,
    earnings_cal: pd.DataFrame | None,
) -> dict[str, object] | None:
    if sub.empty:
        return None

    sub_dates = sorted(sub["date"].dropna().unique().tolist())
    asset_dates: list[date] = []
    asset_spots: list[float] = []
    asset_iv_surfaces: list[np.ndarray] = []
    asset_w_surfaces: list[np.ndarray] = []
    asset_liq_surfaces: list[np.ndarray] = []
    asset_spread_surfaces: list[np.ndarray] = []
    asset_vega_surfaces: list[np.ndarray] = []
    asset_quality_rows: list[dict[str, object]] = []
    contract_rows: list[pd.DataFrame] = []

    for asof in sub_dates:
        day = sub[sub["date"] == asof].copy()
        if len(day) < int(cfg.min_contracts_per_day):
            continue

        surf = _build_day_surface(day, x_grid=x_grid, tenor_days=tenor_days, cfg=cfg)
        iv_grid = np.asarray(surf["iv_surface"], dtype=np.float32)
        w_grid = np.asarray(surf["w_surface"], dtype=np.float32)
        liq_grid = np.asarray(surf["liq_surface"], dtype=np.float32)
        spread_grid = np.asarray(surf["spread_surface"], dtype=np.float32)
        vega_grid = np.asarray(surf["vega_surface"], dtype=np.float32)
        cal = float(calendar_violations(iv_grid[None, ...], tenor_days)[0])
        bfly = float(butterfly_violations(iv_grid[None, ...], x_grid, tenor_days)[0])
        spot = float(pd.to_numeric(day["underlying_close"], errors="coerce").median())
        if not np.isfinite(spot) or spot <= 0.0:
            continue

        asset_dates.append(asof)
        asset_spots.append(spot)
        asset_iv_surfaces.append(iv_grid)
        asset_w_surfaces.append(w_grid)
        asset_liq_surfaces.append(liq_grid)
        asset_spread_surfaces.append(spread_grid)
        asset_vega_surfaces.append(vega_grid)
        asset_quality_rows.append(
            {
                "asset": ticker,
                "date": asof.isoformat(),
                "surface_coverage": float(surf["coverage_ratio"]),
                "avg_rel_spread": float(surf["avg_rel_spread"]),
                "avg_liquidity": float(surf["avg_liquidity"]),
                "calendar_violation_rate": cal,
                "butterfly_violation_rate": bfly,
                "asset_id": int(asset_id),
            }
        )

        sampled = _sample_contracts(
            day,
            max_contracts=int(cfg.max_contracts_per_day),
            rng=np.random.default_rng(_seed_for_asset_day(int(cfg.random_seed), int(asset_id), asof)),
        )
        if not sampled.empty:
            sampled["asset_id"] = int(asset_id)
            sampled["asset"] = ticker
            sampled["date"] = asof.isoformat()
            contract_rows.append(sampled)

    if not asset_dates:
        return None

    quality_df = pd.DataFrame(asset_quality_rows)
    earnings_features = None
    if earnings_cal is not None:
        earnings_features = daily_earnings_features(
            dates=asset_dates,
            ticker=ticker,
            calendar=earnings_cal,
        )
    ctx_np, ctx_names = _build_context(
        dates=asset_dates,
        spots=np.asarray(asset_spots, dtype=np.float32),
        iv_surface=np.stack(asset_iv_surfaces, axis=0).astype(np.float32),
        x_grid=x_grid,
        tenor_days=tenor_days,
        quality=quality_df,
        earnings=earnings_features,
        surface_pca_factors=int(cfg.surface_pca_factors),
    )

    contracts = pd.concat(contract_rows, ignore_index=True) if contract_rows else pd.DataFrame()
    return {
        "asset_id": int(asset_id),
        "asset": ticker,
        "dates": [d.isoformat() for d in asset_dates],
        "spot": asset_spots,
        "iv_surface": asset_iv_surfaces,
        "w_surface": asset_w_surfaces,
        "liq_surface": asset_liq_surfaces,
        "spread_surface": asset_spread_surfaces,
        "vega_surface": asset_vega_surfaces,
        "quality_rows": asset_quality_rows,
        "context_rows": [row.astype(np.float32) for row in ctx_np],
        "context_names": ctx_names,
        "contracts": contracts,
    }


def build_surface_dataset(cfg: SurfaceDatasetBuildConfig) -> dict[str, Path]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.contracts_path)
    if df.empty:
        raise RuntimeError(f"Contracts panel is empty: {cfg.contracts_path}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    if "exdate" in df.columns:
        df["exdate"] = pd.to_datetime(df["exdate"], errors="coerce").dt.date
    else:
        df["exdate"] = pd.NaT
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["call_put"] = df["call_put"].astype(str).str.upper().str[0]
    df["strike"] = pd.to_numeric(df.get("strike"), errors="coerce")
    fwd = pd.to_numeric(df.get("forward_price"), errors="coerce")
    spot = pd.to_numeric(df.get("underlying_close"), errors="coerce")
    fwd = np.where(np.isfinite(fwd) & (fwd > 0.0), fwd, spot)
    df["forward_price"] = fwd
    df["x"] = np.log(df["strike"] / np.clip(df["forward_price"], 1e-6, None))
    if "optionid" not in df.columns:
        df["optionid"] = np.arange(len(df), dtype=np.int64).astype(str)

    start = _parse_date_opt(cfg.start_date)
    end = _parse_date_opt(cfg.end_date)
    if start is not None:
        df = df[df["date"] >= start]
    if end is not None:
        df = df[df["date"] <= end]

    symbols = sorted({s.upper().strip() for s in cfg.symbols if str(s).strip()}) if cfg.symbols else sorted(df["ticker"].unique().tolist())
    if symbols:
        df = df[df["ticker"].isin(symbols)]
    if df.empty:
        raise RuntimeError("No contracts left after symbol/date filters.")

    x_grid = np.asarray(cfg.x_grid, dtype=np.float32)
    tenor_days = np.asarray(cfg.tenor_days, dtype=np.int32)

    earnings_cal: pd.DataFrame | None = None
    if cfg.earnings_path is not None:
        earnings_cal = load_earnings_calendar(cfg.earnings_path)

    asset_tasks: list[tuple[int, str, pd.DataFrame]] = []
    for aid, ticker in enumerate(symbols):
        sub = df[df["ticker"] == ticker].copy()
        if sub.empty:
            continue
        asset_tasks.append((int(aid), ticker, sub))
    if not asset_tasks:
        raise RuntimeError("No contracts left after symbol/date filters.")

    workers = _resolve_num_workers(int(cfg.num_workers), len(asset_tasks))
    parallel_backend = "sequential"
    asset_results: list[dict[str, object]] = []
    if workers <= 1:
        for aid, ticker, sub in asset_tasks:
            out = _build_asset_slice(
                asset_id=aid,
                ticker=ticker,
                sub=sub,
                x_grid=x_grid,
                tenor_days=tenor_days,
                cfg=cfg,
                earnings_cal=earnings_cal,
            )
            if out is not None:
                asset_results.append(out)
    else:
        parallel_backend = "thread"
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    _build_asset_slice,
                    asset_id=aid,
                    ticker=ticker,
                    sub=sub,
                    x_grid=x_grid,
                    tenor_days=tenor_days,
                    cfg=cfg,
                    earnings_cal=earnings_cal,
                )
                for aid, ticker, sub in asset_tasks
            ]
            for fut in as_completed(futs):
                out = fut.result()
                if out is not None:
                    asset_results.append(out)

    if not asset_results:
        raise RuntimeError("No valid daily surfaces were built from the contracts panel.")
    asset_results = sorted(asset_results, key=lambda x: int(x["asset_id"]))
    asset_id_remap = {int(out["asset_id"]): i for i, out in enumerate(asset_results)}

    dates_out: list[str] = []
    asset_ids_out: list[int] = []
    asset_names_out: list[str] = []
    spot_out: list[float] = []
    iv_surface_out: list[np.ndarray] = []
    w_surface_out: list[np.ndarray] = []
    liq_surface_out: list[np.ndarray] = []
    spread_surface_out: list[np.ndarray] = []
    vega_surface_out: list[np.ndarray] = []
    context_rows: list[np.ndarray] = []
    quality_rows_all: list[dict[str, object]] = []
    contract_rows_all: list[pd.DataFrame] = []
    context_names: list[str] | None = None
    date_idx_lookup: dict[tuple[int, str], int] = {}

    for out in asset_results:
        aid_old = int(out["asset_id"])
        aid = int(asset_id_remap[aid_old])
        ticker = str(out["asset"])
        ctx_names = list(out["context_names"])  # type: ignore[arg-type]
        if context_names is None:
            context_names = ctx_names
        elif context_names != ctx_names:
            raise RuntimeError("Context feature mismatch across assets.")
        asset_names_out.append(ticker)

        dates_asset = list(out["dates"])  # type: ignore[arg-type]
        spots_asset = list(out["spot"])  # type: ignore[arg-type]
        iv_asset = list(out["iv_surface"])  # type: ignore[arg-type]
        w_asset = list(out["w_surface"])  # type: ignore[arg-type]
        liq_asset = list(out["liq_surface"])  # type: ignore[arg-type]
        spread_asset = list(out["spread_surface"])  # type: ignore[arg-type]
        vega_asset = list(out["vega_surface"])  # type: ignore[arg-type]
        q_rows = list(out["quality_rows"])  # type: ignore[arg-type]
        ctx_rows = list(out["context_rows"])  # type: ignore[arg-type]

        if not (
            len(dates_asset)
            == len(spots_asset)
            == len(iv_asset)
            == len(w_asset)
            == len(liq_asset)
            == len(spread_asset)
            == len(vega_asset)
            == len(q_rows)
            == len(ctx_rows)
        ):
            raise RuntimeError(f"Asset bundle length mismatch for {ticker}.")

        for i, d in enumerate(dates_asset):
            idx = len(dates_out)
            dates_out.append(str(d))
            asset_ids_out.append(aid)
            spot_out.append(float(spots_asset[i]))
            iv_surface_out.append(np.asarray(iv_asset[i], dtype=np.float32))
            w_surface_out.append(np.asarray(w_asset[i], dtype=np.float32))
            liq_surface_out.append(np.asarray(liq_asset[i], dtype=np.float32))
            spread_surface_out.append(np.asarray(spread_asset[i], dtype=np.float32))
            vega_surface_out.append(np.asarray(vega_asset[i], dtype=np.float32))
            context_rows.append(np.asarray(ctx_rows[i], dtype=np.float32))
            q = dict(q_rows[i])
            q["asset_id"] = int(aid)
            q["date_idx"] = int(idx)
            quality_rows_all.append(q)
            date_idx_lookup[(aid, str(d))] = int(idx)

        contracts_out = out.get("contracts")
        if isinstance(contracts_out, pd.DataFrame) and not contracts_out.empty:
            c = contracts_out.copy()
            c["asset_id"] = c["asset_id"].map(lambda x: asset_id_remap.get(int(x), -1)).astype(np.int32)
            c["date"] = c["date"].astype(str)
            mapped = np.array(
                [date_idx_lookup.get((int(a), d), -1) for a, d in zip(c["asset_id"].tolist(), c["date"].tolist(), strict=False)],
                dtype=np.int32,
            )
            c["date_idx"] = mapped
            c = c[c["date_idx"] >= 0].copy()
            if not c.empty:
                contract_rows_all.append(c)

    if not context_rows:
        raise RuntimeError("Context feature construction produced no rows.")

    n_dates = len(dates_out)
    iv_surface = np.stack(iv_surface_out, axis=0).astype(np.float32)
    w_surface = np.stack(w_surface_out, axis=0).astype(np.float32)
    liq_surface = np.stack(liq_surface_out, axis=0).astype(np.float32)
    spread_surface = np.stack(spread_surface_out, axis=0).astype(np.float32)
    vega_surface = np.stack(vega_surface_out, axis=0).astype(np.float32)
    context = np.stack(context_rows, axis=0).astype(np.float32)
    if len(context) != n_dates:
        raise RuntimeError(f"Context/date length mismatch: context={len(context)} dates={n_dates}")

    contracts = pd.concat(contract_rows_all, ignore_index=True) if contract_rows_all else pd.DataFrame()
    if contracts.empty:
        raise RuntimeError("No contract targets sampled from built surfaces.")

    contract_features = contracts[WRDS_CONTRACT_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    contract_price_target = contracts["price_norm"].to_numpy(dtype=np.float32)
    contract_fill_target = contracts["fill_label"].to_numpy(dtype=np.float32)
    contract_date_idx = contracts["date_idx"].to_numpy(dtype=np.int32)
    contract_asset_id = contracts["asset_id"].to_numpy(dtype=np.int32)

    if cfg.surface_variable.lower().strip() == "iv":
        surface = iv_surface
        surface_variable = "iv"
    else:
        surface = w_surface
        surface_variable = "total_variance"

    dataset_path = cfg.out_dir / "dataset.npz"
    np.savez_compressed(
        dataset_path,
        dates=np.array(dates_out, dtype="<U10"),
        asset_ids=np.asarray(asset_ids_out, dtype=np.int32),
        asset_names=np.array(asset_names_out, dtype="<U16"),
        spot=np.asarray(spot_out, dtype=np.float32),
        x_grid=x_grid,
        tenor_days=tenor_days,
        surface=surface.astype(np.float32),
        surface_variable=np.array(surface_variable),
        iv_surface=iv_surface.astype(np.float32),
        w_surface=w_surface.astype(np.float32),
        liq_surface=liq_surface.astype(np.float32),
        spread_surface=spread_surface.astype(np.float32),
        vega_surface=vega_surface.astype(np.float32),
        context=context.astype(np.float32),
        context_names=np.array(context_names or [], dtype="<U64"),
        contract_features=contract_features,
        contract_feature_names=np.array(WRDS_CONTRACT_FEATURE_COLUMNS, dtype="<U64"),
        contract_price_target=contract_price_target,
        contract_fill_target=contract_fill_target,
        contract_date_index=contract_date_idx,
        contract_asset_id=contract_asset_id,
        contract_symbol=contracts["optionid"].astype(str).to_numpy(dtype="<U64"),
        contract_underlying=contracts["asset"].astype(str).to_numpy(dtype="<U16"),
        contract_call_put=contracts["call_put"].astype(str).to_numpy(dtype="<U4"),
        contract_dte=contracts["dte"].to_numpy(dtype=np.int32),
        contract_mid=contracts["mid"].to_numpy(dtype=np.float32),
        contract_spot=contracts["underlying_close"].to_numpy(dtype=np.float32),
        contract_forward=contracts["forward_price"].to_numpy(dtype=np.float32),
        contract_spread=contracts["spread"].to_numpy(dtype=np.float32),
        contract_vega=pd.to_numeric(contracts["vega"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        contract_strike=contracts["strike"].to_numpy(dtype=np.float32),
        contract_date=contracts["date"].astype(str).to_numpy(dtype="<U10"),
    )

    contracts_path = cfg.out_dir / "contracts.parquet"
    contracts.to_parquet(contracts_path, index=False)

    quality_df = pd.DataFrame(quality_rows_all)
    quality_path = cfg.out_dir / "surface_quality_by_date.parquet"
    quality_df.to_parquet(quality_path, index=False)

    preview = pd.DataFrame(
        {
            "date": dates_out,
            "asset_id": asset_ids_out,
            "asset": [asset_names_out[i] for i in asset_ids_out],
            "spot": spot_out,
            "calendar_violation_rate": quality_df["calendar_violation_rate"].to_numpy(dtype=np.float32),
            "butterfly_violation_rate": quality_df["butterfly_violation_rate"].to_numpy(dtype=np.float32),
            "atm_iv_30": iv_surface[:, int(np.argmin(np.abs(x_grid - 0.0))), int(np.argmin(np.abs(tenor_days - 30)))],
        }
    )
    preview_path = cfg.out_dir / "dataset_preview.parquet"
    preview.to_parquet(preview_path, index=False)

    meta = {
        "config": {
            **asdict(cfg),
            "contracts_path": str(cfg.contracts_path),
            "out_dir": str(cfg.out_dir),
            "earnings_path": str(cfg.earnings_path) if cfg.earnings_path else None,
        },
        "surface_variable": surface_variable,
        "num_workers": int(workers),
        "parallel_backend": parallel_backend,
        "rows_dates": int(n_dates),
        "rows_contracts": int(len(contracts)),
        "date_min": min(dates_out),
        "date_max": max(dates_out),
        "assets": asset_names_out,
        "x_grid": [float(x) for x in x_grid],
        "tenor_days": [int(t) for t in tenor_days],
        "context_features": context_names,
        "contract_features": WRDS_CONTRACT_FEATURE_COLUMNS,
        "calendar_violation_rate_mean": float(np.mean(quality_df["calendar_violation_rate"])),
        "butterfly_violation_rate_mean": float(np.mean(quality_df["butterfly_violation_rate"])),
    }
    meta_path = cfg.out_dir / "dataset_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "dataset": dataset_path,
        "meta": meta_path,
        "contracts": contracts_path,
        "quality": quality_path,
        "preview": preview_path,
    }
