"""Dataset builder over Massive-compatible daily chain snapshots."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ivdyn.data.massive import PluginFactory
from ivdyn.surface import (
    SurfaceConfig,
    build_daily_surface,
    butterfly_violations,
    calendar_violations,
)
from ivdyn.surface.build import enrich_chain


@dataclass(slots=True)
class DatasetBuildConfig:
    data_root: Path
    out_dir: Path
    symbol: str = "SPY"
    plugin: str = "massive_raw_parquet"
    api_key: str | None = None
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
    max_contracts_per_day: int = 900
    random_seed: int = 7
    num_workers: int = 0


BASE_CONTRACT_FEATURE_COLUMNS = [
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


MINUTE_INTRADAY_FEATURE_COLUMNS = [
    "intraday_ret_oc",
    "intraday_range_frac",
    "intraday_rv_1m",
    "intraday_vwap_dev",
    "intraday_volume_cv",
    "intraday_log_bar_count",
    "intraday_log_volume_per_bar",
]


def _contract_feature_columns_for_plugin(plugin_name: str) -> list[str]:
    cols = list(BASE_CONTRACT_FEATURE_COLUMNS)
    if plugin_name.lower().strip() == "massive_flatfile_minute_aggs":
        cols.extend(MINUTE_INTRADAY_FEATURE_COLUMNS)
    return cols


def _parse_date(v: str | None) -> date | None:
    if v is None:
        return None
    return date.fromisoformat(v)


def _load_underlying_close(data_root: Path, symbol: str) -> pd.Series:
    p = data_root / "symbols" / symbol.upper() / "underlying" / f"{symbol.lower()}_eod.parquet"
    if not p.exists():
        return pd.Series(dtype=float)
    udf = pd.read_parquet(p)
    udf["date"] = pd.to_datetime(udf["date"]).dt.date
    close = pd.to_numeric(udf["close"], errors="coerce")
    return pd.Series(close.values, index=udf["date"].values).sort_index()


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


def _build_context(
    dates: list[date],
    spots: np.ndarray,
    iv_surface: np.ndarray,
    x_grid: np.ndarray,
    tenor_days: np.ndarray,
    noarb_df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    idx = pd.Index(dates)
    spot = pd.Series(spots, index=idx).astype(float)
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

    na = noarb_df.set_index("date")
    ctx["calendar_viol_rate"] = na["calendar_violation_rate"].reindex(idx).values
    ctx["butterfly_viol_rate"] = na["butterfly_violation_rate"].reindex(idx).values

    ctx = ctx.ffill().fillna(0.0)
    names = ctx.columns.tolist()
    return ctx.to_numpy(dtype=np.float32), names


def _sample_contracts(
    chain: pd.DataFrame,
    asof: date,
    max_contracts: int,
    rng: np.random.Generator,
    contract_feature_columns: list[str],
) -> pd.DataFrame:
    c = enrich_chain(chain)
    c = c[c["mid"] > 0].copy()
    if c.empty:
        return pd.DataFrame()

    c["spot"] = c["underlying_close"].astype(float)
    c["forward"] = pd.to_numeric(c.get("forward_price"), errors="coerce").where(
        pd.to_numeric(c.get("forward_price"), errors="coerce") > 0.0,
        c["spot"],
    )
    c["price_norm"] = c["mid"] / c["forward"].clip(lower=1e-6)
    c["spread"] = pd.to_numeric(c.get("spread"), errors="coerce")
    c["spread"] = c["spread"].where(c["spread"].notna(), (c["ask"] - c["bid"]).clip(lower=0.0))
    c["rel_spread"] = c["rel_spread"].clip(lower=0.0, upper=5.0)
    c["fill_label"] = (
        (c["volume"].fillna(0) > 0)
        & (c["rel_spread"].fillna(10.0) < c["rel_spread"].quantile(0.65))
    ).astype(int)

    c["log_volume"] = np.log1p(c["volume"].fillna(0.0).clip(lower=0.0))
    c["log_open_interest"] = np.log1p(c["open_interest"].fillna(0.0).clip(lower=0.0))
    c["abs_x"] = np.abs(c["x"])

    for col in contract_feature_columns:
        if col not in c.columns:
            c[col] = 0.0
        c[col] = pd.to_numeric(c[col], errors="coerce")
    c[contract_feature_columns] = c[contract_feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    keep_cols = [
        "symbol",
        "strike",
        "dte",
        "call_put",
        "mid",
        "spot",
        "forward",
        "spread",
        "vega",
        *contract_feature_columns,
        "price_norm",
        "fill_label",
    ]
    keep = c[keep_cols].copy()

    if len(keep) > max_contracts:
        probs = keep["liquidity"].to_numpy()
        probs = np.clip(probs, 1e-6, None)
        probs = probs / probs.sum()
        idx = rng.choice(np.arange(len(keep)), size=max_contracts, replace=False, p=probs)
        keep = keep.iloc[idx]

    keep["date"] = asof.isoformat()
    keep.reset_index(drop=True, inplace=True)
    return keep


def _seed_for_date(base_seed: int, asof: date) -> int:
    return (base_seed * 1_000_003 + int(asof.strftime("%Y%m%d"))) % (2**32 - 1)


def _resolve_num_workers(requested: int, n_tasks: int) -> int:
    if n_tasks <= 1:
        return 1
    if requested == 1:
        return 1
    if requested <= 0:
        cpu = os.cpu_count() or 1
        return max(1, min(cpu - 1, n_tasks))
    return max(1, min(requested, n_tasks))


def _process_one_day(
    *,
    data_root: str,
    plugin_name: str,
    api_key: str | None,
    symbol: str,
    asof_iso: str,
    x_grid: tuple[float, ...],
    tenor_days: tuple[int, ...],
    max_contracts_per_day: int,
    random_seed: int,
) -> dict[str, object] | None:
    asof = date.fromisoformat(asof_iso)
    x_arr = np.asarray(x_grid, dtype=np.float32)
    t_arr = np.asarray(tenor_days, dtype=np.int32)

    factory = PluginFactory(Path(data_root))
    plugin = factory.build(plugin_name=plugin_name, symbol=symbol, api_key=api_key)

    try:
        chain = plugin.load_day(symbol, asof)
    except Exception:
        return None

    if len(chain) < 80:
        return None

    day = build_daily_surface(
        chain,
        SurfaceConfig(
            x_grid=x_arr,
            tenor_days=t_arr,
        ),
    )
    iv = day["iv_surface"]
    w = day.get("w_surface")
    if w is None:
        tau = (t_arr.astype(np.float32) / 365.0).reshape(1, -1)
        w = np.clip(iv, 1e-4, None) ** 2 * np.clip(tau, 1e-6, None)
    px = day["price_surface"]
    liq = day["liq_surface"]

    cal = float(calendar_violations(iv[None, ...], t_arr)[0])
    bfly = float(butterfly_violations(iv[None, ...], x_arr, t_arr)[0])

    sampled = _sample_contracts(
        chain=chain,
        asof=asof,
        max_contracts=max_contracts_per_day,
        rng=np.random.default_rng(_seed_for_date(random_seed, asof)),
        contract_feature_columns=_contract_feature_columns_for_plugin(plugin_name),
    )

    return {
        "date": asof.isoformat(),
        "spot": float(np.nanmedian(chain["underlying_close"])),
        "iv_surface": iv.astype(np.float32),
        "w_surface": np.asarray(w, dtype=np.float32),
        "price_surface": px.astype(np.float32),
        "liq_surface": liq.astype(np.float32),
        "calendar_violation_rate": cal,
        "butterfly_violation_rate": bfly,
        "contracts": sampled,
    }


def build_dataset(cfg: DatasetBuildConfig) -> dict[str, Path]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    contract_feature_columns = _contract_feature_columns_for_plugin(cfg.plugin)

    factory = PluginFactory(cfg.data_root)
    plugin = factory.build(plugin_name=cfg.plugin, symbol=cfg.symbol, api_key=cfg.api_key)

    all_dates = sorted(plugin.list_dates(cfg.symbol))
    start = _parse_date(cfg.start_date)
    end = _parse_date(cfg.end_date)
    if start is not None:
        all_dates = [d for d in all_dates if d >= start]
    if end is not None:
        all_dates = [d for d in all_dates if d <= end]

    if cfg.plugin == "massive_flatfile_aggs":
        spot_series = _load_underlying_close(cfg.data_root, cfg.symbol)
        if spot_series.empty:
            raise RuntimeError(
                "No underlying close file found for flatfile mode. "
                "Expected data/symbols/<SYMBOL>/underlying/<symbol>_eod.parquet. "
                "Run `ivdyn pull-underlying-massive --symbol SPY --start-date ... --end-date ...` first."
            )
        spot_dates = set(spot_series.index.tolist())
        missing = [d for d in all_dates if d not in spot_dates]
        if missing:
            raise RuntimeError(
                "Underlying close data is missing for requested flatfile dates. "
                f"missing={len(missing)} of total={len(all_dates)}; "
                f"first_missing={missing[0].isoformat()} last_missing={missing[-1].isoformat()}. "
                "Run `ivdyn pull-underlying-massive --symbol SPY --start-date ... --end-date ...` "
                "for the same date range before build-dataset."
            )

    x_grid = np.array(cfg.x_grid, dtype=np.float32)
    tenor_days = np.array(cfg.tenor_days, dtype=np.int32)

    workers = _resolve_num_workers(cfg.num_workers, len(all_dates))
    parallel_backend = "sequential"
    date_iso = [d.isoformat() for d in all_dates]
    results_by_date: dict[str, dict[str, object]] = {}

    if workers <= 1:
        for asof in date_iso:
            out = _process_one_day(
                data_root=str(cfg.data_root),
                plugin_name=cfg.plugin,
                api_key=cfg.api_key,
                symbol=cfg.symbol,
                asof_iso=asof,
                x_grid=cfg.x_grid,
                tenor_days=cfg.tenor_days,
                max_contracts_per_day=cfg.max_contracts_per_day,
                random_seed=cfg.random_seed,
            )
            if out is not None:
                results_by_date[asof] = out
    else:
        executor_cls = ProcessPoolExecutor
        parallel_backend = "process"
        try:
            ex_obj = executor_cls(max_workers=workers)
        except (PermissionError, OSError):
            executor_cls = ThreadPoolExecutor
            parallel_backend = "thread"
            ex_obj = executor_cls(max_workers=workers)

        with ex_obj as ex:
            futures = [
                ex.submit(
                    _process_one_day,
                    data_root=str(cfg.data_root),
                    plugin_name=cfg.plugin,
                    api_key=cfg.api_key,
                    symbol=cfg.symbol,
                    asof_iso=asof,
                    x_grid=cfg.x_grid,
                    tenor_days=cfg.tenor_days,
                    max_contracts_per_day=cfg.max_contracts_per_day,
                    random_seed=cfg.random_seed,
                )
                for asof in date_iso
            ]
            for fut in as_completed(futures):
                out = fut.result()
                if out is None:
                    continue
                results_by_date[str(out["date"])] = out

    ordered = [results_by_date[d] for d in date_iso if d in results_by_date]
    if not ordered:
        hint = ""
        if cfg.plugin == "massive_flatfile_aggs":
            hint = (
                " For flatfile mode, ensure day_aggs files exist under "
                "data/options_source/us_options_opra/day_aggs_v1 and "
                "requested dates are present."
            )
        raise RuntimeError(f"No valid dates were built. Check plugin/data filters.{hint}")

    dates = [date.fromisoformat(str(row["date"])) for row in ordered]
    spots = [float(row["spot"]) for row in ordered]
    iv_surface = np.stack([row["iv_surface"] for row in ordered], axis=0).astype(np.float32)
    w_surface = np.stack([row["w_surface"] for row in ordered], axis=0).astype(np.float32)
    price_surface = np.stack([row["price_surface"] for row in ordered], axis=0).astype(np.float32)
    liq_surface = np.stack([row["liq_surface"] for row in ordered], axis=0).astype(np.float32)

    noarb_df = pd.DataFrame(
        [
            {
                "date": str(row["date"]),
                "calendar_violation_rate": float(row["calendar_violation_rate"]),
                "butterfly_violation_rate": float(row["butterfly_violation_rate"]),
            }
            for row in ordered
        ]
    )
    context, context_names = _build_context(
        dates=dates,
        spots=np.array(spots, dtype=np.float32),
        iv_surface=iv_surface,
        x_grid=x_grid,
        tenor_days=tenor_days,
        noarb_df=noarb_df,
    )

    contract_rows = [
        row["contracts"]
        for row in ordered
        if isinstance(row["contracts"], pd.DataFrame) and not row["contracts"].empty
    ]
    contracts = pd.concat(contract_rows, ignore_index=True) if contract_rows else pd.DataFrame()
    if contracts.empty:
        raise RuntimeError("No contracts sampled from available dates.")

    date_to_idx = {d.isoformat(): i for i, d in enumerate(dates)}
    contracts["date_idx"] = contracts["date"].map(date_to_idx).astype(np.int32)
    contracts["contract_idx"] = np.arange(len(contracts))

    contract_features = contracts[contract_feature_columns].to_numpy(dtype=np.float32)
    contract_price_target = contracts["price_norm"].to_numpy(dtype=np.float32)
    contract_fill_target = contracts["fill_label"].to_numpy(dtype=np.float32)
    contract_date_index = contracts["date_idx"].to_numpy(dtype=np.int32)
    contract_asset_id = np.zeros(len(contracts), dtype=np.int32)

    dataset_path = cfg.out_dir / "dataset.npz"
    np.savez_compressed(
        dataset_path,
        dates=np.array([d.isoformat() for d in dates]),
        asset_ids=np.zeros(len(dates), dtype=np.int32),
        asset_names=np.array([cfg.symbol.upper()], dtype="<U16"),
        spot=np.array(spots, dtype=np.float32),
        x_grid=x_grid,
        tenor_days=tenor_days,
        surface=w_surface,
        surface_variable=np.array("total_variance"),
        iv_surface=iv_surface,
        w_surface=w_surface,
        price_surface=price_surface,
        liq_surface=liq_surface,
        context=context,
        context_names=np.array(context_names),
        contract_features=contract_features,
        contract_feature_names=np.array(contract_feature_columns),
        contract_price_target=contract_price_target,
        contract_fill_target=contract_fill_target,
        contract_date_index=contract_date_index,
        contract_asset_id=contract_asset_id,
        contract_symbol=contracts["symbol"].astype(str).to_numpy(dtype="<U64"),
        contract_call_put=contracts["call_put"].astype(str).to_numpy(dtype="<U4"),
        contract_dte=contracts["dte"].to_numpy(dtype=np.int32),
        contract_mid=contracts["mid"].to_numpy(dtype=np.float32),
        contract_spot=contracts["spot"].to_numpy(dtype=np.float32),
        contract_forward=contracts["forward"].to_numpy(dtype=np.float32),
        contract_spread=contracts["spread"].to_numpy(dtype=np.float32),
        contract_vega=pd.to_numeric(contracts["vega"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        contract_strike=contracts["strike"].to_numpy(dtype=np.float32),
        contract_date=np.array(contracts["date"].astype(str).tolist(), dtype="<U10"),
    )

    contracts_path = cfg.out_dir / "contracts.parquet"
    contracts.to_parquet(contracts_path, index=False)

    noarb_path = cfg.out_dir / "noarb_by_date.parquet"
    noarb_df.to_parquet(noarb_path, index=False)

    preview = pd.DataFrame(
        {
            "date": [d.isoformat() for d in dates],
            "spot": spots,
            "calendar_violation_rate": noarb_df["calendar_violation_rate"].values,
            "butterfly_violation_rate": noarb_df["butterfly_violation_rate"].values,
            "atm_iv_30": iv_surface[:, int(np.argmin(np.abs(x_grid - 0.0))), int(np.argmin(np.abs(tenor_days - 30)))],
        }
    )
    preview_path = cfg.out_dir / "dataset_preview.parquet"
    preview.to_parquet(preview_path, index=False)

    meta = {
        "symbol": cfg.symbol,
        "plugin": cfg.plugin,
        "surface_variable": "total_variance",
        "num_workers": workers,
        "parallel_backend": parallel_backend,
        "rows_dates": len(dates),
        "rows_contracts": int(len(contracts)),
        "date_min": min(dates).isoformat(),
        "date_max": max(dates).isoformat(),
        "x_grid": [float(x) for x in x_grid],
        "tenor_days": [int(t) for t in tenor_days],
        "context_features": context_names,
        "contract_features": contract_feature_columns,
        "calendar_violation_rate_mean": float(noarb_df["calendar_violation_rate"].mean()),
        "butterfly_violation_rate_mean": float(noarb_df["butterfly_violation_rate"].mean()),
    }
    meta_path = cfg.out_dir / "dataset_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "dataset": dataset_path,
        "meta": meta_path,
        "contracts": contracts_path,
        "noarb": noarb_path,
        "preview": preview_path,
    }
