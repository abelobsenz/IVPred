"""Evaluation pipeline producing numerical and graphical artifacts."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for evaluation.") from exc

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except Exception:  # pragma: no cover
    HistGradientBoostingRegressor = None

from ivdyn.eval.metrics import brier_score, mae, r2, rmse
from ivdyn.model import ModelBundle, device_auto, to_numpy
from ivdyn.surface import butterfly_violations, calendar_violations


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    npz = np.load(path, allow_pickle=True)
    out: dict[str, np.ndarray] = {}
    for k in npz.files:
        arr = npz[k]
        if arr.dtype == object:
            arr = arr.astype(str)
        out[k] = arr
    return out


def _date_splits(n_dates: int, train_frac: float, val_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i_train = max(2, int(n_dates * train_frac))
    i_val = max(i_train + 1, int(n_dates * (train_frac + val_frac)))
    i_val = min(i_val, n_dates - 1)
    return np.arange(i_train), np.arange(i_train, i_val), np.arange(i_val, n_dates)


def _date_splits_by_asset(
    asset_ids: np.ndarray,
    *,
    train_frac: float,
    val_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tr_all: list[np.ndarray] = []
    va_all: list[np.ndarray] = []
    te_all: list[np.ndarray] = []
    unique_assets = np.unique(asset_ids.astype(np.int32))
    for aid in unique_assets:
        idx = np.where(asset_ids == aid)[0]
        if len(idx) <= 3:
            tr_all.append(idx)
            va_all.append(np.array([], dtype=np.int64))
            te_all.append(np.array([], dtype=np.int64))
            continue
        tr, va, te = _date_splits(len(idx), train_frac, val_frac)
        tr_all.append(idx[tr])
        va_all.append(idx[va])
        te_all.append(idx[te])
    tr_out = np.sort(np.concatenate(tr_all)).astype(np.int64) if tr_all else np.array([], dtype=np.int64)
    va_out = np.sort(np.concatenate(va_all)).astype(np.int64) if va_all else np.array([], dtype=np.int64)
    te_out = np.sort(np.concatenate(te_all)).astype(np.int64) if te_all else np.array([], dtype=np.int64)
    return tr_out, va_out, te_out


def _date_splits_global_time(
    n_dates: int,
    *,
    train_frac: float,
    val_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tr, va, te = _date_splits(n_dates, train_frac, val_frac)
    return tr.astype(np.int64), va.astype(np.int64), te.astype(np.int64)


def _normalize_split_mode(mode: str | None) -> str:
    s = str(mode or "").strip().lower()
    if s in {"global", "global_time", "calendar"}:
        return "global_time"
    return "by_asset_time"


def _date_splits_for_mode(
    *,
    n_dates: int,
    asset_ids: np.ndarray,
    train_frac: float,
    val_frac: float,
    split_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mode = _normalize_split_mode(split_mode)
    if mode == "global_time":
        return _date_splits_global_time(
            n_dates,
            train_frac=train_frac,
            val_frac=val_frac,
        )
    return _date_splits_by_asset(
        asset_ids,
        train_frac=train_frac,
        val_frac=val_frac,
    )


def _split_config_from_train_config(run_dir: Path) -> tuple[float, float, str]:
    default_train = 0.70
    default_val = 0.15
    default_mode = "by_asset_time"
    path = run_dir / "train_config.json"
    if not path.exists():
        return default_train, default_val, default_mode
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_train, default_val, default_mode
    try:
        train_frac = float(raw.get("train_frac", default_train))
    except Exception:
        train_frac = default_train
    try:
        val_frac = float(raw.get("val_frac", default_val))
    except Exception:
        val_frac = default_val
    if not np.isfinite(train_frac):
        train_frac = default_train
    if not np.isfinite(val_frac):
        val_frac = default_val
    train_frac = float(np.clip(train_frac, 0.05, 0.95))
    val_frac = float(np.clip(val_frac, 0.0, max(0.0, 0.99 - train_frac)))
    split_mode = _normalize_split_mode(str(raw.get("split_mode", default_mode)))
    return train_frac, val_frac, split_mode


def _assert_no_train_test_date_overlap(
    *,
    dates: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    if len(train_idx) == 0 or len(test_idx) == 0:
        return
    date_values = np.asarray(dates).astype(str)
    train_dates = set(date_values[np.asarray(train_idx, dtype=np.int64)].tolist())
    test_dates = set(date_values[np.asarray(test_idx, dtype=np.int64)].tolist())
    overlap = train_dates.intersection(test_dates)
    if overlap:
        raise RuntimeError(
            "Date leakage detected in evaluation: train/test sets share calendar dates. "
            "Recheck split configuration."
        )


def _surface_variable_name(v: str | None) -> str:
    if not v:
        return "iv"
    s = str(v).strip().lower()
    if s in {"total_variance", "w", "total_var"}:
        return "total_variance"
    return "iv"


def _surface_to_iv_numpy(surface_raw: np.ndarray, tenor_days: np.ndarray, surface_variable: str) -> np.ndarray:
    if _surface_variable_name(surface_variable) != "total_variance":
        return np.clip(surface_raw, 1e-4, 4.0).astype(np.float32)
    tau = (tenor_days.astype(np.float32) / 365.0).reshape(1, 1, -1)
    w = np.clip(surface_raw, 1e-8, None)
    iv = np.sqrt(np.clip(w / np.clip(tau, 1e-6, None), 1e-8, None))
    return iv.astype(np.float32)


def _surface_forecast_error_profiles(
    *,
    obs_forecast: np.ndarray,
    pred_forecast: np.ndarray,
    x_grid: np.ndarray,
    tenor_days: np.ndarray,
    persistence_forecast: np.ndarray | None = None,
    parametric_forecast: np.ndarray | None = None,
    tree_forecast: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build absolute forecast error profiles by DTE, by moneyness, and on the DTE-x grid."""
    if obs_forecast.size == 0 or pred_forecast.size == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    series_map: dict[str, np.ndarray] = {"model": pred_forecast}
    if persistence_forecast is not None and persistence_forecast.shape == obs_forecast.shape:
        series_map["persistence"] = persistence_forecast
    if parametric_forecast is not None and parametric_forecast.shape == obs_forecast.shape:
        series_map["parametric"] = parametric_forecast
    if tree_forecast is not None and tree_forecast.shape == obs_forecast.shape:
        series_map["tree"] = tree_forecast

    n_days, nx, nt = obs_forecast.shape
    rows_dte: list[dict[str, float | int | str]] = []
    rows_x: list[dict[str, float | int | str]] = []
    rows_grid: list[dict[str, float | int | str]] = []

    for series, pred in series_map.items():
        err = pred - obs_forecast
        abs_err = np.abs(err)
        sq_err = err**2

        mae_dte = np.mean(abs_err, axis=(0, 1))
        rmse_dte = np.sqrt(np.mean(sq_err, axis=(0, 1)))
        for j, dte in enumerate(tenor_days):
            rows_dte.append(
                {
                    "series": series,
                    "dte": int(dte),
                    "mae": float(mae_dte[j]),
                    "rmse": float(rmse_dte[j]),
                    "error_mode": "absolute",
                    "n_points": int(n_days * nx),
                }
            )

        mae_x = np.mean(abs_err, axis=(0, 2))
        rmse_x = np.sqrt(np.mean(sq_err, axis=(0, 2)))
        for i, x in enumerate(x_grid):
            rows_x.append(
                {
                    "series": series,
                    "moneyness_x": float(x),
                    "mae": float(mae_x[i]),
                    "rmse": float(rmse_x[i]),
                    "error_mode": "absolute",
                    "n_points": int(n_days * nt),
                }
            )

        mae_grid = np.mean(abs_err, axis=0)
        rmse_grid = np.sqrt(np.mean(sq_err, axis=0))
        for i, x in enumerate(x_grid):
            for j, dte in enumerate(tenor_days):
                rows_grid.append(
                    {
                        "series": series,
                        "moneyness_x": float(x),
                        "dte": int(dte),
                        "mae": float(mae_grid[i, j]),
                        "rmse": float(rmse_grid[i, j]),
                        "error_mode": "absolute",
                        "n_points": int(n_days),
                    }
                )

    return pd.DataFrame(rows_dte), pd.DataFrame(rows_x), pd.DataFrame(rows_grid)


def _calibrate_regime_surface_blend(
    *,
    obs_surface: np.ndarray,
    model_surface: np.ndarray,
    tree_surface: np.ndarray,
    asset_ids: np.ndarray,
    shrinkage: float = 30.0,
    gate_temperature: float = 0.20,
) -> np.ndarray:
    """Estimate per-asset/per-grid blending weights using validation residual risk.

    Weight convention: blended = w * model + (1 - w) * tree.
    """
    obs = np.asarray(obs_surface, dtype=np.float64)
    pred_m = np.asarray(model_surface, dtype=np.float64)
    pred_t = np.asarray(tree_surface, dtype=np.float64)
    aids = np.asarray(asset_ids, dtype=np.int32).reshape(-1)
    if obs.shape != pred_m.shape or obs.shape != pred_t.shape:
        raise RuntimeError(
            "Blend calibration shape mismatch: "
            f"obs={obs.shape} model={pred_m.shape} tree={pred_t.shape}."
        )
    if obs.ndim != 3 or len(aids) != obs.shape[0]:
        raise RuntimeError(
            "Blend calibration expects [n_days, nx, nt] surfaces and matching asset_ids."
        )

    n_assets = int(max(np.max(aids) + 1, 1)) if len(aids) else 1
    nx = int(obs.shape[1])
    nt = int(obs.shape[2])
    w = np.full((n_assets, nx, nt), 0.5, dtype=np.float64)

    err_m = (pred_m - obs) ** 2
    err_t = (pred_t - obs) ** 2
    global_m = np.mean(err_m, axis=0)
    global_t = np.mean(err_t, axis=0)
    global_score = (global_t - global_m) / np.clip(global_m + global_t, 1e-10, None)

    shrink = max(float(shrinkage), 0.0)
    for aid in range(n_assets):
        idx = np.where(aids == aid)[0]
        if len(idx) == 0:
            score = global_score
        else:
            local_m = np.mean(err_m[idx], axis=0)
            local_t = np.mean(err_t[idx], axis=0)
            local_score = (local_t - local_m) / np.clip(local_m + local_t, 1e-10, None)
            lam = len(idx) / (len(idx) + shrink) if shrink > 0.0 else 1.0
            score = lam * local_score + (1.0 - lam) * global_score

        temp = max(float(gate_temperature), 1e-3)
        w[aid] = 1.0 / (1.0 + np.exp(-np.clip(score / temp, -30.0, 30.0)))

    return np.clip(w, 0.0, 1.0).astype(np.float32)


def _resolve_num_workers(requested: int, n_tasks: int) -> int:
    if n_tasks <= 1:
        return 1
    if requested == 1:
        return 1
    if requested <= 0:
        cpu = os.cpu_count() or 1
        return max(1, min(cpu - 1, n_tasks))
    return max(1, min(requested, n_tasks))


def _noarb_for_day(
    obs_surface: np.ndarray,
    pred_surface: np.ndarray,
    x_grid: np.ndarray,
    tenor_days: np.ndarray,
) -> tuple[float, float, float, float]:
    cal_obs = float(calendar_violations(obs_surface[None, ...], tenor_days)[0])
    cal_pred = float(calendar_violations(pred_surface[None, ...], tenor_days)[0])
    bfly_obs = float(butterfly_violations(obs_surface[None, ...], x_grid, tenor_days)[0])
    bfly_pred = float(butterfly_violations(pred_surface[None, ...], x_grid, tenor_days)[0])
    return cal_obs, cal_pred, bfly_obs, bfly_pred


def _augment_context_with_contract_intraday(
    *,
    context: np.ndarray,
    features: np.ndarray,
    feature_names: list[str],
    date_idx: np.ndarray,
    n_dates: int,
) -> np.ndarray:
    if len(features) == 0 or len(feature_names) == 0:
        return context

    keep = {
        "rel_spread",
        "log_volume",
        "log_open_interest",
        "liquidity",
        "abs_x",
        "intraday_ret_oc",
        "intraday_range_frac",
        "intraday_rv_1m",
        "intraday_vwap_dev",
        "intraday_volume_cv",
        "intraday_log_bar_count",
        "intraday_log_volume_per_bar",
    }
    feat_cols = [i for i, n in enumerate(feature_names) if n in keep]
    if not feat_cols:
        return context

    valid = (date_idx >= 0) & (date_idx < n_dates)
    if not np.any(valid):
        return context
    rows = np.where(valid)[0]
    didx = date_idx[rows]

    k = len(feat_cols)
    sums = np.zeros((n_dates, k), dtype=np.float64)
    counts = np.zeros(n_dates, dtype=np.int32)
    np.add.at(counts, didx, 1)
    for j, col_idx in enumerate(feat_cols):
        np.add.at(sums[:, j], didx, features[rows, col_idx].astype(np.float64))

    agg = np.full((n_dates, k), np.nan, dtype=np.float64)
    has = counts > 0
    if np.any(has):
        agg[has] = sums[has] / counts[has, None]
    ix = np.arange(n_dates, dtype=np.float64)
    for j in range(k):
        col = agg[:, j]
        finite = np.isfinite(col)
        if np.any(finite):
            col[~finite] = np.interp(ix[~finite], ix[finite], col[finite])
        else:
            col[:] = 0.0
        agg[:, j] = col

    return np.concatenate([context.astype(np.float32), agg.astype(np.float32)], axis=1)


def _tree_boost_surface_nextday_forecast(
    *,
    iv_surface_obs: np.ndarray,
    context_scaled: np.ndarray,
    asset_ids: np.ndarray,
    forecast_entry_idx: np.ndarray,
    x_grid: np.ndarray,
    tenor_days: np.ndarray,
    train_date_mask: np.ndarray,
    min_history: int = 60,
    max_factors: int = 8,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Tree-boosted one-step baseline with per-asset PCA factor targets."""
    nx = int(len(x_grid))
    nt = int(len(tenor_days))
    if len(forecast_entry_idx) == 0:
        return np.empty((0, nx, nt), dtype=np.float32), {
            "family": "hist_gradient_boosting_surface_pca",
            "trained_assets": 0,
            "min_history": int(max(10, min_history)),
            "fallback_persistence_days": 0,
        }

    n_dates, sx, st = iv_surface_obs.shape
    nx = int(sx)
    nt = int(st)
    flat = iv_surface_obs.reshape(n_dates, -1).astype(np.float64)
    ctx = np.asarray(context_scaled, dtype=np.float64)
    train_mask = np.asarray(train_date_mask, dtype=bool).reshape(-1)
    if train_mask.shape[0] != n_dates:
        train_mask = np.zeros(n_dates, dtype=bool)

    preds_flat = flat[forecast_entry_idx].copy()  # persistence fallback
    fallback_days = int(len(forecast_entry_idx))
    trained_assets = 0
    min_history = int(max(10, min_history))
    max_factors = int(max(1, max_factors))

    if HistGradientBoostingRegressor is None:
        return np.clip(preds_flat, 1e-4, 4.0).reshape(-1, nx, nt).astype(np.float32), {
            "family": "hist_gradient_boosting_unavailable",
            "trained_assets": 0,
            "min_history": int(min_history),
            "fallback_persistence_days": int(fallback_days),
        }

    same_asset_next = asset_ids[:-1] == asset_ids[1:] if n_dates > 1 else np.array([], dtype=bool)
    train_pairs = np.where(same_asset_next & train_mask[:-1] & train_mask[1:])[0].astype(np.int32)
    if len(train_pairs) == 0:
        return np.clip(preds_flat, 1e-4, 4.0).reshape(-1, nx, nt).astype(np.float32), {
            "family": "hist_gradient_boosting_surface_pca",
            "trained_assets": 0,
            "min_history": int(min_history),
            "fallback_persistence_days": int(fallback_days),
        }

    entry_pos = {int(e): i for i, e in enumerate(forecast_entry_idx)}
    forecast_assets = np.unique(asset_ids[forecast_entry_idx]).astype(np.int32)

    for aid in forecast_assets:
        asset_entries = forecast_entry_idx[asset_ids[forecast_entry_idx] == aid]
        if len(asset_entries) == 0:
            continue
        asset_train_pairs = train_pairs[asset_ids[train_pairs] == aid]
        if len(asset_train_pairs) < min_history:
            continue

        ref_idx = np.unique(np.concatenate([asset_train_pairs, asset_train_pairs + 1]))
        ref = flat[ref_idx]
        ref_mu = np.mean(ref, axis=0)
        ref_centered = ref - ref_mu
        try:
            _, _, vt = np.linalg.svd(ref_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        if vt.size == 0:
            continue
        k = int(max(1, min(max_factors, vt.shape[0], len(asset_train_pairs) - 1)))
        components = vt[:k]

        x_train = np.concatenate([flat[asset_train_pairs], ctx[asset_train_pairs]], axis=1)
        y_train = (flat[asset_train_pairs + 1] - ref_mu) @ components.T
        x_pred = np.concatenate([flat[asset_entries], ctx[asset_entries]], axis=1)
        x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        x_pred = np.nan_to_num(x_pred, nan=0.0, posinf=0.0, neginf=0.0)

        if len(x_train) < min_history or len(x_pred) == 0:
            continue

        pred_factors = np.zeros((len(x_pred), k), dtype=np.float64)
        ok = True
        for j in range(k):
            y_col = y_train[:, j]
            if np.allclose(y_col, y_col[0]):
                pred_factors[:, j] = y_col[0]
                continue
            try:
                model = HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.05,
                    max_iter=300,
                    max_depth=6,
                    min_samples_leaf=20,
                    l2_regularization=1e-3,
                    random_state=13,
                )
                model.fit(x_train, y_col)
                pred_factors[:, j] = model.predict(x_pred)
            except Exception:
                ok = False
                break
        if not ok:
            continue

        pred_asset = np.clip(ref_mu + (pred_factors @ components), 1e-4, 4.0)
        assigned = 0
        for i, e in enumerate(asset_entries):
            pos = entry_pos.get(int(e))
            if pos is None or i >= len(pred_asset):
                continue
            preds_flat[pos] = pred_asset[i]
            assigned += 1
        if assigned > 0:
            trained_assets += 1
            fallback_days -= assigned

    fallback_days = int(max(0, fallback_days))
    return np.clip(preds_flat, 1e-4, 4.0).reshape(-1, nx, nt).astype(np.float32), {
        "family": "hist_gradient_boosting_surface_pca",
        "trained_assets": int(trained_assets),
        "min_history": int(min_history),
        "fallback_persistence_days": int(fallback_days),
    }


def _parametric_factor_har_forecast(
    iv_surface_obs: np.ndarray,
    forecast_entry_idx: np.ndarray,
    *,
    n_factors: int = 3,
    ridge: float = 1e-4,
    min_history: int = 40,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """One-step parametric baseline: PCA factors + expanding HAR dynamics.

    Literature-aligned non-DL baseline:
    - extract a small set of surface factors (PCA)
    - fit HAR(1, 5, 22) dynamics on factors with ridge stabilization
    - forecast next-day factors and reconstruct the full surface
    """
    if len(forecast_entry_idx) == 0:
        return np.empty((0, *iv_surface_obs.shape[1:]), dtype=np.float32), {
            "used_factors": 0,
            "ridge": float(ridge),
            "min_history": int(min_history),
            "fallback_persistence_days": 0,
        }

    t_total, nx, nt = iv_surface_obs.shape
    flat = iv_surface_obs.reshape(t_total, -1).astype(np.float64)
    feature_dim = flat.shape[1]

    # Fit PCA basis on an initial historical block only (no future leakage).
    week_w, month_w = 5, 22
    min_history = max(int(min_history), month_w + 5)
    fit_end = min(t_total - 1, max(min_history, 20))
    ref = flat[: fit_end + 1]
    ref_mu = np.mean(ref, axis=0)
    ref_centered = ref - ref_mu
    try:
        _, _, vt = np.linalg.svd(ref_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        fallback = flat[forecast_entry_idx].reshape(-1, nx, nt).astype(np.float32)
        return fallback, {
            "used_factors": 0,
            "ridge": float(ridge),
            "min_history": int(min_history),
            "fallback_persistence_days": int(len(forecast_entry_idx)),
        }

    k = max(1, min(int(n_factors), int(vt.shape[0])))
    components = vt[:k]  # shape [k, feature_dim]
    factors_all = (flat - ref_mu) @ components.T  # [t_total, k]

    preds = np.empty((len(forecast_entry_idx), feature_dim), dtype=np.float64)
    fallback_days = 0

    for i, entry in enumerate(forecast_entry_idx):
        e = int(entry)
        if e < min_history or e < month_w:
            preds[i] = flat[e]
            fallback_days += 1
            continue

        rows: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for t in range(month_w - 1, e):
            f_1 = factors_all[t]
            f_5 = np.mean(factors_all[t - (week_w - 1) : t + 1], axis=0)
            f_22 = np.mean(factors_all[t - (month_w - 1) : t + 1], axis=0)
            feat = np.concatenate([np.array([1.0], dtype=np.float64), f_1, f_5, f_22], axis=0)
            rows.append(feat)
            targets.append(factors_all[t + 1])

        if len(rows) < (3 * k + 5):
            preds[i] = flat[e]
            fallback_days += 1
            continue

        x_aug = np.vstack(rows)  # [n, 1+3k]
        y = np.vstack(targets)   # [n, k]
        reg = float(max(ridge, 0.0)) * np.eye(x_aug.shape[1], dtype=np.float64)
        reg[0, 0] = 0.0  # do not penalize intercept

        xtx = x_aug.T @ x_aug + reg
        xty = x_aug.T @ y
        try:
            w = np.linalg.solve(xtx, xty)  # [1+k, k]
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(xtx) @ xty

        f_entry = factors_all[e]
        f_entry_5 = np.mean(factors_all[e - (week_w - 1) : e + 1], axis=0)
        f_entry_22 = np.mean(factors_all[e - (month_w - 1) : e + 1], axis=0)
        feat_entry = np.concatenate([np.array([1.0], dtype=np.float64), f_entry, f_entry_5, f_entry_22], axis=0)
        f_pred = feat_entry @ w
        preds[i] = ref_mu + (f_pred @ components)

    preds = np.clip(preds, 1e-4, 4.0).reshape(-1, nx, nt).astype(np.float32)
    return preds, {
        "family": "pca_factor_har",
        "used_factors": int(k),
        "ridge": float(ridge),
        "min_history": int(min_history),
        "windows": "1,5,22",
        "fallback_persistence_days": int(fallback_days),
    }


def evaluate(
    run_dir: Path,
    dataset_path: Path,
    device: str | None = None,
    num_workers: int = 0,
    surface_dynamics_only: bool = True,
    baseline_factor_dim: int = 3,
    baseline_ridge: float = 1e-4,
    baseline_min_history: int = 40,
    tree_baseline_strict: bool = True,
) -> Path:
    run_dir = run_dir.resolve()
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    ds = _load_dataset(dataset_path)

    dev = torch.device(device) if device else device_auto()
    bundle = ModelBundle.load(run_dir / "model.pt", device=dev)
    model = bundle.model.to(dev).eval()

    dates = ds["dates"].astype(str)
    n_dates = len(dates)
    asset_ids = ds.get("asset_ids", np.zeros(n_dates, dtype=np.int32)).astype(np.int32)
    asset_names = ds.get("asset_names", np.array(["ASSET"], dtype=str)).astype(str).tolist()
    x_grid = ds["x_grid"].astype(np.float32)
    tenor_days = ds["tenor_days"].astype(np.int32)

    surface_variable_raw = ds.get("surface_variable")
    if surface_variable_raw is None:
        surface_variable = "total_variance" if "surface" in ds else "iv"
    else:
        sv = surface_variable_raw
        if isinstance(sv, np.ndarray):
            if sv.shape == ():
                sv = sv.item()
            elif sv.size > 0:
                sv = sv.reshape(-1)[0]
        surface_variable = _surface_variable_name(str(sv))

    surface_raw_obs = ds["surface"].astype(np.float32) if "surface" in ds else ds["iv_surface"].astype(np.float32)
    iv_surface_obs = _surface_to_iv_numpy(surface_raw_obs, tenor_days, surface_variable)
    surface_flat = surface_raw_obs.reshape(n_dates, -1)
    surface_scaled = bundle.surface_scaler.transform(surface_flat)

    context = ds["context"].astype(np.float32)
    contract_features = ds["contract_features"].astype(np.float32)
    contract_feature_names = ds.get("contract_feature_names", np.array([], dtype=str)).astype(str).tolist()
    contract_date_idx = ds["contract_date_index"].astype(np.int32)
    contract_asset_id = ds.get("contract_asset_id", asset_ids[contract_date_idx]).astype(np.int32)
    expected_context_dim = int(bundle.context_scaler.mean.shape[1])
    if int(context.shape[1]) != expected_context_dim:
        context_aug = _augment_context_with_contract_intraday(
            context=context,
            features=contract_features,
            feature_names=contract_feature_names,
            date_idx=contract_date_idx,
            n_dates=n_dates,
        )
        if int(context_aug.shape[1]) != expected_context_dim:
            raise RuntimeError(
                "Context dimension mismatch for evaluation dataset and model scaler: "
                f"dataset_context_dim={context.shape[1]}, "
                f"augmented_context_dim={context_aug.shape[1]}, "
                f"expected_context_dim={expected_context_dim}"
            )
        context = context_aug
    context_scaled = bundle.context_scaler.transform(context)

    surface_only = bool(surface_dynamics_only)
    contract_scaled = bundle.contract_scaler.transform(contract_features)
    contract_price_target = ds["contract_price_target"].astype(np.float32)
    contract_fill_target = ds["contract_fill_target"].astype(np.float32)
    contract_symbol = ds["contract_symbol"].astype(str)
    contract_mid = ds["contract_mid"].astype(np.float32)
    contract_forward = ds.get("contract_forward", ds.get("contract_spot")).astype(np.float32)
    contract_mid_norm = contract_mid / np.clip(contract_forward, 1e-6, None)

    split_train_frac, split_val_frac, split_mode = _split_config_from_train_config(run_dir)
    tr_dates, va_dates, te_dates = _date_splits_for_mode(
        n_dates=n_dates,
        asset_ids=asset_ids,
        train_frac=split_train_frac,
        val_frac=split_val_frac,
        split_mode=split_mode,
    )
    train_dates = tr_dates.astype(np.int32)
    val_dates = va_dates.astype(np.int32)
    test_dates = te_dates.astype(np.int32)
    _assert_no_train_test_date_overlap(
        dates=dates,
        train_idx=train_dates.astype(np.int64),
        test_idx=test_dates.astype(np.int64),
    )
    # Use inference_mode for lower overhead than no_grad during evaluation.
    with torch.inference_mode():
        sf = torch.as_tensor(surface_scaled, dtype=torch.float32, device=dev)
        mu, _ = model.encode(sf)
        recon_scaled = model.decode(mu)
        z_all_t = torch.as_tensor(mu, dtype=torch.float32, device=dev)
        ctx_t = torch.as_tensor(context_scaled, dtype=torch.float32, device=dev)
        asset_t = torch.as_tensor(asset_ids, dtype=torch.long, device=dev)
        z_next_t = model.forward_dynamics(z_all_t, ctx_t, asset_id=asset_t)
        forecast_scaled = model.decode(z_next_t)

        recon_raw = bundle.surface_scaler.inverse_transform(to_numpy(recon_scaled)).reshape(surface_raw_obs.shape)
        forecast_raw = bundle.surface_scaler.inverse_transform(to_numpy(forecast_scaled)).reshape(surface_raw_obs.shape)
        recon_iv = _surface_to_iv_numpy(recon_raw.astype(np.float32), tenor_days, surface_variable)
        forecast_iv = _surface_to_iv_numpy(forecast_raw.astype(np.float32), tenor_days, surface_variable)
        z_all = to_numpy(mu)
        z_next = to_numpy(z_next_t)

        price_pred = np.array([], dtype=np.float32)
        price_pred_next = np.array([], dtype=np.float32)
        exec_prob = np.array([], dtype=np.float32)
        if not surface_only:
            cf = torch.as_tensor(contract_scaled, dtype=torch.float32, device=dev)
            z_contract = torch.as_tensor(z_all[contract_date_idx], dtype=torch.float32, device=dev)
            z_contract_next = torch.as_tensor(z_next[contract_date_idx], dtype=torch.float32, device=dev)
            contract_asset_t = torch.as_tensor(contract_asset_id, dtype=torch.long, device=dev)
            price_scaled_pred = to_numpy(model.forward_pricer(z_contract, cf, asset_id=contract_asset_t)).reshape(-1, 1)
            price_scaled_pred_next = to_numpy(model.forward_pricer(z_contract_next, cf, asset_id=contract_asset_t)).reshape(-1, 1)
            exec_logit = to_numpy(model.forward_execution_logit(z_contract, cf, asset_id=contract_asset_t)).reshape(-1)
            price_pred = bundle.price_scaler.inverse_transform(price_scaled_pred).reshape(-1)
            price_pred_next = bundle.price_scaler.inverse_transform(price_scaled_pred_next).reshape(-1)
            exec_prob = 1.0 / (1.0 + np.exp(-np.clip(exec_logit, -60.0, 60.0)))

    metrics: dict[str, float | int | str] = {
        "surface_dynamics_focus_mode": "surface_dynamics_only" if surface_only else "full_contract_metrics",
        "contract_metrics_included": bool(not surface_only),
        "surface_variable": surface_variable,
        "error_metric_mode": "absolute",
        "error_metric_denominator_epsilon": 0.0,
        "n_assets": int(max(asset_ids) + 1) if len(asset_ids) else 1,
        "eval_train_frac": float(split_train_frac),
        "eval_val_frac": float(split_val_frac),
        "eval_split_mode": str(split_mode),
        "eval_test_dates": int(len(test_dates)),
    }

    target_next_price_norm = np.array([], dtype=np.float32)
    target_next_return = np.array([], dtype=np.float32)
    pred_next_return = np.array([], dtype=np.float32)

    if not surface_only:
        mask_test_contracts = np.isin(contract_date_idx, test_dates)

        y_true = contract_price_target[mask_test_contracts]
        y_pred = price_pred[mask_test_contracts]
        e_true = contract_fill_target[mask_test_contracts]
        e_prob = exec_prob[mask_test_contracts]

        metrics.update(
            {
                "test_contracts": int(mask_test_contracts.sum()),
                "price_rmse": rmse(y_pred, y_true),
                "price_mae": mae(y_pred, y_true),
                "price_r2_same_day": r2(y_pred, y_true),
                "price_r2": float("nan"),
                "exec_brier": brier_score(e_prob, e_true),
                "exec_positive_rate": float(np.mean(e_true)) if len(e_true) else float("nan"),
            }
        )

        next_key_mid_norm: dict[tuple[int, str], float] = {}
        for i in range(len(contract_symbol)):
            next_key_mid_norm[(int(contract_date_idx[i]), str(contract_symbol[i]))] = float(contract_mid_norm[i])

        target_next_price_norm = np.full(len(contract_symbol), np.nan, dtype=np.float32)
        for i in range(len(contract_symbol)):
            k = (int(contract_date_idx[i] + 1), str(contract_symbol[i]))
            if k in next_key_mid_norm:
                target_next_price_norm[i] = np.float32(next_key_mid_norm[k])

        pred_next_return = (price_pred_next - contract_price_target) / np.clip(contract_price_target, 1e-6, None)
        target_next_return = (target_next_price_norm - contract_price_target) / np.clip(contract_price_target, 1e-6, None)

        mask_test_next = mask_test_contracts & np.isfinite(target_next_price_norm)
        y_next_true = target_next_price_norm[mask_test_next]
        y_next_pred = price_pred_next[mask_test_next]
        r_next_true = target_next_return[mask_test_next]
        r_next_pred = pred_next_return[mask_test_next]

        if len(y_next_true) > 0:
            metrics["next_test_contracts"] = int(len(y_next_true))
            metrics["next_price_rmse"] = rmse(y_next_pred, y_next_true)
            metrics["next_price_mae"] = mae(y_next_pred, y_next_true)
            metrics["next_price_r2"] = r2(y_next_pred, y_next_true)
            metrics["price_r2"] = metrics["next_price_r2"]
            metrics["price_r2_source"] = "next_day"
            metrics["next_return_directional_acc"] = float(np.mean(np.sign(r_next_pred) == np.sign(r_next_true)))
            metrics["next_return_corr"] = float(np.corrcoef(r_next_pred, r_next_true)[0, 1]) if len(r_next_true) > 1 else float("nan")

            y_next_base = contract_price_target[mask_test_next]
            metrics["next_price_rmse_baseline_midcarry"] = rmse(y_next_base, y_next_true)
            metrics["next_price_mae_baseline_midcarry"] = mae(y_next_base, y_next_true)
            metrics["next_price_r2_baseline_midcarry"] = r2(y_next_base, y_next_true)
        else:
            metrics["next_test_contracts"] = 0
            metrics["next_price_rmse"] = float("nan")
            metrics["next_price_mae"] = float("nan")
            metrics["next_price_r2"] = float("nan")
            metrics["price_r2"] = metrics["price_r2_same_day"]
            metrics["price_r2_source"] = "same_day_fallback"
            metrics["next_return_directional_acc"] = float("nan")
            metrics["next_return_corr"] = float("nan")

        if len(e_true) > 0 and np.isfinite(metrics.get("exec_positive_rate", np.nan)):
            p0 = float(np.clip(metrics["exec_positive_rate"], 1e-6, 1.0 - 1e-6))
            metrics["exec_brier_baseline_constant"] = brier_score(np.full_like(e_true, p0, dtype=float), e_true)
        else:
            metrics["exec_brier_baseline_constant"] = float("nan")

    pred_test = recon_iv[test_dates]
    obs_test = iv_surface_obs[test_dates]
    # Same-day reconstruction quality (helps diagnose representation learning).
    metrics["surface_iv_rmse"] = rmse(pred_test, obs_test)
    metrics["surface_iv_mae"] = mae(pred_test, obs_test)
    metrics["surface_recon_iv_rmse"] = metrics["surface_iv_rmse"]
    metrics["surface_recon_iv_mae"] = metrics["surface_iv_mae"]

    # 1-step ahead surface forecast quality (this is the metric that matters for
    # trading and any forward-looking strategy).
    same_asset_next = asset_ids[:-1] == asset_ids[1:] if n_dates > 1 else np.array([], dtype=bool)
    test_mask = np.zeros(int(n_dates), dtype=bool)
    test_mask[test_dates] = True
    forecast_entry_idx = np.where(same_asset_next & test_mask[:-1] & test_mask[1:])[0].astype(np.int32)
    forecast_target_idx = forecast_entry_idx + 1
    metrics["surface_forecast_days"] = int(len(forecast_entry_idx))
    base_forecast_persistence = np.empty((0, *iv_surface_obs.shape[1:]), dtype=np.float32)
    base_forecast_tree = np.empty((0, *iv_surface_obs.shape[1:]), dtype=np.float32)
    base_forecast_param = np.empty((0, *iv_surface_obs.shape[1:]), dtype=np.float32)
    forecast_err_by_dte = pd.DataFrame()
    forecast_err_by_moneyness = pd.DataFrame()
    forecast_err_grid = pd.DataFrame()
    if len(forecast_entry_idx) > 0:
        pred_forecast = forecast_iv[forecast_entry_idx].astype(np.float32)
        obs_forecast = iv_surface_obs[forecast_target_idx]

        # Baseline: persistence (tomorrow's surface = today's surface).
        base_forecast_persistence = iv_surface_obs[forecast_entry_idx]
        metrics["surface_forecast_iv_rmse_baseline_persistence"] = rmse(base_forecast_persistence, obs_forecast)
        metrics["surface_forecast_iv_mae_baseline_persistence"] = mae(base_forecast_persistence, obs_forecast)

        # Baseline: tree-boosted next-day surface mapping trained on train+val only.
        if tree_baseline_strict and HistGradientBoostingRegressor is None:
            raise RuntimeError(
                "Tree baseline is unavailable because scikit-learn (HistGradientBoostingRegressor) "
                "could not be imported in the active environment. "
                "Install it in this env, e.g. `.venv/bin/python -m pip install scikit-learn`."
            )
        train_val_mask = np.zeros(int(n_dates), dtype=bool)
        train_val_mask[np.setdiff1d(np.arange(n_dates), test_dates)] = True
        base_forecast_tree, tree_meta = _tree_boost_surface_nextday_forecast(
            iv_surface_obs=iv_surface_obs,
            context_scaled=context_scaled,
            asset_ids=asset_ids,
            forecast_entry_idx=forecast_entry_idx,
            x_grid=x_grid,
            tenor_days=tenor_days,
            train_date_mask=train_val_mask,
            min_history=int(max(60, baseline_min_history)),
        )
        if tree_baseline_strict and int(tree_meta.get("trained_assets", 0)) <= 0:
            raise RuntimeError(
                "Tree baseline produced no trained assets and would fall back entirely to persistence. "
                "Refusing to continue in strict mode to avoid silent fallback."
            )

        # Regime-adaptive residual blend: calibrate per-asset/per-grid weights on
        # validation forecasts, then blend model and tree forecasts on test.
        val_same_asset = same_asset_next & np.isin(np.arange(max(n_dates - 1, 0)), val_dates)
        val_same_asset &= np.isin(np.arange(1, n_dates), val_dates)
        val_entry_idx = np.where(val_same_asset)[0].astype(np.int32)
        val_target_idx = val_entry_idx + 1
        blend_family = "model_tree_validation_residual_blend"
        blend_val_days = int(len(val_entry_idx))
        blend_shrinkage = 30.0
        blend_weight_by_asset_grid = np.empty((0, 0, 0), dtype=np.float32)

        if len(val_entry_idx) > 0:
            model_val = forecast_iv[val_entry_idx]
            tree_val, _ = _tree_boost_surface_nextday_forecast(
                iv_surface_obs=iv_surface_obs,
                context_scaled=context_scaled,
                asset_ids=asset_ids,
                forecast_entry_idx=val_entry_idx,
                x_grid=x_grid,
                tenor_days=tenor_days,
                train_date_mask=np.isin(np.arange(n_dates), train_dates),
                min_history=int(max(60, baseline_min_history)),
            )
            blend_weight_by_asset_grid = _calibrate_regime_surface_blend(
                obs_surface=iv_surface_obs[val_target_idx],
                model_surface=model_val,
                tree_surface=tree_val,
                asset_ids=asset_ids[val_entry_idx],
                shrinkage=blend_shrinkage,
            )
            blend_w_test = blend_weight_by_asset_grid[asset_ids[forecast_entry_idx]]
            pred_forecast_blend = (
                blend_w_test * pred_forecast + (1.0 - blend_w_test) * base_forecast_tree
            ).astype(np.float32)
        else:
            pred_forecast_blend = 0.5 * (pred_forecast + base_forecast_tree)
            blend_family = "model_tree_static_equal_blend"

        metrics["surface_forecast_baseline_primary"] = "tree_boost_surface_nextday"
        metrics["surface_forecast_baseline_tree_family"] = str(tree_meta.get("family", "tree_boost_surface_nextday"))
        metrics["surface_forecast_baseline_tree_trained_assets"] = int(tree_meta.get("trained_assets", 0))
        metrics["surface_forecast_baseline_tree_min_history"] = int(tree_meta.get("min_history", 0))
        metrics["surface_forecast_baseline_tree_fallback_days"] = int(tree_meta.get("fallback_persistence_days", 0))
        metrics["surface_forecast_iv_rmse_baseline_tree"] = rmse(base_forecast_tree, obs_forecast)
        metrics["surface_forecast_iv_mae_baseline_tree"] = mae(base_forecast_tree, obs_forecast)

        # Parametric baseline is not computed in this pipeline variant.
        base_forecast_param = np.full_like(base_forecast_tree, np.nan, dtype=np.float32)
        metrics["surface_forecast_baseline_parametric_family"] = "not_computed_replaced_by_tree"
        metrics["surface_forecast_baseline_parametric_factors"] = 0
        metrics["surface_forecast_baseline_parametric_ridge"] = float("nan")
        metrics["surface_forecast_baseline_parametric_min_history"] = 0
        metrics["surface_forecast_baseline_parametric_windows"] = "n/a"
        metrics["surface_forecast_baseline_parametric_fallback_days"] = 0
        metrics["surface_forecast_iv_rmse_baseline_parametric"] = float("nan")
        metrics["surface_forecast_iv_mae_baseline_parametric"] = float("nan")

        metrics["surface_forecast_blend_family"] = blend_family
        metrics["surface_forecast_blend_calibration_days"] = blend_val_days
        metrics["surface_forecast_blend_shrinkage"] = float(blend_shrinkage)
        if blend_weight_by_asset_grid.size > 0:
            metrics["surface_forecast_blend_weight_mean"] = float(np.mean(blend_weight_by_asset_grid))
            metrics["surface_forecast_blend_weight_std"] = float(np.std(blend_weight_by_asset_grid))
        else:
            metrics["surface_forecast_blend_weight_mean"] = float("nan")
            metrics["surface_forecast_blend_weight_std"] = float("nan")

        mse_model = float(np.mean((pred_forecast - obs_forecast) ** 2))
        mse_base_persistence = float(np.mean((base_forecast_persistence - obs_forecast) ** 2))
        mse_base_tree = float(np.mean((base_forecast_tree - obs_forecast) ** 2))
        mse_blend = float(np.mean((pred_forecast_blend - obs_forecast) ** 2))
        metrics["surface_forecast_iv_rmse_model_raw"] = float(np.sqrt(mse_model))
        metrics["surface_forecast_iv_mae_model_raw"] = mae(pred_forecast, obs_forecast)
        metrics["surface_forecast_iv_rmse"] = float(np.sqrt(mse_blend))
        metrics["surface_forecast_iv_mae"] = mae(pred_forecast_blend, obs_forecast)
        metrics["surface_forecast_skill_mse_vs_persistence"] = (
            float(1.0 - (mse_blend / mse_base_persistence)) if mse_base_persistence > 0 else float("nan")
        )
        metrics["surface_forecast_skill_mse_vs_tree"] = (
            float(1.0 - (mse_blend / mse_base_tree)) if mse_base_tree > 0 else float("nan")
        )
        metrics["surface_forecast_skill_mse_vs_tree_model_raw"] = (
            float(1.0 - (mse_model / mse_base_tree)) if mse_base_tree > 0 else float("nan")
        )
        metrics["surface_forecast_skill_mse_vs_parametric"] = float("nan")
        forecast_err_by_dte, forecast_err_by_moneyness, forecast_err_grid = _surface_forecast_error_profiles(
            obs_forecast=obs_forecast.astype(np.float32),
            pred_forecast=pred_forecast_blend.astype(np.float32),
            x_grid=x_grid.astype(np.float32),
            tenor_days=tenor_days.astype(np.int32),
            persistence_forecast=base_forecast_persistence.astype(np.float32),
            tree_forecast=base_forecast_tree.astype(np.float32),
        )
    else:
        metrics["surface_forecast_iv_rmse"] = float("nan")
        metrics["surface_forecast_iv_mae"] = float("nan")
        metrics["surface_forecast_iv_rmse_baseline_persistence"] = float("nan")
        metrics["surface_forecast_iv_mae_baseline_persistence"] = float("nan")
        metrics["surface_forecast_skill_mse_vs_persistence"] = float("nan")
        metrics["surface_forecast_iv_rmse_baseline_tree"] = float("nan")
        metrics["surface_forecast_iv_mae_baseline_tree"] = float("nan")
        metrics["surface_forecast_skill_mse_vs_tree"] = float("nan")
        metrics["surface_forecast_baseline_tree_family"] = (
            "hist_gradient_boosting_surface_pca" if HistGradientBoostingRegressor is not None else "hist_gradient_boosting_unavailable"
        )
        metrics["surface_forecast_baseline_tree_trained_assets"] = 0
        metrics["surface_forecast_baseline_tree_min_history"] = int(max(60, baseline_min_history))
        metrics["surface_forecast_baseline_tree_fallback_days"] = 0
        metrics["surface_forecast_iv_rmse_baseline_parametric"] = float("nan")
        metrics["surface_forecast_iv_mae_baseline_parametric"] = float("nan")
        metrics["surface_forecast_skill_mse_vs_parametric"] = float("nan")
        metrics["surface_forecast_baseline_primary"] = "tree_boost_surface_nextday"
        metrics["surface_forecast_baseline_parametric_family"] = "not_computed_replaced_by_tree"
        metrics["surface_forecast_baseline_parametric_factors"] = 0
        metrics["surface_forecast_baseline_parametric_ridge"] = float("nan")
        metrics["surface_forecast_baseline_parametric_min_history"] = 0
        metrics["surface_forecast_baseline_parametric_windows"] = "n/a"
        metrics["surface_forecast_baseline_parametric_fallback_days"] = 0

    metrics["surface_forecast_error_rows_by_dte"] = int(len(forecast_err_by_dte))
    metrics["surface_forecast_error_rows_by_moneyness"] = int(len(forecast_err_by_moneyness))
    metrics["surface_forecast_error_rows_grid"] = int(len(forecast_err_grid))

    workers = _resolve_num_workers(num_workers, len(test_dates))
    parallel_backend = "sequential"
    if len(test_dates) == 0:
        cal_obs = np.array([], dtype=np.float32)
        cal_pred = np.array([], dtype=np.float32)
        bfly_obs = np.array([], dtype=np.float32)
        bfly_pred = np.array([], dtype=np.float32)
    elif workers <= 1:
        rows = [_noarb_for_day(obs_test[i], pred_test[i], x_grid, tenor_days) for i in range(len(test_dates))]
        cal_obs = np.array([r[0] for r in rows], dtype=np.float32)
        cal_pred = np.array([r[1] for r in rows], dtype=np.float32)
        bfly_obs = np.array([r[2] for r in rows], dtype=np.float32)
        bfly_pred = np.array([r[3] for r in rows], dtype=np.float32)
    else:
        executor_cls = ProcessPoolExecutor
        parallel_backend = "process"
        try:
            ex_obj = executor_cls(max_workers=workers)
        except (PermissionError, OSError):
            ex_obj = ThreadPoolExecutor(max_workers=workers)
            parallel_backend = "thread"
        with ex_obj as ex:
            futures = [
                ex.submit(_noarb_for_day, obs_test[i], pred_test[i], x_grid, tenor_days)
                for i in range(len(test_dates))
            ]
            rows = [f.result() for f in futures]
        cal_obs = np.array([r[0] for r in rows], dtype=np.float32)
        cal_pred = np.array([r[1] for r in rows], dtype=np.float32)
        bfly_obs = np.array([r[2] for r in rows], dtype=np.float32)
        bfly_pred = np.array([r[3] for r in rows], dtype=np.float32)

    metrics["calendar_violation_obs_mean"] = float(np.mean(cal_obs)) if len(cal_obs) else float("nan")
    metrics["calendar_violation_pred_mean"] = float(np.mean(cal_pred)) if len(cal_pred) else float("nan")
    metrics["butterfly_violation_obs_mean"] = float(np.mean(bfly_obs)) if len(bfly_obs) else float("nan")
    metrics["butterfly_violation_pred_mean"] = float(np.mean(bfly_pred)) if len(bfly_pred) else float("nan")

    # No-arbitrage diagnostics for 1-step ahead forecasts (t -> t+1) on test.
    if len(forecast_entry_idx) > 0:
        pred_forecast = forecast_iv[forecast_entry_idx]
        obs_forecast = iv_surface_obs[forecast_target_idx]

        rows_f = [
            _noarb_for_day(obs_forecast[i], pred_forecast[i], x_grid, tenor_days) for i in range(len(forecast_entry_idx))
        ]
        cal_obs_f = np.array([r[0] for r in rows_f], dtype=np.float32)
        cal_pred_f = np.array([r[1] for r in rows_f], dtype=np.float32)
        bfly_obs_f = np.array([r[2] for r in rows_f], dtype=np.float32)
        bfly_pred_f = np.array([r[3] for r in rows_f], dtype=np.float32)

        metrics["calendar_violation_forecast_obs_mean"] = float(np.mean(cal_obs_f)) if len(cal_obs_f) else float("nan")
        metrics["calendar_violation_forecast_pred_mean"] = float(np.mean(cal_pred_f)) if len(cal_pred_f) else float("nan")
        metrics["butterfly_violation_forecast_obs_mean"] = float(np.mean(bfly_obs_f)) if len(bfly_obs_f) else float("nan")
        metrics["butterfly_violation_forecast_pred_mean"] = float(np.mean(bfly_pred_f)) if len(bfly_pred_f) else float("nan")
    else:
        cal_obs_f = np.array([], dtype=np.float32)
        cal_pred_f = np.array([], dtype=np.float32)
        bfly_obs_f = np.array([], dtype=np.float32)
        bfly_pred_f = np.array([], dtype=np.float32)
        metrics["calendar_violation_forecast_obs_mean"] = float("nan")
        metrics["calendar_violation_forecast_pred_mean"] = float("nan")
        metrics["butterfly_violation_forecast_obs_mean"] = float("nan")
        metrics["butterfly_violation_forecast_pred_mean"] = float("nan")

    metrics["num_workers"] = workers
    metrics["parallel_backend"] = parallel_backend

    if not surface_only:
        contract_underlying = ds.get("contract_underlying", np.full(len(contract_date_idx), "", dtype="<U1")).astype(str)
        contract_df = pd.DataFrame(
            {
                "date_idx": contract_date_idx,
                "asset_id": contract_asset_id,
                "asset": contract_underlying,
                "date": ds["contract_date"].astype(str),
                "symbol": ds["contract_symbol"].astype(str),
                "call_put": ds["contract_call_put"].astype(str),
                "dte": ds["contract_dte"].astype(int),
                "strike": ds["contract_strike"].astype(float),
                "spot": ds["contract_spot"].astype(float),
                "forward": ds.get("contract_forward", ds["contract_spot"]).astype(float),
                "spread": ds.get("contract_spread", np.zeros_like(contract_price_target)).astype(float),
                "mid": ds["contract_mid"].astype(float),
                "target_price_norm": contract_price_target,
                "pred_price_norm": price_pred,
                "target_next_price_norm": target_next_price_norm,
                "pred_next_price_norm": price_pred_next,
                "target_next_return": target_next_return,
                "pred_next_return": pred_next_return,
                "target_fill": contract_fill_target,
                "pred_fill_prob": exec_prob,
            }
        )
        contract_df.to_parquet(eval_dir / "contract_predictions.parquet", index=False)
    else:
        legacy_contract_path = eval_dir / "contract_predictions.parquet"
        if legacy_contract_path.exists():
            legacy_contract_path.unlink()

    noarb_dates = pd.DataFrame(
        {
            "date": dates[test_dates],
            "asset_id": asset_ids[test_dates],
            "asset": np.array(
                [
                    asset_names[int(i)] if 0 <= int(i) < len(asset_names) else f"asset_{int(i)}"
                    for i in asset_ids[test_dates]
                ],
                dtype=object,
            ),
            "calendar_obs": cal_obs,
            "calendar_pred": cal_pred,
            "butterfly_obs": bfly_obs,
            "butterfly_pred": bfly_pred,
        }
    )
    noarb_dates.to_parquet(eval_dir / "noarb_test_dates.parquet", index=False)

    if len(forecast_entry_idx) > 0:
        noarb_forecast = pd.DataFrame(
            {
                "date_entry": dates[forecast_entry_idx],
                "date_target": dates[forecast_target_idx],
                "asset_id": asset_ids[forecast_entry_idx],
                "asset": np.array(
                    [
                        asset_names[int(i)] if 0 <= int(i) < len(asset_names) else f"asset_{int(i)}"
                        for i in asset_ids[forecast_entry_idx]
                    ],
                    dtype=object,
                ),
                "calendar_obs": cal_obs_f,
                "calendar_pred": cal_pred_f,
                "butterfly_obs": bfly_obs_f,
                "butterfly_pred": bfly_pred_f,
            }
        )
        noarb_forecast.to_parquet(eval_dir / "noarb_forecast_test_dates.parquet", index=False)

    latent = pd.DataFrame(z_all, columns=[f"z_{i}" for i in range(z_all.shape[1])])
    latent.insert(0, "date", dates)
    latent.insert(1, "asset_id", asset_ids)
    latent.insert(
        2,
        "asset",
        np.array(
            [
                asset_names[int(i)] if 0 <= int(i) < len(asset_names) else f"asset_{int(i)}"
                for i in asset_ids
            ],
            dtype=object,
        ),
    )
    latent.to_parquet(eval_dir / "latent_states.parquet", index=False)

    profile_artifacts: list[tuple[str, pd.DataFrame]] = [
        ("surface_forecast_error_by_dte.parquet", forecast_err_by_dte),
        ("surface_forecast_error_by_moneyness.parquet", forecast_err_by_moneyness),
        ("surface_forecast_error_grid.parquet", forecast_err_grid),
    ]
    for name, frame in profile_artifacts:
        path = eval_dir / name
        if not frame.empty:
            frame.to_parquet(path, index=False)
        elif path.exists():
            path.unlink()

    np.savez_compressed(
        eval_dir / "surface_predictions.npz",
        dates=dates,
        asset_ids=asset_ids,
        asset_names=np.array(asset_names, dtype="<U32"),
        surface_variable=np.array(surface_variable),
        surface_obs=surface_raw_obs.astype(np.float32),
        surface_pred=recon_raw.astype(np.float32),
        surface_forecast=forecast_raw.astype(np.float32),
        iv_surface_obs=iv_surface_obs,
        iv_surface_pred=recon_iv.astype(np.float32),
        iv_surface_forecast=forecast_iv.astype(np.float32),
        iv_surface_forecast_baseline_persistence=base_forecast_persistence.astype(np.float32),
        iv_surface_forecast_baseline_tree=base_forecast_tree.astype(np.float32),
        iv_surface_forecast_baseline_parametric=base_forecast_param.astype(np.float32),
        x_grid=x_grid,
        tenor_days=tenor_days,
        test_date_index=test_dates,
        forecast_entry_index=forecast_entry_idx,
        forecast_target_index=forecast_target_idx,
    )

    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return eval_dir
