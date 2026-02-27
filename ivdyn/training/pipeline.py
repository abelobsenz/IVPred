"""Tree-boosted training pipeline for IV surface dynamics."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = object

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "scikit-learn is required for tree-based training. Install scikit-learn in the active environment."
    ) from exc
try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None

from ivdyn.model.scalers import ArrayScaler
from ivdyn.utils import make_run_dir


@dataclass(slots=True)
class TrainingConfig:
    out_dir: Path
    seed: int = 7
    train_frac: float = 0.70
    val_frac: float = 0.15
    split_mode: str = "by_asset_time"

    # Backward-compatible fields retained from the previous neural pipeline.
    latent_dim: int = 32
    vae_hidden: tuple[int, int] = (384, 192)
    dynamics_hidden: tuple[int, ...] = (256, 128)
    pricing_hidden: tuple[int, int] = (256, 128)
    execution_hidden: tuple[int, int] = (192, 96)
    model_dropout: float = 0.08

    vae_epochs: int = 120
    vae_batch_size: int = 32
    vae_lr: float = 2e-3
    vae_kl_beta: float = 0.02
    kl_warmup_epochs: int = 20
    noarb_lambda: float = 0.01
    noarb_butterfly_lambda: float = 0.005
    recon_huber_beta: float = 0.015

    head_epochs: int = 130
    dyn_batch_size: int = 64
    contract_batch_size: int = 2048
    head_lr: float = 1e-3
    rollout_steps: int = 3
    rollout_surface_lambda: float = 0.65
    rollout_calendar_lambda: float = 0.03
    rollout_butterfly_lambda: float = 0.02
    rollout_random_horizon: bool = True
    rollout_min_steps: int = 1
    rollout_teacher_forcing_start: float = 0.35
    rollout_teacher_forcing_end: float = 0.10
    rollout_surface_huber_beta: float = 0.015
    surface_weight_liq_alpha: float = 0.0
    surface_weight_spread_alpha: float = 0.0
    surface_weight_vega_alpha: float = 0.0
    surface_weight_clip_min: float = 1.0
    surface_weight_clip_max: float = 4.0
    surface_focus_alpha: float = 1.25
    surface_focus_x_min: float = 0.10
    surface_focus_x_scale: float = 0.03
    surface_focus_dte_scale_days: float = 21.0
    surface_focus_dte_max_days: float = 30.0
    surface_focus_neg_x_max: float = -0.20
    surface_focus_neg_weight_ratio: float = 0.35
    surface_focus_density_alpha: float = 0.0
    surface_focus_density_map_path: str | None = None

    joint_epochs: int = 120
    joint_lr: float = 5e-4
    joint_contract_batch_size: int = 4096
    joint_dyn_lambda: float = 1.0
    joint_price_lambda: float = 0.15
    joint_exec_lambda: float = 0.0
    joint_use_mu_deterministic: bool = True

    weight_decay: float = 1e-5
    price_risk_weight: float = 1.0
    exec_risk_weight: float = 0.5
    price_spread_inv_lambda: float = 0.35
    price_spread_clip_min: float = 0.02
    price_spread_clip_max: float = 3.0
    price_vega_power: float = 0.25
    price_vega_cap: float = 4.0
    risk_focus_abs_x: float = 0.06
    risk_focus_tau_days: float = 20.0
    exec_label_smoothing: float = 0.03
    exec_logit_l2: float = 2e-4
    surface_dynamics_only: bool = True
    context_winsor_quantile: float = 0.01
    context_z_clip: float = 5.0
    context_augment_from_contracts: bool = True
    context_augment_surface_history: bool = True
    dynamics_residual: bool = True
    asset_embed_dim: int = 8
    early_stop_patience: int = 20
    early_stop_min_delta: float = 1e-4
    lr_plateau_patience: int = 6
    lr_plateau_factor: float = 0.5
    min_lr: float = 1e-6

    # Tree-pipeline-specific knobs.
    tree_min_history: int = 60
    tree_pca_factors: int = 8
    tree_learning_rate: float = 0.03
    tree_max_iter: int = 260
    tree_max_depth: int = 7
    tree_min_samples_leaf: int = 16
    tree_l2_regularization: float = 1e-3
    tree_calibration_frac: float = 0.25
    tree_min_calibration_rows: int = 24
    tree_use_sample_weight: bool = True
    tree_ensemble_seeds: tuple[int, ...] = (13,)
    max_cpu_threads: int = 2
    tree_enable_residual_corrector: bool = False
    tree_residual_factors: int = 6
    tree_residual_learning_rate: float = 0.04
    tree_residual_max_iter: int = 180
    tree_residual_max_depth: int = 5
    tree_residual_min_samples_leaf: int = 24
    tree_residual_l2_regularization: float = 1e-3
    tree_residual_scale_max: float = 1.5
    tree_enable_regime_experts: bool = False
    tree_regime_min_rows: int = 80
    tree_enable_tenor_calibration: bool = False
    tree_tenor_calibration_shrinkage: float = 0.35
    tree_tenor_slope_min: float = 0.70
    tree_tenor_slope_max: float = 1.30

    # Architecture selector.
    model_arch: str = "tree_boost"

    # Option A: PCA factors + shared neural dynamics (TCN).
    option_a_seq_len: int = 20
    option_a_epochs: int = 80
    option_a_batch_size: int = 128
    option_a_eval_batch_size: int = 512
    option_a_lr: float = 8e-4
    option_a_weight_decay: float = 1e-5
    option_a_hidden_dim: int = 192
    option_a_tcn_layers: int = 4
    option_a_tcn_kernel_size: int = 3
    option_a_dropout: float = 0.08
    option_a_early_stop_patience: int = 12
    option_a_blend_alpha_min: float = 0.6
    option_a_blend_alpha_max: float = 1.4
    option_a_blend_alpha_steps: int = 9
    option_a_device: str = "auto"


def _load_dataset(dataset_path: Path) -> dict[str, np.ndarray]:
    z = np.load(dataset_path, allow_pickle=True)
    out: dict[str, np.ndarray] = {}
    for k in z.files:
        arr = z[k]
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
    overlap = sorted(train_dates.intersection(test_dates))
    if not overlap:
        return

    sample = ", ".join(overlap[:8])
    if len(overlap) > 8:
        sample = f"{sample}, ..."
    raise RuntimeError(
        "Date leakage detected: train/test sets share calendar dates "
        f"(count={len(overlap)}; sample={sample}). "
        "Use a strict time-forward split configuration (e.g. split_mode=global_time)."
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
    return np.clip(iv, 1e-4, 4.0).astype(np.float32)


def _iv_to_surface_raw_numpy(iv_surface: np.ndarray, tenor_days: np.ndarray, surface_variable: str) -> np.ndarray:
    if _surface_variable_name(surface_variable) != "total_variance":
        return np.clip(iv_surface, 1e-4, 4.0).astype(np.float32)
    tau = (tenor_days.astype(np.float32) / 365.0).reshape(1, 1, -1)
    w = np.square(np.clip(iv_surface, 1e-4, 4.0)) * np.clip(tau, 1e-6, None)
    return np.clip(w, 1e-8, 8.0).astype(np.float32)


def _load_focus_density_by_asset(
    *,
    map_path: Path,
    asset_names: list[str],
    nx: int,
    nt: int,
) -> np.ndarray:
    if not map_path.exists():
        raise RuntimeError(f"Focus density map not found: {map_path}")
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Focus density map must be a JSON object keyed by asset name.")

    missing = [a for a in asset_names if a not in payload]
    extra = [k for k in payload.keys() if k not in set(asset_names)]
    if missing or extra:
        raise RuntimeError(
            "Focus density map keys must match dataset assets exactly. "
            f"missing={missing} extra={extra}"
        )

    out = np.empty((len(asset_names), nx, nt), dtype=np.float64)
    for i, asset in enumerate(asset_names):
        arr = np.asarray(payload[asset], dtype=np.float64)
        if arr.shape == (nx * nt,):
            arr = arr.reshape(nx, nt)
        if arr.shape != (nx, nt):
            raise RuntimeError(
                f"Focus density for asset `{asset}` must have shape ({nx}, {nt}) "
                f"or flat length {nx * nt}; got {arr.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"Focus density for asset `{asset}` contains non-finite values.")
        if np.any(arr <= 0.0):
            raise RuntimeError(f"Focus density for asset `{asset}` must be strictly positive.")
        m = float(np.mean(arr))
        if m <= 0.0:
            raise RuntimeError(f"Focus density for asset `{asset}` has non-positive mean.")
        out[i] = arr / m

    return out.astype(np.float32)


def _augment_context_with_contract_intraday(
    *,
    context: np.ndarray,
    features: np.ndarray,
    feature_names: list[str],
    date_idx: np.ndarray,
    n_dates: int,
) -> tuple[np.ndarray, list[str]]:
    if len(features) == 0 or len(feature_names) == 0:
        return context, []

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
        return context, []

    valid = (date_idx >= 0) & (date_idx < n_dates)
    if not np.any(valid):
        return context, []
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

    added_names = [f"ctx_contract_mean_{feature_names[i]}" for i in feat_cols]
    context_aug = np.concatenate([context.astype(np.float32), agg.astype(np.float32)], axis=1)
    return context_aug, added_names


def _augment_context_with_surface_history(
    *,
    context: np.ndarray,
    surface_raw: np.ndarray,
    tenor_days: np.ndarray,
    asset_ids: np.ndarray,
    surface_variable: str,
) -> tuple[np.ndarray, list[str]]:
    """Augment context with lagged per-asset surface summary features."""
    n_dates = int(surface_raw.shape[0])
    if n_dates <= 1:
        return context, []

    iv = _surface_to_iv_numpy(surface_raw.astype(np.float32), tenor_days, surface_variable).astype(np.float64)
    nx = int(iv.shape[1])
    nt = int(iv.shape[2])
    if nx <= 0 or nt <= 0:
        return context, []

    x_idx_atm = int(nx // 2)
    x_idx_put = max(0, int(round(0.15 * (nx - 1))))
    x_idx_call = min(nx - 1, int(round(0.85 * (nx - 1))))
    t_idx_short = int(np.argmin(np.abs(np.asarray(tenor_days, dtype=np.float64) - 14.0)))
    t_idx_mid = int(np.argmin(np.abs(np.asarray(tenor_days, dtype=np.float64) - 30.0)))
    t_idx_long = int(np.argmin(np.abs(np.asarray(tenor_days, dtype=np.float64) - 90.0)))

    level = np.mean(iv, axis=(1, 2))
    atm_short = iv[:, x_idx_atm, t_idx_short]
    atm_mid = iv[:, x_idx_atm, t_idx_mid]
    atm_long = iv[:, x_idx_atm, t_idx_long]
    rr_short = iv[:, x_idx_call, t_idx_short] - iv[:, x_idx_put, t_idx_short]
    term_slope = atm_long - atm_short
    smile_curv = iv[:, x_idx_put, t_idx_mid] + iv[:, x_idx_call, t_idx_mid] - 2.0 * atm_mid

    base = np.stack([level, atm_short, atm_mid, atm_long, rr_short, term_slope, smile_curv], axis=1)
    lag1 = np.zeros_like(base)
    d1 = np.zeros_like(base)
    ewm5 = np.zeros_like(base)
    alpha = 2.0 / (5.0 + 1.0)

    aid = np.asarray(asset_ids, dtype=np.int32).reshape(-1)
    for a in np.unique(aid):
        idx = np.where(aid == a)[0]
        if len(idx) == 0:
            continue
        seq = np.sort(idx)
        prev = base[seq[0]].copy()
        ema = prev.copy()
        lag1[seq[0]] = prev
        ewm5[seq[0]] = ema
        d1[seq[0]] = 0.0
        for j in range(1, len(seq)):
            cur = base[seq[j]]
            lag1[seq[j]] = prev
            d1[seq[j]] = cur - prev
            ema = alpha * prev + (1.0 - alpha) * ema
            ewm5[seq[j]] = ema
            prev = cur

    out = np.concatenate([lag1, d1, ewm5], axis=1).astype(np.float32)
    names0 = [
        "surface_level",
        "surface_atm_short",
        "surface_atm_mid",
        "surface_atm_long",
        "surface_rr_short",
        "surface_term_slope",
        "surface_smile_curvature",
    ]
    added_names = [f"ctx_lag1_{n}" for n in names0] + [f"ctx_d1_{n}" for n in names0] + [f"ctx_ewm5_{n}" for n in names0]
    context_aug = np.concatenate([context.astype(np.float32), out], axis=1)
    return context_aug, added_names


def _safe_stats(x: np.ndarray) -> tuple[float, float, float]:
    if x.size == 0:
        return 0.0, 0.0, 0.0
    ax = np.abs(x)
    return float(np.mean(ax)), float(np.std(x)), float(np.max(ax))


def _cnn_surface_features(iv_surface: np.ndarray, tenor_days: np.ndarray) -> np.ndarray:
    n, nx, nt = iv_surface.shape
    out = np.zeros((n, 17), dtype=np.float32)

    x_idx_atm = int(nx // 2)
    x_idx_put = max(0, int(round(0.15 * (nx - 1))))
    x_idx_call = min(nx - 1, int(round(0.85 * (nx - 1))))
    t_idx_short = int(np.argmin(np.abs(np.asarray(tenor_days, dtype=np.float64) - 14.0)))
    t_idx_mid = int(np.argmin(np.abs(np.asarray(tenor_days, dtype=np.float64) - 30.0)))
    t_idx_long = int(np.argmin(np.abs(np.asarray(tenor_days, dtype=np.float64) - 90.0)))

    for i in range(n):
        g = iv_surface[i].astype(np.float64)
        gx = np.diff(g, axis=0)
        gt = np.diff(g, axis=1)
        if nx >= 3 and nt >= 3:
            lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
        else:
            lap = np.zeros((0, 0), dtype=np.float64)

        gx_ma, gx_std, gx_max = _safe_stats(gx)
        gt_ma, gt_std, gt_max = _safe_stats(gt)
        lap_ma, lap_std, lap_max = _safe_stats(lap)

        atm_short = float(g[x_idx_atm, t_idx_short])
        atm_mid = float(g[x_idx_atm, t_idx_mid])
        atm_long = float(g[x_idx_atm, t_idx_long])
        rr_short = float(g[x_idx_call, t_idx_short] - g[x_idx_put, t_idx_short])
        term_slope = float(atm_long - atm_short)
        smile_curv = float(g[x_idx_put, t_idx_mid] + g[x_idx_call, t_idx_mid] - 2.0 * atm_mid)

        out[i] = np.asarray(
            [
                float(np.mean(g)),
                float(np.std(g)),
                gx_ma,
                gx_std,
                gx_max,
                gt_ma,
                gt_std,
                gt_max,
                lap_ma,
                lap_std,
                lap_max,
                atm_short,
                atm_mid,
                atm_long,
                rr_short,
                term_slope,
                smile_curv,
            ],
            dtype=np.float32,
        )
    return out


def _build_factor_feature_matrix(
    factors: np.ndarray,
    cnn_feat: np.ndarray,
    context_scaled: np.ndarray,
) -> np.ndarray:
    n, k = factors.shape
    c = cnn_feat.shape[1]
    p = context_scaled.shape[1]
    out = np.zeros((n, (k * 5) + (c * 2) + p), dtype=np.float32)
    for i in range(n):
        f_cur = factors[i]
        f_lag1 = factors[i - 1] if i > 0 else f_cur
        f_d1 = f_cur - f_lag1
        f_m5 = np.mean(factors[max(0, i - 4) : i + 1], axis=0)
        f_m22 = np.mean(factors[max(0, i - 21) : i + 1], axis=0)

        c_cur = cnn_feat[i]
        c_lag1 = cnn_feat[i - 1] if i > 0 else c_cur
        c_d1 = c_cur - c_lag1

        out[i] = np.concatenate([f_cur, f_lag1, f_d1, f_m5, f_m22, c_cur, c_d1, context_scaled[i]], axis=0)
    return out.astype(np.float32)


def _focus_grid(
    *,
    x_grid: np.ndarray,
    tenor_days: np.ndarray,
    focus_x_min: float,
    focus_x_scale: float,
    focus_dte_scale_days: float,
    focus_dte_max_days: float,
    focus_neg_x_max: float,
    focus_neg_weight_ratio: float,
) -> np.ndarray:
    x = np.asarray(x_grid, dtype=np.float64).reshape(-1, 1)
    dte = np.asarray(tenor_days, dtype=np.float64).reshape(1, -1)

    x_scale = max(float(focus_x_scale), 1e-4)
    pos = 1.0 / (1.0 + np.exp(-(x - float(focus_x_min)) / x_scale))
    neg = 1.0 / (1.0 + np.exp((x - float(focus_neg_x_max)) / x_scale))

    dte_scale = max(float(focus_dte_scale_days), 1e-3)
    dte_decay = np.exp(-np.clip(dte, 0.0, None) / dte_scale)
    dte_mask = (dte <= float(focus_dte_max_days)).astype(np.float64)

    w = (pos + max(float(focus_neg_weight_ratio), 0.0) * neg) * dte_decay * dte_mask
    m = float(np.mean(w))
    if (not np.isfinite(m)) or m <= 1e-12:
        return np.ones_like(w, dtype=np.float32)
    return (w / m).astype(np.float32)


def _fit_pca_basis(
    flat_surface: np.ndarray,
    pair_pos: np.ndarray,
    max_factors: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(pair_pos) == 0:
        ref = flat_surface
    else:
        ref_idx = np.unique(np.concatenate([pair_pos, pair_pos + 1]))
        ref = flat_surface[ref_idx]

    ref_mu = np.mean(ref, axis=0)
    ref_centered = ref - ref_mu

    try:
        _, _, vt = np.linalg.svd(ref_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        d = flat_surface.shape[1]
        return ref_mu.astype(np.float32), np.eye(d, dtype=np.float32)[:1]

    if vt.size == 0:
        d = flat_surface.shape[1]
        return ref_mu.astype(np.float32), np.eye(d, dtype=np.float32)[:1]

    k = int(max(1, min(max_factors, vt.shape[0], flat_surface.shape[1])))
    comps = vt[:k].astype(np.float32)
    return ref_mu.astype(np.float32), comps


def _fit_factor_ensemble(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    seeds: tuple[int, ...],
    learning_rate: float,
    max_iter: int,
    max_depth: int,
    min_samples_leaf: int,
    l2_regularization: float,
    sample_weight: np.ndarray | None,
    max_cpu_threads: int,
) -> list[list[dict[str, object]]]:
    k = int(y_train.shape[1])
    ensemble: list[list[dict[str, object]]] = []

    for s in seeds:
        member: list[dict[str, object]] = []
        for j in range(k):
            y_col = y_train[:, j].astype(np.float64)
            if len(y_col) == 0 or np.allclose(y_col, y_col[0]):
                member.append({"kind": "const", "value": float(y_col[0] if len(y_col) else 0.0)})
                continue

            model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=float(learning_rate),
                max_iter=int(max_iter),
                max_depth=int(max_depth),
                min_samples_leaf=int(min_samples_leaf),
                l2_regularization=float(l2_regularization),
                random_state=int(s + (j * 9973)),
            )
            limit = int(max(1, max_cpu_threads))
            pool_ctx = (
                threadpool_limits(limits=limit)
                if threadpool_limits is not None
                else nullcontext()
            )
            try:
                with pool_ctx:
                    if sample_weight is not None:
                        model.fit(x_train, y_col, sample_weight=sample_weight)
                    else:
                        model.fit(x_train, y_col)
                member.append({"kind": "hgb", "model": model})
            except Exception:
                member.append({"kind": "const", "value": float(np.mean(y_col))})
        ensemble.append(member)
    return ensemble


def _predict_factor_delta(
    ensemble: list[list[dict[str, object]]],
    x: np.ndarray,
) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if len(ensemble) == 0:
        return np.zeros((len(x), 1), dtype=np.float32)

    k = len(ensemble[0])
    acc = np.zeros((len(x), k), dtype=np.float64)
    for member in ensemble:
        pred = np.zeros((len(x), k), dtype=np.float64)
        for j, spec in enumerate(member):
            kind = str(spec.get("kind", "const"))
            if kind == "hgb":
                mdl = spec.get("model")
                if mdl is None:
                    pred[:, j] = 0.0
                else:
                    pred[:, j] = np.asarray(mdl.predict(x), dtype=np.float64)
            else:
                pred[:, j] = float(spec.get("value", 0.0))
        acc += pred
    return (acc / float(len(ensemble))).astype(np.float32)


def _fit_regime_ensembles(
    *,
    pair_pos: np.ndarray,
    feats: np.ndarray,
    factors: np.ndarray,
    level_series: np.ndarray,
    sample_weight: np.ndarray | None,
    min_rows: int,
    seeds: tuple[int, ...],
    learning_rate: float,
    max_iter: int,
    max_depth: int,
    min_samples_leaf: int,
    l2_regularization: float,
    max_cpu_threads: int,
    use_sample_weight: bool,
) -> tuple[dict[int, list[list[dict[str, object]]]], tuple[float, float]]:
    if len(pair_pos) == 0:
        return {}, (0.0, 0.0)

    levels = level_series[pair_pos].astype(np.float64)
    if len(levels) >= 3:
        q1, q2 = np.quantile(levels, [1.0 / 3.0, 2.0 / 3.0])
    else:
        med = float(np.median(levels)) if len(levels) else 0.0
        q1, q2 = med, med
    bounds = (float(q1), float(q2))
    labels = np.digitize(levels, [bounds[0], bounds[1]]).astype(np.int32)

    out: dict[int, list[list[dict[str, object]]]] = {}
    for regime in (0, 1, 2):
        mask = labels == regime
        if int(np.sum(mask)) < int(max(min_rows, 1)):
            continue
        rows = pair_pos[mask]
        x_r = feats[rows].astype(np.float32)
        y_r = (factors[rows + 1] - factors[rows]).astype(np.float32)
        sw_r = None
        if use_sample_weight and sample_weight is not None and len(sample_weight) == len(pair_pos):
            sw_r = sample_weight[mask].astype(np.float32)
        out[regime] = _fit_factor_ensemble(
            x_train=x_r,
            y_train=y_r,
            seeds=seeds,
            learning_rate=float(learning_rate),
            max_iter=int(max_iter),
            max_depth=int(max_depth),
            min_samples_leaf=int(min_samples_leaf),
            l2_regularization=float(l2_regularization),
            sample_weight=sw_r,
            max_cpu_threads=int(max_cpu_threads),
        )
    return out, bounds


def _predict_factor_delta_with_regimes(
    *,
    x: np.ndarray,
    level_values: np.ndarray,
    global_ensemble: list[list[dict[str, object]]],
    regime_ensembles: dict[int, list[list[dict[str, object]]]],
    regime_bounds: tuple[float, float],
    blend_weight: float,
) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if len(x) == 0:
        return np.zeros((0, 1), dtype=np.float32)

    delta_global = _predict_factor_delta(global_ensemble, x).astype(np.float32)
    if len(regime_ensembles) == 0:
        return delta_global
    w = float(np.clip(float(blend_weight), 0.0, 1.0))
    if w <= 0.0:
        return delta_global

    levels = np.asarray(level_values, dtype=np.float32).reshape(-1)
    if len(levels) != len(x):
        raise RuntimeError(
            f"Regime level length mismatch: levels={len(levels)} x_rows={len(x)}"
        )
    labels = np.digitize(levels.astype(np.float64), [float(regime_bounds[0]), float(regime_bounds[1])]).astype(np.int32)

    out = delta_global.copy()
    for regime, ens in regime_ensembles.items():
        mask = labels == int(regime)
        if not np.any(mask):
            continue
        reg_pred = _predict_factor_delta(ens, x[mask]).astype(np.float32)
        out[mask] = (1.0 - w) * out[mask] + w * reg_pred
    return out


def _sample_weight_from_focus(
    *,
    iv_asset: np.ndarray,
    pair_pos: np.ndarray,
    base_focus_grid: np.ndarray,
    focus_alpha: float,
    density_grid: np.ndarray | None,
    density_alpha: float,
) -> np.ndarray:
    if len(pair_pos) == 0:
        return np.array([], dtype=np.float32)
    if focus_alpha <= 0.0:
        return np.ones(len(pair_pos), dtype=np.float32)

    g = base_focus_grid.astype(np.float64)
    if density_grid is not None and density_alpha > 0.0:
        d = np.clip(density_grid.astype(np.float64), 1e-8, None)
        g = g * np.power(d, float(density_alpha))
    g = g / max(float(np.mean(g)), 1e-8)

    w = np.ones(len(pair_pos), dtype=np.float64)
    for i, p in enumerate(pair_pos):
        move = np.abs(iv_asset[p + 1] - iv_asset[p]).astype(np.float64)
        w[i] = 1.0 + float(focus_alpha) * float(np.mean(move * g))
    w = np.clip(w, 1.0, 8.0)
    return w.astype(np.float32)


def _normalize_model_arch(mode: str | None) -> str:
    s = str(mode or "").strip().lower()
    if s in {"", "tree", "tree_boost", "tree_boost_surface"}:
        return "tree_boost"
    if s in {"option_a", "option_a_pca_tcn", "pca_tcn", "pca_neural"}:
        return "option_a_pca_tcn"
    return "tree_boost"


def _build_option_a_sequence_records(
    *,
    asset_ids: np.ndarray,
    seq_len: int,
    entry_target_mask: np.ndarray,
    hist_allowed_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hist_rows: list[np.ndarray] = []
    target_rows: list[int] = []
    entry_rows: list[int] = []
    asset_rows: list[int] = []

    aid_arr = np.asarray(asset_ids, dtype=np.int32).reshape(-1)
    et_mask = np.asarray(entry_target_mask, dtype=bool).reshape(-1)
    hist_mask = np.asarray(hist_allowed_mask, dtype=bool).reshape(-1)
    seq_len = int(max(2, seq_len))

    for aid in np.unique(aid_arr):
        seq = np.where(aid_arr == aid)[0]
        if len(seq) <= seq_len:
            continue
        for j in range(seq_len - 1, len(seq) - 1):
            hist = seq[j - seq_len + 1 : j + 1]
            entry = int(seq[j])
            target = int(seq[j + 1])
            if not (et_mask[entry] and et_mask[target]):
                continue
            if not np.all(hist_mask[hist]):
                continue
            hist_rows.append(hist.astype(np.int32))
            target_rows.append(target)
            entry_rows.append(entry)
            asset_rows.append(int(aid))

    if len(target_rows) == 0:
        return (
            np.empty((0, seq_len), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    return (
        np.stack(hist_rows, axis=0).astype(np.int32),
        np.asarray(target_rows, dtype=np.int32),
        np.asarray(entry_rows, dtype=np.int32),
        np.asarray(asset_rows, dtype=np.int32),
    )


def _option_a_sample_weight_from_focus(
    *,
    iv_obs: np.ndarray,
    entry_idx: np.ndarray,
    target_idx: np.ndarray,
    focus_grid: np.ndarray,
    focus_alpha: float,
) -> np.ndarray:
    n = int(len(entry_idx))
    if n == 0:
        return np.empty((0,), dtype=np.float32)
    if focus_alpha <= 0.0:
        return np.ones(n, dtype=np.float32)
    g = np.asarray(focus_grid, dtype=np.float64)
    g = g / max(float(np.mean(g)), 1e-8)
    move = np.abs(iv_obs[target_idx] - iv_obs[entry_idx]).astype(np.float64)
    w = 1.0 + float(focus_alpha) * np.mean(move * g.reshape(1, g.shape[0], g.shape[1]), axis=(1, 2))
    return np.clip(w, 1.0, 8.0).astype(np.float32)


class _OptionASequenceDataset(Dataset):
    def __init__(
        self,
        *,
        x_seq: np.ndarray,
        y_delta: np.ndarray,
        y_mask: np.ndarray,
        asset_id: np.ndarray,
        sample_weight: np.ndarray,
    ) -> None:
        self.x_seq = x_seq.astype(np.float32)
        self.y_delta = y_delta.astype(np.float32)
        self.y_mask = y_mask.astype(np.float32)
        self.asset_id = asset_id.astype(np.int64)
        self.sample_weight = sample_weight.astype(np.float32)

    def __len__(self) -> int:
        return int(len(self.asset_id))

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.int64, np.float32]:
        return (
            self.x_seq[idx],
            self.y_delta[idx],
            self.y_mask[idx],
            np.int64(self.asset_id[idx]),
            np.float32(self.sample_weight[idx]),
        )


_OPTION_A_NN_BASE = nn.Module if nn is not None else object


class _OptionATcnBlock(_OPTION_A_NN_BASE):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        k = int(max(2, kernel_size))
        d = int(max(1, dilation))
        self.pad = int((k - 1) * d)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=k, dilation=d)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=k, dilation=d)
        self.dropout = nn.Dropout(float(max(0.0, dropout)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(x, (self.pad, 0))
        y = self.conv1(y)
        y = F.gelu(y)
        y = self.dropout(y)
        y = F.pad(y, (self.pad, 0))
        y = self.conv2(y)
        y = self.dropout(y)
        return F.gelu(x + y)


class _OptionATcnModel(_OPTION_A_NN_BASE):
    def __init__(
        self,
        *,
        input_dim: int,
        out_dim: int,
        n_assets: int,
        asset_embed_dim: int,
        hidden_dim: int,
        n_layers: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        emb_dim = int(max(1, asset_embed_dim))
        hid = int(max(16, hidden_dim))
        self.asset_emb = nn.Embedding(int(max(1, n_assets)), emb_dim)
        self.in_proj = nn.Conv1d(int(input_dim + emb_dim), hid, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                _OptionATcnBlock(
                    channels=hid,
                    kernel_size=int(max(2, kernel_size)),
                    dilation=int(2**i),
                    dropout=float(max(0.0, dropout)),
                )
                for i in range(int(max(1, n_layers)))
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(hid, int(max(1, out_dim))),
        )

    def forward(self, x_seq: torch.Tensor, asset_id: torch.Tensor) -> torch.Tensor:
        b, l, _ = x_seq.shape
        a = self.asset_emb(asset_id).unsqueeze(1).expand(-1, l, -1)
        x = torch.cat([x_seq, a], dim=-1).transpose(1, 2)
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        out = self.head(h[:, :, -1])
        return torch.tanh(out) * 1.5


def _option_a_masked_huber_loss(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    d = float(max(beta, 1e-4))
    abs_err = torch.abs(pred - target)
    quad = torch.clamp(abs_err, max=d)
    lin = abs_err - quad
    hub = (0.5 * torch.square(quad) / d) + lin
    hub = hub * mask
    den = torch.sum(mask, dim=1).clamp(min=1.0)
    per = torch.sum(hub, dim=1) / den
    sw = torch.clamp(sample_weight, min=1e-6)
    return torch.sum(per * sw) / torch.sum(sw)


def _train_option_a(
    *,
    dataset_path: Path,
    cfg: TrainingConfig,
    dates: np.ndarray,
    asset_ids: np.ndarray,
    asset_names: list[str],
    surface_variable: str,
    iv_obs: np.ndarray,
    context: np.ndarray,
    context_scaled: np.ndarray,
    tenor_days: np.ndarray,
    x_grid: np.ndarray,
    tr_dates: np.ndarray,
    va_dates: np.ndarray,
    te_dates: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    trainval_mask: np.ndarray,
    split_mode: str,
    original_context_dim: int,
    added_context_names: list[str],
    added_surface_context_names: list[str],
    focus_grid: np.ndarray,
    context_scaler: ArrayScaler,
    max_factors: int,
) -> Path:
    if torch is None or nn is None or DataLoader is None or F is None:
        raise RuntimeError("Option A requires PyTorch in the active environment.")

    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))

    max_threads = int(max(1, cfg.max_cpu_threads))
    try:
        torch.set_num_threads(max_threads)
    except Exception:
        pass
    try:
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, min(4, max_threads)))
    except Exception:
        pass

    device_raw = str(getattr(cfg, "option_a_device", "auto") or "auto").strip().lower()
    if device_raw in {"", "auto"}:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device_raw)

    n_dates = int(len(dates))
    n_assets = int(np.max(asset_ids)) + 1 if len(asset_ids) else 1
    nx = int(iv_obs.shape[1])
    nt = int(iv_obs.shape[2])
    iv_flat = iv_obs.reshape(n_dates, -1).astype(np.float32)
    seq_len = int(max(4, cfg.option_a_seq_len))

    latent = np.zeros((n_dates, max_factors), dtype=np.float32)
    recon_flat = iv_flat.copy()
    forecast_flat = iv_flat.copy()
    step_feat = np.zeros((n_dates, max_factors + 17 + int(context_scaled.shape[1])), dtype=np.float32)
    asset_factor_k = np.zeros(n_assets, dtype=np.int32)

    asset_models_internal: list[dict[str, object]] = []
    for aid in range(n_assets):
        seq = np.where(asset_ids == aid)[0].astype(np.int32)
        if len(seq) <= 1:
            continue
        iv_asset = iv_obs[seq].astype(np.float32)
        flat_asset = iv_asset.reshape(len(seq), -1).astype(np.float32)
        ctx_asset = context_scaled[seq].astype(np.float32)
        pair_pos = np.where(trainval_mask[seq[:-1]] & trainval_mask[seq[1:]])[0].astype(np.int32)

        ref_mu, components = _fit_pca_basis(flat_asset, pair_pos, max_factors)
        k = int(components.shape[0])
        factors = ((flat_asset - ref_mu) @ components.T).astype(np.float32)
        factors_pad = np.zeros((len(seq), max_factors), dtype=np.float32)
        factors_pad[:, :k] = factors

        recon_local = np.clip(ref_mu + (factors @ components), 1e-4, 4.0).astype(np.float32)
        cnn_feat = _cnn_surface_features(iv_asset, tenor_days)
        feat_local = np.concatenate([factors_pad, cnn_feat, ctx_asset], axis=1).astype(np.float32)

        latent[seq] = factors_pad
        recon_flat[seq] = recon_local
        step_feat[seq] = feat_local
        asset_factor_k[aid] = np.int32(k)

        asset_models_internal.append(
            {
                "asset_id": int(aid),
                "asset_name": str(asset_names[aid] if 0 <= aid < len(asset_names) else f"asset_{aid}"),
                "seq": seq,
                "factor_k": int(k),
                "ref_mu": ref_mu.astype(np.float32),
                "components": components.astype(np.float32),
                "blend_alpha": 1.0,
                "trained": False,
                "n_train_rows": 0,
                "n_val_rows": 0,
                "fallback_persistence_days": int(len(seq) - 1),
            }
        )

    if len(asset_models_internal) == 0:
        raise RuntimeError("Option A could not initialize any per-asset PCA models.")

    tr_hist, tr_tgt, tr_entry, tr_aid = _build_option_a_sequence_records(
        asset_ids=asset_ids,
        seq_len=seq_len,
        entry_target_mask=train_mask,
        hist_allowed_mask=train_mask,
    )
    va_hist, va_tgt, va_entry, va_aid = _build_option_a_sequence_records(
        asset_ids=asset_ids,
        seq_len=seq_len,
        entry_target_mask=val_mask,
        hist_allowed_mask=trainval_mask,
    )

    if len(tr_tgt) == 0:
        raise RuntimeError(
            "Option A has no train sequences after split/history constraints. "
            "Reduce --option-a-seq-len or adjust split."
        )

    if len(va_tgt) == 0 and len(tr_tgt) >= 32:
        cut = int(max(16, round(0.10 * len(tr_tgt))))
        cut = min(cut, len(tr_tgt) - 16)
        va_hist = tr_hist[-cut:].copy()
        va_tgt = tr_tgt[-cut:].copy()
        va_entry = tr_entry[-cut:].copy()
        va_aid = tr_aid[-cut:].copy()
        tr_hist = tr_hist[:-cut]
        tr_tgt = tr_tgt[:-cut]
        tr_entry = tr_entry[:-cut]
        tr_aid = tr_aid[:-cut]

    factor_mask_by_asset = np.zeros((n_assets, max_factors), dtype=np.float32)
    for aid in range(n_assets):
        k = int(asset_factor_k[aid])
        if k > 0:
            factor_mask_by_asset[aid, :k] = 1.0

    tr_x = step_feat[tr_hist]
    tr_y = (latent[tr_tgt] - latent[tr_entry]).astype(np.float32)
    tr_m = factor_mask_by_asset[tr_aid].astype(np.float32)
    tr_sw = _option_a_sample_weight_from_focus(
        iv_obs=iv_obs,
        entry_idx=tr_entry,
        target_idx=tr_tgt,
        focus_grid=focus_grid,
        focus_alpha=float(max(0.0, cfg.surface_focus_alpha)),
    )

    va_x = step_feat[va_hist] if len(va_tgt) > 0 else np.empty((0, seq_len, step_feat.shape[1]), dtype=np.float32)
    va_y = (latent[va_tgt] - latent[va_entry]).astype(np.float32) if len(va_tgt) > 0 else np.empty((0, max_factors), dtype=np.float32)
    va_m = factor_mask_by_asset[va_aid].astype(np.float32) if len(va_tgt) > 0 else np.empty((0, max_factors), dtype=np.float32)
    va_sw = _option_a_sample_weight_from_focus(
        iv_obs=iv_obs,
        entry_idx=va_entry,
        target_idx=va_tgt,
        focus_grid=focus_grid,
        focus_alpha=float(max(0.0, cfg.surface_focus_alpha)),
    ) if len(va_tgt) > 0 else np.empty((0,), dtype=np.float32)

    train_ds = _OptionASequenceDataset(
        x_seq=tr_x,
        y_delta=tr_y,
        y_mask=tr_m,
        asset_id=tr_aid,
        sample_weight=tr_sw,
    )
    val_ds = _OptionASequenceDataset(
        x_seq=va_x,
        y_delta=va_y,
        y_mask=va_m,
        asset_id=va_aid,
        sample_weight=(va_sw if len(va_sw) == len(va_tgt) else np.ones(len(va_tgt), dtype=np.float32)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(max(8, cfg.option_a_batch_size)),
        shuffle=True,
        num_workers=0,
        pin_memory=(dev.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(max(8, cfg.option_a_batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=(dev.type == "cuda"),
    )

    model = _OptionATcnModel(
        input_dim=int(step_feat.shape[1]),
        out_dim=int(max_factors),
        n_assets=int(n_assets),
        asset_embed_dim=int(max(1, cfg.asset_embed_dim)),
        hidden_dim=int(max(32, cfg.option_a_hidden_dim)),
        n_layers=int(max(1, cfg.option_a_tcn_layers)),
        kernel_size=int(max(2, cfg.option_a_tcn_kernel_size)),
        dropout=float(max(0.0, cfg.option_a_dropout)),
    ).to(dev)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(max(1e-6, cfg.option_a_lr)),
        weight_decay=float(max(0.0, cfg.option_a_weight_decay)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(np.clip(cfg.lr_plateau_factor, 0.1, 0.9)),
        patience=int(max(1, cfg.lr_plateau_patience)),
        min_lr=float(max(1e-7, cfg.min_lr)),
    )

    huber_beta = float(max(1e-4, cfg.rollout_surface_huber_beta))
    epochs = int(max(1, cfg.option_a_epochs))
    patience = int(max(2, cfg.option_a_early_stop_patience))
    min_delta = float(max(0.0, cfg.early_stop_min_delta))

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    epoch_rows: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss_sum = 0.0
        tr_batches = 0
        for x_seq_b, y_delta_b, y_mask_b, aid_b, sw_b in train_loader:
            x_t = x_seq_b.to(dev, dtype=torch.float32, non_blocking=True)
            y_t = y_delta_b.to(dev, dtype=torch.float32, non_blocking=True)
            m_t = y_mask_b.to(dev, dtype=torch.float32, non_blocking=True)
            aid_t = aid_b.to(dev, dtype=torch.long, non_blocking=True)
            sw_t = sw_b.to(dev, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred_t = model(x_t, aid_t)
            loss = _option_a_masked_huber_loss(
                pred=pred_t,
                target=y_t,
                mask=m_t,
                sample_weight=sw_t,
                beta=huber_beta,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            tr_loss_sum += float(loss.detach().cpu().item())
            tr_batches += 1

        tr_loss = float(tr_loss_sum / max(1, tr_batches))

        model.eval()
        va_loss_sum = 0.0
        va_batches = 0
        with torch.inference_mode():
            for x_seq_b, y_delta_b, y_mask_b, aid_b, sw_b in val_loader:
                x_t = x_seq_b.to(dev, dtype=torch.float32, non_blocking=True)
                y_t = y_delta_b.to(dev, dtype=torch.float32, non_blocking=True)
                m_t = y_mask_b.to(dev, dtype=torch.float32, non_blocking=True)
                aid_t = aid_b.to(dev, dtype=torch.long, non_blocking=True)
                sw_t = sw_b.to(dev, dtype=torch.float32, non_blocking=True)
                pred_t = model(x_t, aid_t)
                loss = _option_a_masked_huber_loss(
                    pred=pred_t,
                    target=y_t,
                    mask=m_t,
                    sample_weight=sw_t,
                    beta=huber_beta,
                )
                va_loss_sum += float(loss.detach().cpu().item())
                va_batches += 1

        va_loss = float(va_loss_sum / max(1, va_batches)) if len(val_ds) > 0 else tr_loss
        scheduler.step(va_loss)
        lr_cur = float(optimizer.param_groups[0]["lr"])
        epoch_rows.append(
            {
                "epoch": int(epoch),
                "train_loss": float(tr_loss),
                "val_loss": float(va_loss),
                "lr": float(lr_cur),
                "train_rows": int(len(train_ds)),
                "val_rows": int(len(val_ds)),
            }
        )

        if va_loss + min_delta < best_val:
            best_val = float(va_loss)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    blend_steps = int(max(2, cfg.option_a_blend_alpha_steps))
    blend_grid = np.linspace(
        float(cfg.option_a_blend_alpha_min),
        float(cfg.option_a_blend_alpha_max),
        blend_steps,
    ).astype(np.float32)
    if len(va_tgt) > 0:
        va_pred_delta = np.zeros((len(va_tgt), max_factors), dtype=np.float32)
        eval_bs = int(max(16, cfg.option_a_eval_batch_size))
        model.eval()
        with torch.inference_mode():
            for s in range(0, len(va_tgt), eval_bs):
                e = min(len(va_tgt), s + eval_bs)
                x_t = torch.as_tensor(va_x[s:e], dtype=torch.float32, device=dev)
                aid_t = torch.as_tensor(va_aid[s:e], dtype=torch.long, device=dev)
                va_pred_delta[s:e] = model(x_t, aid_t).detach().cpu().numpy().astype(np.float32)

        for meta in asset_models_internal:
            aid = int(meta["asset_id"])
            idx = np.where(va_aid == aid)[0]
            if len(idx) == 0:
                continue
            k = int(meta["factor_k"])
            if k <= 0:
                continue
            base = latent[va_entry[idx], :k].astype(np.float32)
            delta = va_pred_delta[idx, :k].astype(np.float32)
            true_next = latent[va_tgt[idx], :k].astype(np.float32)
            best_alpha = 1.0
            best_mse = float("inf")
            for a in blend_grid:
                pred_next = base + float(a) * delta
                mse = float(np.mean(np.square(pred_next - true_next)))
                if mse < best_mse:
                    best_mse = mse
                    best_alpha = float(a)
            meta["blend_alpha"] = float(best_alpha)

    train_rows_per_asset = np.bincount(tr_aid, minlength=n_assets) if len(tr_aid) > 0 else np.zeros(n_assets, dtype=np.int32)
    val_rows_per_asset = np.bincount(va_aid, minlength=n_assets) if len(va_aid) > 0 else np.zeros(n_assets, dtype=np.int32)

    model.eval()
    eval_bs = int(max(16, cfg.option_a_eval_batch_size))
    for meta in asset_models_internal:
        aid = int(meta["asset_id"])
        seq = np.asarray(meta["seq"], dtype=np.int32)
        n_pairs = int(max(0, len(seq) - 1))
        if n_pairs <= 0:
            continue

        meta["n_train_rows"] = int(train_rows_per_asset[aid]) if aid < len(train_rows_per_asset) else 0
        meta["n_val_rows"] = int(val_rows_per_asset[aid]) if aid < len(val_rows_per_asset) else 0
        k = int(meta["factor_k"])
        if int(meta["n_train_rows"]) <= 0 or k <= 0 or len(seq) <= seq_len:
            meta["trained"] = False
            meta["fallback_persistence_days"] = int(n_pairs)
            continue

        eligible_local = np.arange(seq_len - 1, len(seq) - 1, dtype=np.int32)
        if len(eligible_local) == 0:
            meta["trained"] = False
            meta["fallback_persistence_days"] = int(n_pairs)
            continue

        pred_delta = np.zeros((len(eligible_local), max_factors), dtype=np.float32)
        with torch.inference_mode():
            for s in range(0, len(eligible_local), eval_bs):
                e = min(len(eligible_local), s + eval_bs)
                loc = eligible_local[s:e]
                hist_idx = np.stack([seq[j - seq_len + 1 : j + 1] for j in loc], axis=0).astype(np.int32)
                x_t = torch.as_tensor(step_feat[hist_idx], dtype=torch.float32, device=dev)
                aid_t = torch.full((len(loc),), int(aid), dtype=torch.long, device=dev)
                pred_delta[s:e] = model(x_t, aid_t).detach().cpu().numpy().astype(np.float32)

        entry_idx = seq[eligible_local]
        alpha = float(meta["blend_alpha"])
        base_fac = latent[entry_idx, :k].astype(np.float32)
        next_fac = base_fac + alpha * pred_delta[:, :k]
        ref_mu = np.asarray(meta["ref_mu"], dtype=np.float32)
        components = np.asarray(meta["components"], dtype=np.float32)
        pred_flat = np.clip(ref_mu + (next_fac @ components), 1e-4, 4.0).astype(np.float32)
        forecast_flat[entry_idx] = pred_flat

        meta["trained"] = True
        meta["fallback_persistence_days"] = int(max(0, n_pairs - len(entry_idx)))

    history_rows: list[dict[str, object]] = []
    asset_model_payloads: list[dict[str, object]] = []
    for meta in asset_models_internal:
        aid = int(meta["asset_id"])
        seq = np.asarray(meta["seq"], dtype=np.int32)
        k = int(meta["factor_k"])
        history_rows.append(
            {
                "asset_id": int(aid),
                "asset": str(meta["asset_name"]),
                "trained": bool(meta["trained"]),
                "n_asset_dates": int(len(seq)),
                "n_train_rows": int(meta["n_train_rows"]),
                "n_val_rows": int(meta["n_val_rows"]),
                "factor_k": int(k),
                "blend_alpha": float(meta["blend_alpha"]),
                "fallback_persistence_days": int(meta["fallback_persistence_days"]),
            }
        )
        asset_model_payloads.append(
            {
                "asset_id": int(aid),
                "asset_name": str(meta["asset_name"]),
                "trained": bool(meta["trained"]),
                "factor_k": int(k),
                "blend_alpha": float(meta["blend_alpha"]),
                "ref_mu": np.asarray(meta["ref_mu"], dtype=np.float32),
                "components": np.asarray(meta["components"], dtype=np.float32),
                "min_history": int(seq_len),
                "fallback_persistence_days": int(meta["fallback_persistence_days"]),
            }
        )

    recon_iv = recon_flat.reshape(n_dates, nx, nt).astype(np.float32)
    forecast_iv = forecast_flat.reshape(n_dates, nx, nt).astype(np.float32)
    recon_raw = _iv_to_surface_raw_numpy(recon_iv, tenor_days, surface_variable)
    forecast_raw = _iv_to_surface_raw_numpy(forecast_iv, tenor_days, surface_variable)

    run_dir = make_run_dir(cfg.out_dir, prefix="run")

    epoch_df = pd.DataFrame(epoch_rows)
    if len(epoch_df) == 0:
        epoch_df = pd.DataFrame([{"epoch": 0, "train_loss": float("nan"), "val_loss": float("nan"), "lr": float("nan")}])
    epoch_df.to_csv(run_dir / "train_history.csv", index=False)

    latent_df = pd.DataFrame(latent, columns=[f"z_{i}" for i in range(latent.shape[1])])
    latent_df.insert(0, "date", dates)
    latent_df.insert(1, "asset_id", asset_ids.astype(np.int32))
    latent_df.insert(
        2,
        "asset",
        np.array([asset_names[int(i)] if 0 <= int(i) < len(asset_names) else f"asset_{int(i)}" for i in asset_ids], dtype=object),
    )
    latent_df.to_parquet(run_dir / "latent_states.parquet", index=False)

    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "model_kwargs": {
                "input_dim": int(step_feat.shape[1]),
                "out_dim": int(max_factors),
                "n_assets": int(n_assets),
                "asset_embed_dim": int(max(1, cfg.asset_embed_dim)),
                "hidden_dim": int(max(32, cfg.option_a_hidden_dim)),
                "n_layers": int(max(1, cfg.option_a_tcn_layers)),
                "kernel_size": int(max(2, cfg.option_a_tcn_kernel_size)),
                "dropout": float(max(0.0, cfg.option_a_dropout)),
            },
            "seq_len": int(seq_len),
        },
        run_dir / "option_a_model.pt",
    )

    artifact = {
        "model_type": "option_a_pca_tcn_v1",
        "dataset_path": str(dataset_path.resolve()),
        "surface_variable": str(surface_variable),
        "x_grid": x_grid.astype(np.float32),
        "tenor_days": tenor_days.astype(np.int32),
        "context_scaler_state": context_scaler.state(),
        "context_augment_from_contracts": bool(cfg.context_augment_from_contracts),
        "context_augment_surface_history": bool(cfg.context_augment_surface_history),
        "context_added_features": added_context_names + added_surface_context_names,
        "tree_pca_factors": int(max_factors),
        "option_a_model_path": str((run_dir / "option_a_model.pt").resolve()),
        "option_a_seq_len": int(seq_len),
        "asset_models": asset_model_payloads,
        "forecast_surface_raw": forecast_raw.astype(np.float32),
        "recon_surface_raw": recon_raw.astype(np.float32),
        "forecast_iv": forecast_iv.astype(np.float32),
        "recon_iv": recon_iv.astype(np.float32),
        "latent": latent.astype(np.float32),
        "asset_ids": asset_ids.astype(np.int32),
        "dates": dates,
    }
    with (run_dir / "tree_model.pkl").open("wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

    cfg_payload = asdict(cfg)
    cfg_payload["out_dir"] = str(cfg.out_dir)
    cfg_payload["model_family"] = "option_a_pca_tcn_v1"
    (run_dir / "train_config.json").write_text(json.dumps(cfg_payload, indent=2), encoding="utf-8")

    trainval_rmse = _forecast_pairs_rmse(
        iv_obs=iv_obs,
        iv_forecast=forecast_iv,
        asset_ids=asset_ids,
        date_mask=trainval_mask,
    )
    test_rmse = _forecast_pairs_rmse(
        iv_obs=iv_obs,
        iv_forecast=forecast_iv,
        asset_ids=asset_ids,
        date_mask=test_mask,
    )

    summary = {
        "model_path": str((run_dir / "tree_model.pkl").resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "model_family": "option_a_pca_tcn_v1",
        "surface_variable": str(surface_variable),
        "split_mode": str(split_mode),
        "assets": asset_names,
        "n_assets": int(n_assets),
        "n_dates_total": int(n_dates),
        "n_dates_train": int(len(tr_dates)),
        "n_dates_val": int(len(va_dates)),
        "n_dates_test": int(len(te_dates)),
        "context_dim_original": int(original_context_dim),
        "context_dim_used": int(context.shape[1]),
        "context_augmented_from_contracts": bool(len(added_context_names) > 0),
        "context_augmented_from_surface_history": bool(len(added_surface_context_names) > 0),
        "context_added_features": added_context_names + added_surface_context_names,
        "tree_pca_factors": int(max_factors),
        "option_a_seq_len": int(seq_len),
        "option_a_epochs": int(cfg.option_a_epochs),
        "option_a_batch_size": int(cfg.option_a_batch_size),
        "option_a_lr": float(cfg.option_a_lr),
        "option_a_weight_decay": float(cfg.option_a_weight_decay),
        "option_a_hidden_dim": int(cfg.option_a_hidden_dim),
        "option_a_tcn_layers": int(cfg.option_a_tcn_layers),
        "option_a_tcn_kernel_size": int(cfg.option_a_tcn_kernel_size),
        "option_a_dropout": float(cfg.option_a_dropout),
        "max_cpu_threads": int(max(1, cfg.max_cpu_threads)),
        "device_used": str(dev),
        "train_sequences": int(len(train_ds)),
        "val_sequences": int(len(val_ds)),
        "surface_forecast_iv_rmse_trainval": float(trainval_rmse),
        "surface_forecast_iv_rmse_test": float(test_rmse),
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir


def _forecast_pairs_rmse(
    *,
    iv_obs: np.ndarray,
    iv_forecast: np.ndarray,
    asset_ids: np.ndarray,
    date_mask: np.ndarray,
) -> float:
    same_asset_next = asset_ids[:-1] == asset_ids[1:] if len(asset_ids) > 1 else np.array([], dtype=bool)
    entry_idx = np.where(same_asset_next & date_mask[:-1] & date_mask[1:])[0].astype(np.int32)
    if len(entry_idx) == 0:
        return float("nan")
    err = iv_forecast[entry_idx] - iv_obs[entry_idx + 1]
    return float(np.sqrt(np.mean(np.square(err))))


def train(dataset_path: Path, cfg: TrainingConfig) -> Path:
    rng = np.random.default_rng(cfg.seed)
    _ = rng  # Reserved for future stochastic extensions.

    ds = _load_dataset(dataset_path)

    dates = ds["dates"].astype(str)
    n_dates = int(len(dates))
    asset_ids = ds.get("asset_ids", np.zeros(n_dates, dtype=np.int32)).astype(np.int32)
    asset_names = ds.get("asset_names", np.array(["ASSET"], dtype=str)).astype(str).tolist()
    n_assets = int(np.max(asset_ids)) + 1 if len(asset_ids) else 1
    if len(asset_names) != n_assets:
        raise RuntimeError(
            "asset_names length must match inferred n_assets: "
            f"asset_names={len(asset_names)} n_assets={n_assets}."
        )

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

    surface_key = "surface" if "surface" in ds else "iv_surface"
    surface_raw_obs = ds[surface_key].astype(np.float32)
    context = ds["context"].astype(np.float32)
    context_names = ds.get("context_names", np.array([], dtype=str)).astype(str).tolist()
    contract_features = ds["contract_features"].astype(np.float32)
    contract_feature_names = ds.get("contract_feature_names", np.array([], dtype=str)).astype(str).tolist()
    contract_date_idx = ds["contract_date_index"].astype(np.int32)

    tenor_days = ds["tenor_days"].astype(np.int32)
    x_grid = ds.get("x_grid")
    if x_grid is None:
        raise RuntimeError("Dataset must include x_grid for tree-based surface training.")
    x_grid = np.asarray(x_grid, dtype=np.float32).reshape(-1)

    nx = int(surface_raw_obs.shape[1])
    nt = int(surface_raw_obs.shape[2])
    if len(x_grid) != nx:
        raise RuntimeError(f"x_grid length {len(x_grid)} does not match surface x dimension {nx}.")

    split_mode = _normalize_split_mode(cfg.split_mode)
    tr_dates, va_dates, te_dates = _date_splits_for_mode(
        n_dates=n_dates,
        asset_ids=asset_ids,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        split_mode=split_mode,
    )
    _assert_no_train_test_date_overlap(
        dates=dates,
        train_idx=tr_dates,
        test_idx=te_dates,
    )
    if len(tr_dates) == 0:
        raise RuntimeError("No training dates after split.")

    train_mask = np.zeros(n_dates, dtype=bool)
    train_mask[tr_dates] = True
    val_mask = np.zeros(n_dates, dtype=bool)
    val_mask[va_dates] = True
    test_mask = np.zeros(n_dates, dtype=bool)
    test_mask[te_dates] = True
    trainval_mask = ~test_mask

    original_context_dim = int(context.shape[1])
    added_context_names: list[str] = []
    added_surface_context_names: list[str] = []
    if bool(cfg.context_augment_from_contracts):
        context, added_context_names = _augment_context_with_contract_intraday(
            context=context,
            features=contract_features,
            feature_names=contract_feature_names,
            date_idx=contract_date_idx,
            n_dates=n_dates,
        )
        if added_context_names:
            context_names = context_names + added_context_names
    if bool(cfg.context_augment_surface_history):
        context, added_surface_context_names = _augment_context_with_surface_history(
            context=context,
            surface_raw=surface_raw_obs,
            tenor_days=tenor_days,
            asset_ids=asset_ids,
            surface_variable=surface_variable,
        )
        if added_surface_context_names:
            context_names = context_names + added_surface_context_names

    context_scaler = ArrayScaler.fit(
        context[tr_dates],
        winsor_quantile=float(max(0.0, cfg.context_winsor_quantile)),
        z_clip=(float(cfg.context_z_clip) if float(cfg.context_z_clip) > 0 else None),
    )
    context_scaled = context_scaler.transform(context)

    iv_obs = _surface_to_iv_numpy(surface_raw_obs.astype(np.float32), tenor_days, surface_variable)
    iv_flat = iv_obs.reshape(n_dates, -1).astype(np.float32)

    max_factors = int(max(1, min(cfg.tree_pca_factors, cfg.latent_dim, iv_flat.shape[1])))
    min_history = int(max(10, cfg.tree_min_history))

    focus_grid = _focus_grid(
        x_grid=x_grid,
        tenor_days=tenor_days,
        focus_x_min=float(cfg.surface_focus_x_min),
        focus_x_scale=float(cfg.surface_focus_x_scale),
        focus_dte_scale_days=float(cfg.surface_focus_dte_scale_days),
        focus_dte_max_days=float(cfg.surface_focus_dte_max_days),
        focus_neg_x_max=float(cfg.surface_focus_neg_x_max),
        focus_neg_weight_ratio=float(cfg.surface_focus_neg_weight_ratio),
    )

    model_arch = _normalize_model_arch(getattr(cfg, "model_arch", "tree_boost"))
    if model_arch == "option_a_pca_tcn":
        return _train_option_a(
            dataset_path=dataset_path,
            cfg=cfg,
            dates=dates,
            asset_ids=asset_ids,
            asset_names=asset_names,
            surface_variable=surface_variable,
            iv_obs=iv_obs,
            context=context,
            context_scaled=context_scaled,
            tenor_days=tenor_days,
            x_grid=x_grid,
            tr_dates=tr_dates,
            va_dates=va_dates,
            te_dates=te_dates,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            trainval_mask=trainval_mask,
            split_mode=split_mode,
            original_context_dim=original_context_dim,
            added_context_names=added_context_names,
            added_surface_context_names=added_surface_context_names,
            focus_grid=focus_grid,
            context_scaler=context_scaler,
            max_factors=max_factors,
        )

    focus_density_by_asset: np.ndarray | None = None
    focus_density_alpha = max(float(cfg.surface_focus_density_alpha), 0.0)
    focus_density_map_path_used: str | None = None
    if focus_density_alpha > 0.0:
        map_path_raw = cfg.surface_focus_density_map_path
        if map_path_raw is None or str(map_path_raw).strip() == "":
            raise RuntimeError(
                "surface_focus_density_alpha > 0 requires --surface-focus-density-map with a valid map path."
            )
        map_path = Path(str(map_path_raw)).expanduser().resolve()
        focus_density_map_path_used = str(map_path)
        focus_density_by_asset = _load_focus_density_by_asset(
            map_path=map_path,
            asset_names=asset_names,
            nx=nx,
            nt=nt,
        )
    residual_enabled = bool(cfg.tree_enable_residual_corrector)
    residual_factors = max(int(cfg.tree_residual_factors), 1)
    residual_scale_max = max(float(cfg.tree_residual_scale_max), 0.0)

    latent = np.zeros((n_dates, max_factors), dtype=np.float32)
    recon_flat = iv_flat.copy()
    forecast_flat = iv_flat.copy()

    history_rows: list[dict[str, object]] = []
    asset_model_payloads: list[dict[str, object]] = []

    for aid in range(n_assets):
        seq = np.where(asset_ids == aid)[0]
        if len(seq) <= 1:
            continue

        iv_asset = iv_obs[seq].astype(np.float32)
        flat_asset = iv_asset.reshape(len(seq), -1).astype(np.float32)
        ctx_asset = context_scaled[seq].astype(np.float32)
        cnn_asset = _cnn_surface_features(iv_asset, tenor_days)

        pair_pos = np.where(trainval_mask[seq[:-1]] & trainval_mask[seq[1:]])[0].astype(np.int32)

        ref_mu, components = _fit_pca_basis(flat_asset, pair_pos, max_factors)
        k = int(components.shape[0])

        factors = ((flat_asset - ref_mu) @ components.T).astype(np.float32)
        recon_local = np.clip(ref_mu + (factors @ components), 1e-4, 4.0).astype(np.float32)

        feats = _build_factor_feature_matrix(factors, cnn_asset, ctx_asset)

        trained = False
        blend_alpha = 0.0
        ensemble: list[list[dict[str, object]]] = []
        residual_trained = False
        residual_scale = 0.0
        residual_components = np.zeros((1, flat_asset.shape[1]), dtype=np.float32)
        residual_ensemble: list[list[dict[str, object]]] = []
        regime_trained = False
        regime_blend_weight = 0.0
        regime_bounds = (0.0, 0.0)
        regime_ensembles: dict[int, list[list[dict[str, object]]]] = {}
        level_series = cnn_asset[:, 0].astype(np.float32)
        tenor_cal_a = np.zeros(nt, dtype=np.float32)
        tenor_cal_b = np.ones(nt, dtype=np.float32)

        if len(pair_pos) >= min_history:
            cal_frac = float(np.clip(float(cfg.tree_calibration_frac), 0.0, 0.45))
            cal_n = int(max(cfg.tree_min_calibration_rows, round(len(pair_pos) * cal_frac)))
            cal_n = min(cal_n, max(0, len(pair_pos) - min_history))
            fit_n = int(len(pair_pos) - cal_n)

            pair_fit = pair_pos[:fit_n] if fit_n > 0 else pair_pos
            pair_cal = pair_pos[fit_n:] if cal_n > 0 else np.array([], dtype=np.int32)

            x_fit = feats[pair_fit]
            y_fit = (factors[pair_fit + 1] - factors[pair_fit]).astype(np.float32)

            density_grid = (
                focus_density_by_asset[aid]
                if (focus_density_by_asset is not None and 0 <= aid < len(focus_density_by_asset))
                else None
            )
            w_fit = _sample_weight_from_focus(
                iv_asset=iv_asset,
                pair_pos=pair_fit,
                base_focus_grid=focus_grid,
                focus_alpha=max(float(cfg.surface_focus_alpha), 0.0),
                density_grid=density_grid,
                density_alpha=focus_density_alpha,
            )

            seeds = tuple(int(s) for s in cfg.tree_ensemble_seeds if int(s) >= 0)
            if not seeds:
                seeds = (13,)

            ensemble_fit = _fit_factor_ensemble(
                x_train=x_fit,
                y_train=y_fit,
                seeds=seeds,
                learning_rate=float(cfg.tree_learning_rate),
                max_iter=int(cfg.tree_max_iter),
                max_depth=int(cfg.tree_max_depth),
                min_samples_leaf=int(cfg.tree_min_samples_leaf),
                l2_regularization=float(cfg.tree_l2_regularization),
                sample_weight=(w_fit if bool(cfg.tree_use_sample_weight) else None),
                max_cpu_threads=int(cfg.max_cpu_threads),
            )

            regime_fit_ensembles: dict[int, list[list[dict[str, object]]]] = {}
            regime_fit_bounds = (0.0, 0.0)
            if bool(cfg.tree_enable_regime_experts):
                regime_fit_ensembles, regime_fit_bounds = _fit_regime_ensembles(
                    pair_pos=pair_fit,
                    feats=feats,
                    factors=factors,
                    level_series=level_series,
                    sample_weight=w_fit,
                    min_rows=int(max(20, cfg.tree_regime_min_rows // 2)),
                    seeds=seeds,
                    learning_rate=float(cfg.tree_learning_rate),
                    max_iter=int(cfg.tree_max_iter),
                    max_depth=int(cfg.tree_max_depth),
                    min_samples_leaf=int(cfg.tree_min_samples_leaf),
                    l2_regularization=float(cfg.tree_l2_regularization),
                    max_cpu_threads=int(cfg.max_cpu_threads),
                    use_sample_weight=bool(cfg.tree_use_sample_weight),
                )

            if len(pair_cal) > 0:
                x_cal = feats[pair_cal]
                y_true_cal = flat_asset[pair_cal + 1]
                alpha_grid = np.linspace(0.2, 1.2, 11)
                regime_grid = np.linspace(0.0, 1.0, 5) if len(regime_fit_ensembles) > 0 else np.array([0.0], dtype=np.float32)
                best_alpha = 1.0
                best_regime_w = 0.0
                best_mse = float("inf")
                for regime_w in regime_grid:
                    delta_cal = _predict_factor_delta_with_regimes(
                        x=x_cal,
                        level_values=level_series[pair_cal],
                        global_ensemble=ensemble_fit,
                        regime_ensembles=regime_fit_ensembles,
                        regime_bounds=regime_fit_bounds,
                        blend_weight=float(regime_w),
                    )
                    for alpha in alpha_grid:
                        f_pred = factors[pair_cal] + float(alpha) * delta_cal
                        y_pred = np.clip(ref_mu + (f_pred @ components), 1e-4, 4.0)
                        mse = float(np.mean((y_pred - y_true_cal) ** 2))
                        if mse < best_mse:
                            best_mse = mse
                            best_alpha = float(alpha)
                            best_regime_w = float(regime_w)
                blend_alpha = float(best_alpha)
                regime_blend_weight = float(best_regime_w)
            else:
                blend_alpha = 1.0
                regime_blend_weight = 0.0

            if bool(cfg.tree_enable_tenor_calibration) and len(pair_cal) > 0:
                delta_cal_for_tenor = _predict_factor_delta_with_regimes(
                    x=feats[pair_cal],
                    level_values=level_series[pair_cal],
                    global_ensemble=ensemble_fit,
                    regime_ensembles=regime_fit_ensembles,
                    regime_bounds=regime_fit_bounds,
                    blend_weight=float(regime_blend_weight),
                ).astype(np.float32)
                pred_cal_flat = np.clip(
                    ref_mu + ((factors[pair_cal] + float(blend_alpha) * delta_cal_for_tenor) @ components),
                    1e-4,
                    4.0,
                ).astype(np.float32)
                true_cal_flat = flat_asset[pair_cal + 1].astype(np.float32)
                pred_cal_grid = pred_cal_flat.reshape(len(pair_cal), nx, nt)
                true_cal_grid = true_cal_flat.reshape(len(pair_cal), nx, nt)
                shrink = float(np.clip(float(cfg.tree_tenor_calibration_shrinkage), 0.0, 1.0))
                slope_min = float(cfg.tree_tenor_slope_min)
                slope_max = max(float(cfg.tree_tenor_slope_max), slope_min)
                for tj in range(nt):
                    p = pred_cal_grid[:, :, tj].reshape(-1).astype(np.float64)
                    y = true_cal_grid[:, :, tj].reshape(-1).astype(np.float64)
                    if len(p) == 0:
                        continue
                    p_mean = float(np.mean(p))
                    y_mean = float(np.mean(y))
                    var_p = float(np.mean((p - p_mean) ** 2))
                    if var_p > 1e-12:
                        cov_py = float(np.mean((p - p_mean) * (y - y_mean)))
                        slope = cov_py / var_p
                    else:
                        slope = 1.0
                    slope = (1.0 - shrink) * slope + shrink * 1.0
                    slope = float(np.clip(slope, slope_min, slope_max))
                    intercept = float(y_mean - slope * p_mean)
                    tenor_cal_b[tj] = np.float32(slope)
                    tenor_cal_a[tj] = np.float32(intercept)

            x_all = feats[pair_pos]
            y_all = (factors[pair_pos + 1] - factors[pair_pos]).astype(np.float32)
            w_all = _sample_weight_from_focus(
                iv_asset=iv_asset,
                pair_pos=pair_pos,
                base_focus_grid=focus_grid,
                focus_alpha=max(float(cfg.surface_focus_alpha), 0.0),
                density_grid=density_grid,
                density_alpha=focus_density_alpha,
            )
            ensemble = _fit_factor_ensemble(
                x_train=x_all,
                y_train=y_all,
                seeds=seeds,
                learning_rate=float(cfg.tree_learning_rate),
                max_iter=int(cfg.tree_max_iter),
                max_depth=int(cfg.tree_max_depth),
                min_samples_leaf=int(cfg.tree_min_samples_leaf),
                l2_regularization=float(cfg.tree_l2_regularization),
                sample_weight=(w_all if bool(cfg.tree_use_sample_weight) else None),
                max_cpu_threads=int(cfg.max_cpu_threads),
            )
            if bool(cfg.tree_enable_regime_experts):
                regime_ensembles, regime_bounds = _fit_regime_ensembles(
                    pair_pos=pair_pos,
                    feats=feats,
                    factors=factors,
                    level_series=level_series,
                    sample_weight=w_all,
                    min_rows=int(max(20, cfg.tree_regime_min_rows)),
                    seeds=seeds,
                    learning_rate=float(cfg.tree_learning_rate),
                    max_iter=int(cfg.tree_max_iter),
                    max_depth=int(cfg.tree_max_depth),
                    min_samples_leaf=int(cfg.tree_min_samples_leaf),
                    l2_regularization=float(cfg.tree_l2_regularization),
                    max_cpu_threads=int(cfg.max_cpu_threads),
                    use_sample_weight=bool(cfg.tree_use_sample_weight),
                )
                regime_trained = bool(len(regime_ensembles) > 0 and regime_blend_weight > 0.0)
            trained = True

            if residual_enabled and len(pair_fit) >= max(24, min_history // 2):
                delta_fit_primary = _predict_factor_delta_with_regimes(
                    x=feats[pair_fit],
                    level_values=level_series[pair_fit],
                    global_ensemble=ensemble_fit,
                    regime_ensembles=regime_fit_ensembles,
                    regime_bounds=regime_fit_bounds,
                    blend_weight=float(regime_blend_weight),
                ).astype(np.float32)
                base_fit = np.clip(
                    ref_mu + ((factors[pair_fit] + float(blend_alpha) * delta_fit_primary) @ components),
                    1e-4,
                    4.0,
                ).astype(np.float32)
                residual_fit_flat = (flat_asset[pair_fit + 1] - base_fit).astype(np.float32)

                res_basis_ok = False
                if residual_fit_flat.shape[0] >= 4:
                    try:
                        _, _, vt_res = np.linalg.svd(residual_fit_flat.astype(np.float64), full_matrices=False)
                        if vt_res.size > 0:
                            kr = int(min(residual_factors, vt_res.shape[0], max(1, residual_fit_flat.shape[0] - 1)))
                            residual_components = vt_res[:kr].astype(np.float32)
                            res_basis_ok = kr > 0
                    except np.linalg.LinAlgError:
                        res_basis_ok = False

                if res_basis_ok:
                    x_res_fit = np.concatenate([feats[pair_fit], delta_fit_primary, factors[pair_fit]], axis=1).astype(np.float32)
                    y_res_fit = (residual_fit_flat @ residual_components.T).astype(np.float32)
                    residual_ensemble_fit = _fit_factor_ensemble(
                        x_train=x_res_fit,
                        y_train=y_res_fit,
                        seeds=seeds,
                        learning_rate=float(cfg.tree_residual_learning_rate),
                        max_iter=int(cfg.tree_residual_max_iter),
                        max_depth=int(cfg.tree_residual_max_depth),
                        min_samples_leaf=int(cfg.tree_residual_min_samples_leaf),
                        l2_regularization=float(cfg.tree_residual_l2_regularization),
                        sample_weight=(w_fit if bool(cfg.tree_use_sample_weight) else None),
                        max_cpu_threads=int(cfg.max_cpu_threads),
                    )

                    residual_scale = 1.0
                    if len(pair_cal) > 0 and residual_scale_max > 0.0:
                        delta_cal_primary = _predict_factor_delta_with_regimes(
                            x=feats[pair_cal],
                            level_values=level_series[pair_cal],
                            global_ensemble=ensemble_fit,
                            regime_ensembles=regime_fit_ensembles,
                            regime_bounds=regime_fit_bounds,
                            blend_weight=float(regime_blend_weight),
                        ).astype(np.float32)
                        base_cal = np.clip(
                            ref_mu + ((factors[pair_cal] + float(blend_alpha) * delta_cal_primary) @ components),
                            1e-4,
                            4.0,
                        ).astype(np.float32)
                        x_res_cal = np.concatenate([feats[pair_cal], delta_cal_primary, factors[pair_cal]], axis=1).astype(np.float32)
                        res_cal_coef = _predict_factor_delta(residual_ensemble_fit, x_res_cal).astype(np.float32)
                        res_cal_flat = (res_cal_coef @ residual_components).astype(np.float32)
                        y_true_cal = flat_asset[pair_cal + 1].astype(np.float32)

                        best_gamma = 1.0
                        best_mse = float("inf")
                        gamma_grid = np.linspace(0.0, float(residual_scale_max), 13)
                        for gamma in gamma_grid:
                            y_pred = np.clip(base_cal + float(gamma) * res_cal_flat, 1e-4, 4.0)
                            mse = float(np.mean((y_pred - y_true_cal) ** 2))
                            if mse < best_mse:
                                best_mse = mse
                                best_gamma = float(gamma)
                        residual_scale = float(best_gamma)

                    delta_all_primary = _predict_factor_delta_with_regimes(
                        x=feats[pair_pos],
                        level_values=level_series[pair_pos],
                        global_ensemble=ensemble,
                        regime_ensembles=regime_ensembles,
                        regime_bounds=regime_bounds,
                        blend_weight=float(regime_blend_weight),
                    ).astype(np.float32)
                    base_all = np.clip(
                        ref_mu + ((factors[pair_pos] + float(blend_alpha) * delta_all_primary) @ components),
                        1e-4,
                        4.0,
                    ).astype(np.float32)
                    residual_all_flat = (flat_asset[pair_pos + 1] - base_all).astype(np.float32)
                    x_res_all = np.concatenate([feats[pair_pos], delta_all_primary, factors[pair_pos]], axis=1).astype(np.float32)
                    y_res_all = (residual_all_flat @ residual_components.T).astype(np.float32)

                    residual_ensemble = _fit_factor_ensemble(
                        x_train=x_res_all,
                        y_train=y_res_all,
                        seeds=seeds,
                        learning_rate=float(cfg.tree_residual_learning_rate),
                        max_iter=int(cfg.tree_residual_max_iter),
                        max_depth=int(cfg.tree_residual_max_depth),
                        min_samples_leaf=int(cfg.tree_residual_min_samples_leaf),
                        l2_regularization=float(cfg.tree_residual_l2_regularization),
                        sample_weight=(w_all if bool(cfg.tree_use_sample_weight) else None),
                        max_cpu_threads=int(cfg.max_cpu_threads),
                    )
                    residual_trained = True

        forecast_local = flat_asset.copy()
        fallback_days = int(len(seq) - 1)
        if trained and len(ensemble) > 0:
            for i in range(len(seq) - 1):
                delta = _predict_factor_delta_with_regimes(
                    x=feats[i : i + 1],
                    level_values=level_series[i : i + 1],
                    global_ensemble=ensemble,
                    regime_ensembles=regime_ensembles,
                    regime_bounds=regime_bounds,
                    blend_weight=float(regime_blend_weight),
                )[0]
                f_pred = factors[i] + float(blend_alpha) * delta
                pred_flat = ref_mu + (f_pred @ components)
                if residual_trained and len(residual_ensemble) > 0:
                    x_res_i = np.concatenate(
                        [
                            feats[i : i + 1],
                            delta.reshape(1, -1).astype(np.float32),
                            factors[i : i + 1],
                        ],
                        axis=1,
                    ).astype(np.float32)
                    res_coef_i = _predict_factor_delta(residual_ensemble, x_res_i)[0].astype(np.float32)
                    pred_flat = pred_flat + float(residual_scale) * (res_coef_i @ residual_components)
                if bool(cfg.tree_enable_tenor_calibration):
                    pred_grid = pred_flat.reshape(nx, nt)
                    pred_grid = tenor_cal_a.reshape(1, nt) + tenor_cal_b.reshape(1, nt) * pred_grid
                    pred_flat = pred_grid.reshape(-1)
                forecast_local[i] = np.clip(pred_flat, 1e-4, 4.0)
                fallback_days -= 1
        fallback_days = int(max(0, fallback_days))

        recon_flat[seq] = recon_local
        forecast_flat[seq] = forecast_local
        latent[seq, :k] = factors[:, :k]

        asset_model_payloads.append(
            {
                "asset_id": int(aid),
                "asset_name": str(asset_names[aid] if 0 <= aid < len(asset_names) else f"asset_{aid}"),
                "trained": bool(trained),
                "factor_k": int(k),
                "blend_alpha": float(blend_alpha),
                "regime_trained": bool(regime_trained),
                "regime_blend_weight": float(regime_blend_weight),
                "regime_bounds": [float(regime_bounds[0]), float(regime_bounds[1])],
                "tenor_calibration_enabled": bool(cfg.tree_enable_tenor_calibration),
                "tenor_calibration_a": tenor_cal_a.astype(np.float32),
                "tenor_calibration_b": tenor_cal_b.astype(np.float32),
                "residual_trained": bool(residual_trained),
                "residual_scale": float(residual_scale),
                "residual_factor_k": int(residual_components.shape[0] if residual_trained else 0),
                "ref_mu": ref_mu.astype(np.float32),
                "components": components.astype(np.float32),
                "ensemble": ensemble,
                "regime_ensembles": regime_ensembles if regime_trained else {},
                "residual_components": (residual_components.astype(np.float32) if residual_trained else np.empty((0, flat_asset.shape[1]), dtype=np.float32)),
                "residual_ensemble": residual_ensemble if residual_trained else [],
                "min_history": int(min_history),
                "fallback_persistence_days": int(fallback_days),
            }
        )

        history_rows.append(
            {
                "asset_id": int(aid),
                "asset": str(asset_names[aid] if 0 <= aid < len(asset_names) else f"asset_{aid}"),
                "trained": bool(trained),
                "n_asset_dates": int(len(seq)),
                "n_trainval_pairs": int(len(pair_pos)),
                "factor_k": int(k),
                "blend_alpha": float(blend_alpha),
                "regime_trained": bool(regime_trained),
                "regime_blend_weight": float(regime_blend_weight),
                "tenor_calibration_enabled": bool(cfg.tree_enable_tenor_calibration),
                "tenor_calibration_mean_abs_a": float(np.mean(np.abs(tenor_cal_a))),
                "tenor_calibration_mean_b": float(np.mean(tenor_cal_b)),
                "residual_trained": bool(residual_trained),
                "residual_scale": float(residual_scale),
                "residual_factor_k": int(residual_components.shape[0] if residual_trained else 0),
                "fallback_persistence_days": int(fallback_days),
            }
        )

    recon_iv = recon_flat.reshape(n_dates, nx, nt).astype(np.float32)
    forecast_iv = forecast_flat.reshape(n_dates, nx, nt).astype(np.float32)

    recon_raw = _iv_to_surface_raw_numpy(recon_iv, tenor_days, surface_variable)
    forecast_raw = _iv_to_surface_raw_numpy(forecast_iv, tenor_days, surface_variable)

    run_dir = make_run_dir(cfg.out_dir, prefix="run")

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(run_dir / "train_history.csv", index=False)

    latent_df = pd.DataFrame(latent, columns=[f"z_{i}" for i in range(latent.shape[1])])
    latent_df.insert(0, "date", dates)
    latent_df.insert(1, "asset_id", asset_ids.astype(np.int32))
    latent_df.insert(
        2,
        "asset",
        np.array([asset_names[int(i)] if 0 <= int(i) < len(asset_names) else f"asset_{int(i)}" for i in asset_ids], dtype=object),
    )
    latent_df.to_parquet(run_dir / "latent_states.parquet", index=False)

    artifact = {
        "model_type": "tree_boost_surface_v4_regime_residual",
        "dataset_path": str(dataset_path.resolve()),
        "surface_variable": str(surface_variable),
        "x_grid": x_grid.astype(np.float32),
        "tenor_days": tenor_days.astype(np.int32),
        "context_scaler_state": context_scaler.state(),
        "context_augment_from_contracts": bool(cfg.context_augment_from_contracts),
        "context_augment_surface_history": bool(cfg.context_augment_surface_history),
        "context_added_features": added_context_names + added_surface_context_names,
        "tree_pca_factors": int(max_factors),
        "tree_min_history": int(min_history),
        "tree_enable_regime_experts": bool(cfg.tree_enable_regime_experts),
        "tree_regime_min_rows": int(max(1, cfg.tree_regime_min_rows)),
        "tree_enable_tenor_calibration": bool(cfg.tree_enable_tenor_calibration),
        "tree_tenor_calibration_shrinkage": float(cfg.tree_tenor_calibration_shrinkage),
        "tree_tenor_slope_min": float(cfg.tree_tenor_slope_min),
        "tree_tenor_slope_max": float(cfg.tree_tenor_slope_max),
        "tree_enable_residual_corrector": bool(residual_enabled),
        "tree_residual_factors": int(residual_factors),
        "asset_models": asset_model_payloads,
        "forecast_surface_raw": forecast_raw.astype(np.float32),
        "recon_surface_raw": recon_raw.astype(np.float32),
        "forecast_iv": forecast_iv.astype(np.float32),
        "recon_iv": recon_iv.astype(np.float32),
        "latent": latent.astype(np.float32),
        "asset_ids": asset_ids.astype(np.int32),
        "dates": dates,
    }
    with (run_dir / "tree_model.pkl").open("wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

    cfg_payload = asdict(cfg)
    cfg_payload["out_dir"] = str(cfg.out_dir)
    cfg_payload["model_family"] = "tree_boost_surface_v4_regime_residual"
    (run_dir / "train_config.json").write_text(json.dumps(cfg_payload, indent=2), encoding="utf-8")

    trainval_rmse = _forecast_pairs_rmse(
        iv_obs=iv_obs,
        iv_forecast=forecast_iv,
        asset_ids=asset_ids,
        date_mask=trainval_mask,
    )
    test_rmse = _forecast_pairs_rmse(
        iv_obs=iv_obs,
        iv_forecast=forecast_iv,
        asset_ids=asset_ids,
        date_mask=test_mask,
    )

    summary = {
        "model_path": str((run_dir / "tree_model.pkl").resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "model_family": "tree_boost_surface_v4_regime_residual",
        "surface_variable": str(surface_variable),
        "split_mode": str(split_mode),
        "assets": asset_names,
        "n_assets": int(n_assets),
        "n_dates_total": int(n_dates),
        "n_dates_train": int(len(tr_dates)),
        "n_dates_val": int(len(va_dates)),
        "n_dates_test": int(len(te_dates)),
        "context_dim_original": int(original_context_dim),
        "context_dim_used": int(context.shape[1]),
        "context_augmented_from_contracts": bool(len(added_context_names) > 0),
        "context_augmented_from_surface_history": bool(len(added_surface_context_names) > 0),
        "context_added_features": added_context_names + added_surface_context_names,
        "tree_pca_factors": int(max_factors),
        "tree_min_history": int(min_history),
        "tree_learning_rate": float(cfg.tree_learning_rate),
        "tree_max_iter": int(cfg.tree_max_iter),
        "tree_max_depth": int(cfg.tree_max_depth),
        "tree_min_samples_leaf": int(cfg.tree_min_samples_leaf),
        "tree_l2_regularization": float(cfg.tree_l2_regularization),
        "tree_ensemble_seeds": [int(v) for v in cfg.tree_ensemble_seeds],
        "max_cpu_threads": int(max(1, cfg.max_cpu_threads)),
        "tree_enable_regime_experts": bool(cfg.tree_enable_regime_experts),
        "tree_regime_min_rows": int(max(1, cfg.tree_regime_min_rows)),
        "tree_enable_tenor_calibration": bool(cfg.tree_enable_tenor_calibration),
        "tree_tenor_calibration_shrinkage": float(cfg.tree_tenor_calibration_shrinkage),
        "tree_tenor_slope_min": float(cfg.tree_tenor_slope_min),
        "tree_tenor_slope_max": float(cfg.tree_tenor_slope_max),
        "tree_enable_residual_corrector": bool(cfg.tree_enable_residual_corrector),
        "tree_residual_factors": int(max(1, cfg.tree_residual_factors)),
        "tree_residual_learning_rate": float(cfg.tree_residual_learning_rate),
        "tree_residual_max_iter": int(cfg.tree_residual_max_iter),
        "tree_residual_max_depth": int(cfg.tree_residual_max_depth),
        "tree_residual_min_samples_leaf": int(cfg.tree_residual_min_samples_leaf),
        "tree_residual_l2_regularization": float(cfg.tree_residual_l2_regularization),
        "tree_residual_scale_max": float(cfg.tree_residual_scale_max),
        "surface_focus_density_alpha": float(focus_density_alpha),
        "surface_focus_density_map_path": focus_density_map_path_used,
        "surface_forecast_iv_rmse_trainval": float(trainval_rmse),
        "surface_forecast_iv_rmse_test": float(test_rmse),
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return run_dir


def derive_focus_density_map_from_run(
    *,
    run_dir: Path,
    dataset_path: Path,
    cfg: TrainingConfig,
    out_path: Path,
    error_power: float = 1.0,
    device: str | None = None,
) -> dict[str, object]:
    """Build strict per-asset focus-density map from non-test forecast errors.

    Uses only train+val dates (never test) to avoid leakage into OOS evaluation.
    """
    _ = device  # Compatibility with the previous signature.

    power = float(error_power)
    if not np.isfinite(power) or power <= 0.0:
        raise RuntimeError(f"adaptive focus error power must be > 0, got {error_power}.")

    model_path = run_dir / "tree_model.pkl"
    if not model_path.exists():
        raise RuntimeError(f"Tree model artifact not found: {model_path}")

    with model_path.open("rb") as f:
        artifact = pickle.load(f)
    if not isinstance(artifact, dict):
        raise RuntimeError("Invalid tree model artifact payload.")

    ds = _load_dataset(dataset_path)
    dates = ds["dates"].astype(str)
    n_dates = int(len(dates))
    asset_ids = ds.get("asset_ids", np.zeros(n_dates, dtype=np.int32)).astype(np.int32)
    asset_names = ds.get("asset_names", np.array(["ASSET"], dtype=str)).astype(str).tolist()
    n_assets = int(np.max(asset_ids)) + 1 if len(asset_ids) else 1
    if len(asset_names) != n_assets:
        raise RuntimeError(
            "asset_names length must match inferred n_assets for strict density map build: "
            f"asset_names={len(asset_names)} n_assets={n_assets}."
        )

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

    surface_key = "surface" if "surface" in ds else "iv_surface"
    surface_raw_obs = ds[surface_key].astype(np.float32)
    tenor_days = ds["tenor_days"].astype(np.int32)

    forecast_raw = np.asarray(artifact.get("forecast_surface_raw"), dtype=np.float32)
    if forecast_raw.shape != surface_raw_obs.shape:
        raise RuntimeError(
            "Tree artifact forecast shape mismatch for focus map build: "
            f"forecast={forecast_raw.shape} obs={surface_raw_obs.shape}."
        )

    split_mode = _normalize_split_mode(cfg.split_mode)
    tr_dates, va_dates, te_dates = _date_splits_for_mode(
        n_dates=n_dates,
        asset_ids=asset_ids,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        split_mode=split_mode,
    )
    _assert_no_train_test_date_overlap(
        dates=dates,
        train_idx=tr_dates,
        test_idx=te_dates,
    )
    trainval_mask = np.zeros(n_dates, dtype=bool)
    trainval_mask[np.setdiff1d(np.arange(n_dates), te_dates)] = True

    iv_obs = _surface_to_iv_numpy(surface_raw_obs.astype(np.float32), tenor_days, surface_variable)
    iv_forecast = _surface_to_iv_numpy(forecast_raw.astype(np.float32), tenor_days, surface_variable)

    same_asset_next = asset_ids[:-1] == asset_ids[1:] if n_dates > 1 else np.array([], dtype=bool)
    entry_idx = np.where(same_asset_next & trainval_mask[:-1] & trainval_mask[1:])[0].astype(np.int32)
    if len(entry_idx) == 0:
        raise RuntimeError("No train/val one-step forecast rows available to build focus density map.")
    target_idx = entry_idx + 1
    entry_aid = asset_ids[entry_idx].astype(np.int32)

    obs = iv_obs[target_idx]
    pred = iv_forecast[entry_idx]
    sq_err = (pred - obs) ** 2

    density_payload: dict[str, list[list[float]]] = {}
    counts_by_asset: dict[str, int] = {}
    for aid in range(n_assets):
        rows = np.where(entry_aid == aid)[0]
        asset_name = asset_names[aid]
        if len(rows) == 0:
            raise RuntimeError(
                f"No train/val forecast rows for asset `{asset_name}` when building focus density map."
            )
        rmse_grid = np.sqrt(np.mean(sq_err[rows], axis=0))
        rmse_grid = np.clip(rmse_grid, 1e-8, None)
        if power != 1.0:
            rmse_grid = np.power(rmse_grid, power)
        mean_rmse = float(np.mean(rmse_grid))
        if (not np.isfinite(mean_rmse)) or mean_rmse <= 0.0:
            raise RuntimeError(f"Invalid mean RMSE for asset `{asset_name}` while building focus density map.")
        density = rmse_grid / mean_rmse
        if not np.all(np.isfinite(density)) or np.any(density <= 0.0):
            raise RuntimeError(f"Invalid density values for asset `{asset_name}` while building focus density map.")
        density_payload[asset_name] = density.astype(np.float64).tolist()
        counts_by_asset[asset_name] = int(len(rows))

    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(density_payload, indent=2), encoding="utf-8")

    meta = {
        "source_run_dir": str(run_dir.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "split_mode": split_mode,
        "n_dates_total": int(n_dates),
        "n_dates_train": int(len(tr_dates)),
        "n_dates_val": int(len(va_dates)),
        "n_dates_test": int(len(te_dates)),
        "forecast_rows_used_trainval": int(len(entry_idx)),
        "error_metric": "absolute_rmse",
        "error_power": float(power),
        "assets": asset_names,
        "rows_by_asset": counts_by_asset,
        "x_grid_len": int(surface_raw_obs.shape[1]),
        "tenor_days_len": int(surface_raw_obs.shape[2]),
        "map_path": str(out_path),
        "trainval_only": True,
    }
    meta_path = out_path.with_name(f"{out_path.stem}_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
