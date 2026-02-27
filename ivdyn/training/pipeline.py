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
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.isotonic import IsotonicRegression
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

    # --- Tree-v5 extensions (default-safe, feature-flagged). ---
    # Target space for PCA + forecasting. Keeps scoring in IV space, regardless.
    #   - "iv":             model PCA + deltas directly in IV
    #   - "total_variance": model in w = IV^2 * tau
    #   - "log_total_variance": model in log(w)
    tree_target_space: str = "iv"

    # Pooled cross-asset global model (single ensemble across assets) using a shared PCA basis.
    tree_enable_global_model: bool = False
    tree_global_use_asset_onehot: bool = True

    # Confidence-aware shrinkage using ensemble disagreement -> calibration-only isotonic scaling.
    tree_enable_uncertainty_shrinkage: bool = False
    tree_uncertainty_scale_min: float = 0.0
    tree_uncertainty_scale_max: float = 1.25
    tree_uncertainty_min_calibration_rows: int = 48

    # Time-decayed sample weighting to handle non-stationarity (multiplies existing focus weights).
    tree_enable_time_decay_weight: bool = False
    tree_time_decay_halflife_pairs: float = 120.0
    tree_time_decay_min_weight: float = 0.25

    # Arbitrage-inspired postprocessing (calendar monotonicity + convexity in strike) in total-variance space.
    tree_postprocess_noarb: bool = False
    tree_postprocess_noarb_strength: float = 1.0
    tree_postprocess_noarb_calendar: bool = True
    tree_postprocess_noarb_convex: bool = True



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


def _target_space_name(v: str | None) -> str:
    s = str(v or "").strip().lower()
    if s in {"w", "total_variance", "total_var"}:
        return "total_variance"
    if s in {"log_total_variance", "log_w", "logw"}:
        return "log_total_variance"
    return "iv"


def _iv_to_total_variance_numpy(iv_surface: np.ndarray, tenor_days: np.ndarray) -> np.ndarray:
    tau = (tenor_days.astype(np.float32) / 365.0).reshape(1, 1, -1)
    iv = np.clip(iv_surface.astype(np.float32), 1e-4, 4.0)
    return (iv * iv) * np.clip(tau, 1e-6, None)


def _total_variance_to_iv_numpy(w_surface: np.ndarray, tenor_days: np.ndarray) -> np.ndarray:
    tau = (tenor_days.astype(np.float32) / 365.0).reshape(1, 1, -1)
    w = np.clip(w_surface.astype(np.float32), 1e-8, None)
    iv = np.sqrt(np.clip(w / np.clip(tau, 1e-6, None), 1e-8, None))
    return np.clip(iv, 1e-4, 4.0).astype(np.float32)


def _iv_to_target_numpy(iv_surface: np.ndarray, tenor_days: np.ndarray, target_space: str | None) -> np.ndarray:
    t = _target_space_name(target_space)
    if t == "iv":
        return np.clip(iv_surface, 1e-4, 4.0).astype(np.float32)
    w = _iv_to_total_variance_numpy(iv_surface, tenor_days).astype(np.float32)
    if t == "total_variance":
        return np.clip(w, 1e-8, 8.0).astype(np.float32)
    # log_total_variance
    return np.log(np.clip(w, 1e-8, None)).astype(np.float32)


def _target_to_iv_numpy(target_surface: np.ndarray, tenor_days: np.ndarray, target_space: str | None) -> np.ndarray:
    t = _target_space_name(target_space)
    if t == "iv":
        return np.clip(target_surface, 1e-4, 4.0).astype(np.float32)
    if t == "total_variance":
        return _total_variance_to_iv_numpy(target_surface, tenor_days)
    # log_total_variance
    w = np.exp(np.clip(target_surface, np.log(1e-8), np.log(8.0))).astype(np.float32)
    return _total_variance_to_iv_numpy(w, tenor_days)


def _clip_target_flat(x: np.ndarray, target_space: str | None) -> np.ndarray:
    t = _target_space_name(target_space)
    if t == "iv":
        return np.clip(x, 1e-4, 4.0).astype(np.float32)
    if t == "total_variance":
        return np.clip(x, 1e-8, 8.0).astype(np.float32)
    lo = float(np.log(1e-8))
    hi = float(np.log(8.0))
    return np.clip(x, lo, hi).astype(np.float32)


def _time_decay_weights(pair_pos: np.ndarray, halflife_pairs: float, min_weight: float) -> np.ndarray:
    if len(pair_pos) == 0:
        return np.array([], dtype=np.float32)
    hl = float(halflife_pairs)
    if not np.isfinite(hl) or hl <= 0.0:
        return np.ones(len(pair_pos), dtype=np.float32)
    min_w = float(np.clip(float(min_weight), 0.0, 1.0))
    age = float(np.max(pair_pos)) - pair_pos.astype(np.float64)
    decay = np.exp(-np.log(2.0) * age / hl)
    decay = np.clip(decay, min_w, 1.0)
    return decay.astype(np.float32)


def _predict_factor_delta_members(
    ensemble: list[list[dict[str, object]]],
    x: np.ndarray,
) -> np.ndarray:
    """Return per-member predictions with shape (n_members, n_rows, k)."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if len(ensemble) == 0:
        return np.zeros((0, len(x), 1), dtype=np.float32)
    n_members = len(ensemble)
    k = len(ensemble[0])
    out = np.zeros((n_members, len(x), k), dtype=np.float32)
    for m, member in enumerate(ensemble):
        for j, spec in enumerate(member):
            kind = str(spec.get("kind", "const"))
            if kind == "hgb":
                mdl = spec.get("model")
                out[m, :, j] = 0.0 if mdl is None else np.asarray(mdl.predict(x), dtype=np.float32)
            else:
                out[m, :, j] = float(spec.get("value", 0.0))
    return out.astype(np.float32)


def _delta_uncertainty_scalar(delta_members: np.ndarray) -> np.ndarray:
    """Scalar uncertainty from per-member delta preds; output shape (n_rows,)."""
    if delta_members.ndim != 3 or delta_members.shape[0] <= 1:
        n_rows = int(delta_members.shape[1]) if delta_members.ndim == 3 else 0
        return np.zeros(n_rows, dtype=np.float32)
    var = np.var(delta_members.astype(np.float64), axis=0)  # (n_rows, k)
    u = np.sqrt(np.mean(var, axis=1))
    return u.astype(np.float32)


def _predict_factor_delta_with_regimes_and_members(
    *,
    x: np.ndarray,
    level_values: np.ndarray,
    global_ensemble: list[list[dict[str, object]]],
    regime_ensembles: dict[int, list[list[dict[str, object]]]],
    regime_bounds: tuple[float, float],
    blend_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict delta mean and a scalar disagreement-based uncertainty."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if len(x) == 0:
        return np.zeros((0, 1), dtype=np.float32), np.zeros(0, dtype=np.float32)

    global_members = _predict_factor_delta_members(global_ensemble, x)
    delta_global = np.mean(global_members, axis=0).astype(np.float32)
    unc_global = _delta_uncertainty_scalar(global_members)

    if len(regime_ensembles) == 0:
        return delta_global, unc_global
    w = float(np.clip(float(blend_weight), 0.0, 1.0))
    if w <= 0.0:
        return delta_global, unc_global

    levels = np.asarray(level_values, dtype=np.float32).reshape(-1)
    if len(levels) != len(x):
        raise RuntimeError(
            f"Regime level length mismatch: levels={len(levels)} x_rows={len(x)}"
        )
    labels = np.digitize(
        levels.astype(np.float64),
        [float(regime_bounds[0]), float(regime_bounds[1])],
    ).astype(np.int32)

    out_mean = delta_global.copy()
    out_unc = unc_global.copy()

    for regime, ens in regime_ensembles.items():
        mask = labels == int(regime)
        if not np.any(mask):
            continue
        # If we can compute members for regimes with matching ensemble size, blend per-member.
        if len(ens) == len(global_ensemble) and len(global_ensemble) > 0:
            reg_members = _predict_factor_delta_members(ens, x[mask])  # (m, rows, k)
            glob_members = global_members[:, mask, :]
            blended_members = (1.0 - w) * glob_members + w * reg_members
            out_mean[mask] = np.mean(blended_members, axis=0).astype(np.float32)
            out_unc[mask] = _delta_uncertainty_scalar(blended_members)
        else:
            reg_mean = _predict_factor_delta(ens, x[mask]).astype(np.float32)
            out_mean[mask] = (1.0 - w) * out_mean[mask] + w * reg_mean
            # keep out_unc from global
    return out_mean, out_unc


def _optimal_row_scale(
    *,
    delta: np.ndarray,
    y_true: np.ndarray,
    base_scale: float,
    scale_min: float,
    scale_max: float,
) -> np.ndarray:
    """Row-wise scalar g that best fits base_scale * g * delta ~= y_true (LS)."""
    d = delta.astype(np.float64)
    y = y_true.astype(np.float64)
    denom = np.sum(d * d, axis=1) + 1e-12
    num = np.sum(d * y, axis=1)
    base = float(max(abs(float(base_scale)), 1e-6))
    g = num / (denom * base)
    g = np.clip(g, float(scale_min), float(scale_max))
    return g.astype(np.float32)


def _isotonic_increasing(y: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Pure-numpy PAV isotonic regression (increasing)."""
    v = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(len(v))
    if n <= 1:
        return v.copy()

    start: list[int] = []
    end: list[int] = []
    weight: list[float] = []
    value: list[float] = []

    for i in range(n):
        start.append(i)
        end.append(i)
        weight.append(1.0)
        value.append(float(v[i]))
        while len(value) >= 2 and value[-2] > value[-1] + float(eps):
            w1 = weight[-2]
            w2 = weight[-1]
            new_w = w1 + w2
            new_v = (w1 * value[-2] + w2 * value[-1]) / max(new_w, float(eps))
            end[-2] = end[-1]
            weight[-2] = new_w
            value[-2] = float(new_v)
            start.pop()
            end.pop()
            weight.pop()
            value.pop()

    out = np.empty(n, dtype=np.float64)
    for s, e, val in zip(start, end, value):
        out[s : e + 1] = float(val)
    return out


def _convexify_1d(y: np.ndarray) -> np.ndarray:
    """Approx convex projection for uniform x-grid via isotonic on first-differences."""
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(len(yy))
    if n <= 2:
        return yy.copy()
    d = np.diff(yy)  # slopes
    d_hat = _isotonic_increasing(d)
    y_hat = yy[0] + np.concatenate([np.array([0.0], dtype=np.float64), np.cumsum(d_hat)])
    # allow constant shift to reduce bias (does not affect convexity)
    y_hat = y_hat + (np.mean(yy) - np.mean(y_hat))
    return y_hat


def _noarb_project_total_variance(
    w: np.ndarray,
    *,
    enforce_calendar: bool,
    enforce_convex: bool,
) -> np.ndarray:
    w0 = np.asarray(w, dtype=np.float64)
    if w0.ndim != 2:
        raise RuntimeError(f"Expected 2D (nx, nt) total-variance grid, got {w0.shape}")
    out = w0.copy()

    if bool(enforce_calendar):
        out = np.maximum.accumulate(out, axis=1)

    if bool(enforce_convex):
        nx, nt = out.shape
        for t in range(nt):
            out[:, t] = _convexify_1d(out[:, t])

    # Re-apply calendar monotonicity in case convexification perturbed it slightly.
    if bool(enforce_calendar):
        out = np.maximum.accumulate(out, axis=1)

    return np.clip(out, 1e-8, 8.0).astype(np.float32)


def _postprocess_noarb_iv(
    iv_grid: np.ndarray,
    tenor_days: np.ndarray,
    *,
    strength: float,
    enforce_calendar: bool,
    enforce_convex: bool,
) -> np.ndarray:
    iv0 = np.asarray(iv_grid, dtype=np.float32)
    s = float(np.clip(float(strength), 0.0, 1.0))
    if s <= 0.0:
        return np.clip(iv0, 1e-4, 4.0).astype(np.float32)

    w0 = _iv_to_total_variance_numpy(iv0.reshape(1, *iv0.shape), tenor_days)[0]
    w_proj = _noarb_project_total_variance(
        w0,
        enforce_calendar=bool(enforce_calendar),
        enforce_convex=bool(enforce_convex),
    )
    w_blend = (1.0 - s) * w0.astype(np.float32) + s * w_proj.astype(np.float32)
    # Ensure final is still projected (blend can slightly violate constraints).
    w_final = _noarb_project_total_variance(
        w_blend,
        enforce_calendar=bool(enforce_calendar),
        enforce_convex=bool(enforce_convex),
    )
    iv_final = _total_variance_to_iv_numpy(w_final.reshape(1, *w_final.shape), tenor_days)[0]
    return np.clip(iv_final, 1e-4, 4.0).astype(np.float32)


def _apply_noarb_postprocess_to_target_flat(
    *,
    pred_flat: np.ndarray,
    nx: int,
    nt: int,
    tenor_days: np.ndarray,
    target_space: str | None,
    strength: float,
    enforce_calendar: bool,
    enforce_convex: bool,
) -> np.ndarray:
    pred_flat = np.asarray(pred_flat, dtype=np.float32).reshape(-1)
    if pred_flat.size != int(nx * nt):
        raise RuntimeError(
            f"Pred flat length mismatch: got {pred_flat.size}, expected {nx * nt}"
        )
    iv = _target_to_iv_numpy(pred_flat.reshape(1, nx, nt), tenor_days, target_space)[0]
    iv_pp = _postprocess_noarb_iv(
        iv,
        tenor_days,
        strength=float(strength),
        enforce_calendar=bool(enforce_calendar),
        enforce_convex=bool(enforce_convex),
    )
    tgt = _iv_to_target_numpy(iv_pp.reshape(1, nx, nt), tenor_days, target_space)[0]
    return tgt.reshape(-1).astype(np.float32)


def _fit_global_pca_basis_by_asset(
    flat_surface: np.ndarray,
    asset_ids: np.ndarray,
    trainval_mask: np.ndarray,
    max_factors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared PCA basis on per-asset demeaned surfaces using train+val only."""
    flat = np.asarray(flat_surface, dtype=np.float32)
    aid = np.asarray(asset_ids, dtype=np.int32).reshape(-1)
    mask = np.asarray(trainval_mask, dtype=bool).reshape(-1)
    if len(flat) != len(aid) or len(flat) != len(mask):
        raise RuntimeError("Global PCA inputs length mismatch.")

    n_assets = int(np.max(aid)) + 1 if len(aid) else 1
    d = int(flat.shape[1])
    ref_mu_by_asset = np.zeros((n_assets, d), dtype=np.float32)

    residuals: list[np.ndarray] = []
    for a in range(n_assets):
        rows = np.where((aid == a) & mask)[0]
        if len(rows) == 0:
            rows = np.where(aid == a)[0]
        if len(rows) == 0:
            continue
        mu = np.mean(flat[rows].astype(np.float64), axis=0).astype(np.float32)
        ref_mu_by_asset[a] = mu
        residuals.append((flat[rows] - mu).astype(np.float32))

    if not residuals:
        return ref_mu_by_asset, np.eye(d, dtype=np.float32)[:1]

    ref = np.concatenate(residuals, axis=0).astype(np.float64)
    try:
        _, _, vt = np.linalg.svd(ref, full_matrices=False)
    except np.linalg.LinAlgError:
        return ref_mu_by_asset, np.eye(d, dtype=np.float32)[:1]
    if vt.size == 0:
        return ref_mu_by_asset, np.eye(d, dtype=np.float32)[:1]
    k = int(max(1, min(int(max_factors), vt.shape[0], d)))
    comps = vt[:k].astype(np.float32)
    return ref_mu_by_asset.astype(np.float32), comps


def _asset_onehot(asset_ids: np.ndarray, n_assets: int) -> np.ndarray:
    aid = np.asarray(asset_ids, dtype=np.int32).reshape(-1)
    n = int(len(aid))
    k = int(max(1, n_assets))
    out = np.zeros((n, k), dtype=np.float32)
    valid = (aid >= 0) & (aid < k)
    if np.any(valid):
        rows = np.where(valid)[0]
        out[rows, aid[rows]] = 1.0
    return out.astype(np.float32)


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

    # Always score in IV space.
    iv_obs = _surface_to_iv_numpy(surface_raw_obs.astype(np.float32), tenor_days, surface_variable)

    # Optional target-space transform for factor/PCA modeling.
    target_space = _target_space_name(cfg.tree_target_space)
    target_obs = _iv_to_target_numpy(iv_obs.astype(np.float32), tenor_days, target_space)
    target_flat = target_obs.reshape(n_dates, -1).astype(np.float32)

    # Tenor calibration is defined in IV space only; disable silently otherwise.
    tenor_calibration_enabled = bool(cfg.tree_enable_tenor_calibration) and target_space == "iv"

    max_factors = int(max(1, min(cfg.tree_pca_factors, cfg.latent_dim, target_flat.shape[1])))
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

    # Output buffers in target space; converted back to IV at the end.
    latent = np.zeros((n_dates, max_factors), dtype=np.float32)
    recon_flat = target_flat.copy()
    forecast_flat = target_flat.copy()

    history_rows: list[dict[str, object]] = []
    asset_model_payloads: list[dict[str, object]] = []

    # ------------------------------------------------------------
    # Global cross-asset model (shared PCA basis + pooled delta model).
    # ------------------------------------------------------------
    if bool(cfg.tree_enable_global_model):
        # Shared PCA basis on train+val only, after per-asset demeaning.
        ref_mu_by_asset, components = _fit_global_pca_basis_by_asset(
            target_flat,
            asset_ids=asset_ids,
            trainval_mask=trainval_mask,
            max_factors=max_factors,
        )
        k = int(components.shape[0])

        # Factors for all dates in the shared basis.
        ref_mu_for_rows = ref_mu_by_asset[asset_ids.astype(np.int32)]
        factors_all = ((target_flat - ref_mu_for_rows) @ components.T).astype(np.float32)

        # Build per-asset lagged features (same logic as per-asset model), then append optional one-hot.
        cnn_all = _cnn_surface_features(iv_obs.astype(np.float32), tenor_days)
        feats_all = np.zeros(
            (n_dates, (k * 5) + (cnn_all.shape[1] * 2) + context_scaled.shape[1]),
            dtype=np.float32,
        )
        for aid in range(n_assets):
            seq = np.where(asset_ids == aid)[0]
            if len(seq) == 0:
                continue
            seq = np.sort(seq)
            feats_all[seq] = _build_factor_feature_matrix(
                factors_all[seq],
                cnn_all[seq],
                context_scaled[seq].astype(np.float32),
            )

        if bool(cfg.tree_global_use_asset_onehot) and n_assets > 1:
            feats_all = np.concatenate([feats_all, _asset_onehot(asset_ids, n_assets)], axis=1).astype(np.float32)

        # Build train+val pair indices by asset (never cross assets).
        # NOTE: keep per-asset alignment so calibration splits can be done per asset safely.
        pair_entry_by_asset: list[np.ndarray] = [np.array([], dtype=np.int64) for _ in range(n_assets)]
        pair_target_by_asset: list[np.ndarray] = [np.array([], dtype=np.int64) for _ in range(n_assets)]
        pair_asset_by_asset: list[np.ndarray] = [np.array([], dtype=np.int32) for _ in range(n_assets)]
        for aid in range(n_assets):
            seq = np.where(asset_ids == aid)[0]
            if len(seq) <= 1:
                continue
            seq = np.sort(seq)
            pos = np.where(trainval_mask[seq[:-1]] & trainval_mask[seq[1:]])[0].astype(np.int32)
            if len(pos) == 0:
                continue
            pair_entry_by_asset[aid] = seq[pos].astype(np.int64)
            pair_target_by_asset[aid] = seq[pos + 1].astype(np.int64)
            pair_asset_by_asset[aid] = np.full(len(pos), aid, dtype=np.int32)

        valid_assets = [aid for aid in range(n_assets) if len(pair_entry_by_asset[aid]) > 0]
        if not valid_assets:
            raise RuntimeError("Global model: no train/val one-step pairs available.")

        pair_entry = np.concatenate([pair_entry_by_asset[aid] for aid in valid_assets]).astype(np.int64)
        pair_target = np.concatenate([pair_target_by_asset[aid] for aid in valid_assets]).astype(np.int64)
        pair_asset = np.concatenate([pair_asset_by_asset[aid] for aid in valid_assets]).astype(np.int32)


        # Calibration split per asset: hold out the tail of each asset's trainval pairs.
        cal_frac = float(np.clip(float(cfg.tree_calibration_frac), 0.0, 0.45))
        min_cal = int(max(0, cfg.tree_min_calibration_rows))
        pair_fit_mask = np.ones(len(pair_entry), dtype=bool)
        pair_cal_mask = np.zeros(len(pair_entry), dtype=bool)

        # We compute the mask by iterating assets and selecting tail indices within that asset's pairs.
        offset = 0
        for aid in valid_assets:
            # Pairs are concatenated in valid_assets order (ascending aid), so we can slice.
            n_pairs = int(len(pair_entry_by_asset[aid]))
            if n_pairs <= 0:
                continue
            sl = slice(offset, offset + n_pairs)
            offset += n_pairs

            cal_n = int(max(min_cal, round(n_pairs * cal_frac))) if cal_frac > 0 else 0
            cal_n = min(cal_n, max(0, n_pairs - min_history))
            if cal_n <= 0:
                continue
            # Use the *most recent* pairs (tail of sequence).
            idx_local = np.arange(n_pairs - cal_n, n_pairs, dtype=np.int64)
            pair_fit_mask[sl][idx_local] = False
            pair_cal_mask[sl][idx_local] = True

        # Ensure we have fit samples.
        if not np.any(pair_fit_mask):
            pair_fit_mask[:] = True
            pair_cal_mask[:] = False

        entry_fit = pair_entry[pair_fit_mask]
        target_fit = pair_target[pair_fit_mask]
        aid_fit = pair_asset[pair_fit_mask]

        x_fit = feats_all[entry_fit].astype(np.float32)
        y_fit = (factors_all[target_fit] - factors_all[entry_fit]).astype(np.float32)

        # Sample weights: focus (and optional density map) + optional time decay by asset.
        w_fit = np.ones(len(entry_fit), dtype=np.float32)
        for i, (e_idx, t_idx, aid) in enumerate(zip(entry_fit, target_fit, aid_fit, strict=False)):
            move = np.abs(iv_obs[t_idx] - iv_obs[e_idx]).astype(np.float64)
            density_grid = (
                focus_density_by_asset[int(aid)]
                if (focus_density_by_asset is not None and 0 <= int(aid) < len(focus_density_by_asset))
                else None
            )
            g = focus_grid.astype(np.float64)
            if density_grid is not None and focus_density_alpha > 0.0:
                d = np.clip(density_grid.astype(np.float64), 1e-8, None)
                g = g * np.power(d, float(focus_density_alpha))
            g = g / max(float(np.mean(g)), 1e-8)
            w_fit[i] = np.float32(1.0 + max(float(cfg.surface_focus_alpha), 0.0) * float(np.mean(move * g)))
        w_fit = np.clip(w_fit, 1.0, 8.0).astype(np.float32)

        # Optional time decay (within each asset, on entry position rank).
        if bool(cfg.tree_enable_time_decay_weight):
            # Build an "age" proxy from per-asset pair order.
            w_time = np.ones(len(entry_fit), dtype=np.float32)
            for aid in range(n_assets):
                mask = aid_fit == aid
                if not np.any(mask):
                    continue
                # Use the order in the original asset sequence.
                seq = np.where(asset_ids == aid)[0]
                seq = np.sort(seq)
                rank = {int(v): int(i) for i, v in enumerate(seq)}
                pos = np.asarray([rank.get(int(e), 0) for e in entry_fit[mask]], dtype=np.int64)
                w_time[mask] = _time_decay_weights(
                    pos.astype(np.int32),
                    halflife_pairs=float(cfg.tree_time_decay_halflife_pairs),
                    min_weight=float(cfg.tree_time_decay_min_weight),
                )
            w_fit = (w_fit * w_time).astype(np.float32)

        seeds = tuple(int(s) for s in cfg.tree_ensemble_seeds if int(s) >= 0) or (13,)

        global_ensemble = _fit_factor_ensemble(
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

        # Calibrate per-asset blend alpha on that asset's held-out tail pairs.
        alpha_grid = np.linspace(0.2, 1.2, 11)
        blend_alpha_by_asset = np.ones(n_assets, dtype=np.float32)
        if np.any(pair_cal_mask):
            entry_cal = pair_entry[pair_cal_mask]
            target_cal = pair_target[pair_cal_mask]
            aid_cal = pair_asset[pair_cal_mask]
            delta_cal = _predict_factor_delta(global_ensemble, feats_all[entry_cal].astype(np.float32)).astype(np.float32)
            for aid in range(n_assets):
                mask = aid_cal == aid
                if not np.any(mask):
                    continue
                best_alpha = 1.0
                best_mse = float("inf")
                y_true = target_flat[target_cal[mask]]
                f0 = factors_all[entry_cal[mask]]
                d0 = delta_cal[mask]
                for a in alpha_grid:
                    f_pred = f0 + float(a) * d0
                    y_pred = ref_mu_by_asset[aid] + (f_pred @ components)
                    y_pred = _clip_target_flat(y_pred, target_space)
                    mse = float(np.mean((y_pred - y_true) ** 2))
                    if mse < best_mse:
                        best_mse = mse
                        best_alpha = float(a)
                blend_alpha_by_asset[aid] = np.float32(best_alpha)

        # Optional uncertainty shrinkage: fit a single isotonic map on pooled calibration pairs.
        uncertainty_model = None
        if bool(cfg.tree_enable_uncertainty_shrinkage) and np.any(pair_cal_mask) and len(seeds) > 1:
            entry_cal = pair_entry[pair_cal_mask]
            target_cal = pair_target[pair_cal_mask]
            x_cal = feats_all[entry_cal].astype(np.float32)
            y_true_delta = (factors_all[target_cal] - factors_all[entry_cal]).astype(np.float32)

            delta_mean, unc = _predict_factor_delta_with_regimes_and_members(
                x=x_cal,
                level_values=cnn_all[entry_cal, 0],
                global_ensemble=global_ensemble,
                regime_ensembles={},
                regime_bounds=(0.0, 0.0),
                blend_weight=0.0,
            )
            # Use per-asset base alpha in the scale target.
            base_scale = blend_alpha_by_asset[aid_cal.astype(np.int32)]
            base_scale = np.clip(base_scale, 0.05, 3.0)
            g_opt = np.zeros(len(entry_cal), dtype=np.float32)
            for i in range(len(entry_cal)):
                g_opt[i] = _optimal_row_scale(
                    delta=delta_mean[i : i + 1],
                    y_true=y_true_delta[i : i + 1],
                    base_scale=float(base_scale[i]),
                    scale_min=float(cfg.tree_uncertainty_scale_min),
                    scale_max=float(cfg.tree_uncertainty_scale_max),
                )[0]
            if len(g_opt) >= int(max(8, cfg.tree_uncertainty_min_calibration_rows)):
                # IsotonicRegression is monotone increasing; use x = -unc so larger uncertainty => smaller x.
                iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
                iso.fit(-unc.astype(np.float64), g_opt.astype(np.float64))
                uncertainty_model = iso

        # Forecast in target space for all assets.
        forecast_local = target_flat.copy()
        for aid in range(n_assets):
            seq = np.where(asset_ids == aid)[0]
            if len(seq) <= 1:
                continue
            seq = np.sort(seq)
            x_pred = feats_all[seq[:-1]].astype(np.float32)

            if bool(cfg.tree_enable_uncertainty_shrinkage) and uncertainty_model is not None and len(seeds) > 1:
                delta_mean, unc = _predict_factor_delta_with_regimes_and_members(
                    x=x_pred,
                    level_values=cnn_all[seq[:-1], 0],
                    global_ensemble=global_ensemble,
                    regime_ensembles={},
                    regime_bounds=(0.0, 0.0),
                    blend_weight=0.0,
                )
                scale = uncertainty_model.predict(-unc.astype(np.float64)).astype(np.float32)
            else:
                delta_mean = _predict_factor_delta(global_ensemble, x_pred).astype(np.float32)
                scale = np.ones(len(seq) - 1, dtype=np.float32)

            alpha = float(blend_alpha_by_asset[aid])
            for j in range(len(seq) - 1):
                f_pred = factors_all[seq[j]] + alpha * float(scale[j]) * delta_mean[j]
                pred_flat = ref_mu_by_asset[aid] + (f_pred @ components)

                # Optional postprocess (in IV / total-variance space, then map back to target).
                if bool(cfg.tree_postprocess_noarb):
                    pred_flat = _apply_noarb_postprocess_to_target_flat(
                        pred_flat=pred_flat,
                        nx=nx,
                        nt=nt,
                        tenor_days=tenor_days,
                        target_space=target_space,
                        strength=float(cfg.tree_postprocess_noarb_strength),
                        enforce_calendar=bool(cfg.tree_postprocess_noarb_calendar),
                        enforce_convex=bool(cfg.tree_postprocess_noarb_convex),
                    )

                forecast_local[seq[j]] = _clip_target_flat(pred_flat, target_space)

            # History summary rows per asset.
            history_rows.append(
                {
                    "asset_id": int(aid),
                    "asset": str(asset_names[aid] if 0 <= aid < len(asset_names) else f"asset_{aid}"),
                    "trained": True,
                    "n_asset_dates": int(len(seq)),
                    "n_trainval_pairs": int(np.sum(trainval_mask[seq[:-1]] & trainval_mask[seq[1:]])),
                    "factor_k": int(k),
                    "blend_alpha": float(blend_alpha_by_asset[aid]),
                    "regime_trained": False,
                    "regime_blend_weight": 0.0,
                    "tenor_calibration_enabled": False,
                    "tenor_calibration_mean_abs_a": 0.0,
                    "tenor_calibration_mean_b": 1.0,
                    "residual_trained": False,
                    "residual_scale": 0.0,
                    "residual_factor_k": 0,
                    "fallback_persistence_days": 0,
                    "global_model": True,
                    "target_space": str(target_space),
                }
            )
            asset_model_payloads.append(
                {
                    "asset_id": int(aid),
                    "asset_name": str(asset_names[aid] if 0 <= aid < len(asset_names) else f"asset_{aid}"),
                    "trained": True,
                    "global_model": True,
                    "target_space": str(target_space),
                    "factor_k": int(k),
                    "blend_alpha": float(blend_alpha_by_asset[aid]),
                    "ref_mu": ref_mu_by_asset[aid].astype(np.float32),
                    "components": components,
                    "ensemble": global_ensemble,
                    "uncertainty_shrinkage_enabled": bool(cfg.tree_enable_uncertainty_shrinkage and uncertainty_model is not None),
                    "uncertainty_model": uncertainty_model,
                    "fallback_persistence_days": 0,
                }
            )

        # Write outputs.
        forecast_flat = forecast_local.astype(np.float32)
        recon_flat = _clip_target_flat((ref_mu_for_rows + (factors_all @ components)), target_space).astype(np.float32)
        latent[:, :k] = factors_all[:, :k].astype(np.float32)

    # ------------------------------------------------------------
    # Per-asset model (existing v4 pipeline + v5 feature flags).
    # ------------------------------------------------------------
    else:
        for aid in range(n_assets):
            seq = np.where(asset_ids == aid)[0]
            if len(seq) <= 1:
                continue

            seq = np.sort(seq)
            iv_asset = iv_obs[seq].astype(np.float32)
            tgt_asset = target_obs[seq].astype(np.float32)
            flat_asset = tgt_asset.reshape(len(seq), -1).astype(np.float32)

            ctx_asset = context_scaled[seq].astype(np.float32)
            cnn_asset = _cnn_surface_features(iv_asset, tenor_days)

            # Train+val one-step pairs within this asset (local positions).
            pair_pos = np.where(trainval_mask[seq[:-1]] & trainval_mask[seq[1:]])[0].astype(np.int32)

            ref_mu, components = _fit_pca_basis(flat_asset, pair_pos, max_factors)
            k = int(components.shape[0])

            factors = ((flat_asset - ref_mu) @ components.T).astype(np.float32)
            recon_local = _clip_target_flat(ref_mu + (factors @ components), target_space).astype(np.float32)

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

            # Confidence shrink mapping (optional).
            uncertainty_model = None

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
                if bool(cfg.tree_enable_time_decay_weight):
                    w_fit = (w_fit * _time_decay_weights(
                        pair_fit,
                        halflife_pairs=float(cfg.tree_time_decay_halflife_pairs),
                        min_weight=float(cfg.tree_time_decay_min_weight),
                    )).astype(np.float32)

                seeds = tuple(int(s) for s in cfg.tree_ensemble_seeds if int(s) >= 0) or (13,)

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

                # Calibration for blend alpha (and regime blend weight, if enabled).
                if len(pair_cal) > 0:
                    x_cal = feats[pair_cal]
                    y_true_cal = flat_asset[pair_cal + 1]
                    alpha_grid = np.linspace(0.2, 1.2, 11)
                    regime_grid = (
                        np.linspace(0.0, 1.0, 5)
                        if len(regime_fit_ensembles) > 0
                        else np.array([0.0], dtype=np.float32)
                    )
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
                            y_pred = ref_mu + (f_pred @ components)
                            y_pred = _clip_target_flat(y_pred, target_space)
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

                # Optional tenor calibration (IV-space only).
                if tenor_calibration_enabled and len(pair_cal) > 0:
                    delta_cal_for_tenor = _predict_factor_delta_with_regimes(
                        x=feats[pair_cal],
                        level_values=level_series[pair_cal],
                        global_ensemble=ensemble_fit,
                        regime_ensembles=regime_fit_ensembles,
                        regime_bounds=regime_fit_bounds,
                        blend_weight=float(regime_blend_weight),
                    ).astype(np.float32)
                    pred_cal_flat = _clip_target_flat(
                        ref_mu + ((factors[pair_cal] + float(blend_alpha) * delta_cal_for_tenor) @ components),
                        target_space,
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

                # Fit final models on all train+val pairs for this asset.
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
                if bool(cfg.tree_enable_time_decay_weight):
                    w_all = (w_all * _time_decay_weights(
                        pair_pos,
                        halflife_pairs=float(cfg.tree_time_decay_halflife_pairs),
                        min_weight=float(cfg.tree_time_decay_min_weight),
                    )).astype(np.float32)

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

                # Optional uncertainty shrink calibration (calibration-only, uses member disagreement).
                if (
                    bool(cfg.tree_enable_uncertainty_shrinkage)
                    and len(pair_cal) >= int(max(8, cfg.tree_uncertainty_min_calibration_rows))
                    and len(seeds) > 1
                ):
                    x_cal = feats[pair_cal].astype(np.float32)
                    y_true_delta = (factors[pair_cal + 1] - factors[pair_cal]).astype(np.float32)
                    delta_mean, unc = _predict_factor_delta_with_regimes_and_members(
                        x=x_cal,
                        level_values=level_series[pair_cal],
                        global_ensemble=ensemble_fit,
                        regime_ensembles=regime_fit_ensembles,
                        regime_bounds=regime_fit_bounds,
                        blend_weight=float(regime_blend_weight),
                    )
                    g_opt = _optimal_row_scale(
                        delta=delta_mean,
                        y_true=y_true_delta,
                        base_scale=float(blend_alpha),
                        scale_min=float(cfg.tree_uncertainty_scale_min),
                        scale_max=float(cfg.tree_uncertainty_scale_max),
                    )
                    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
                    iso.fit(-unc.astype(np.float64), g_opt.astype(np.float64))
                    uncertainty_model = iso

                # Residual corrector (unchanged logic, operates in target space).
                if residual_enabled and len(pair_fit) >= max(24, min_history // 2):
                    delta_fit_primary = _predict_factor_delta_with_regimes(
                        x=feats[pair_fit],
                        level_values=level_series[pair_fit],
                        global_ensemble=ensemble_fit,
                        regime_ensembles=regime_fit_ensembles,
                        regime_bounds=regime_fit_bounds,
                        blend_weight=float(regime_blend_weight),
                    ).astype(np.float32)
                    base_fit = _clip_target_flat(
                        ref_mu + ((factors[pair_fit] + float(blend_alpha) * delta_fit_primary) @ components),
                        target_space,
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
                            base_cal = _clip_target_flat(
                                ref_mu + ((factors[pair_cal] + float(blend_alpha) * delta_cal_primary) @ components),
                                target_space,
                            ).astype(np.float32)
                            x_res_cal = np.concatenate([feats[pair_cal], delta_cal_primary, factors[pair_cal]], axis=1).astype(np.float32)
                            res_cal_coef = _predict_factor_delta(residual_ensemble_fit, x_res_cal).astype(np.float32)
                            res_cal_flat = (res_cal_coef @ residual_components).astype(np.float32)
                            y_true_cal = flat_asset[pair_cal + 1].astype(np.float32)

                            best_gamma = 1.0
                            best_mse = float("inf")
                            gamma_grid = np.linspace(0.0, float(residual_scale_max), 13)
                            for gamma in gamma_grid:
                                y_pred = _clip_target_flat(base_cal + float(gamma) * res_cal_flat, target_space)
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
                        base_all = _clip_target_flat(
                            ref_mu + ((factors[pair_pos] + float(blend_alpha) * delta_all_primary) @ components),
                            target_space,
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

            # Forecast generation (batch delta prediction for speed).
            forecast_local = flat_asset.copy()
            fallback_days = int(len(seq) - 1)
            if trained and len(ensemble) > 0 and len(seq) > 1:
                x_pred = feats[: len(seq) - 1].astype(np.float32)
                if bool(cfg.tree_enable_uncertainty_shrinkage) and uncertainty_model is not None and len(cfg.tree_ensemble_seeds) > 1:
                    delta_pred, unc_pred = _predict_factor_delta_with_regimes_and_members(
                        x=x_pred,
                        level_values=level_series[: len(seq) - 1],
                        global_ensemble=ensemble,
                        regime_ensembles=regime_ensembles,
                        regime_bounds=regime_bounds,
                        blend_weight=float(regime_blend_weight),
                    )
                    scale_pred = uncertainty_model.predict(-unc_pred.astype(np.float64)).astype(np.float32)
                else:
                    delta_pred = _predict_factor_delta_with_regimes(
                        x=x_pred,
                        level_values=level_series[: len(seq) - 1],
                        global_ensemble=ensemble,
                        regime_ensembles=regime_ensembles,
                        regime_bounds=regime_bounds,
                        blend_weight=float(regime_blend_weight),
                    ).astype(np.float32)
                    scale_pred = np.ones(len(seq) - 1, dtype=np.float32)

                for i in range(len(seq) - 1):
                    delta = delta_pred[i]
                    f_pred = factors[i] + float(blend_alpha) * float(scale_pred[i]) * delta
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

                    if tenor_calibration_enabled:
                        pred_grid = pred_flat.reshape(nx, nt)
                        pred_grid = tenor_cal_a.reshape(1, nt) + tenor_cal_b.reshape(1, nt) * pred_grid
                        pred_flat = pred_grid.reshape(-1)

                    if bool(cfg.tree_postprocess_noarb):
                        pred_flat = _apply_noarb_postprocess_to_target_flat(
                            pred_flat=pred_flat,
                            nx=nx,
                            nt=nt,
                            tenor_days=tenor_days,
                            target_space=target_space,
                            strength=float(cfg.tree_postprocess_noarb_strength),
                            enforce_calendar=bool(cfg.tree_postprocess_noarb_calendar),
                            enforce_convex=bool(cfg.tree_postprocess_noarb_convex),
                        )

                    forecast_local[i] = _clip_target_flat(pred_flat, target_space)
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
                    "global_model": False,
                    "target_space": str(target_space),
                    "factor_k": int(k),
                    "blend_alpha": float(blend_alpha),
                    "regime_trained": bool(regime_trained),
                    "regime_blend_weight": float(regime_blend_weight),
                    "regime_bounds": [float(regime_bounds[0]), float(regime_bounds[1])],
                    "tenor_calibration_enabled": bool(tenor_calibration_enabled),
                    "tenor_calibration_a": tenor_cal_a.astype(np.float32),
                    "tenor_calibration_b": tenor_cal_b.astype(np.float32),
                    "residual_trained": bool(residual_trained),
                    "residual_scale": float(residual_scale),
                    "residual_factor_k": int(residual_components.shape[0] if residual_trained else 0),
                    "uncertainty_shrinkage_enabled": bool(cfg.tree_enable_uncertainty_shrinkage and uncertainty_model is not None),
                    "uncertainty_model": uncertainty_model,
                    "ref_mu": ref_mu.astype(np.float32),
                    "components": components.astype(np.float32),
                    "ensemble": ensemble,
                    "regime_ensembles": regime_ensembles if regime_trained else {},
                    "residual_components": (
                        residual_components.astype(np.float32)
                        if residual_trained
                        else np.empty((0, flat_asset.shape[1]), dtype=np.float32)
                    ),
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
                    "tenor_calibration_enabled": bool(tenor_calibration_enabled),
                    "tenor_calibration_mean_abs_a": float(np.mean(np.abs(tenor_cal_a))) if tenor_calibration_enabled else 0.0,
                    "tenor_calibration_mean_b": float(np.mean(tenor_cal_b)) if tenor_calibration_enabled else 1.0,
                    "residual_trained": bool(residual_trained),
                    "residual_scale": float(residual_scale),
                    "residual_factor_k": int(residual_components.shape[0] if residual_trained else 0),
                    "fallback_persistence_days": int(fallback_days),
                    "global_model": False,
                    "target_space": str(target_space),
                }
            )

    # Convert back to IV space for metrics and for surface_variable serialization.
    recon_target = recon_flat.reshape(n_dates, nx, nt).astype(np.float32)
    forecast_target = forecast_flat.reshape(n_dates, nx, nt).astype(np.float32)
    recon_iv = _target_to_iv_numpy(recon_target, tenor_days, target_space).astype(np.float32)
    forecast_iv = _target_to_iv_numpy(forecast_target, tenor_days, target_space).astype(np.float32)

    recon_raw = _iv_to_surface_raw_numpy(recon_iv, tenor_days, surface_variable)
    forecast_raw = _iv_to_surface_raw_numpy(forecast_iv, tenor_days, surface_variable)

    # Tag the model family by enabled v5 extensions for easier tracking.
    model_family = "tree_boost_surface_v5"
    if bool(cfg.tree_enable_global_model):
        model_family = f"{model_family}_global"
    if target_space != "iv":
        model_family = f"{model_family}_{target_space}"
    if bool(cfg.tree_enable_uncertainty_shrinkage):
        model_family = f"{model_family}_uncert"
    if bool(cfg.tree_enable_time_decay_weight):
        model_family = f"{model_family}_timedecay"
    if bool(cfg.tree_postprocess_noarb):
        model_family = f"{model_family}_noarb"

    run_dir = make_run_dir(cfg.out_dir, prefix="run")

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(run_dir / "train_history.csv", index=False)

    latent_df = pd.DataFrame(latent, columns=[f"z_{i}" for i in range(latent.shape[1])])
    latent_df.insert(0, "date", dates)
    latent_df.insert(1, "asset_id", asset_ids.astype(np.int32))
    latent_df.insert(
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
    latent_df.to_parquet(run_dir / "latent_states.parquet", index=False)

    artifact = {
        "model_type": model_family,
        "dataset_path": str(dataset_path.resolve()),
        "surface_variable": str(surface_variable),
        "tree_target_space": str(target_space),
        "tree_enable_global_model": bool(cfg.tree_enable_global_model),
        "tree_global_use_asset_onehot": bool(cfg.tree_global_use_asset_onehot),
        "tree_enable_uncertainty_shrinkage": bool(cfg.tree_enable_uncertainty_shrinkage),
        "tree_enable_time_decay_weight": bool(cfg.tree_enable_time_decay_weight),
        "tree_postprocess_noarb": bool(cfg.tree_postprocess_noarb),
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
        "tree_enable_tenor_calibration": bool(tenor_calibration_enabled),
        "tree_tenor_calibration_shrinkage": float(cfg.tree_tenor_calibration_shrinkage),
        "tree_tenor_slope_min": float(cfg.tree_tenor_slope_min),
        "tree_tenor_slope_max": float(cfg.tree_tenor_slope_max),
        "tree_enable_residual_corrector": bool(residual_enabled),
        "tree_residual_factors": int(residual_factors),
        "tree_enable_time_decay_weight": bool(cfg.tree_enable_time_decay_weight),
        "tree_time_decay_halflife_pairs": float(cfg.tree_time_decay_halflife_pairs),
        "tree_time_decay_min_weight": float(cfg.tree_time_decay_min_weight),
        "tree_postprocess_noarb_strength": float(cfg.tree_postprocess_noarb_strength),
        "tree_postprocess_noarb_calendar": bool(cfg.tree_postprocess_noarb_calendar),
        "tree_postprocess_noarb_convex": bool(cfg.tree_postprocess_noarb_convex),
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
    cfg_payload["model_family"] = model_family
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
        "model_family": model_family,
        "surface_variable": str(surface_variable),
        "tree_target_space": str(target_space),
        "tree_enable_global_model": bool(cfg.tree_enable_global_model),
        "tree_global_use_asset_onehot": bool(cfg.tree_global_use_asset_onehot),
        "tree_enable_uncertainty_shrinkage": bool(cfg.tree_enable_uncertainty_shrinkage),
        "tree_enable_time_decay_weight": bool(cfg.tree_enable_time_decay_weight),
        "tree_postprocess_noarb": bool(cfg.tree_postprocess_noarb),
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
        "tree_enable_tenor_calibration": bool(tenor_calibration_enabled),
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
