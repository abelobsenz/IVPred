"""PyTorch training pipeline for IV dynamics architecture."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for training. Install torch first.") from exc

from ivdyn.model import IVDynamicsTorchModel, ModelBundle, ModelConfig, device_auto, to_numpy
from ivdyn.model.scalers import ArrayScaler
from ivdyn.utils import make_run_dir


@dataclass(slots=True)
class TrainingConfig:
    out_dir: Path
    seed: int = 7
    train_frac: float = 0.70
    val_frac: float = 0.15
    split_mode: str = "by_asset_time"

    latent_dim: int = 32
    vae_hidden: tuple[int, int] = (384, 192)
    dynamics_hidden: tuple[int, ...] = (256, 128)
    pricing_hidden: tuple[int, int] = (256, 128)
    execution_hidden: tuple[int, int] = (192, 96)
    model_dropout: float = 0.08

    vae_epochs: int = 150
    vae_batch_size: int = 32
    vae_lr: float = 2e-3
    vae_kl_beta: float = 0.02
    kl_warmup_epochs: int = 20
    noarb_lambda: float = 0.01
    noarb_butterfly_lambda: float = 0.005
    recon_huber_beta: float = 0.015

    head_epochs: int = 150
    dyn_batch_size: int = 64
    contract_batch_size: int = 2048
    head_lr: float = 1e-3
    rollout_steps: int = 3
    rollout_surface_lambda: float = 0.5
    rollout_calendar_lambda: float = 0.03
    rollout_butterfly_lambda: float = 0.02
    rollout_random_horizon: bool = True
    rollout_min_steps: int = 1
    rollout_teacher_forcing_start: float = 0.35
    rollout_teacher_forcing_end: float = 0.10
    rollout_surface_huber_beta: float = 0.02
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

    joint_epochs: int = 150
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
    surface_dynamics_only: bool = False
    context_winsor_quantile: float = 0.01
    context_z_clip: float = 5.0
    context_augment_from_contracts: bool = True
    dynamics_residual: bool = True
    asset_embed_dim: int = 8
    early_stop_patience: int = 20
    early_stop_min_delta: float = 1e-4
    lr_plateau_patience: int = 6
    lr_plateau_factor: float = 0.5
    min_lr: float = 1e-6


def _load_dataset(dataset_path: Path) -> dict[str, np.ndarray]:
    z = np.load(dataset_path, allow_pickle=True)
    out: dict[str, np.ndarray] = {}
    for k in z.files:
        arr = z[k]
        if arr.dtype == object:
            arr = arr.astype(str)
        out[k] = arr
    return out


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


def _rollout_windows_by_asset(
    indices: np.ndarray,
    asset_ids: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Return rollout windows [n_windows, steps+1] in global index space.

    This works for both asset-major and date-major datasets by building
    contiguous windows inside each asset-specific timeline.
    """
    if len(indices) == 0:
        return np.empty((0, max(int(steps), 1) + 1), dtype=np.int64)

    steps = max(int(steps), 1)
    idx = np.asarray(indices, dtype=np.int64)
    windows: list[np.ndarray] = []
    for aid in np.unique(asset_ids[idx].astype(np.int32)):
        seq = idx[asset_ids[idx] == aid]
        if len(seq) <= steps:
            continue
        seq = np.sort(seq)
        for pos in range(len(seq) - steps):
            windows.append(seq[pos : pos + steps + 1])
    if not windows:
        return np.empty((0, steps + 1), dtype=np.int64)
    return np.stack(windows, axis=0).astype(np.int64)


def _contract_splits(contract_date_idx: np.ndarray, tr: np.ndarray, va: np.ndarray, te: np.ndarray):
    idx = np.arange(len(contract_date_idx))
    c_tr = idx[np.isin(contract_date_idx, tr)]
    c_va = idx[np.isin(contract_date_idx, va)]
    c_te = idx[np.isin(contract_date_idx, te)]
    return c_tr, c_va, c_te


def _recon_raw_from_scaled(
    recon_scaled: torch.Tensor,
    surface_scaler: ArrayScaler,
    nx: int,
    nt: int,
) -> torch.Tensor:
    mean = torch.as_tensor(surface_scaler.mean, dtype=recon_scaled.dtype, device=recon_scaled.device)
    std = torch.as_tensor(surface_scaler.std, dtype=recon_scaled.dtype, device=recon_scaled.device)
    return (recon_scaled * std + mean).view(-1, nx, nt)


def _surface_variable_name(v: str | None) -> str:
    if not v:
        return "iv"
    s = str(v).strip().lower()
    if s in {"total_variance", "w", "total_var"}:
        return "total_variance"
    return "iv"


def _surface_to_total_variance_torch(
    surface_raw: torch.Tensor,
    tenor_days: np.ndarray,
    surface_variable: str,
) -> torch.Tensor:
    tau = torch.as_tensor(
        tenor_days.astype(np.float32) / 365.0,
        dtype=surface_raw.dtype,
        device=surface_raw.device,
    ).view(1, 1, -1)
    if _surface_variable_name(surface_variable) == "total_variance":
        # Smooth positivity transform with scale keeps values near identity
        # in the typical total-variance range while avoiding clamp dead-zones.
        softplus_scale = 0.02
        return F.softplus(surface_raw / softplus_scale) * softplus_scale + 1e-8
    vol = torch.clamp(surface_raw, min=1e-4, max=4.0)
    return vol.pow(2) * torch.clamp(tau, min=1e-6)


def _surface_to_iv_torch(
    surface_raw: torch.Tensor,
    tenor_days: np.ndarray,
    surface_variable: str,
) -> torch.Tensor:
    if _surface_variable_name(surface_variable) != "total_variance":
        return torch.clamp(surface_raw, min=1e-4, max=4.0)
    tau = torch.as_tensor(
        tenor_days.astype(np.float32) / 365.0,
        dtype=surface_raw.dtype,
        device=surface_raw.device,
    ).view(1, 1, -1)
    w = _surface_to_total_variance_torch(surface_raw, tenor_days=tenor_days, surface_variable=surface_variable)
    return torch.sqrt(torch.clamp(w / torch.clamp(tau, min=1e-6), min=1e-8))


def _surface_to_iv_numpy(
    surface_raw: np.ndarray,
    tenor_days: np.ndarray,
    surface_variable: str,
) -> np.ndarray:
    if _surface_variable_name(surface_variable) != "total_variance":
        return np.clip(surface_raw, 1e-4, 4.0).astype(np.float32)
    tau = (tenor_days.astype(np.float32) / 365.0).reshape(1, 1, -1)
    w = np.clip(surface_raw, 1e-8, None)
    iv = np.sqrt(np.clip(w / np.clip(tau, 1e-6, None), 1e-8, None))
    return iv.astype(np.float32)


def _calendar_penalty_torch(
    recon_scaled: torch.Tensor,
    surface_scaler: ArrayScaler,
    nx: int,
    nt: int,
    tenor_days: np.ndarray,
    surface_variable: str,
) -> torch.Tensor:
    recon_raw = _recon_raw_from_scaled(recon_scaled, surface_scaler, nx, nt)
    return _calendar_penalty_from_raw(recon_raw, tenor_days, surface_variable=surface_variable)


def _calendar_penalty_from_raw(
    recon_raw: torch.Tensor,
    tenor_days: np.ndarray,
    *,
    surface_variable: str,
) -> torch.Tensor:
    total_var = _surface_to_total_variance_torch(
        recon_raw,
        tenor_days=tenor_days,
        surface_variable=surface_variable,
    )
    viol = torch.relu(total_var[:, :, :-1] - total_var[:, :, 1:])
    return viol.pow(2).mean()


def _norm_cdf_torch(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


def _butterfly_penalty_torch(
    recon_scaled: torch.Tensor,
    surface_scaler: ArrayScaler,
    nx: int,
    nt: int,
    x_grid: np.ndarray,
    tenor_days: np.ndarray,
    surface_variable: str,
) -> torch.Tensor:
    recon_raw = _recon_raw_from_scaled(recon_scaled, surface_scaler, nx, nt)
    return _butterfly_penalty_from_raw(
        recon_raw,
        x_grid=x_grid,
        tenor_days=tenor_days,
        surface_variable=surface_variable,
    )


def _butterfly_penalty_from_raw(
    recon_raw: torch.Tensor,
    *,
    x_grid: np.ndarray,
    tenor_days: np.ndarray,
    surface_variable: str,
) -> torch.Tensor:
    # Forward-normalized proxy with k=K/F=exp(x). Penalize negative discrete
    # second differences in call prices across strikes.
    dtype = recon_raw.dtype
    device = recon_raw.device

    tau = torch.as_tensor(
        np.asarray(tenor_days, dtype=np.float32) / 365.0,
        dtype=dtype,
        device=device,
    ).clamp_min(1e-5)
    tau = tau.view(1, 1, -1)
    sqrt_tau = torch.sqrt(tau)

    k = torch.exp(
        torch.as_tensor(np.asarray(x_grid, dtype=np.float32), dtype=dtype, device=device)
    ).clamp_min(1e-6)
    k = k.view(1, -1, 1)
    ln_sk = -torch.log(k)

    vol = _surface_to_iv_torch(
        recon_raw,
        tenor_days=tenor_days,
        surface_variable=surface_variable,
    ).clamp(min=1e-4, max=4.0)
    vol_sqrt_tau = (vol * sqrt_tau).clamp_min(1e-8)
    d1 = (ln_sk + 0.5 * vol.pow(2) * tau) / vol_sqrt_tau
    d2 = d1 - vol_sqrt_tau
    call = _norm_cdf_torch(d1) - k * _norm_cdf_torch(d2)

    second = call[:, :-2, :] - 2.0 * call[:, 1:-1, :] + call[:, 2:, :]
    viol = torch.relu(-second)
    return viol.pow(2).mean()


def _weighted_surface_huber_loss(
    pred_scaled: torch.Tensor,
    target_scaled: torch.Tensor,
    *,
    point_weight: torch.Tensor | None = None,
    beta: float = 0.02,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred_scaled, target_scaled, beta=max(float(beta), 1e-4), reduction="none")
    if point_weight is not None and point_weight.shape != pred_scaled.shape:
        raise RuntimeError(
            f"Surface point weight shape mismatch: weight={tuple(point_weight.shape)} pred={tuple(pred_scaled.shape)}"
        )
    if point_weight is not None:
        loss = loss * point_weight
    return loss.mean()


def _surface_recon_loss(
    pred_scaled: torch.Tensor,
    target_scaled: torch.Tensor,
    *,
    point_weight: torch.Tensor | None = None,
    beta: float = 0.015,
) -> torch.Tensor:
    return _weighted_surface_huber_loss(
        pred_scaled,
        target_scaled,
        point_weight=point_weight,
        beta=beta,
    )


def _rollout_losses_torch(
    *,
    model: IVDynamicsTorchModel,
    z_init: torch.Tensor,
    z_target: torch.Tensor,
    context_scaled: torch.Tensor,
    surface_scaled_target: torch.Tensor,
    rollout_windows: np.ndarray,
    steps: int,
    surface_scaler: ArrayScaler,
    nx: int,
    nt: int,
    tenor_days: np.ndarray,
    x_grid: np.ndarray,
    surface_variable: str,
    surface_point_weight: torch.Tensor | None = None,
    rollout_surface_huber_beta: float = 0.02,
    teacher_forcing_prob: float = 0.0,
    asset_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    zero = torch.tensor(0.0, dtype=z_init.dtype, device=z_init.device)
    if len(rollout_windows) == 0:
        return zero, zero, zero, zero

    steps = max(int(steps), 1)
    windows = np.asarray(rollout_windows, dtype=np.int64)
    if windows.ndim != 2 or windows.shape[1] != steps + 1:
        raise RuntimeError(
            f"Rollout windows shape mismatch: got {windows.shape}, expected [n, {steps + 1}]"
        )
    start_t = torch.as_tensor(windows[:, 0], dtype=torch.long, device=z_init.device)
    z_cur = z_init.index_select(0, start_t)

    dyn_losses: list[torch.Tensor] = []
    surf_losses: list[torch.Tensor] = []
    cal_losses: list[torch.Tensor] = []
    bfly_losses: list[torch.Tensor] = []
    tf_prob = float(np.clip(float(teacher_forcing_prob), 0.0, 1.0))

    for step in range(steps):
        idx_now = torch.as_tensor(windows[:, step], dtype=torch.long, device=z_init.device)
        idx_next = torch.as_tensor(windows[:, step + 1], dtype=torch.long, device=z_init.device)
        aid_now = asset_ids.index_select(0, idx_now) if asset_ids is not None else None

        pred_next = model.forward_dynamics(
            z_cur,
            context_scaled.index_select(0, idx_now),
            asset_id=aid_now,
        )
        target_next = z_target.index_select(0, idx_next)
        dyn_losses.append(F.mse_loss(pred_next, target_next))

        surf_pred = model.decode(pred_next)
        surf_true = surface_scaled_target.index_select(0, idx_next)
        point_w = surface_point_weight.index_select(0, idx_next) if surface_point_weight is not None else None
        surf_losses.append(
            _weighted_surface_huber_loss(
                surf_pred,
                surf_true,
                point_weight=point_w,
                beta=rollout_surface_huber_beta,
            )
        )

        surf_pred_raw = _recon_raw_from_scaled(surf_pred, surface_scaler, nx, nt)
        cal_losses.append(
            _calendar_penalty_from_raw(
                surf_pred_raw,
                tenor_days,
                surface_variable=surface_variable,
            )
        )
        bfly_losses.append(
            _butterfly_penalty_from_raw(
                surf_pred_raw,
                x_grid=x_grid,
                tenor_days=tenor_days,
                surface_variable=surface_variable,
            )
        )

        if tf_prob > 0.0 and step < steps - 1:
            # Scheduled sampling: mix teacher-forced latent targets with
            # model roll-forward states to stabilize longer-horizon training.
            tf_mask = (
                torch.rand((pred_next.shape[0], 1), device=pred_next.device, dtype=pred_next.dtype)
                < tf_prob
            ).to(pred_next.dtype)
            z_cur = tf_mask * target_next + (1.0 - tf_mask) * pred_next
        else:
            z_cur = pred_next

    return (
        torch.stack(dyn_losses).mean() if dyn_losses else zero,
        torch.stack(surf_losses).mean() if surf_losses else zero,
        torch.stack(cal_losses).mean() if cal_losses else zero,
        torch.stack(bfly_losses).mean() if bfly_losses else zero,
    )


def _kl_beta_for_epoch(base_beta: float, epoch: int, warmup_epochs: int) -> float:
    w = max(int(warmup_epochs), 0)
    if w == 0:
        return float(base_beta)
    frac = min(1.0, max(float(epoch), 0.0) / float(w))
    return float(base_beta) * frac


def _linear_schedule_value(
    start: float,
    end: float,
    epoch: int,
    total_epochs: int,
) -> float:
    if total_epochs <= 1:
        return float(end)
    t = float(np.clip((int(epoch) - 1) / max(int(total_epochs) - 1, 1), 0.0, 1.0))
    return float(start) + (float(end) - float(start)) * t


def _sample_rollout_steps(
    *,
    rng: np.random.Generator,
    max_steps: int,
    min_steps: int,
    random_horizon: bool,
) -> int:
    max_s = max(int(max_steps), 1)
    min_s = min(max_s, max(int(min_steps), 1))
    if (not bool(random_horizon)) or max_s <= min_s:
        return max_s
    return int(rng.integers(min_s, max_s + 1))


def _iter_batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    if len(indices) == 0:
        return []
    order = rng.permutation(indices)
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _eval_recon(
    model: IVDynamicsTorchModel,
    surface_scaled: torch.Tensor,
    idx: np.ndarray,
) -> float:
    if len(idx) == 0:
        return float("nan")
    with torch.no_grad():
        mu, _ = model.encode(surface_scaled[idx])
        recon = model.decode(mu)
        return float(F.mse_loss(recon, surface_scaled[idx]).item())


def _build_surface_point_weights(
    *,
    liq_surface: np.ndarray | None,
    spread_surface: np.ndarray | None,
    vega_surface: np.ndarray | None,
    liq_alpha: float,
    spread_alpha: float,
    vega_alpha: float,
    clip_min: float,
    clip_max: float,
    x_grid: np.ndarray | None = None,
    tenor_days: np.ndarray | None = None,
    focus_alpha: float = 0.0,
    focus_x_min: float = 0.10,
    focus_x_scale: float = 0.03,
    focus_dte_scale_days: float = 21.0,
    focus_dte_max_days: float = 30.0,
    focus_neg_x_max: float = -0.20,
    focus_neg_weight_ratio: float = 0.0,
    asset_ids: np.ndarray | None = None,
    focus_density_by_asset: np.ndarray | None = None,
    focus_density_alpha: float = 0.0,
    train_date_mask: np.ndarray | None = None,
    n_dates_hint: int | None = None,
) -> np.ndarray:
    n_dates = 0
    for arr in (liq_surface, spread_surface, vega_surface):
        if arr is not None:
            n_dates = int(arr.shape[0])
            break

    ref_shape = None
    for arr in (liq_surface, spread_surface, vega_surface):
        if arr is not None:
            ref_shape = tuple(arr.shape)
            break
    if ref_shape is None:
        if n_dates_hint is None or x_grid is None or tenor_days is None:
            return np.ones((0, 0), dtype=np.float32)
        nx = int(len(np.asarray(x_grid).reshape(-1)))
        nt = int(len(np.asarray(tenor_days).reshape(-1)))
        n_dates = int(max(int(n_dates_hint), 0))
    else:
        n_dates, nx, nt = ref_shape
    if n_dates <= 0:
        return np.ones((0, 0), dtype=np.float32)
    w = np.ones((n_dates, nx, nt), dtype=np.float64)

    if liq_surface is not None and float(liq_alpha) > 0.0:
        liq = np.asarray(liq_surface, dtype=np.float64)
        med = np.nanmedian(liq[np.isfinite(liq)]) if np.isfinite(liq).any() else 1.0
        med = max(float(med), 1e-6)
        liq_norm = np.clip(liq / med, 0.25, 16.0)
        w *= np.power(liq_norm, max(float(liq_alpha), 0.0))

    if spread_surface is not None and float(spread_alpha) > 0.0:
        sp = np.abs(np.asarray(spread_surface, dtype=np.float64))
        inv_sp = 1.0 / np.clip(sp, 0.01, 5.0)
        w *= np.power(inv_sp, max(float(spread_alpha), 0.0))

    if vega_surface is not None and float(vega_alpha) > 0.0:
        vg = np.abs(np.asarray(vega_surface, dtype=np.float64))
        med = np.nanmedian(vg[np.isfinite(vg)]) if np.isfinite(vg).any() else 1.0
        med = max(float(med), 1e-6)
        vg_norm = np.clip(vg / med, 0.25, 16.0)
        w *= np.power(vg_norm, max(float(vega_alpha), 0.0))

    # Optional focal weighting to emphasize high positive moneyness and short tenor.
    # This helps prioritize nonlinear front-wing structure where errors can concentrate.
    if (
        float(focus_alpha) > 0.0
        and x_grid is not None
        and tenor_days is not None
        and len(np.asarray(x_grid).reshape(-1)) == nx
        and len(np.asarray(tenor_days).reshape(-1)) == nt
    ):
        x = np.asarray(x_grid, dtype=np.float64).reshape(1, nx, 1)
        dte = np.asarray(tenor_days, dtype=np.float64).reshape(1, 1, nt)
        x_scale = max(float(focus_x_scale), 1e-4)
        dte_scale = max(float(focus_dte_scale_days), 1e-3)
        logits = np.clip((x - float(focus_x_min)) / x_scale, -30.0, 30.0)
        x_focus_pos = 1.0 / (1.0 + np.exp(-logits))
        neg_ratio = max(float(focus_neg_weight_ratio), 0.0)
        x_focus_neg = np.zeros_like(x_focus_pos, dtype=np.float64)
        if neg_ratio > 0.0:
            neg_logits = np.clip((float(focus_neg_x_max) - x) / x_scale, -30.0, 30.0)
            x_focus_neg = 1.0 / (1.0 + np.exp(-neg_logits))
        dte_focus = np.exp(-dte / dte_scale)
        dte_max = float(focus_dte_max_days)
        if np.isfinite(dte_max) and dte_max > 0.0:
            dte_gate = (dte < dte_max).astype(np.float64)
        else:
            dte_gate = np.ones_like(dte, dtype=np.float64)
        region_focus = (x_focus_pos + neg_ratio * x_focus_neg) * dte_focus * dte_gate
        w *= 1.0 + max(float(focus_alpha), 0.0) * region_focus

    density_alpha = max(float(focus_density_alpha), 0.0)
    if density_alpha > 0.0:
        if focus_density_by_asset is None:
            raise RuntimeError(
                "surface_focus_density_alpha > 0 requires focus_density_by_asset. "
                "Provide a valid --surface-focus-density-map with per-asset density grids."
            )
        if asset_ids is None:
            raise RuntimeError(
                "surface_focus_density_alpha > 0 requires asset_ids to map each date to an asset."
            )
        aid = np.asarray(asset_ids, dtype=np.int32).reshape(-1)
        if len(aid) != n_dates:
            raise RuntimeError(
                f"asset_ids length mismatch for focus density: got {len(aid)} expected {n_dates}."
            )
        density = np.asarray(focus_density_by_asset, dtype=np.float64)
        if density.ndim != 3:
            raise RuntimeError(
                f"Focus density map must be rank-3 [n_assets, nx, nt], got shape={density.shape}."
            )
        if density.shape[1] != nx or density.shape[2] != nt:
            raise RuntimeError(
                "Focus density map shape mismatch: "
                f"map grid=({density.shape[1]}, {density.shape[2]}) dataset grid=({nx}, {nt})."
            )
        if np.min(aid) < 0 or np.max(aid) >= density.shape[0]:
            raise RuntimeError(
                "asset_ids contain values outside focus density map asset dimension: "
                f"asset_id range=[{int(np.min(aid))}, {int(np.max(aid))}] map_assets={density.shape[0]}."
            )
        density_by_date = density[aid]
        if train_date_mask is not None:
            tmask = np.asarray(train_date_mask, dtype=bool).reshape(-1)
            if len(tmask) != n_dates:
                raise RuntimeError(
                    f"train_date_mask length mismatch for focus density: got {len(tmask)} expected {n_dates}."
                )
            density_by_date = np.where(tmask[:, None, None], density_by_date, 1.0)

        # Exponential reweighting around neutral density=1 keeps factors strictly
        # positive while allowing smooth asset-specific emphasis/de-emphasis.
        w *= np.exp(density_alpha * (density_by_date - 1.0))

    w = np.where(np.isfinite(w), w, 1.0)
    w = np.clip(
        w,
        max(float(clip_min), 1e-3),
        max(float(clip_max), max(float(clip_min), 1e-3)),
    )
    return w.reshape(n_dates, nx * nt).astype(np.float32)


def _contract_risk_focus_weights(
    *,
    features: np.ndarray,
    feature_names: list[str],
    price_risk_weight: float,
    exec_risk_weight: float,
    price_spread_inv_lambda: float,
    price_spread_clip_min: float,
    price_spread_clip_max: float,
    price_vega_power: float,
    price_vega_cap: float,
    risk_focus_abs_x: float,
    risk_focus_tau_days: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(features)
    price_w = np.ones(n, dtype=np.float32)
    exec_w = np.ones(n, dtype=np.float32)

    req = {"x", "tau", "cp_sign"}
    if not req.issubset(set(feature_names)):
        return price_w, exec_w

    ix_x = feature_names.index("x")
    ix_tau = feature_names.index("tau")
    ix_cp = feature_names.index("cp_sign")

    abs_x = np.abs(features[:, ix_x].astype(np.float64))
    tau = np.clip(features[:, ix_tau].astype(np.float64), 1e-6, None)
    cp_sign = features[:, ix_cp].astype(np.float64)

    x_scale = max(float(risk_focus_abs_x), 1e-4)
    tau_scale = max(float(risk_focus_tau_days) / 365.0, 1e-6)

    near_atm = np.exp(-np.square(abs_x / x_scale))
    short_tenor = np.exp(-np.square(tau / tau_scale))
    is_put = (cp_sign < 0.0).astype(np.float64)
    focus = is_put * near_atm * short_tenor

    price_w += np.clip(float(price_risk_weight), 0.0, None) * focus
    exec_w += np.clip(float(exec_risk_weight), 0.0, None) * focus

    if "rel_spread" in feature_names:
        ix_rs = feature_names.index("rel_spread")
        rel_spread = np.abs(features[:, ix_rs].astype(np.float64))
        rs_min = max(float(price_spread_clip_min), 1e-4)
        rs_max = max(float(price_spread_clip_max), rs_min)
        inv_spread = 1.0 / np.clip(rel_spread, rs_min, rs_max)
        spread_lambda = max(float(price_spread_inv_lambda), 0.0)
        if spread_lambda > 0.0:
            price_w *= np.clip(np.power(inv_spread, spread_lambda), 0.5, 10.0)

    if "vega" in feature_names:
        ix_v = feature_names.index("vega")
        vega = np.abs(features[:, ix_v].astype(np.float64))
        med = float(np.nanmedian(vega[np.isfinite(vega)])) if np.isfinite(vega).any() else 1.0
        med = max(med, 1e-6)
        vega_norm = np.clip(vega / med, 0.25, max(float(price_vega_cap), 0.25))
        vega_power = max(float(price_vega_power), 0.0)
        if vega_power > 0.0:
            price_w *= np.power(vega_norm, vega_power)

    return price_w.astype(np.float32), exec_w.astype(np.float32)


def _weighted_smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    beta: float = 0.02,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def _smooth_binary_targets(target: torch.Tensor, smoothing: float) -> torch.Tensor:
    s = float(np.clip(smoothing, 0.0, 0.25))
    if s <= 0.0:
        return target
    return target * (1.0 - s) + 0.5 * s


def _weighted_bce_with_logits(
    pred_logit: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(pred_logit, target, reduction="none")
    if weight is not None:
        loss = loss * weight
    return loss.mean()


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
    power = float(error_power)
    if not np.isfinite(power) or power <= 0.0:
        raise RuntimeError(f"adaptive focus error power must be > 0, got {error_power}.")

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
    if "x_grid" not in ds or "tenor_days" not in ds:
        raise RuntimeError("Dataset must include x_grid and tenor_days to build focus density map.")
    x_grid = np.asarray(ds["x_grid"], dtype=np.float32).reshape(-1)
    tenor_days = np.asarray(ds["tenor_days"], dtype=np.int32).reshape(-1)

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

    context = ds["context"].astype(np.float32)
    contract_features = ds["contract_features"].astype(np.float32)
    contract_feature_names = ds.get("contract_feature_names", np.array([], dtype=str)).astype(str).tolist()
    contract_date_idx = ds["contract_date_index"].astype(np.int32)

    dev = torch.device(device) if device else device_auto()
    bundle = ModelBundle.load(run_dir / "model.pt", device=dev)
    model = bundle.model.to(dev).eval()

    expected_context_dim = int(bundle.context_scaler.mean.shape[1])
    if int(context.shape[1]) != expected_context_dim:
        if not bool(cfg.context_augment_from_contracts):
            raise RuntimeError(
                "Context dimension mismatch and context augmentation disabled: "
                f"dataset={context.shape[1]} expected={expected_context_dim}."
            )
        context_aug, _ = _augment_context_with_contract_intraday(
            context=context,
            features=contract_features,
            feature_names=contract_feature_names,
            date_idx=contract_date_idx,
            n_dates=n_dates,
        )
        if int(context_aug.shape[1]) != expected_context_dim:
            raise RuntimeError(
                "Augmented context dimension mismatch while building focus density map: "
                f"augmented={context_aug.shape[1]} expected={expected_context_dim}."
            )
        context = context_aug

    surface_flat = surface_raw_obs.reshape(n_dates, -1)
    surface_scaled = bundle.surface_scaler.transform(surface_flat)
    context_scaled = bundle.context_scaler.transform(context)

    with torch.inference_mode():
        sf = torch.as_tensor(surface_scaled, dtype=torch.float32, device=dev)
        mu, _ = model.encode(sf)
        ctx_t = torch.as_tensor(context_scaled, dtype=torch.float32, device=dev)
        aid_t = torch.as_tensor(asset_ids, dtype=torch.long, device=dev)
        z_next = model.forward_dynamics(mu, ctx_t, asset_id=aid_t)
        forecast_scaled = model.decode(z_next)
        forecast_raw = bundle.surface_scaler.inverse_transform(to_numpy(forecast_scaled)).reshape(surface_raw_obs.shape)

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
        "x_grid_len": int(len(x_grid)),
        "tenor_days_len": int(len(tenor_days)),
        "map_path": str(out_path),
        "trainval_only": True,
    }
    meta_path = out_path.with_name(f"{out_path.stem}_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def train(dataset_path: Path, cfg: TrainingConfig) -> Path:
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    ds = _load_dataset(dataset_path)

    dates = ds["dates"].astype(str)
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
    surface_raw = ds[surface_key].astype(np.float32)
    liq_surface = ds.get("liq_surface")
    spread_surface = ds.get("spread_surface")
    vega_surface = ds.get("vega_surface")
    liq_surface = liq_surface.astype(np.float32) if isinstance(liq_surface, np.ndarray) else None
    spread_surface = spread_surface.astype(np.float32) if isinstance(spread_surface, np.ndarray) else None
    vega_surface = vega_surface.astype(np.float32) if isinstance(vega_surface, np.ndarray) else None
    context = ds["context"].astype(np.float32)
    context_names = ds.get("context_names", np.array([], dtype=str)).astype(str).tolist()
    features = ds["contract_features"].astype(np.float32)
    feature_names = ds.get("contract_feature_names", np.array([], dtype=str)).astype(str).tolist()
    price_target = ds["contract_price_target"].astype(np.float32)
    fill_target = ds["contract_fill_target"].astype(np.float32)
    date_idx = ds["contract_date_index"].astype(np.int32)
    asset_ids = ds.get("asset_ids", np.zeros(len(dates), dtype=np.int32)).astype(np.int32)
    contract_asset_id = ds.get("contract_asset_id", asset_ids[date_idx]).astype(np.int32)
    asset_names = ds.get("asset_names", np.array(["ASSET"], dtype=str)).astype(str).tolist()
    n_assets = max(1, int(np.max(asset_ids)) + 1) if len(asset_ids) > 0 else 1
    tenor_days = ds["tenor_days"].astype(np.int32)
    x_grid_raw = ds.get("x_grid")

    if len(asset_ids) != len(dates):
        raise RuntimeError(f"asset_ids length {len(asset_ids)} must match dates length {len(dates)}.")

    n_dates, nx, nt = surface_raw.shape
    surface_flat = surface_raw.reshape(n_dates, nx * nt)
    if x_grid_raw is None:
        if (
            float(cfg.noarb_butterfly_lambda) > 0.0
            or float(cfg.rollout_butterfly_lambda) > 0.0
        ):
            raise RuntimeError(
                "Dataset is missing x_grid required for butterfly penalties. "
                "Rebuild dataset.npz with current build-dataset command."
            )
        x_grid = np.linspace(-0.30, 0.30, num=nx, dtype=np.float32)
    else:
        x_grid = np.asarray(x_grid_raw, dtype=np.float32).reshape(-1)
    if len(x_grid) != nx:
        raise RuntimeError(f"x_grid length {len(x_grid)} does not match surface x dimension {nx}.")

    kl_warmup_epochs = max(int(cfg.kl_warmup_epochs), 0)
    rollout_steps = max(int(cfg.rollout_steps), 1)
    rollout_min_steps = min(rollout_steps, max(int(cfg.rollout_min_steps), 1))
    rollout_random_horizon = bool(cfg.rollout_random_horizon) and rollout_steps > rollout_min_steps
    rollout_teacher_forcing_start = float(np.clip(float(cfg.rollout_teacher_forcing_start), 0.0, 1.0))
    rollout_teacher_forcing_end = float(np.clip(float(cfg.rollout_teacher_forcing_end), 0.0, 1.0))
    rollout_surface_lambda = max(float(cfg.rollout_surface_lambda), 0.0)
    rollout_calendar_lambda = max(float(cfg.rollout_calendar_lambda), 0.0)
    rollout_butterfly_lambda = max(float(cfg.rollout_butterfly_lambda), 0.0)
    rollout_surface_huber_beta = max(float(cfg.rollout_surface_huber_beta), 1e-4)
    recon_huber_beta = max(float(cfg.recon_huber_beta), 1e-4)
    noarb_lambda = max(float(cfg.noarb_lambda), 0.0)
    noarb_butterfly_lambda = max(float(cfg.noarb_butterfly_lambda), 0.0)
    focus_alpha = max(float(cfg.surface_focus_alpha), 0.0)
    clip_max = float(cfg.surface_weight_clip_max)
    if focus_alpha > 0.0 and clip_max <= 1.0:
        raise RuntimeError(
            "surface_focus_alpha > 0 but surface_weight_clip_max <= 1.0, "
            "so focus weights are clipped to 1 and have no effect. "
            "Set --surface-weight-clip-max > 1.0 (e.g., 4.0)."
        )
    early_stop_patience = max(int(cfg.early_stop_patience), 0)
    early_stop_min_delta = max(float(cfg.early_stop_min_delta), 0.0)
    lr_plateau_patience = max(int(cfg.lr_plateau_patience), 1)
    lr_plateau_factor = float(np.clip(float(cfg.lr_plateau_factor), 0.1, 0.99))
    min_lr = max(float(cfg.min_lr), 1e-8)

    original_context_dim = int(context.shape[1])
    added_context_names: list[str] = []
    if bool(cfg.context_augment_from_contracts):
        context, added_context_names = _augment_context_with_contract_intraday(
            context=context,
            features=features,
            feature_names=feature_names,
            date_idx=date_idx,
            n_dates=n_dates,
        )
        if added_context_names:
            context_names = context_names + added_context_names

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
    train_date_mask = np.zeros(int(n_dates), dtype=bool)
    train_date_mask[tr_dates] = True
    c_tr, c_va, c_te = _contract_splits(date_idx, tr_dates, va_dates, te_dates)
    if len(tr_dates) == 0:
        raise RuntimeError("No training dates after date split.")

    surface_scaler = ArrayScaler.fit(surface_flat[tr_dates])
    context_scaler = ArrayScaler.fit(
        context[tr_dates],
        winsor_quantile=float(max(0.0, cfg.context_winsor_quantile)),
        z_clip=float(cfg.context_z_clip) if float(cfg.context_z_clip) > 0 else None,
    )
    fit_contract_idx = c_tr if len(c_tr) > 0 else np.arange(len(features), dtype=np.int64)
    if len(fit_contract_idx) == 0:
        raise RuntimeError("No contract rows available for training.")
    contract_scaler = ArrayScaler.fit(features[fit_contract_idx])
    price_scaler = ArrayScaler.fit(price_target[fit_contract_idx].reshape(-1, 1))

    surface_scaled_np = surface_scaler.transform(surface_flat)
    context_scaled_np = context_scaler.transform(context)
    feature_scaled_np = contract_scaler.transform(features)
    price_scaled_np = price_scaler.transform(price_target.reshape(-1, 1)).reshape(-1)

    price_weight_np, exec_weight_np = _contract_risk_focus_weights(
        features=features,
        feature_names=feature_names,
        price_risk_weight=cfg.price_risk_weight,
        exec_risk_weight=cfg.exec_risk_weight,
        price_spread_inv_lambda=cfg.price_spread_inv_lambda,
        price_spread_clip_min=cfg.price_spread_clip_min,
        price_spread_clip_max=cfg.price_spread_clip_max,
        price_vega_power=cfg.price_vega_power,
        price_vega_cap=cfg.price_vega_cap,
        risk_focus_abs_x=cfg.risk_focus_abs_x,
        risk_focus_tau_days=cfg.risk_focus_tau_days,
    )
    focus_density_alpha = max(float(cfg.surface_focus_density_alpha), 0.0)
    focus_density_map_path_used: str | None = None
    focus_density_by_asset: np.ndarray | None = None
    if focus_density_alpha > 0.0:
        map_path_raw = cfg.surface_focus_density_map_path
        if map_path_raw is None or str(map_path_raw).strip() == "":
            raise RuntimeError(
                "surface_focus_density_alpha > 0 requires --surface-focus-density-map "
                "with a valid per-asset density JSON file."
            )
        map_path = Path(str(map_path_raw)).expanduser().resolve()
        focus_density_map_path_used = str(map_path)
        focus_density_by_asset = _load_focus_density_by_asset(
            map_path=map_path,
            asset_names=asset_names,
            nx=nx,
            nt=nt,
        )
    surface_point_weight_np = _build_surface_point_weights(
        liq_surface=liq_surface,
        spread_surface=spread_surface,
        vega_surface=vega_surface,
        liq_alpha=float(cfg.surface_weight_liq_alpha),
        spread_alpha=float(cfg.surface_weight_spread_alpha),
        vega_alpha=float(cfg.surface_weight_vega_alpha),
        clip_min=float(cfg.surface_weight_clip_min),
        clip_max=float(cfg.surface_weight_clip_max),
        x_grid=x_grid,
        tenor_days=tenor_days,
        focus_alpha=float(cfg.surface_focus_alpha),
        focus_x_min=float(cfg.surface_focus_x_min),
        focus_x_scale=float(cfg.surface_focus_x_scale),
        focus_dte_scale_days=float(cfg.surface_focus_dte_scale_days),
        focus_dte_max_days=float(cfg.surface_focus_dte_max_days),
        focus_neg_x_max=float(cfg.surface_focus_neg_x_max),
        focus_neg_weight_ratio=float(cfg.surface_focus_neg_weight_ratio),
        asset_ids=asset_ids,
        focus_density_by_asset=focus_density_by_asset,
        focus_density_alpha=focus_density_alpha,
        train_date_mask=train_date_mask,
        n_dates_hint=n_dates,
    )
    if surface_point_weight_np.size == 0:
        surface_point_weight_np = np.ones((n_dates, nx * nt), dtype=np.float32)
    elif surface_point_weight_np.shape[0] != n_dates:
        raise RuntimeError(
            f"Surface weight/date length mismatch: weights={surface_point_weight_np.shape[0]} dates={n_dates}"
        )

    device = device_auto()

    surface_scaled = _to_tensor(surface_scaled_np, device)
    context_scaled = _to_tensor(context_scaled_np, device)
    feature_scaled = _to_tensor(feature_scaled_np, device)
    price_scaled = _to_tensor(price_scaled_np.reshape(-1, 1), device)
    fill_t = _to_tensor(fill_target.reshape(-1, 1), device)
    price_w_t = _to_tensor(price_weight_np.reshape(-1, 1), device)
    exec_w_t = _to_tensor(exec_weight_np.reshape(-1, 1), device)
    surface_w_t = _to_tensor(surface_point_weight_np, device)
    asset_ids_t = torch.as_tensor(asset_ids, dtype=torch.long, device=device)
    contract_asset_t = torch.as_tensor(contract_asset_id, dtype=torch.long, device=device)

    model_cfg = ModelConfig(
        latent_dim=cfg.latent_dim,
        vae_hidden=tuple(int(v) for v in cfg.vae_hidden),
        dynamics_hidden=tuple(int(v) for v in cfg.dynamics_hidden),
        pricing_hidden=tuple(int(v) for v in cfg.pricing_hidden),
        execution_hidden=tuple(int(v) for v in cfg.execution_hidden),
        dropout=float(cfg.model_dropout),
        dynamics_residual=bool(cfg.dynamics_residual),
        n_assets=int(n_assets),
        asset_embed_dim=max(int(cfg.asset_embed_dim), 0),
    )
    model = IVDynamicsTorchModel(
        surface_dim=surface_scaled.shape[1],
        context_dim=context_scaled.shape[1],
        contract_dim=feature_scaled.shape[1],
        config=model_cfg,
    ).to(device)

    hist_rows: list[dict[str, object]] = []

    # -------------------------- Stage 1: VAE pretrain --------------------------
    vae_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    opt_vae = torch.optim.AdamW(vae_params, lr=cfg.vae_lr, weight_decay=cfg.weight_decay)

    for epoch in range(1, cfg.vae_epochs + 1):
        model.train()
        losses = []
        recon_losses = []
        kl_losses = []
        cal_losses = []
        bfly_losses = []
        kl_beta = _kl_beta_for_epoch(cfg.vae_kl_beta, epoch, kl_warmup_epochs)

        for batch in _iter_batches(tr_dates, cfg.vae_batch_size, rng):
            x = surface_scaled[batch]
            mu, logvar = model.encode(x)
            z = model.reparameterize(mu, logvar)
            recon = model.decode(z)

            batch_w = surface_w_t.index_select(0, torch.as_tensor(batch, dtype=torch.long, device=device))
            recon_loss = _surface_recon_loss(recon, x, point_weight=batch_w, beta=recon_huber_beta)
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            cal = _calendar_penalty_torch(
                recon,
                surface_scaler,
                nx,
                nt,
                tenor_days,
                surface_variable=surface_variable,
            )
            bfly = _butterfly_penalty_torch(
                recon,
                surface_scaler,
                nx,
                nt,
                x_grid=x_grid,
                tenor_days=tenor_days,
                surface_variable=surface_variable,
            )
            loss = (
                recon_loss
                + kl_beta * kl
                + noarb_lambda * cal
                + noarb_butterfly_lambda * bfly
            )

            opt_vae.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae_params, 2.5)
            opt_vae.step()

            losses.append(float(loss.item()))
            recon_losses.append(float(recon_loss.item()))
            kl_losses.append(float(kl.item()))
            cal_losses.append(float(cal.item()))
            bfly_losses.append(float(bfly.item()))

        model.eval()
        val_recon = _eval_recon(
            model,
            surface_scaled,
            va_dates,
        )

        hist_rows.append(
            {
                "stage": "vae",
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "recon_loss": float(np.mean(recon_losses)),
                "kl_loss": float(np.mean(kl_losses)),
                "kl_beta": float(kl_beta),
                "calendar_loss": float(np.mean(cal_losses)),
                "butterfly_loss": float(np.mean(bfly_losses)),
                "val_recon_loss": float(val_recon),
            }
        )

    # ---------------------- Stage 2: heads on frozen latent --------------------
    surface_only = bool(cfg.surface_dynamics_only)
    price_head_enabled = (not surface_only) and float(cfg.joint_price_lambda) > 0.0
    exec_head_enabled = (not surface_only) and float(cfg.joint_exec_lambda) > 0.0

    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = False

    head_params = list(model.dynamics.parameters())
    if price_head_enabled:
        head_params += list(model.pricer.parameters())
    if exec_head_enabled:
        head_params += list(model.execution.parameters())
    opt_head = torch.optim.AdamW(head_params, lr=cfg.head_lr, weight_decay=cfg.weight_decay)

    model.eval()
    with torch.no_grad():
        z_all, _ = model.encode(surface_scaled)

    # Dynamics input is aligned as (z_t, ctx_t) -> z_{t+1}.
    # Build rollout windows in asset-local time so date-major datasets are handled.
    dyn_train_windows = _rollout_windows_by_asset(tr_dates, asset_ids, rollout_steps)
    dyn_val_windows_step1 = _rollout_windows_by_asset(va_dates, asset_ids, 1)
    dyn_val_windows_roll = _rollout_windows_by_asset(va_dates, asset_ids, rollout_steps)
    dyn_train_start_idx = dyn_train_windows[:, 0] if len(dyn_train_windows) > 0 else np.array([], dtype=np.int64)
    train_window_row_by_start = {int(w[0]): i for i, w in enumerate(dyn_train_windows)}

    scheduler_head = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_head,
        mode="min",
        factor=lr_plateau_factor,
        patience=lr_plateau_patience,
        min_lr=min_lr,
    )
    best_head_metric = float("inf")
    best_head_epoch = 0
    best_head_state: dict[str, torch.Tensor] | None = None
    head_early_stopped = False
    head_bad_epochs = 0

    for epoch in range(1, cfg.head_epochs + 1):
        model.train()
        head_teacher_forcing_prob = _linear_schedule_value(
            rollout_teacher_forcing_start,
            rollout_teacher_forcing_end,
            epoch,
            cfg.head_epochs,
        )
        dyn_losses = []
        dyn_roll_surface_losses = []
        dyn_roll_cal_losses = []
        dyn_roll_bfly_losses = []
        head_rollout_steps_used: list[int] = []
        price_losses = []
        exec_losses = []

        for batch in _iter_batches(dyn_train_start_idx, cfg.dyn_batch_size, rng):
            row_idx = np.asarray([train_window_row_by_start[int(s)] for s in batch], dtype=np.int64)
            batch_windows = dyn_train_windows[row_idx]
            batch_rollout_steps = _sample_rollout_steps(
                rng=rng,
                max_steps=rollout_steps,
                min_steps=rollout_min_steps,
                random_horizon=rollout_random_horizon,
            )
            head_rollout_steps_used.append(int(batch_rollout_steps))
            batch_windows_eff = batch_windows[:, : batch_rollout_steps + 1]
            dyn_loss, roll_surface_loss, roll_cal_loss, roll_bfly_loss = _rollout_losses_torch(
                model=model,
                z_init=z_all,
                z_target=z_all,
                context_scaled=context_scaled,
                surface_scaled_target=surface_scaled,
                rollout_windows=batch_windows_eff,
                steps=batch_rollout_steps,
                surface_scaler=surface_scaler,
                nx=nx,
                nt=nt,
                tenor_days=tenor_days,
                x_grid=x_grid,
                surface_variable=surface_variable,
                surface_point_weight=surface_w_t,
                rollout_surface_huber_beta=rollout_surface_huber_beta,
                teacher_forcing_prob=head_teacher_forcing_prob,
                asset_ids=asset_ids_t,
            )
            loss = (
                dyn_loss
                + rollout_surface_lambda * roll_surface_loss
                + rollout_calendar_lambda * roll_cal_loss
                + rollout_butterfly_lambda * roll_bfly_loss
            )

            opt_head.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, 2.0)
            opt_head.step()

            dyn_losses.append(float(dyn_loss.item()))
            dyn_roll_surface_losses.append(float(roll_surface_loss.item()))
            dyn_roll_cal_losses.append(float(roll_cal_loss.item()))
            dyn_roll_bfly_losses.append(float(roll_bfly_loss.item()))

        if price_head_enabled or exec_head_enabled:
            for batch in _iter_batches(c_tr, cfg.contract_batch_size, rng):
                z = z_all[date_idx[batch]]
                feat = feature_scaled[batch]
                aid = contract_asset_t[batch]
                p_t = price_scaled[batch]
                e_t = fill_t[batch]
                p_w = price_w_t[batch]
                e_w = exec_w_t[batch]

                p_loss = torch.tensor(0.0, device=device)
                if price_head_enabled:
                    p_pred = model.forward_pricer(z, feat, asset_id=aid)
                    p_loss = _weighted_smooth_l1_loss(p_pred, p_t, p_w, beta=0.02)

                e_loss = torch.tensor(0.0, device=device)
                if exec_head_enabled:
                    e_pred = model.forward_execution_logit(z, feat, asset_id=aid)
                    e_t_s = _smooth_binary_targets(e_t, cfg.exec_label_smoothing)
                    e_loss = _weighted_bce_with_logits(e_pred, e_t_s, e_w) + cfg.exec_logit_l2 * torch.mean(
                        e_pred.pow(2)
                    )

                loss = p_loss + e_loss

                opt_head.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head_params, 2.0)
                opt_head.step()

                if price_head_enabled:
                    price_losses.append(float(p_loss.item()))
                if exec_head_enabled:
                    exec_losses.append(float(e_loss.item()))

        model.eval()
        with torch.no_grad():
            val_dyn = np.nan
            if len(dyn_val_windows_step1) > 0:
                idx_t = torch.as_tensor(dyn_val_windows_step1[:, 0], dtype=torch.long, device=device)
                next_t = torch.as_tensor(dyn_val_windows_step1[:, 1], dtype=torch.long, device=device)
                pred = model.forward_dynamics(
                    z_all.index_select(0, idx_t),
                    context_scaled.index_select(0, idx_t),
                    asset_id=asset_ids_t.index_select(0, idx_t),
                )
                val_dyn = float(F.mse_loss(pred, z_all.index_select(0, next_t)).item())

            val_roll_surface = np.nan
            val_roll_calendar = np.nan
            val_roll_bfly = np.nan
            if len(dyn_val_windows_roll) > 0:
                _, vr_surf, vr_cal, vr_bfly = _rollout_losses_torch(
                    model=model,
                    z_init=z_all,
                    z_target=z_all,
                    context_scaled=context_scaled,
                    surface_scaled_target=surface_scaled,
                    rollout_windows=dyn_val_windows_roll,
                    steps=rollout_steps,
                    surface_scaler=surface_scaler,
                    nx=nx,
                    nt=nt,
                    tenor_days=tenor_days,
                    x_grid=x_grid,
                    surface_variable=surface_variable,
                    surface_point_weight=surface_w_t,
                    rollout_surface_huber_beta=rollout_surface_huber_beta,
                    asset_ids=asset_ids_t,
                )
                val_roll_surface = float(vr_surf.item())
                val_roll_calendar = float(vr_cal.item())
                val_roll_bfly = float(vr_bfly.item())

            val_price = np.nan
            val_exec = np.nan
            if len(c_va) > 0 and (price_head_enabled or exec_head_enabled):
                zv = z_all[date_idx[c_va]]
                fv = feature_scaled[c_va]
                av = contract_asset_t[c_va]
                pv = price_scaled[c_va]
                ev = fill_t[c_va]
                if price_head_enabled:
                    val_price = float(F.mse_loss(model.forward_pricer(zv, fv, asset_id=av), pv).item())
                if exec_head_enabled:
                    val_exec = float(
                        F.binary_cross_entropy_with_logits(
                            model.forward_execution_logit(zv, fv, asset_id=av),
                            ev,
                        ).item()
                    )

        val_forecast_metric = np.nan
        if np.isfinite(val_roll_surface):
            val_forecast_metric = float(val_roll_surface)
            if np.isfinite(val_roll_calendar):
                val_forecast_metric += float(rollout_calendar_lambda) * float(val_roll_calendar)
            if np.isfinite(val_roll_bfly):
                val_forecast_metric += float(rollout_butterfly_lambda) * float(val_roll_bfly)
        elif np.isfinite(val_dyn):
            val_forecast_metric = float(val_dyn)

        if np.isfinite(val_forecast_metric):
            scheduler_head.step(float(val_forecast_metric))
            if float(val_forecast_metric) < (best_head_metric - early_stop_min_delta):
                best_head_metric = float(val_forecast_metric)
                best_head_epoch = int(epoch)
                best_head_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                head_bad_epochs = 0
            else:
                head_bad_epochs += 1
        else:
            head_bad_epochs += 1

        hist_rows.append(
            {
                "stage": "heads",
                "epoch": epoch,
                "dyn_loss": float(np.mean(dyn_losses) if dyn_losses else np.nan),
                "rollout_surface_loss": float(np.mean(dyn_roll_surface_losses) if dyn_roll_surface_losses else np.nan),
                "rollout_calendar_loss": float(np.mean(dyn_roll_cal_losses) if dyn_roll_cal_losses else np.nan),
                "rollout_butterfly_loss": float(np.mean(dyn_roll_bfly_losses) if dyn_roll_bfly_losses else np.nan),
                "price_loss": float(np.mean(price_losses) if price_losses else np.nan),
                "exec_loss": float(np.mean(exec_losses) if exec_losses else np.nan),
                "val_dyn_loss": float(val_dyn),
                "val_rollout_surface_loss": float(val_roll_surface),
                "val_rollout_calendar_loss": float(val_roll_calendar),
                "val_rollout_butterfly_loss": float(val_roll_bfly),
                "val_price_loss": float(val_price),
                "val_exec_loss": float(val_exec),
                "val_forecast_metric": float(val_forecast_metric),
                "lr": float(opt_head.param_groups[0]["lr"]),
                "rollout_train_windows": int(len(dyn_train_windows)),
                "rollout_steps_train_avg": float(np.mean(head_rollout_steps_used) if head_rollout_steps_used else rollout_steps),
                "rollout_val_windows_step1": int(len(dyn_val_windows_step1)),
                "rollout_val_windows_k": int(len(dyn_val_windows_roll)),
                "teacher_forcing_prob": float(head_teacher_forcing_prob),
            }
        )
        if early_stop_patience > 0 and head_bad_epochs >= early_stop_patience:
            head_early_stopped = True
            break

    if best_head_state is not None:
        model.load_state_dict(best_head_state)

    # ------------------------- Stage 3: joint fine-tune ------------------------
    for p in model.encoder.parameters():
        p.requires_grad = True
    for p in model.decoder.parameters():
        p.requires_grad = True

    joint_params = list(model.parameters())
    opt_joint = torch.optim.AdamW(joint_params, lr=cfg.joint_lr, weight_decay=cfg.weight_decay)
    scheduler_joint = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_joint,
        mode="min",
        factor=lr_plateau_factor,
        patience=lr_plateau_patience,
        min_lr=min_lr,
    )
    best_joint_metric = float("inf")
    best_joint_epoch = 0
    best_joint_state: dict[str, torch.Tensor] | None = None
    joint_bad_epochs = 0
    joint_early_stopped = False

    local_idx_map = np.full(n_dates, -1, dtype=np.int32)
    local_idx_map[tr_dates] = np.arange(len(tr_dates), dtype=np.int32)
    tr_asset_ids = asset_ids[tr_dates].astype(np.int32)
    tr_asset_ids_t = torch.as_tensor(tr_asset_ids, dtype=torch.long, device=device)
    joint_rollout_windows_local = _rollout_windows_by_asset(
        np.arange(len(tr_dates), dtype=np.int64),
        tr_asset_ids,
        rollout_steps,
    )
    tr_date_t = torch.as_tensor(tr_dates.astype(np.int64), dtype=torch.long, device=device)
    surface_w_tr = surface_w_t.index_select(0, tr_date_t)

    for epoch in range(1, cfg.joint_epochs + 1):
        model.train()
        joint_teacher_forcing_prob = _linear_schedule_value(
            rollout_teacher_forcing_start,
            rollout_teacher_forcing_end,
            epoch,
            cfg.joint_epochs,
        )
        joint_rollout_steps = _sample_rollout_steps(
            rng=rng,
            max_steps=rollout_steps,
            min_steps=rollout_min_steps,
            random_horizon=rollout_random_horizon,
        )
        joint_rollout_windows_epoch = (
            joint_rollout_windows_local[:, : joint_rollout_steps + 1]
            if len(joint_rollout_windows_local) > 0
            else joint_rollout_windows_local
        )

        x = surface_scaled[tr_dates]
        ctx = context_scaled[tr_dates]

        mu, logvar = model.encode(x)
        z = mu if bool(cfg.joint_use_mu_deterministic) else model.reparameterize(mu, logvar)
        recon = model.decode(z)

        recon_loss = _surface_recon_loss(recon, x, point_weight=surface_w_tr, beta=recon_huber_beta)
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        cal = _calendar_penalty_torch(
            recon,
            surface_scaler,
            nx,
            nt,
            tenor_days,
            surface_variable=surface_variable,
        )
        bfly = _butterfly_penalty_torch(
            recon,
            surface_scaler,
            nx,
            nt,
            x_grid=x_grid,
            tenor_days=tenor_days,
            surface_variable=surface_variable,
        )
        kl_beta = _kl_beta_for_epoch(cfg.vae_kl_beta, epoch, kl_warmup_epochs)

        dyn_loss = torch.tensor(0.0, device=device)
        roll_surface_loss = torch.tensor(0.0, device=device)
        roll_cal_loss = torch.tensor(0.0, device=device)
        roll_bfly_loss = torch.tensor(0.0, device=device)
        if len(joint_rollout_windows_epoch) > 0:
            dyn_loss, roll_surface_loss, roll_cal_loss, roll_bfly_loss = _rollout_losses_torch(
                model=model,
                z_init=z,
                z_target=z,
                context_scaled=ctx,
                surface_scaled_target=x,
                rollout_windows=joint_rollout_windows_epoch,
                steps=joint_rollout_steps,
                surface_scaler=surface_scaler,
                nx=nx,
                nt=nt,
                tenor_days=tenor_days,
                x_grid=x_grid,
                surface_variable=surface_variable,
                surface_point_weight=surface_w_tr,
                rollout_surface_huber_beta=rollout_surface_huber_beta,
                teacher_forcing_prob=joint_teacher_forcing_prob,
                asset_ids=tr_asset_ids_t,
            )

        p_loss = torch.tensor(0.0, device=device)
        e_loss = torch.tensor(0.0, device=device)
        if len(c_tr) > 0 and (price_head_enabled or exec_head_enabled):
            pick_size = min(len(c_tr), cfg.joint_contract_batch_size)
            pick = rng.choice(c_tr, size=pick_size, replace=False)
            local = local_idx_map[date_idx[pick]]
            local_t = torch.as_tensor(local, dtype=torch.long, device=device)

            zc = z[local_t]
            fc = feature_scaled[pick]
            ac = contract_asset_t[pick]
            pt = price_scaled[pick]
            et = fill_t[pick]
            pw = price_w_t[pick]
            ew = exec_w_t[pick]

            if price_head_enabled:
                p_pred = model.forward_pricer(zc, fc, asset_id=ac)
                p_loss = _weighted_smooth_l1_loss(p_pred, pt, pw, beta=0.02)

            if exec_head_enabled:
                e_pred = model.forward_execution_logit(zc, fc, asset_id=ac)
                et_s = _smooth_binary_targets(et, cfg.exec_label_smoothing)
                e_loss = _weighted_bce_with_logits(e_pred, et_s, ew) + cfg.exec_logit_l2 * torch.mean(
                    e_pred.pow(2)
                )

        loss = (
            recon_loss
            + kl_beta * kl
            + noarb_lambda * cal
            + noarb_butterfly_lambda * bfly
            + cfg.joint_dyn_lambda * dyn_loss
            + rollout_surface_lambda * roll_surface_loss
            + rollout_calendar_lambda * roll_cal_loss
            + rollout_butterfly_lambda * roll_bfly_loss
            + (cfg.joint_price_lambda if price_head_enabled else 0.0) * p_loss
            + (cfg.joint_exec_lambda if exec_head_enabled else 0.0) * e_loss
        )

        opt_joint.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(joint_params, 2.5)
        opt_joint.step()

        model.eval()
        with torch.no_grad():
            val_recon = _eval_recon(
                model,
                surface_scaled,
                va_dates,
            )
            z_val_all, _ = model.encode(surface_scaled)

            val_dyn = np.nan
            if len(dyn_val_windows_step1) > 0:
                idx_t = torch.as_tensor(dyn_val_windows_step1[:, 0], dtype=torch.long, device=device)
                next_t = torch.as_tensor(dyn_val_windows_step1[:, 1], dtype=torch.long, device=device)
                pred = model.forward_dynamics(
                    z_val_all.index_select(0, idx_t),
                    context_scaled.index_select(0, idx_t),
                    asset_id=asset_ids_t.index_select(0, idx_t),
                )
                val_dyn = float(F.mse_loss(pred, z_val_all.index_select(0, next_t)).item())

            val_roll_surface = np.nan
            val_roll_calendar = np.nan
            val_roll_bfly = np.nan
            if len(dyn_val_windows_roll) > 0:
                _, vr_surf, vr_cal, vr_bfly = _rollout_losses_torch(
                    model=model,
                    z_init=z_val_all,
                    z_target=z_val_all,
                    context_scaled=context_scaled,
                    surface_scaled_target=surface_scaled,
                    rollout_windows=dyn_val_windows_roll,
                    steps=rollout_steps,
                    surface_scaler=surface_scaler,
                    nx=nx,
                    nt=nt,
                    tenor_days=tenor_days,
                    x_grid=x_grid,
                    surface_variable=surface_variable,
                    surface_point_weight=surface_w_t,
                    rollout_surface_huber_beta=rollout_surface_huber_beta,
                    asset_ids=asset_ids_t,
                )
                val_roll_surface = float(vr_surf.item())
                val_roll_calendar = float(vr_cal.item())
                val_roll_bfly = float(vr_bfly.item())

            val_forecast_metric = np.nan
            if np.isfinite(val_roll_surface):
                val_forecast_metric = float(val_roll_surface)
                if np.isfinite(val_roll_calendar):
                    val_forecast_metric += float(rollout_calendar_lambda) * float(val_roll_calendar)
                if np.isfinite(val_roll_bfly):
                    val_forecast_metric += float(rollout_butterfly_lambda) * float(val_roll_bfly)
            elif np.isfinite(val_dyn):
                val_forecast_metric = float(val_dyn)

        if np.isfinite(val_forecast_metric):
            scheduler_joint.step(float(val_forecast_metric))
            if float(val_forecast_metric) < (best_joint_metric - early_stop_min_delta):
                best_joint_metric = float(val_forecast_metric)
                best_joint_epoch = int(epoch)
                best_joint_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                joint_bad_epochs = 0
            else:
                joint_bad_epochs += 1
        else:
            joint_bad_epochs += 1

        hist_rows.append(
            {
                "stage": "joint",
                "epoch": epoch,
                "loss": float(loss.item()),
                "recon_loss": float(recon_loss.item()),
                "kl_loss": float(kl.item()),
                "kl_beta": float(kl_beta),
                "calendar_loss": float(cal.item()),
                "butterfly_loss": float(bfly.item()),
                "dyn_loss": float(dyn_loss.item()),
                "rollout_surface_loss": float(roll_surface_loss.item()),
                "rollout_calendar_loss": float(roll_cal_loss.item()),
                "rollout_butterfly_loss": float(roll_bfly_loss.item()),
                "price_loss": float(p_loss.item()),
                "exec_loss": float(e_loss.item()),
                "val_recon_loss": float(val_recon),
                "val_dyn_loss": float(val_dyn),
                "val_rollout_surface_loss": float(val_roll_surface),
                "val_rollout_calendar_loss": float(val_roll_calendar),
                "val_rollout_butterfly_loss": float(val_roll_bfly),
                "val_forecast_metric": float(val_forecast_metric),
                "lr": float(opt_joint.param_groups[0]["lr"]),
                "rollout_train_windows": int(len(joint_rollout_windows_local)),
                "rollout_steps_train_avg": float(joint_rollout_steps),
                "rollout_val_windows_step1": int(len(dyn_val_windows_step1)),
                "rollout_val_windows_k": int(len(dyn_val_windows_roll)),
                "teacher_forcing_prob": float(joint_teacher_forcing_prob),
            }
        )
        if early_stop_patience > 0 and joint_bad_epochs >= early_stop_patience:
            joint_early_stopped = True
            break

    if best_joint_state is not None:
        model.load_state_dict(best_joint_state)

    # Save artifacts
    run_dir = make_run_dir(cfg.out_dir, prefix="run")

    bundle = ModelBundle(
        model=model.eval().cpu(),
        surface_scaler=surface_scaler,
        context_scaler=context_scaler,
        contract_scaler=contract_scaler,
        price_scaler=price_scaler,
    )
    model_path = run_dir / "model.pt"
    bundle.save(model_path)

    hist = pd.DataFrame(hist_rows)
    hist.to_csv(run_dir / "train_history.csv", index=False)

    with torch.no_grad():
        model_cpu = bundle.model
        sf = torch.as_tensor(surface_scaled_np, dtype=torch.float32)
        mu_all, _ = model_cpu.encode(sf)
        latent = to_numpy(mu_all)

    latent_df = pd.DataFrame(latent, columns=[f"z_{i}" for i in range(latent.shape[1])])
    latent_df.insert(0, "date", dates)
    latent_df.to_parquet(run_dir / "latent_states.parquet", index=False)

    cfg_payload = asdict(cfg)
    cfg_payload["out_dir"] = str(cfg.out_dir)
    cfg_payload["device"] = str(device)
    (run_dir / "train_config.json").write_text(json.dumps(cfg_payload, indent=2), encoding="utf-8")

    summary = {
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "device": str(device),
        "surface_variable": surface_variable,
        "split_mode": split_mode,
        "assets": asset_names,
        "n_assets": int(n_assets),
        "n_dates_total": int(n_dates),
        "n_dates_train": int(len(tr_dates)),
        "n_dates_val": int(len(va_dates)),
        "n_dates_test": int(len(te_dates)),
        "n_contracts_train": int(len(c_tr)),
        "n_contracts_val": int(len(c_va)),
        "n_contracts_test": int(len(c_te)),
        "surface_dynamics_only": bool(surface_only),
        "price_head_enabled": bool(price_head_enabled),
        "exec_head_enabled": bool(exec_head_enabled),
        "dynamics_residual_enabled": bool(cfg.dynamics_residual),
        "asset_embedding_dim": int(max(cfg.asset_embed_dim, 0)),
        "context_dim_original": int(original_context_dim),
        "context_dim_used": int(context.shape[1]),
        "context_augmented_from_contracts": bool(len(added_context_names) > 0),
        "context_added_features": added_context_names,
        "price_spread_inv_lambda": float(max(cfg.price_spread_inv_lambda, 0.0)),
        "price_spread_clip_min": float(max(cfg.price_spread_clip_min, 1e-4)),
        "price_spread_clip_max": float(max(cfg.price_spread_clip_max, cfg.price_spread_clip_min)),
        "price_vega_power": float(max(cfg.price_vega_power, 0.0)),
        "price_vega_cap": float(max(cfg.price_vega_cap, 0.0)),
        "context_winsor_quantile": float(max(0.0, cfg.context_winsor_quantile)),
        "context_z_clip": float(cfg.context_z_clip),
        "kl_warmup_epochs": int(kl_warmup_epochs),
        "rollout_steps": int(rollout_steps),
        "rollout_min_steps": int(rollout_min_steps),
        "rollout_random_horizon": bool(rollout_random_horizon),
        "rollout_teacher_forcing_start": float(rollout_teacher_forcing_start),
        "rollout_teacher_forcing_end": float(rollout_teacher_forcing_end),
        "rollout_surface_lambda": float(rollout_surface_lambda),
        "rollout_calendar_lambda": float(rollout_calendar_lambda),
        "rollout_butterfly_lambda": float(rollout_butterfly_lambda),
        "surface_focus_density_alpha": float(focus_density_alpha),
        "surface_focus_density_map_path": focus_density_map_path_used,
        "surface_focus_density_train_only": True,
        "rollout_train_windows": int(len(dyn_train_windows)),
        "rollout_val_windows_step1": int(len(dyn_val_windows_step1)),
        "rollout_val_windows_k": int(len(dyn_val_windows_roll)),
        "rollout_joint_train_windows": int(len(joint_rollout_windows_local)),
        "head_best_epoch": int(best_head_epoch),
        "head_best_val_forecast_metric": float(best_head_metric) if np.isfinite(best_head_metric) else None,
        "head_early_stopped": bool(head_early_stopped),
        "joint_best_epoch": int(best_joint_epoch),
        "joint_best_val_forecast_metric": float(best_joint_metric) if np.isfinite(best_joint_metric) else None,
        "joint_early_stopped": bool(joint_early_stopped),
        "joint_use_mu_deterministic": bool(cfg.joint_use_mu_deterministic),
        "noarb_butterfly_lambda": float(noarb_butterfly_lambda),
        "final_val_recon": float(hist[hist["stage"] == "joint"]["val_recon_loss"].iloc[-1]) if (hist["stage"] == "joint").any() else None,
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return run_dir
