"""Command-line interface for ivdyn."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import date, timedelta
from pathlib import Path
import os
import subprocess
import sys
from typing import Any, Sequence

from ivdyn.utils.paths import resolve_latest, utc_timestamp


def _to_path(v: str) -> Path:
    return Path(v).expanduser()


def _autoload_dotenv() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [Path.cwd() / ".env", repo_root / ".env"]
    seen: set[Path] = set()
    for candidate in candidates:
        p = candidate.resolve()
        if p in seen or not p.exists() or not p.is_file():
            continue
        seen.add(p)

        for raw_line in p.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue

            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                q = value[0]
                value = value[1:-1]
                if q == '"':
                    value = value.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
            else:
                comment_pos = value.find(" #")
                if comment_pos >= 0:
                    value = value[:comment_pos].rstrip()

            os.environ.setdefault(key, value)


def _resolve_run_dir(raw: str | None) -> Path:
    if raw:
        return _to_path(raw).resolve()
    latest = resolve_latest(Path("outputs/runs"))
    if latest is None:
        raise RuntimeError("No run directory found. Provide --run-dir explicitly.")
    return latest


def _resolve_dataset(dataset_arg: str | None, run_dir: Path) -> Path:
    if dataset_arg:
        return _to_path(dataset_arg).resolve()
    train_summary = run_dir / "train_summary.json"
    if not train_summary.exists():
        raise RuntimeError("No dataset argument and no train_summary.json found in run dir.")
    import json

    payload = json.loads(train_summary.read_text(encoding="utf-8"))
    if not (dataset := payload.get("dataset_path")):
        raise RuntimeError("train_summary.json exists but missing dataset_path.")
    return _to_path(str(dataset)).resolve()


def _load_npz_dict(path: Path) -> dict[str, "np.ndarray"]:
    import numpy as np

    z = np.load(path, allow_pickle=True)
    out: dict[str, np.ndarray] = {}
    for k in z.files:
        arr = z[k]
        if arr.dtype == object:
            arr = arr.astype(str)
        out[k] = arr
    return out


def _dataset_symbols(path: Path) -> list[str]:
    ds = _load_npz_dict(path)
    if "asset_names" in ds and len(ds["asset_names"]) > 0:
        return sorted({str(x).upper() for x in ds["asset_names"].astype(str).tolist()})
    if "contract_underlying" in ds and len(ds["contract_underlying"]) > 0:
        return sorted({str(x).upper() for x in ds["contract_underlying"].astype(str).tolist() if str(x).strip()})
    return []


def _subset_dataset_for_symbol(dataset_path: Path, symbol: str, out_path: Path) -> Path:
    import numpy as np

    ds = _load_npz_dict(dataset_path)
    sym_u = symbol.upper().strip()
    if not sym_u:
        raise RuntimeError("Empty symbol requested for dataset subset.")

    if "asset_ids" not in ds:
        raise RuntimeError("Dataset does not contain asset_ids; cannot subset per symbol automatically.")

    n_dates = int(len(ds["asset_ids"]))
    if n_dates == 0:
        raise RuntimeError("Dataset has zero date rows.")

    asset_ids = ds["asset_ids"].astype(np.int32)
    asset_names = ds.get("asset_names", np.array([], dtype=str)).astype(str)

    if len(asset_names) > 0:
        matches = np.where(np.char.upper(asset_names.astype(str)) == sym_u)[0]
        if len(matches) == 0:
            raise RuntimeError(f"Symbol '{sym_u}' not found in dataset asset_names={asset_names.tolist()}.")
        asset_id = int(matches[0])
        date_mask = asset_ids == asset_id
    else:
        contract_underlying = ds.get("contract_underlying")
        if contract_underlying is None:
            raise RuntimeError("Dataset missing both asset_names and contract_underlying for symbol subsetting.")
        date_idx = ds["contract_date_index"].astype(np.int32)
        cu = np.char.upper(contract_underlying.astype(str))
        date_mask = np.zeros(n_dates, dtype=bool)
        date_mask[np.unique(date_idx[cu == sym_u])] = True
        if not np.any(date_mask):
            raise RuntimeError(f"Symbol '{sym_u}' not found in contract_underlying.")
        asset_id = int(np.bincount(asset_ids[date_mask]).argmax())

    if not np.any(date_mask):
        raise RuntimeError(f"No date rows found for symbol '{sym_u}'.")

    date_old_idx = np.where(date_mask)[0].astype(np.int32)
    remap = np.full(n_dates, -1, dtype=np.int32)
    remap[date_old_idx] = np.arange(len(date_old_idx), dtype=np.int32)

    contract_date_idx = ds["contract_date_index"].astype(np.int32)
    contract_asset_id = ds.get("contract_asset_id", asset_ids[contract_date_idx]).astype(np.int32)
    contract_mask = (contract_asset_id == asset_id) & date_mask[contract_date_idx]

    out: dict[str, np.ndarray] = {}
    date_keys = {
        "dates",
        "asset_ids",
        "spot",
        "surface",
        "iv_surface",
        "w_surface",
        "liq_surface",
        "spread_surface",
        "vega_surface",
        "context",
    }
    for k, arr in ds.items():
        if k == "asset_names":
            out[k] = np.array([sym_u], dtype="<U32")
            continue
        if k == "asset_ids":
            out[k] = np.zeros(np.sum(date_mask), dtype=np.int32)
            continue
        if k == "contract_asset_id":
            out[k] = np.zeros(np.sum(contract_mask), dtype=np.int32)
            continue
        if k == "contract_date_index":
            out[k] = remap[contract_date_idx[contract_mask]].astype(np.int32)
            continue
        if k.startswith("contract_"):
            if arr.ndim == 0:
                out[k] = arr
            elif len(arr) == len(contract_date_idx):
                out[k] = arr[contract_mask]
            else:
                out[k] = arr
            continue
        if k in date_keys:
            out[k] = arr[date_mask]
            continue
        if arr.ndim > 0 and len(arr) == n_dates and k not in {"x_grid", "tenor_days", "context_names"}:
            out[k] = arr[date_mask]
            continue
        out[k] = arr

    if "contract_underlying" in out:
        out["contract_underlying"] = np.array([sym_u] * len(out["contract_underlying"]), dtype="<U16")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    return out_path


def _build_training_config_from_ns(ns: Any, *, seed: int, out_dir: Path):
    from ivdyn.training import TrainingConfig

    price_exec_enabled = (
        bool(getattr(ns, "enable_price_exec_heads", False))
        or float(getattr(ns, "joint_price_lambda", 0.0)) > 0.0
        or float(getattr(ns, "joint_exec_lambda", 0.0)) > 0.0
    )
    surface_only_forced = bool(getattr(ns, "surface_dynamics_only", False))
    surface_only = surface_only_forced or (not price_exec_enabled)

    kwargs = dict(
        out_dir=out_dir,
        seed=seed,
        train_frac=ns.train_frac,
        val_frac=ns.val_frac,
        split_mode=str(getattr(ns, "split_mode", "by_asset_time")),
        latent_dim=ns.latent_dim,
        vae_hidden=(ns.vae_hidden_dim_1, ns.vae_hidden_dim_2),
        dynamics_hidden=(ns.dynamics_hidden_dim_1, ns.dynamics_hidden_dim_2),
        pricing_hidden=(ns.pricing_hidden_dim_1, ns.pricing_hidden_dim_2),
        execution_hidden=(ns.execution_hidden_dim_1, ns.execution_hidden_dim_2),
        model_dropout=float(getattr(ns, "model_dropout", 0.08)),
        vae_epochs=ns.vae_epochs,
        vae_batch_size=ns.vae_batch_size,
        vae_lr=ns.vae_lr,
        vae_kl_beta=ns.vae_kl_beta,
        kl_warmup_epochs=ns.kl_warmup_epochs,
        noarb_lambda=ns.noarb_lambda,
        noarb_butterfly_lambda=ns.noarb_butterfly_lambda,
        recon_huber_beta=float(getattr(ns, "recon_huber_beta", 0.015)),
        head_epochs=ns.head_epochs,
        dyn_batch_size=ns.dyn_batch_size,
        contract_batch_size=ns.contract_batch_size,
        head_lr=ns.head_lr,
        rollout_steps=ns.rollout_steps,
        rollout_random_horizon=bool(getattr(ns, "rollout_random_horizon", False)),
        rollout_min_steps=int(getattr(ns, "rollout_min_steps", 1)),
        rollout_teacher_forcing_start=float(getattr(ns, "rollout_teacher_forcing_start", 0.0)),
        rollout_teacher_forcing_end=float(getattr(ns, "rollout_teacher_forcing_end", 0.0)),
        rollout_surface_lambda=ns.rollout_surface_lambda,
        rollout_calendar_lambda=ns.rollout_calendar_lambda,
        rollout_butterfly_lambda=ns.rollout_butterfly_lambda,
        rollout_surface_huber_beta=float(getattr(ns, "rollout_surface_huber_beta", 0.02)),
        rollout_slope_lambda=float(getattr(ns, "rollout_slope_lambda", 0.08)),
        rollout_curvature_lambda=float(getattr(ns, "rollout_curvature_lambda", 0.02)),
        surface_weight_liq_alpha=float(getattr(ns, "surface_weight_liq_alpha", 0.0)),
        surface_weight_spread_alpha=float(getattr(ns, "surface_weight_spread_alpha", 0.0)),
        surface_weight_vega_alpha=float(getattr(ns, "surface_weight_vega_alpha", 0.0)),
        surface_weight_clip_min=float(getattr(ns, "surface_weight_clip_min", 1.0)),
        surface_weight_clip_max=float(getattr(ns, "surface_weight_clip_max", 1.0)),
        surface_focus_alpha=float(getattr(ns, "surface_focus_alpha", 0.0)),
        surface_focus_x_min=float(getattr(ns, "surface_focus_x_min", 0.10)),
        surface_focus_x_scale=float(getattr(ns, "surface_focus_x_scale", 0.03)),
        surface_focus_dte_scale_days=float(getattr(ns, "surface_focus_dte_scale_days", 21.0)),
        surface_focus_dte_max_days=float(getattr(ns, "surface_focus_dte_max_days", 30.0)),
        surface_focus_neg_x_max=float(getattr(ns, "surface_focus_neg_x_max", -0.20)),
        surface_focus_neg_weight_ratio=float(getattr(ns, "surface_focus_neg_weight_ratio", 0.0)),
        surface_focus_density_alpha=float(getattr(ns, "surface_focus_density_alpha", 0.0)),
        surface_focus_density_map_path=getattr(ns, "surface_focus_density_map", None),
        joint_epochs=ns.joint_epochs,
        joint_lr=ns.joint_lr,
        joint_contract_batch_size=ns.joint_contract_batch_size,
        joint_dyn_lambda=ns.joint_dyn_lambda,
        joint_price_lambda=ns.joint_price_lambda,
        joint_exec_lambda=ns.joint_exec_lambda,
        joint_use_mu_deterministic=bool(getattr(ns, "joint_use_mu_deterministic", True)),
        weight_decay=ns.weight_decay,
        price_risk_weight=ns.price_risk_weight,
        exec_risk_weight=ns.exec_risk_weight,
        price_spread_inv_lambda=float(getattr(ns, "price_spread_inv_lambda", 0.35)),
        price_spread_clip_min=float(getattr(ns, "price_spread_clip_min", 0.02)),
        price_spread_clip_max=float(getattr(ns, "price_spread_clip_max", 3.0)),
        price_vega_power=float(getattr(ns, "price_vega_power", 0.25)),
        price_vega_cap=float(getattr(ns, "price_vega_cap", 4.0)),
        risk_focus_abs_x=ns.risk_focus_abs_x,
        risk_focus_tau_days=ns.risk_focus_tau_days,
        exec_label_smoothing=ns.exec_label_smoothing,
        exec_logit_l2=ns.exec_logit_l2,
        surface_dynamics_only=surface_only,
        context_winsor_quantile=float(getattr(ns, "context_winsor_quantile", 0.01)),
        context_z_clip=float(getattr(ns, "context_z_clip", 5.0)),
        context_augment_from_contracts=not bool(getattr(ns, "disable_context_augment", False)),
        context_augment_surface_history=not bool(getattr(ns, "disable_surface_history_augment", False)),
        dynamics_residual=not bool(getattr(ns, "disable_dynamics_residual", False)),
        asset_embed_dim=int(getattr(ns, "asset_embed_dim", 8)),
        surface_refiner_hidden=(
            int(getattr(ns, "surface_refiner_hidden_1", 256)),
            int(getattr(ns, "surface_refiner_hidden_2", 128)),
        ),
        disable_surface_refiner=bool(getattr(ns, "disable_surface_refiner", False)),
        early_stop_patience=int(getattr(ns, "early_stop_patience", 20)),
        early_stop_min_delta=float(getattr(ns, "early_stop_min_delta", 1e-4)),
        lr_plateau_patience=int(getattr(ns, "lr_plateau_patience", 6)),
        lr_plateau_factor=float(getattr(ns, "lr_plateau_factor", 0.5)),
        min_lr=float(getattr(ns, "min_lr", 1e-6)),
        max_cpu_threads=int(getattr(ns, "max_cpu_threads", 2)),
        model_arch=str(getattr(ns, "model_arch", "tree_boost")),
        option_a_seq_len=int(getattr(ns, "option_a_seq_len", 20)),
        option_a_epochs=int(getattr(ns, "option_a_epochs", 80)),
        option_a_batch_size=int(getattr(ns, "option_a_batch_size", 128)),
        option_a_eval_batch_size=int(getattr(ns, "option_a_eval_batch_size", 512)),
        option_a_lr=float(getattr(ns, "option_a_lr", 8e-4)),
        option_a_weight_decay=float(getattr(ns, "option_a_weight_decay", 1e-5)),
        option_a_hidden_dim=int(getattr(ns, "option_a_hidden_dim", 192)),
        option_a_tcn_layers=int(getattr(ns, "option_a_tcn_layers", 4)),
        option_a_tcn_kernel_size=int(getattr(ns, "option_a_tcn_kernel_size", 3)),
        option_a_dropout=float(getattr(ns, "option_a_dropout", 0.08)),
        option_a_early_stop_patience=int(getattr(ns, "option_a_early_stop_patience", 12)),
        option_a_blend_alpha_min=float(getattr(ns, "option_a_blend_alpha_min", 0.6)),
        option_a_blend_alpha_max=float(getattr(ns, "option_a_blend_alpha_max", 1.4)),
        option_a_blend_alpha_steps=int(getattr(ns, "option_a_blend_alpha_steps", 9)),
        option_a_device=str(getattr(ns, "option_a_device", "auto")),
    )
    # Keep CLI backward/forward compatible with the installed training pipeline.
    # Some branches expose a smaller/larger TrainingConfig field set.
    valid = set(getattr(TrainingConfig, "__dataclass_fields__", {}).keys())
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return TrainingConfig(**filtered)


def _train_command(ns: Any) -> None:
    from ivdyn.training import derive_focus_density_map_from_run, train

    dataset_path = _to_path(ns.dataset).resolve()
    out_dir = _to_path(ns.out_dir)
    seed = int(ns.seed)
    adaptive_focus = bool(getattr(ns, "adaptive_focus_rerun", False))

    if adaptive_focus and bool(getattr(ns, "per_symbol", False)):
        raise RuntimeError("--adaptive-focus-rerun does not support --per-symbol mode.")

    if not bool(getattr(ns, "per_symbol", False)):
        if not adaptive_focus:
            run_dir = train(
                dataset_path,
                _build_training_config_from_ns(ns, seed=seed, out_dir=out_dir),
            )
            print(run_dir)
            return

        from ivdyn.eval import evaluate
        import json

        focus_alpha = float(getattr(ns, "adaptive_focus_density_alpha", 0.0))
        if not (focus_alpha > 0.0):
            raise RuntimeError(
                "--adaptive-focus-rerun requires --adaptive-focus-density-alpha > 0."
            )
        focus_power = float(getattr(ns, "adaptive_focus_error_power", 1.0))
        if not (focus_power > 0.0):
            raise RuntimeError(
                "--adaptive-focus-error-power must be > 0."
            )

        cfg_stage1 = _build_training_config_from_ns(ns, seed=seed, out_dir=out_dir)
        cfg_stage1.surface_focus_density_alpha = 0.0
        cfg_stage1.surface_focus_density_map_path = None
        run_stage1 = train(dataset_path, cfg_stage1)

        map_arg = str(getattr(ns, "adaptive_focus_map_path", "") or "").strip()
        if map_arg:
            map_path = _to_path(map_arg).resolve()
        else:
            map_path = (run_stage1 / f"adaptive_focus_density_map_{utc_timestamp()}.json").resolve()
        focus_meta = derive_focus_density_map_from_run(
            run_dir=run_stage1,
            dataset_path=dataset_path,
            cfg=cfg_stage1,
            out_path=map_path,
            error_power=focus_power,
        )

        cfg_stage2 = _build_training_config_from_ns(ns, seed=seed, out_dir=out_dir)
        cfg_stage2.surface_focus_density_alpha = focus_alpha
        cfg_stage2.surface_focus_density_map_path = str(map_path)
        run_stage2 = train(dataset_path, cfg_stage2)
        eval_dir = evaluate(
            run_dir=run_stage2,
            dataset_path=dataset_path,
            surface_dynamics_only=not bool(getattr(ns, "include_contract_metrics", False)),
        )

        summary = {
            "dataset": str(dataset_path),
            "stage1_run_dir": str(run_stage1),
            "focus_map_path": str(map_path),
            "focus_map_meta": focus_meta,
            "stage2_run_dir": str(run_stage2),
            "stage2_eval_dir": str(eval_dir),
            "adaptive_focus_density_alpha": float(focus_alpha),
            "adaptive_focus_error_power": float(focus_power),
            "trainval_only_for_focus_map": True,
            "test_used_for_focus_map": False,
        }
        summary_path = out_dir / f"adaptive_focus_rerun_{utc_timestamp()}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(f"stage1_run\t{run_stage1}")
        print(f"focus_map\t{map_path}")
        print(f"stage2_run\t{run_stage2}")
        print(f"stage2_eval\t{eval_dir}")
        print(summary_path)
        return

    all_syms = _dataset_symbols(dataset_path)
    requested = [str(s).upper().strip() for s in (getattr(ns, "symbols", None) or []) if str(s).strip()]
    targets = requested if requested else all_syms
    if not targets:
        raise RuntimeError("No symbols available for --per-symbol run.")

    ds_root = (
        _to_path(str(getattr(ns, "per_symbol_dataset_dir", ""))).resolve()
        if getattr(ns, "per_symbol_dataset_dir", None)
        else (out_dir / "_per_symbol_datasets").resolve()
    )
    ds_root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for sym in targets:
        sub_dataset = _subset_dataset_for_symbol(
            dataset_path=dataset_path,
            symbol=sym,
            out_path=ds_root / f"dataset_{sym}.npz",
        )
        run_dir = train(
            sub_dataset,
            _build_training_config_from_ns(ns, seed=seed, out_dir=out_dir / sym),
        )
        rows.append(
            {
                "symbol": sym,
                "dataset": str(sub_dataset),
                "run_dir": str(run_dir),
            }
        )
        print(f"{sym}\t{run_dir}")

    summary_path = out_dir / f"per_symbol_runs_{utc_timestamp()}.json"
    import json

    summary_path.write_text(
        json.dumps(
            {
                "base_dataset": str(dataset_path),
                "symbols": targets,
                "runs": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_path)


def _build_dataset_command(ns: Any) -> None:
    from ivdyn.data import DatasetBuildConfig, build_dataset

    out = build_dataset(
        DatasetBuildConfig(
            data_root=_to_path(ns.data_root),
            out_dir=_to_path(ns.out_dir),
            symbol=ns.symbol,
            plugin=ns.plugin,
            api_key=ns.api_key,
            start_date=ns.start_date,
            end_date=ns.end_date,
            x_grid=tuple(ns.x_grid),
            tenor_days=tuple(ns.tenor_days),
            max_contracts_per_day=ns.max_contracts_per_day,
            random_seed=ns.random_seed,
            num_workers=ns.num_workers,
        )
    )
    print(out["dataset"])


def _extract_wrds_command(ns: Any) -> None:
    from ivdyn.data import WRDSExtractConfig, build_wrds_contract_panel

    out = build_wrds_contract_panel(
        WRDSExtractConfig(
            options_path=_to_path(ns.options_path).resolve(),
            underlying_path=_to_path(ns.underlying_path).resolve(),
            forward_path=(_to_path(ns.forward_path).resolve() if ns.forward_path else None),
            out_dir=_to_path(ns.out_dir).resolve(),
            symbols=tuple(ns.symbols or ()),
            start_date=ns.start_date,
            end_date=ns.end_date,
            chunksize=int(ns.chunksize),
            min_iv=float(ns.min_iv),
            max_iv=float(ns.max_iv),
        )
    )
    print(out["contracts"])


def _build_surface_dataset_command(ns: Any) -> None:
    from ivdyn.data import SurfaceDatasetBuildConfig, build_surface_dataset

    out = build_surface_dataset(
        SurfaceDatasetBuildConfig(
            contracts_path=_to_path(ns.contracts_path).resolve(),
            out_dir=_to_path(ns.out_dir).resolve(),
            earnings_path=(_to_path(ns.earnings_path).resolve() if ns.earnings_path else None),
            symbols=tuple(ns.symbols or ()),
            start_date=ns.start_date,
            end_date=ns.end_date,
            x_grid=tuple(ns.x_grid),
            tenor_days=tuple(ns.tenor_days),
            max_contracts_per_day=int(ns.max_contracts_per_day),
            min_contracts_per_day=int(ns.min_contracts_per_day),
            max_neighbors=int(ns.max_neighbors),
            max_dte_distance=int(ns.max_dte_distance),
            max_rel_spread=float(ns.max_rel_spread),
            random_seed=int(ns.random_seed),
            surface_variable=str(ns.surface_variable),
            num_workers=int(getattr(ns, "num_workers", 0)),
            surface_pca_factors=int(getattr(ns, "surface_pca_factors", 3)),
        )
    )
    print(out["dataset"])


def _evaluate_command(ns: Any) -> None:
    from ivdyn.eval import evaluate

    run_dir = _resolve_run_dir(ns.run_dir)
    dataset = _resolve_dataset(ns.dataset, run_dir)
    out_dir = evaluate(
        run_dir=run_dir,
        dataset_path=dataset,
        device=ns.device,
        num_workers=ns.num_workers,
        surface_dynamics_only=not bool(getattr(ns, "include_contract_metrics", False)),
        baseline_factor_dim=int(getattr(ns, "baseline_factor_dim", 3)),
        baseline_ridge=float(getattr(ns, "baseline_ridge", 1e-4)),
        baseline_min_history=int(getattr(ns, "baseline_min_history", 40)),
    )
    print(out_dir)


def _experiment_plan_bundles() -> list[dict[str, Any]]:
    return [
        {
            "name": "baseline",
            "description": "Reference run with current surface-dynamics-focused defaults.",
            "overrides": {},
            "unsupported_suggestions": [],
        },
        {
            "name": "front_tenor_focus",
            "description": "Approximate recommendation #1 with supported knobs.",
            "overrides": {
                "risk_focus_tau_days": 30.0,
            },
            "unsupported_suggestions": [
                "surface_mask_min_tenor_days",
                "surface_liq_weight_alpha",
            ],
        },
        {
            "name": "dynamics_bundle",
            "description": "Recommendation #2: strengthen latent transition and forecast learning.",
            "overrides": {
                "joint_dyn_lambda": 2.0,
                "latent_dim": 24,
                "vae_kl_beta": 0.03,
                "joint_lr": 3.5e-4,
            },
            "unsupported_suggestions": [
                "surface_forecast_huber_beta",
            ],
        },
        {
            "name": "noarb_bundle",
            "description": "Recommendation #4: increase no-arbitrage regularization.",
            "overrides": {
                "noarb_lambda": 0.35,
            },
            "unsupported_suggestions": [],
        },
        {
            "name": "stability_bundle",
            "description": "Recommendation #5: improve optimization stability.",
            "overrides": {
                "head_lr": 7e-4,
                "joint_lr": 3.5e-4,
                "weight_decay": 5e-5,
                "joint_contract_batch_size": 3072,
            },
            "unsupported_suggestions": [],
        },
    ]


def _metric_subset_keys() -> list[str]:
    return [
        "surface_dynamics_focus_mode",
        "error_metric_mode",
        "surface_forecast_iv_rmse",
        "surface_forecast_baseline_primary",
        "surface_forecast_iv_rmse_baseline_parametric",
        "surface_forecast_skill_mse_vs_parametric",
        "surface_forecast_iv_rmse_baseline_persistence",
        "surface_forecast_skill_mse_vs_persistence",
        "surface_recon_iv_rmse",
        "calendar_violation_pred_mean",
        "butterfly_violation_pred_mean",
        "calendar_violation_obs_mean",
        "butterfly_violation_obs_mean",
        "calendar_violation_forecast_pred_mean",
        "calendar_violation_forecast_tree_mean",
        "butterfly_violation_forecast_pred_mean",
        "butterfly_violation_forecast_tree_mean",
        "calendar_violation_forecast_obs_mean",
        "butterfly_violation_forecast_obs_mean",
    ]


def _metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = _metric_subset_keys()
    out: dict[str, Any] = {}
    for key in keys:
        if key in metrics:
            out[key] = metrics[key]
    return out


def _experiment_plan_command(ns: Any) -> None:
    import csv
    import json
    import time
    import traceback

    dataset = _to_path(ns.dataset).resolve()
    if not dataset.exists():
        raise RuntimeError(f"Dataset file not found: {dataset}")

    out_dir = _to_path(ns.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plan_root = _to_path(ns.plan_dir).resolve() / f"plan_{utc_timestamp()}"
    plan_root.mkdir(parents=True, exist_ok=True)

    bundles = _experiment_plan_bundles()
    if bool(getattr(ns, "skip_baseline", False)):
        bundles = [b for b in bundles if b["name"] != "baseline"]
    if not bundles:
        raise RuntimeError("No bundles left to execute. Remove --skip-baseline.")

    seeds: list[int] = list(getattr(ns, "seeds", []) or [7])
    plan_manifest = {
        "created_utc": utc_timestamp(),
        "dataset": str(dataset),
        "out_dir": str(out_dir),
        "plan_dir": str(plan_root),
        "execute": bool(getattr(ns, "execute", False)),
        "device": getattr(ns, "device", None),
        "num_workers": int(getattr(ns, "num_workers", 0)),
        "baseline_factor_dim": int(getattr(ns, "baseline_factor_dim", 3)),
        "baseline_ridge": float(getattr(ns, "baseline_ridge", 1e-4)),
        "baseline_min_history": int(getattr(ns, "baseline_min_history", 40)),
        "seeds": seeds,
        "bundles": bundles,
        "notes": [
            "This plan applies only parameters currently supported by ivdyn training config.",
            "By default, runs include price supervision when joint lambdas are > 0; use --surface-dynamics-only to force surface-only.",
            "Primary forecast baseline is tree-boosted next-day surface mapping, with persistence retained for reference.",
            "Unsupported suggestions are recorded per bundle in `unsupported_suggestions`.",
            "For front-tenor changes, rebuild dataset with custom --tenor-days if needed.",
        ],
    }
    manifest_path = plan_root / "plan_manifest.json"
    manifest_path.write_text(json.dumps(plan_manifest, indent=2), encoding="utf-8")
    print(f"Plan manifest: {manifest_path}")

    if not bool(getattr(ns, "execute", False)):
        print("Dry-run only. Re-run with --execute to train/evaluate all bundles.")
        return

    from ivdyn.eval import evaluate
    from ivdyn.training import train

    jsonl_path = plan_root / "runs.jsonl"
    summary_json_path = plan_root / "summary.json"
    summary_csv_path = plan_root / "summary.csv"

    rows: list[dict[str, Any]] = []
    metric_keys = _metric_subset_keys()

    total = len(seeds) * len(bundles)
    i = 0
    for seed in seeds:
        for bundle in bundles:
            i += 1
            name = str(bundle["name"])
            overrides = dict(bundle.get("overrides", {}))
            unsupported = list(bundle.get("unsupported_suggestions", []))
            started = time.time()
            row: dict[str, Any] = {
                "index": i,
                "total": total,
                "bundle": name,
                "seed": int(seed),
                "status": "failed",
                "run_dir": "",
                "eval_dir": "",
                "duration_sec": 0.0,
                "overrides": overrides,
                "unsupported_suggestions": unsupported,
                "error": "",
                "metrics": {},
            }
            print(f"[{i}/{total}] bundle={name} seed={seed} starting...")

            try:
                cfg = _build_training_config_from_ns(ns, seed=int(seed), out_dir=out_dir)
                for key, value in overrides.items():
                    if not hasattr(cfg, key):
                        raise RuntimeError(f"Unsupported TrainingConfig field `{key}` in bundle `{name}`.")
                    setattr(cfg, key, value)

                run_dir = train(dataset, cfg).resolve()
                eval_dir = evaluate(
                    run_dir=run_dir,
                    dataset_path=dataset,
                    device=getattr(ns, "device", None),
                    num_workers=int(getattr(ns, "num_workers", 0)),
                    surface_dynamics_only=not bool(getattr(ns, "include_contract_metrics", False)),
                    baseline_factor_dim=int(getattr(ns, "baseline_factor_dim", 3)),
                    baseline_ridge=float(getattr(ns, "baseline_ridge", 1e-4)),
                    baseline_min_history=int(getattr(ns, "baseline_min_history", 40)),
                ).resolve()

                metrics_path = eval_dir / "metrics.json"
                metrics = {}
                if metrics_path.exists():
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                subset = _metric_subset(metrics)

                row["status"] = "ok"
                row["run_dir"] = str(run_dir)
                row["eval_dir"] = str(eval_dir)
                row["metrics"] = subset
                print(f"[{i}/{total}] bundle={name} seed={seed} complete.")
            except Exception as exc:
                row["error"] = f"{exc}\n\n{traceback.format_exc()}"
                print(f"[{i}/{total}] bundle={name} seed={seed} failed: {exc}")
            finally:
                row["duration_sec"] = round(float(time.time() - started), 3)

            rows.append(row)
            with jsonl_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(row) + "\n")

    summary_payload = {
        "manifest_path": str(manifest_path),
        "jsonl_path": str(jsonl_path),
        "rows": rows,
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    fieldnames = [
        "index",
        "total",
        "bundle",
        "seed",
        "status",
        "duration_sec",
        "run_dir",
        "eval_dir",
        "error",
        "overrides",
        "unsupported_suggestions",
    ] + metric_keys

    with summary_csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat: dict[str, Any] = {
                "index": row.get("index"),
                "total": row.get("total"),
                "bundle": row.get("bundle"),
                "seed": row.get("seed"),
                "status": row.get("status"),
                "duration_sec": row.get("duration_sec"),
                "run_dir": row.get("run_dir"),
                "eval_dir": row.get("eval_dir"),
                "error": row.get("error"),
                "overrides": json.dumps(row.get("overrides", {}), sort_keys=True),
                "unsupported_suggestions": json.dumps(row.get("unsupported_suggestions", [])),
            }
            metrics = row.get("metrics", {})
            if isinstance(metrics, dict):
                for key in metric_keys:
                    flat[key] = metrics.get(key)
            writer.writerow(flat)

    ok_runs = sum(1 for row in rows if row.get("status") == "ok")
    print(f"Plan complete: {ok_runs}/{len(rows)} successful.")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary CSV:  {summary_csv_path}")


def _ui_command(ns: Any) -> None:
    env = os.environ.copy()
    if ns.run_dir:
        env["IVDYN_DEFAULT_RUN_DIR"] = _to_path(ns.run_dir).resolve().as_posix()
    app_path = Path(__file__).resolve().parent.parent / "ui" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    rc = subprocess.call(cmd, env=env)
    raise SystemExit(rc)


def _resolve_massive_api_key(explicit: str | None) -> str:
    api_key = explicit or os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Set MASSIVE_API_KEY (or POLYGON_API_KEY) or pass --api-key.")
    return api_key


def _normalize_flatfiles_prefix(bucket: str, prefix: str) -> str:
    p = str(prefix).strip().lstrip("/")
    b = str(bucket).strip().strip("/")
    if p.startswith(f"{b}/"):
        p = p[len(b) + 1 :]
    return p.strip("/")


def _resolve_flatfiles_credentials(ns: Any) -> dict[str, str | None]:
    access_key = ns.access_key or os.environ.get("MASSIVE_FLATFILES_ACCESS_KEY") or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = ns.secret_key or os.environ.get("MASSIVE_FLATFILES_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    session_token = ns.session_token or os.environ.get("AWS_SESSION_TOKEN")

    if not access_key or not secret_key:
        raise RuntimeError(
            "Missing flatfiles credentials. Set MASSIVE_FLATFILES_ACCESS_KEY and "
            "MASSIVE_FLATFILES_SECRET_ACCESS_KEY (or pass --access-key / --secret-key)."
        )

    endpoint_url = (
        ns.endpoint_url
        or os.environ.get("MASSIVE_FLATFILES_ENDPOINT_URL")
        or "https://files.massive.com"
    )
    bucket = ns.bucket or os.environ.get("MASSIVE_FLATFILES_BUCKET") or "flatfiles"
    prefix = ns.prefix or os.environ.get("MASSIVE_FLATFILES_PREFIX") or "us_options_opra/day_aggs_v1"
    prefix = _normalize_flatfiles_prefix(bucket=bucket, prefix=prefix)

    return {
        "access_key": access_key,
        "secret_key": secret_key,
        "session_token": session_token,
        "endpoint_url": endpoint_url,
        "bucket": bucket,
        "prefix": prefix,
    }


def _flatfile_key(prefix: str, asof: date) -> str:
    return f"{prefix}/{asof.year:04d}/{asof.month:02d}/{asof.isoformat()}.csv.gz"


def _pull_flatfiles_command(ns: Any) -> None:
    import json

    import boto3
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError

    cfg = _resolve_flatfiles_credentials(ns)
    data_root = _to_path(ns.data_root)
    out_root = data_root / "options_source"
    out_root.mkdir(parents=True, exist_ok=True)

    session = boto3.Session(
        aws_access_key_id=str(cfg["access_key"]),
        aws_secret_access_key=str(cfg["secret_key"]),
        aws_session_token=str(cfg["session_token"]) if cfg["session_token"] else None,
    )
    s3 = session.client(
        "s3",
        endpoint_url=str(cfg["endpoint_url"]),
        config=Config(signature_version="s3v4"),
    )

    all_days = _iter_weekdays(ns.start_date, ns.end_date)
    if ns.max_days > 0:
        all_days = all_days[: ns.max_days]

    downloaded = 0
    skipped_existing = 0
    missing_or_error = 0
    failures: list[dict[str, str]] = []

    prefix = str(cfg["prefix"])
    bucket = str(cfg["bucket"])
    for asof in all_days:
        rel_key = _flatfile_key(prefix=prefix, asof=asof)
        local_path = out_root / rel_key
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists() and not ns.overwrite:
            skipped_existing += 1
            continue

        try:
            s3.download_file(bucket, rel_key, str(local_path))
            downloaded += 1
        except (ClientError, BotoCoreError, OSError) as exc:
            missing_or_error += 1
            failures.append(
                {
                    "date": asof.isoformat(),
                    "key": rel_key,
                    "error": str(exc),
                }
            )
            if ns.fail_fast:
                raise RuntimeError(f"Failed to download {rel_key}: {exc}") from exc

    summary = {
        "source": "massive_flatfiles_s3",
        "endpoint_url": cfg["endpoint_url"],
        "bucket": bucket,
        "prefix": prefix,
        "start_date": ns.start_date,
        "end_date": ns.end_date,
        "trading_days_considered": len(all_days),
        "downloaded": downloaded,
        "skipped_existing": skipped_existing,
        "missing_or_error": missing_or_error,
        "output_root": str(out_root.resolve()),
        "failures_sample": failures[:25],
    }
    summary_path = out_root / f"_flatfile_pull_summary_{utc_timestamp()}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


def _pull_options_symbol_command(ns: Any) -> None:
    import json

    import pandas as pd

    from ivdyn.data.massive import parse_opra_ticker
    from ivdyn.data.schemas import normalize_chain_df

    symbol = str(ns.symbol).upper().strip()
    if not symbol:
        raise RuntimeError("Symbol is required.")

    data_root = _to_path(ns.data_root)
    if ns.source_root:
        source_root = _to_path(ns.source_root)
    else:
        source_prefix = ns.source_prefix or os.environ.get("MASSIVE_FLATFILES_PREFIX") or "us_options_opra/day_aggs_v1"
        source_prefix = _normalize_flatfiles_prefix(bucket="flatfiles", prefix=source_prefix)
        source_root = data_root / "options_source" / source_prefix

    out_raw_root = data_root / "symbols" / symbol / "options" / "raw"
    out_raw_root.mkdir(parents=True, exist_ok=True)

    underlying_path = data_root / "symbols" / symbol / "underlying" / f"{symbol.lower()}_eod.parquet"
    spot_lookup: dict[date, float] = {}
    if underlying_path.exists():
        udf = pd.read_parquet(underlying_path)
        if "date" in udf.columns and "close" in udf.columns:
            udf = udf.copy()
            udf["date"] = pd.to_datetime(udf["date"], errors="coerce").dt.date
            close = pd.to_numeric(udf["close"], errors="coerce")
            mask = udf["date"].notna() & close.notna()
            spot_lookup = dict(zip(udf.loc[mask, "date"], close.loc[mask], strict=False))
    if not spot_lookup and not ns.allow_missing_underlying:
        raise RuntimeError(
            f"No usable underlying closes found at {underlying_path}. "
            "Run pull-underlying-massive first or pass --allow-missing-underlying."
        )

    all_days = _iter_weekdays(ns.start_date, ns.end_date)
    if ns.max_days > 0:
        all_days = all_days[: ns.max_days]

    saved_days = 0
    skipped_existing = 0
    missing_source = 0
    zero_rows = 0
    total_rows_saved = 0

    for asof in all_days:
        day_str = asof.isoformat()
        source_path = source_root / f"{asof.year:04d}" / f"{asof.month:02d}" / f"{day_str}.csv.gz"
        out_path = out_raw_root / f"{day_str}.parquet"
        meta_path = out_raw_root / f"{day_str}.metadata.json"

        if out_path.exists() and not ns.overwrite:
            skipped_existing += 1
            continue

        if not source_path.exists():
            missing_source += 1
            meta = {
                "symbol": symbol,
                "as_of": day_str,
                "source_path": str(source_path),
                "rows_source": 0,
                "rows_symbol_raw": 0,
                "rows_saved": 0,
                "missing_source": True,
                "path": None,
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            continue

        raw = pd.read_csv(source_path, compression="gzip")
        if "ticker" not in raw.columns:
            meta = {
                "symbol": symbol,
                "as_of": day_str,
                "source_path": str(source_path),
                "rows_source": int(len(raw)),
                "rows_symbol_raw": 0,
                "rows_saved": 0,
                "missing_source": False,
                "error": "missing ticker column",
                "path": None,
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            zero_rows += 1
            continue

        ticker = raw["ticker"].astype(str)
        scoped = raw.loc[ticker.str.startswith(f"O:{symbol}", na=False)].copy()
        if scoped.empty:
            rows_saved = 0
            norm = pd.DataFrame()
            rows_symbol_raw = 0
        else:
            parsed = scoped["ticker"].astype(str).apply(parse_opra_ticker)
            mask = parsed.notna()
            scoped = scoped.loc[mask].copy()
            parsed = parsed.loc[mask]

            if not scoped.empty:
                under = parsed.apply(lambda x: x["underlying"])  # type: ignore[index]
                scoped = scoped.loc[under == symbol].copy()
                parsed = parsed.loc[scoped.index]

            if scoped.empty:
                rows_saved = 0
                norm = pd.DataFrame()
                rows_symbol_raw = 0
            else:
                expiry = parsed.apply(lambda x: x["expiry"])  # type: ignore[index]
                dte = parsed.apply(lambda x: (x["expiry"] - asof).days)  # type: ignore[index]
                call_put = parsed.apply(lambda x: x["call_put"])  # type: ignore[index]
                strike = parsed.apply(lambda x: x["strike"])  # type: ignore[index]

                mid = pd.to_numeric(scoped.get("close"), errors="coerce")
                volume = pd.to_numeric(scoped.get("volume"), errors="coerce")
                open_interest = pd.to_numeric(scoped.get("open_interest"), errors="coerce")
                spot = float(spot_lookup.get(asof, float("nan")))

                day_df = pd.DataFrame(
                    {
                        "date": asof,
                        "expiry": expiry,
                        "dte": dte,
                        "call_put": call_put,
                        "symbol": scoped["ticker"].astype(str),
                        "strike": strike,
                        "bid": pd.NA,
                        "ask": pd.NA,
                        "mid": mid,
                        "last": mid,
                        "volume": volume,
                        "open_interest": open_interest,
                        "underlying_close": spot,
                        "delta": pd.NA,
                        "gamma": pd.NA,
                        "theta": pd.NA,
                        "vega": pd.NA,
                        "iv": pd.NA,
                    }
                )
                rows_symbol_raw = int(len(day_df))
                norm = normalize_chain_df(day_df, asof=asof, symbol=symbol)
                rows_saved = int(len(norm))

        if rows_saved > 0:
            norm.to_parquet(out_path, index=False)
            saved_days += 1
            total_rows_saved += rows_saved
        else:
            if out_path.exists() and ns.overwrite:
                out_path.unlink()
            zero_rows += 1

        meta = {
            "symbol": symbol,
            "as_of": day_str,
            "source_path": str(source_path),
            "rows_source": int(len(raw)),
            "rows_symbol_raw": int(rows_symbol_raw),
            "rows_saved": int(rows_saved),
            "missing_source": False,
            "path": str(out_path.resolve()) if rows_saved > 0 else None,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    summary = {
        "symbol": symbol,
        "start_date": ns.start_date,
        "end_date": ns.end_date,
        "source_root": str(source_root),
        "output_root": str(out_raw_root),
        "trading_days_considered": len(all_days),
        "saved_days": saved_days,
        "skipped_existing_days": skipped_existing,
        "missing_source_days": missing_source,
        "zero_row_days": zero_rows,
        "rows_saved_total": total_rows_saved,
    }
    summary_path = out_raw_root / f"_symbol_pull_summary_{utc_timestamp()}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


def _pull_underlying_massive_command(ns: Any) -> None:
    import json

    import pandas as pd
    import requests

    symbol = str(ns.symbol).upper().strip()
    if not symbol:
        raise RuntimeError("Symbol is required.")

    api_key = _resolve_massive_api_key(ns.api_key)
    base_url = str(ns.base_url).rstrip("/")

    out_dir = _to_path(ns.data_root) / "symbols" / symbol / "underlying"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol.lower()}_eod.parquet"
    meta_path = out_dir / f"{symbol.lower()}_eod.metadata.json"

    url = f"{base_url}/v2/aggs/ticker/{symbol}/range/1/day/{ns.start_date}/{ns.end_date}"
    params: dict[str, Any] | None = {
        "adjusted": "true" if ns.adjusted else "false",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }

    rows: list[dict[str, Any]] = []
    pages = 0
    while url:
        resp = requests.get(url, params=params, timeout=ns.timeout_seconds)
        params = None
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            snippet = resp.text[:500]
            raise RuntimeError(f"Massive request failed ({resp.status_code}): {snippet}") from exc

        payload = resp.json()
        batch = payload.get("results", [])
        if isinstance(batch, list):
            rows.extend(batch)

        pages += 1
        if ns.max_pages > 0 and pages >= ns.max_pages:
            break

        next_url = payload.get("next_url")
        if next_url and "apiKey=" not in str(next_url):
            sep = "&" if "?" in str(next_url) else "?"
            next_url = f"{next_url}{sep}apiKey={api_key}"
        url = next_url

    if not rows:
        raise RuntimeError(
            "No underlying bars returned for the requested range. "
            "Check symbol/date range and API permissions."
        )

    raw = pd.DataFrame(rows)
    if "t" not in raw.columns or "c" not in raw.columns:
        raise RuntimeError("Unexpected Massive response schema: expected fields 't' and 'c'.")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["t"], unit="ms", utc=True).dt.date,
            "open": pd.to_numeric(raw.get("o"), errors="coerce"),
            "high": pd.to_numeric(raw.get("h"), errors="coerce"),
            "low": pd.to_numeric(raw.get("l"), errors="coerce"),
            "close": pd.to_numeric(raw.get("c"), errors="coerce"),
            "volume": pd.to_numeric(raw.get("v"), errors="coerce"),
            "vwap": pd.to_numeric(raw.get("vw"), errors="coerce"),
            "trades": pd.to_numeric(raw.get("n"), errors="coerce"),
            "timestamp_ms": pd.to_numeric(raw.get("t"), errors="coerce"),
        }
    )
    out = out.dropna(subset=["date", "close"])
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    if out_path.exists() and not ns.overwrite:
        prev = pd.read_parquet(out_path)
        if not prev.empty and "date" in prev.columns:
            prev = prev.copy()
            prev["date"] = pd.to_datetime(prev["date"]).dt.date
            out = pd.concat([prev, out], ignore_index=True)
            out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    out.to_parquet(out_path, index=False)

    meta = {
        "symbol": symbol,
        "source": "massive_v2_aggs",
        "base_url": base_url,
        "start_date": ns.start_date,
        "end_date": ns.end_date,
        "adjusted": bool(ns.adjusted),
        "rows": int(len(out)),
        "path": str(out_path.resolve()),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(out_path)


def _iter_weekdays(start_date: str, end_date: str) -> list[date]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if end < start:
        raise RuntimeError(f"Invalid date range: start-date {start_date} is after end-date {end_date}.")

    out: list[date] = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def _pull_massive_command(ns: Any) -> None:
    import json

    import pandas as pd
    import requests

    from ivdyn.data.schemas import normalize_chain_df

    symbol = str(ns.symbol).upper().strip()
    if not symbol:
        raise RuntimeError("Symbol is required.")

    api_key = _resolve_massive_api_key(ns.api_key)
    base_url = str(ns.base_url).rstrip("/")

    out_dir = _to_path(ns.data_root) / "symbols" / symbol / "options" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    weekdays = _iter_weekdays(ns.start_date, ns.end_date)
    saved_days = 0
    skipped_existing = 0
    empty_days = 0
    total_rows = 0

    for asof in weekdays:
        day_str = asof.isoformat()
        day_path = out_dir / f"{day_str}.parquet"
        day_meta_path = out_dir / f"{day_str}.metadata.json"

        if day_path.exists() and not ns.overwrite:
            skipped_existing += 1
            continue

        url = f"{base_url}/v3/snapshot/options/{symbol}"
        params: dict[str, Any] | None = {
            "as_of": day_str,
            "limit": int(ns.limit),
            "sort": "ticker",
            "order": "asc",
            "expired": "true" if ns.include_expired else "false",
            "apiKey": api_key,
        }

        page_count = 0
        chain_rows: list[dict[str, Any]] = []
        while url:
            resp = requests.get(url, params=params, timeout=ns.timeout_seconds)
            params = None
            try:
                resp.raise_for_status()
            except requests.HTTPError as exc:
                snippet = resp.text[:500]
                raise RuntimeError(f"Massive options request failed on {day_str} ({resp.status_code}): {snippet}") from exc

            payload = resp.json()
            results = payload.get("results", [])
            if isinstance(results, list):
                for row in results:
                    details = row.get("details", {}) if isinstance(row, dict) else {}
                    quote = row.get("last_quote", {}) if isinstance(row, dict) else {}
                    trade = row.get("last_trade", {}) if isinstance(row, dict) else {}
                    greeks = row.get("greeks", {}) if isinstance(row, dict) else {}
                    day = row.get("day", {}) if isinstance(row, dict) else {}
                    under = row.get("underlying_asset", {}) if isinstance(row, dict) else {}

                    bid = quote.get("bid_price")
                    ask = quote.get("ask_price")
                    mid = None
                    if bid is not None and ask is not None:
                        try:
                            mid = (float(bid) + float(ask)) / 2.0
                        except Exception:
                            mid = None

                    chain_rows.append(
                        {
                            "date": day_str,
                            "expiry": details.get("expiration_date"),
                            "call_put": str(details.get("contract_type", "")).upper()[:1],
                            "symbol": details.get("ticker"),
                            "strike": details.get("strike_price"),
                            "bid": bid,
                            "ask": ask,
                            "mid": mid,
                            "last": trade.get("price"),
                            "volume": day.get("volume"),
                            "open_interest": row.get("open_interest"),
                            "underlying_close": under.get("price"),
                            "delta": greeks.get("delta"),
                            "gamma": greeks.get("gamma"),
                            "theta": greeks.get("theta"),
                            "vega": greeks.get("vega"),
                            "iv": row.get("implied_volatility"),
                        }
                    )

            page_count += 1
            if ns.max_pages > 0 and page_count >= ns.max_pages:
                break

            next_url = payload.get("next_url")
            if next_url and "apiKey=" not in str(next_url):
                sep = "&" if "?" in str(next_url) else "?"
                next_url = f"{next_url}{sep}apiKey={api_key}"
            url = next_url

        if chain_rows:
            norm = normalize_chain_df(pd.DataFrame(chain_rows), asof=asof, symbol=symbol)
        else:
            norm = pd.DataFrame()

        rows_saved = int(len(norm))
        if rows_saved > 0:
            norm.to_parquet(day_path, index=False)
            saved_days += 1
            total_rows += rows_saved
        else:
            if day_path.exists() and ns.overwrite:
                day_path.unlink()
            empty_days += 1

        meta = {
            "symbol": symbol,
            "as_of": day_str,
            "source": "massive_v3_snapshot_options",
            "base_url": base_url,
            "include_expired": bool(ns.include_expired),
            "rows_raw": int(len(chain_rows)),
            "rows_saved": rows_saved,
            "pages": page_count,
            "path": str(day_path.resolve()) if rows_saved > 0 else None,
        }
        day_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    summary = {
        "symbol": symbol,
        "start_date": ns.start_date,
        "end_date": ns.end_date,
        "trading_days_considered": len(weekdays),
        "saved_days": saved_days,
        "skipped_existing_days": skipped_existing,
        "empty_days": empty_days,
        "rows_saved_total": total_rows,
    }
    summary_path = out_dir / f"_massive_pull_summary_{utc_timestamp()}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", default="outputs/runs")
    p.add_argument(
        "--per-symbol",
        action="store_true",
        default=False,
        help="Train one model per symbol (separate runs) from a multi-asset dataset.",
    )
    p.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Optional symbol subset for --per-symbol mode.",
    )
    p.add_argument(
        "--per-symbol-dataset-dir",
        default=None,
        help="Optional directory for generated per-symbol dataset slices.",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument(
        "--split-mode",
        choices=["by_asset_time", "global_time"],
        default="by_asset_time",
        help="Date split mode: by_asset_time uses per-asset chronology, global_time uses shared calendar chronology.",
    )
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--vae-hidden-dim-1", type=int, default=384)
    p.add_argument("--vae-hidden-dim-2", type=int, default=192)
    p.add_argument("--dynamics-hidden-dim-1", type=int, default=256)
    p.add_argument("--dynamics-hidden-dim-2", type=int, default=128)
    p.add_argument("--pricing-hidden-dim-1", type=int, default=256)
    p.add_argument("--pricing-hidden-dim-2", type=int, default=128)
    p.add_argument("--execution-hidden-dim-1", type=int, default=192)
    p.add_argument("--execution-hidden-dim-2", type=int, default=96)
    p.add_argument("--model-dropout", type=float, default=0.08)
    p.add_argument("--vae-epochs", type=int, default=120)
    p.add_argument("--vae-batch-size", type=int, default=32)
    p.add_argument("--vae-lr", type=float, default=2e-3)
    p.add_argument("--vae-kl-beta", type=float, default=0.02)
    p.add_argument("--kl-warmup-epochs", type=int, default=20)
    p.add_argument("--noarb-lambda", type=float, default=0.01)
    p.add_argument("--noarb-butterfly-lambda", type=float, default=0.005)
    p.add_argument("--head-epochs", type=int, default=130)
    p.add_argument("--dyn-batch-size", type=int, default=64)
    p.add_argument("--contract-batch-size", type=int, default=2048)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--rollout-steps", type=int, default=3)
    p.add_argument(
        "--rollout-random-horizon",
        dest="rollout_random_horizon",
        action="store_true",
        default=True,
        help="Sample rollout horizon uniformly from [rollout_min_steps, rollout_steps] each train batch/epoch.",
    )
    p.add_argument(
        "--no-rollout-random-horizon",
        dest="rollout_random_horizon",
        action="store_false",
        help="Disable random rollout horizon and always use fixed --rollout-steps.",
    )
    p.add_argument(
        "--rollout-min-steps",
        type=int,
        default=1,
        help="Minimum horizon used when --rollout-random-horizon is enabled.",
    )
    p.add_argument(
        "--rollout-teacher-forcing-start",
        type=float,
        default=0.35,
        help="Teacher-forcing probability at epoch 1 for rollout dynamics training.",
    )
    p.add_argument(
        "--rollout-teacher-forcing-end",
        type=float,
        default=0.10,
        help="Teacher-forcing probability at final epoch for rollout dynamics training.",
    )
    p.add_argument("--rollout-surface-lambda", type=float, default=0.65)
    p.add_argument("--rollout-calendar-lambda", type=float, default=0.03)
    p.add_argument("--rollout-butterfly-lambda", type=float, default=0.02)
    p.add_argument("--rollout-surface-huber-beta", type=float, default=0.015)
    p.add_argument("--recon-huber-beta", type=float, default=0.015)
    p.add_argument("--rollout-slope-lambda", type=float, default=0.0)
    p.add_argument("--rollout-curvature-lambda", type=float, default=0.0)
    p.add_argument("--surface-weight-liq-alpha", type=float, default=0.0)
    p.add_argument("--surface-weight-spread-alpha", type=float, default=0.0)
    p.add_argument("--surface-weight-vega-alpha", type=float, default=0.0)
    p.add_argument("--surface-weight-clip-min", type=float, default=1.0)
    p.add_argument("--surface-weight-clip-max", type=float, default=4.0)
    p.add_argument(
        "--surface-focus-alpha",
        type=float,
        default=1.25,
        help="Extra weighting strength for high positive moneyness + low DTE region (0 disables).",
    )
    p.add_argument(
        "--surface-focus-x-min",
        type=float,
        default=0.10,
        help="Moneyness threshold x=ln(K/S) where focus weighting starts to activate.",
    )
    p.add_argument(
        "--surface-focus-x-scale",
        type=float,
        default=0.03,
        help="Smoothness scale for the positive-moneyness focus transition.",
    )
    p.add_argument(
        "--surface-focus-dte-scale-days",
        type=float,
        default=21.0,
        help="Decay scale (days) for short-tenor focus weighting.",
    )
    p.add_argument(
        "--surface-focus-dte-max-days",
        type=float,
        default=30.0,
        help="Hard DTE ceiling for focus region (only points with DTE < this value are emphasized).",
    )
    p.add_argument(
        "--surface-focus-neg-x-max",
        type=float,
        default=-0.20,
        help="Negative-moneyness boundary for second focus wing (points with x < this value are emphasized).",
    )
    p.add_argument(
        "--surface-focus-neg-weight-ratio",
        type=float,
        default=0.35,
        help="Relative strength of negative-moneyness focus wing vs primary positive wing.",
    )
    p.add_argument(
        "--surface-focus-density-alpha",
        type=float,
        default=0.0,
        help="Asset-specific focus-density strength. Requires --surface-focus-density-map when > 0.",
    )
    p.add_argument(
        "--surface-focus-density-map",
        default=None,
        help=(
            "Path to strict JSON focus-density map keyed by asset symbol. "
            "Each value must be an (nx x nt) positive grid (or flat nx*nt vector). "
            "Applied to train dates only; missing/extra keys raise errors."
        ),
    )
    p.add_argument(
        "--adaptive-focus-rerun",
        dest="adaptive_focus_rerun",
        action="store_true",
        default=False,
        help=(
            "Two-pass training: fit stage-1 model, build focus-density map from train+val "
            "absolute RMSE, retrain stage-2 with that map, then evaluate on test."
        ),
    )
    p.add_argument(
        "--no-adaptive-focus-rerun",
        dest="adaptive_focus_rerun",
        action="store_false",
        help="Disable adaptive two-pass focus rerun and run a single-pass training.",
    )
    p.add_argument(
        "--adaptive-focus-density-alpha",
        type=float,
        default=0.0,
        help="Density-map weighting strength used in stage-2 when --adaptive-focus-rerun is enabled.",
    )
    p.add_argument(
        "--adaptive-focus-error-power",
        type=float,
        default=1.0,
        help="Power transform applied to train+val RMSE grid before normalization when building the density map.",
    )
    p.add_argument(
        "--adaptive-focus-map-path",
        default=None,
        help="Optional explicit output path for generated adaptive focus map JSON.",
    )
    p.add_argument("--joint-epochs", type=int, default=120)
    p.add_argument("--joint-lr", type=float, default=5e-4)
    p.add_argument("--joint-contract-batch-size", type=int, default=4096)
    p.add_argument("--joint-dyn-lambda", type=float, default=1.0)
    p.add_argument("--joint-price-lambda", type=float, default=0.15)
    p.add_argument("--joint-exec-lambda", type=float, default=0.0)
    p.add_argument(
        "--joint-use-mu-deterministic",
        action="store_true",
        default=True,
        help="Use encoder mean (mu) instead of sampling z during joint stage dynamics updates.",
    )
    p.add_argument(
        "--joint-use-z-sampling",
        dest="joint_use_mu_deterministic",
        action="store_false",
        help="Override default and use sampled z (stochastic) during joint stage dynamics updates.",
    )
    p.add_argument(
        "--enable-price-exec-heads",
        action="store_true",
        default=False,
        help="Legacy opt-in flag; pricing/execution heads are auto-enabled when joint lambdas are > 0.",
    )
    p.add_argument(
        "--surface-dynamics-only",
        action="store_true",
        default=False,
        help="Force surface-only training by disabling price/execution heads.",
    )
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--price-risk-weight", type=float, default=1.0)
    p.add_argument("--exec-risk-weight", type=float, default=0.5)
    p.add_argument("--price-spread-inv-lambda", type=float, default=0.35)
    p.add_argument("--price-spread-clip-min", type=float, default=0.02)
    p.add_argument("--price-spread-clip-max", type=float, default=3.0)
    p.add_argument("--price-vega-power", type=float, default=0.25)
    p.add_argument("--price-vega-cap", type=float, default=4.0)
    p.add_argument("--risk-focus-abs-x", type=float, default=0.06)
    p.add_argument("--risk-focus-tau-days", type=float, default=20.0)
    p.add_argument("--exec-label-smoothing", type=float, default=0.03)
    p.add_argument("--exec-logit-l2", type=float, default=2e-4)
    p.add_argument(
        "--context-winsor-quantile",
        type=float,
        default=0.01,
        help="Winsorization quantile for context scaling (e.g., 0.01 clips to 1%%-99%%).",
    )
    p.add_argument(
        "--context-z-clip",
        type=float,
        default=5.0,
        help="Clip scaled context features to [-z, z]; <=0 disables z-clip.",
    )
    p.add_argument(
        "--disable-context-augment",
        action="store_true",
        default=False,
        help="Disable augmentation of context with per-day aggregates from contract minute features.",
    )
    p.add_argument(
        "--disable-surface-history-augment",
        action="store_true",
        default=False,
        help="Disable augmentation of context with lagged per-asset surface state summaries.",
    )
    p.add_argument(
        "--disable-dynamics-residual",
        action="store_true",
        default=False,
        help="Disable residual latent dynamics (z_next = z_prev + f(...)).",
    )
    p.add_argument("--asset-embed-dim", type=int, default=8)
    p.add_argument("--surface-refiner-hidden-1", type=int, default=256)
    p.add_argument("--surface-refiner-hidden-2", type=int, default=128)
    p.add_argument(
        "--disable-surface-refiner",
        action="store_true",
        default=False,
        help="Disable context/surface-conditioned residual refiner for one-step surface forecasts.",
    )
    p.add_argument("--early-stop-patience", type=int, default=20)
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    p.add_argument("--lr-plateau-patience", type=int, default=6)
    p.add_argument("--lr-plateau-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument(
        "--max-cpu-threads",
        type=int,
        default=2,
        help="Cap CPU threads used by training.",
    )
    p.add_argument(
        "--model-arch",
        choices=["tree_boost", "option_a_pca_tcn"],
        default="tree_boost",
        help="Training architecture. `tree_boost` keeps current baseline; `option_a_pca_tcn` uses shared neural factor dynamics.",
    )
    p.add_argument("--option-a-seq-len", type=int, default=20, help="History length L for Option-A sequence model.")
    p.add_argument("--option-a-epochs", type=int, default=80)
    p.add_argument("--option-a-batch-size", type=int, default=128)
    p.add_argument("--option-a-eval-batch-size", type=int, default=512)
    p.add_argument("--option-a-lr", type=float, default=8e-4)
    p.add_argument("--option-a-weight-decay", type=float, default=1e-5)
    p.add_argument("--option-a-hidden-dim", type=int, default=192)
    p.add_argument("--option-a-tcn-layers", type=int, default=4)
    p.add_argument("--option-a-tcn-kernel-size", type=int, default=3)
    p.add_argument("--option-a-dropout", type=float, default=0.08)
    p.add_argument("--option-a-early-stop-patience", type=int, default=12)
    p.add_argument("--option-a-blend-alpha-min", type=float, default=0.6)
    p.add_argument("--option-a-blend-alpha-max", type=float, default=1.4)
    p.add_argument("--option-a-blend-alpha-steps", type=int, default=9)
    p.add_argument("--option-a-device", default="auto")
    p.set_defaults(func=_train_command)

    p = sub.add_parser("build-dataset", help="Legacy Massive-based dataset builder.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--plugin", default="massive_raw_parquet")
    p.add_argument("--api-key")
    p.add_argument("--start-date")
    p.add_argument("--end-date")
    p.add_argument(
        "--x-grid",
        nargs="+",
        type=float,
        default=[-0.35, -0.30, -0.25, -0.20, -0.16, -0.12, -0.09, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.09, 0.12, 0.16, 0.20, 0.25, 0.30, 0.35],
    )
    p.add_argument("--tenor-days", nargs="+", type=int, default=[7, 14, 30, 60, 90, 180])
    p.add_argument("--max-contracts-per-day", type=int, default=900)
    p.add_argument("--random-seed", type=int, default=7)
    p.add_argument("--num-workers", type=int, default=0)
    p.set_defaults(func=_build_dataset_command)

    p = sub.add_parser("extract-wrds")
    p.add_argument("--options-path", default="Options_chain.csv.gz")
    p.add_argument("--underlying-path", default="Underlying.csv.gz")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--symbols", nargs="+", default=None, help="Optional ticker filter list, e.g. AAPL MSFT NVDA.")
    p.add_argument("--forward-path", default=None, help="Optional forward curve CSV (secid,date,expiration,ForwardPrice).")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--chunksize", type=int, default=300000)
    p.add_argument("--min-iv", type=float, default=1e-4)
    p.add_argument("--max-iv", type=float, default=4.0)
    p.set_defaults(func=_extract_wrds_command)

    p = sub.add_parser("build-surface-dataset")
    p.add_argument("--contracts-path", required=True, help="Parquet from `extract-wrds` (contracts_daily.parquet).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--earnings-path", default=None, help="Optional earnings CSV (e.g. Earnings.csv).")
    p.add_argument("--symbols", nargs="+", default=None, help="Optional ticker subset.")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument(
        "--x-grid",
        nargs="+",
        type=float,
        default=[-0.35, -0.30, -0.25, -0.20, -0.16, -0.12, -0.09, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.09, 0.12, 0.16, 0.20, 0.25, 0.30, 0.35],
    )
    p.add_argument("--tenor-days", nargs="+", type=int, default=[7, 14, 30, 60, 90, 180])
    p.add_argument("--max-contracts-per-day", type=int, default=1200)
    p.add_argument("--min-contracts-per-day", type=int, default=80)
    p.add_argument("--max-neighbors", type=int, default=20)
    p.add_argument("--max-dte-distance", type=int, default=30)
    p.add_argument("--max-rel-spread", type=float, default=2.0)
    p.add_argument("--random-seed", type=int, default=7)
    p.add_argument("--num-workers", type=int, default=0, help="Parallel workers for per-asset surface build (0=auto).")
    p.add_argument("--surface-variable", choices=["total_variance", "iv"], default="total_variance")
    p.add_argument("--surface-pca-factors", type=int, default=3)
    p.set_defaults(func=_build_surface_dataset_command)

    p = sub.add_parser("evaluate")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--device")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--include-contract-metrics",
        action="store_true",
        default=False,
        help="Also compute legacy contract-level pricing/execution metrics (surface-only mode is default).",
    )
    p.add_argument("--baseline-factor-dim", type=int, default=3, help="Number of PCA factors for parametric surface baseline.")
    p.add_argument("--baseline-ridge", type=float, default=1e-4, help="Ridge regularization for factor HAR(1,5,22) baseline.")
    p.add_argument("--baseline-min-history", type=int, default=40, help="Minimum history days before parametric baseline fit.")
    p.set_defaults(func=_evaluate_command)

    p = sub.add_parser("experiment-plan")
    p.add_argument("--dataset", required=True, help="Path to dataset.npz used for all plan runs.")
    p.add_argument("--out-dir", default="outputs/runs", help="Root output directory for model runs.")
    p.add_argument(
        "--plan-dir",
        default="outputs/experiment_plans",
        help="Directory where plan manifest and summary artifacts are stored.",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[7], help="Seeds to run for each bundle.")
    p.add_argument("--execute", action="store_true", default=False, help="Run training+evaluation; default is dry-run manifest.")
    p.add_argument("--skip-baseline", action="store_true", default=False, help="Skip baseline bundle.")
    p.add_argument("--device", default=None, help="Optional evaluation device override.")
    p.add_argument("--num-workers", type=int, default=0, help="Evaluation workers (0=auto).")

    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument(
        "--split-mode",
        choices=["by_asset_time", "global_time"],
        default="by_asset_time",
        help="Date split mode: by_asset_time uses per-asset chronology, global_time uses shared calendar chronology.",
    )
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--vae-hidden-dim-1", type=int, default=384)
    p.add_argument("--vae-hidden-dim-2", type=int, default=192)
    p.add_argument("--dynamics-hidden-dim-1", type=int, default=256)
    p.add_argument("--dynamics-hidden-dim-2", type=int, default=128)
    p.add_argument("--pricing-hidden-dim-1", type=int, default=256)
    p.add_argument("--pricing-hidden-dim-2", type=int, default=128)
    p.add_argument("--execution-hidden-dim-1", type=int, default=192)
    p.add_argument("--execution-hidden-dim-2", type=int, default=96)
    p.add_argument("--model-dropout", type=float, default=0.08)
    p.add_argument("--vae-epochs", type=int, default=120)
    p.add_argument("--vae-batch-size", type=int, default=32)
    p.add_argument("--vae-lr", type=float, default=2e-3)
    p.add_argument("--vae-kl-beta", type=float, default=0.02)
    p.add_argument("--kl-warmup-epochs", type=int, default=20)
    p.add_argument("--noarb-lambda", type=float, default=0.01)
    p.add_argument("--noarb-butterfly-lambda", type=float, default=0.005)
    p.add_argument("--head-epochs", type=int, default=130)
    p.add_argument("--dyn-batch-size", type=int, default=64)
    p.add_argument("--contract-batch-size", type=int, default=2048)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--rollout-steps", type=int, default=3)
    p.add_argument(
        "--rollout-random-horizon",
        dest="rollout_random_horizon",
        action="store_true",
        default=True,
        help="Sample rollout horizon uniformly from [rollout_min_steps, rollout_steps] each train batch/epoch.",
    )
    p.add_argument(
        "--no-rollout-random-horizon",
        dest="rollout_random_horizon",
        action="store_false",
        help="Disable random rollout horizon and always use fixed --rollout-steps.",
    )
    p.add_argument("--rollout-min-steps", type=int, default=1)
    p.add_argument("--rollout-teacher-forcing-start", type=float, default=0.35)
    p.add_argument("--rollout-teacher-forcing-end", type=float, default=0.10)
    p.add_argument("--rollout-surface-lambda", type=float, default=0.65)
    p.add_argument("--rollout-calendar-lambda", type=float, default=0.03)
    p.add_argument("--rollout-butterfly-lambda", type=float, default=0.02)
    p.add_argument("--rollout-surface-huber-beta", type=float, default=0.015)
    p.add_argument("--recon-huber-beta", type=float, default=0.015)
    p.add_argument("--rollout-slope-lambda", type=float, default=0.0)
    p.add_argument("--rollout-curvature-lambda", type=float, default=0.0)
    p.add_argument("--surface-weight-liq-alpha", type=float, default=0.0)
    p.add_argument("--surface-weight-spread-alpha", type=float, default=0.0)
    p.add_argument("--surface-weight-vega-alpha", type=float, default=0.0)
    p.add_argument("--surface-weight-clip-min", type=float, default=1.0)
    p.add_argument("--surface-weight-clip-max", type=float, default=4.0)
    p.add_argument("--surface-focus-alpha", type=float, default=1.25)
    p.add_argument("--surface-focus-x-min", type=float, default=0.10)
    p.add_argument("--surface-focus-x-scale", type=float, default=0.03)
    p.add_argument("--surface-focus-dte-scale-days", type=float, default=21.0)
    p.add_argument("--surface-focus-dte-max-days", type=float, default=30.0)
    p.add_argument("--surface-focus-neg-x-max", type=float, default=-0.20)
    p.add_argument("--surface-focus-neg-weight-ratio", type=float, default=0.35)
    p.add_argument("--joint-epochs", type=int, default=120)
    p.add_argument("--joint-lr", type=float, default=5e-4)
    p.add_argument("--joint-contract-batch-size", type=int, default=4096)
    p.add_argument("--joint-dyn-lambda", type=float, default=1.0)
    p.add_argument("--joint-price-lambda", type=float, default=0.15)
    p.add_argument("--joint-exec-lambda", type=float, default=0.0)
    p.add_argument(
        "--joint-use-mu-deterministic",
        dest="joint_use_mu_deterministic",
        action="store_true",
        default=True,
        help="Use encoder mean (mu) instead of sampling z during joint stage dynamics updates.",
    )
    p.add_argument(
        "--joint-use-z-sampling",
        dest="joint_use_mu_deterministic",
        action="store_false",
        help="Override default and use sampled z (stochastic) during joint stage dynamics updates.",
    )
    p.add_argument(
        "--enable-price-exec-heads",
        action="store_true",
        default=False,
        help="Legacy opt-in flag; heads are auto-enabled when joint lambdas are > 0.",
    )
    p.add_argument(
        "--surface-dynamics-only",
        action="store_true",
        default=False,
        help="Force surface-only training by disabling price/execution heads.",
    )
    p.add_argument(
        "--include-contract-metrics",
        action="store_true",
        default=False,
        help="Also compute legacy contract-level pricing/execution metrics during plan evaluation.",
    )
    p.add_argument("--baseline-factor-dim", type=int, default=3, help="Number of PCA factors for parametric surface baseline.")
    p.add_argument("--baseline-ridge", type=float, default=1e-4, help="Ridge regularization for factor HAR(1,5,22) baseline.")
    p.add_argument("--baseline-min-history", type=int, default=40, help="Minimum history days before parametric baseline fit.")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--price-risk-weight", type=float, default=1.0)
    p.add_argument("--exec-risk-weight", type=float, default=0.5)
    p.add_argument("--price-spread-inv-lambda", type=float, default=0.35)
    p.add_argument("--price-spread-clip-min", type=float, default=0.02)
    p.add_argument("--price-spread-clip-max", type=float, default=3.0)
    p.add_argument("--price-vega-power", type=float, default=0.25)
    p.add_argument("--price-vega-cap", type=float, default=4.0)
    p.add_argument("--risk-focus-abs-x", type=float, default=0.06)
    p.add_argument("--risk-focus-tau-days", type=float, default=20.0)
    p.add_argument("--exec-label-smoothing", type=float, default=0.03)
    p.add_argument("--exec-logit-l2", type=float, default=2e-4)
    p.add_argument(
        "--context-winsor-quantile",
        type=float,
        default=0.01,
        help="Winsorization quantile for context scaling (e.g., 0.01 clips to 1%%-99%%).",
    )
    p.add_argument(
        "--context-z-clip",
        type=float,
        default=5.0,
        help="Clip scaled context features to [-z, z]; <=0 disables z-clip.",
    )
    p.add_argument(
        "--disable-context-augment",
        action="store_true",
        default=False,
        help="Disable augmentation of context with per-day aggregates from contract minute features.",
    )
    p.add_argument(
        "--disable-surface-history-augment",
        action="store_true",
        default=False,
        help="Disable augmentation of context with lagged per-asset surface state summaries.",
    )
    p.add_argument(
        "--disable-dynamics-residual",
        action="store_true",
        default=False,
        help="Disable residual latent dynamics (z_next = z_prev + f(...)).",
    )
    p.add_argument("--asset-embed-dim", type=int, default=8)
    p.add_argument("--surface-refiner-hidden-1", type=int, default=256)
    p.add_argument("--surface-refiner-hidden-2", type=int, default=128)
    p.add_argument(
        "--disable-surface-refiner",
        action="store_true",
        default=False,
        help="Disable context/surface-conditioned residual refiner for one-step surface forecasts.",
    )
    p.add_argument("--early-stop-patience", type=int, default=20)
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    p.add_argument("--lr-plateau-patience", type=int, default=6)
    p.add_argument("--lr-plateau-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument(
        "--max-cpu-threads",
        type=int,
        default=2,
        help="Cap CPU threads used by training.",
    )
    p.add_argument(
        "--model-arch",
        choices=["tree_boost", "option_a_pca_tcn"],
        default="tree_boost",
        help="Architecture used for each experiment-plan run.",
    )
    p.add_argument("--option-a-seq-len", type=int, default=20)
    p.add_argument("--option-a-epochs", type=int, default=80)
    p.add_argument("--option-a-batch-size", type=int, default=128)
    p.add_argument("--option-a-eval-batch-size", type=int, default=512)
    p.add_argument("--option-a-lr", type=float, default=8e-4)
    p.add_argument("--option-a-weight-decay", type=float, default=1e-5)
    p.add_argument("--option-a-hidden-dim", type=int, default=192)
    p.add_argument("--option-a-tcn-layers", type=int, default=4)
    p.add_argument("--option-a-tcn-kernel-size", type=int, default=3)
    p.add_argument("--option-a-dropout", type=float, default=0.08)
    p.add_argument("--option-a-early-stop-patience", type=int, default=12)
    p.add_argument("--option-a-blend-alpha-min", type=float, default=0.6)
    p.add_argument("--option-a-blend-alpha-max", type=float, default=1.4)
    p.add_argument("--option-a-blend-alpha-steps", type=int, default=9)
    p.add_argument("--option-a-device", default="auto")
    p.set_defaults(func=_experiment_plan_command)

    p = sub.add_parser("ui")
    p.add_argument("--run-dir", default=None)
    p.set_defaults(func=_ui_command)

    p = sub.add_parser("pull-underlying-massive")
    p.add_argument("--data-root", default="data")
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default="https://api.massive.com")
    p.add_argument("--timeout-seconds", type=int, default=30)
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--max-pages", type=int, default=0, help="Optional page cap for debugging (0 = no cap).")
    p.add_argument("--adjusted", dest="adjusted", action="store_true")
    p.add_argument("--unadjusted", dest="adjusted", action="store_false")
    p.set_defaults(adjusted=True, func=_pull_underlying_massive_command)

    p = sub.add_parser("pull-massive")
    p.add_argument("--data-root", default="data")
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default="https://api.massive.com")
    p.add_argument("--timeout-seconds", type=int, default=30)
    p.add_argument("--limit", type=int, default=250, help="API page size (max depends on Massive plan).")
    p.add_argument("--max-pages", type=int, default=0, help="Optional page cap per day (0 = no cap).")
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--include-expired", dest="include_expired", action="store_true")
    p.add_argument("--exclude-expired", dest="include_expired", action="store_false")
    p.set_defaults(include_expired=True, func=_pull_massive_command)

    p = sub.add_parser("pull-flatfiles")
    p.add_argument("--data-root", default="data")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--access-key", default=None)
    p.add_argument("--secret-key", default=None)
    p.add_argument("--session-token", default=None)
    p.add_argument("--endpoint-url", default=None)
    p.add_argument("--bucket", default=None)
    p.add_argument("--prefix", default=None)
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--fail-fast", action="store_true", default=False)
    p.add_argument("--max-days", type=int, default=0, help="Optional cap for debugging (0 = no cap).")
    p.set_defaults(func=_pull_flatfiles_command)

    p = sub.add_parser("pull-options-symbol")
    p.add_argument("--data-root", default="data")
    p.add_argument("--symbol", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument(
        "--source-root",
        default=None,
        help="Root containing full OPRA day files. Default: <data-root>/options_source/<source-prefix>.",
    )
    p.add_argument(
        "--source-prefix",
        default=None,
        help="Used when --source-root is omitted. Default env MASSIVE_FLATFILES_PREFIX or us_options_opra/day_aggs_v1.",
    )
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--allow-missing-underlying", action="store_true", default=False)
    p.add_argument("--max-days", type=int, default=0, help="Optional cap for debugging (0 = no cap).")
    p.set_defaults(func=_pull_options_symbol_command)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    _autoload_dotenv()
    parser = _build_parser()
    ns = parser.parse_args(argv)
    ns.func(ns)


if __name__ == "__main__":
    main()
