"""Streamlit dashboard focused on training and evaluation workflows."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

OPENAI_MODEL = "gpt-5"
RUN_REPORTS_KEY = "_te_run_reports"
ACTIVE_RUN_KEY = "_te_active_run_dir"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _inject_style() -> None:
    st.markdown(
        """
<style>
:root {
  --bg0: #f4f7f3;
  --panel: #fbfdf9;
  --ink: #1d2a23;
  --accent: #14532d;
  --line: #dbe4dc;
}

.stApp {
  font-family: "Avenir Next", "Segoe UI", sans-serif;
  background:
    radial-gradient(1100px 500px at -10% -20%, #dff3e4 0%, rgba(223,243,228,0.15) 60%, transparent 100%),
    radial-gradient(900px 380px at 110% -10%, #e4efe9 0%, rgba(228,239,233,0.10) 55%, transparent 100%),
    linear-gradient(180deg, var(--bg0) 0%, #eef3ef 100%);
  color: var(--ink);
}

div[data-testid="stMetric"] {
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--panel);
  padding: 6px 10px;
}

.block-note {
  border-left: 4px solid var(--accent);
  background: #f2f8f3;
  border-radius: 10px;
  padding: 10px 12px;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _load_dotenv() -> None:
    """Best-effort .env loader for direct `streamlit run` usage."""
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


def _ensure_state() -> None:
    if RUN_REPORTS_KEY not in st.session_state:
        st.session_state[RUN_REPORTS_KEY] = []
    if ACTIVE_RUN_KEY not in st.session_state:
        st.session_state[ACTIVE_RUN_KEY] = ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _read_parquet_or_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def _resolve_existing_path(path_raw: Any, *, run_dir: Path | None = None) -> Path | None:
    if path_raw is None:
        return None
    s = str(path_raw).strip()
    if not s:
        return None
    p = Path(s).expanduser()
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        if run_dir is not None:
            candidates.append((run_dir / p).resolve())
        candidates.append((Path.cwd() / p).resolve())
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    return None


def _load_surface_axes_from_run(
    run_dir: Path,
    *,
    nx_hint: int = 21,
    nt_hint: int = 6,
    dataset_path_hint: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    candidates: list[Path] = []
    ds_hint = _resolve_existing_path(dataset_path_hint, run_dir=run_dir)
    if ds_hint is not None:
        candidates.append(ds_hint)

    train_summary = _read_json(run_dir / "train_summary.json")
    ds_summary = _resolve_existing_path(train_summary.get("dataset_path"), run_dir=run_dir)
    if ds_summary is not None:
        candidates.append(ds_summary)

    surf_pred = run_dir / "evaluation" / "surface_predictions.npz"
    if surf_pred.exists():
        candidates.append(surf_pred.resolve())

    seen: set[Path] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        try:
            payload = np.load(cand, allow_pickle=False)
        except Exception:
            continue
        if "x_grid" in payload.files and "tenor_days" in payload.files:
            try:
                x_grid = np.asarray(payload["x_grid"], dtype=np.float32).reshape(-1)
                tenor_days = np.asarray(payload["tenor_days"], dtype=np.float32).reshape(-1)
            except Exception:
                continue
            if len(x_grid) > 0 and len(tenor_days) > 0:
                return x_grid, tenor_days

    nx = max(int(nx_hint), 2)
    nt = max(int(nt_hint), 2)
    return (
        np.linspace(-0.30, 0.30, nx, dtype=np.float32),
        np.arange(1, nt + 1, dtype=np.float32),
    )


def _compute_static_focus_multiplier(
    x_grid: np.ndarray,
    tenor_days: np.ndarray,
    *,
    focus_alpha: float,
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
    dte_scale = max(float(focus_dte_scale_days), 1e-3)

    logits_pos = np.clip((x - float(focus_x_min)) / x_scale, -30.0, 30.0)
    x_focus_pos = 1.0 / (1.0 + np.exp(-logits_pos))

    neg_ratio = max(float(focus_neg_weight_ratio), 0.0)
    x_focus_neg = np.zeros_like(x_focus_pos, dtype=np.float64)
    if neg_ratio > 0.0:
        logits_neg = np.clip((float(focus_neg_x_max) - x) / x_scale, -30.0, 30.0)
        x_focus_neg = 1.0 / (1.0 + np.exp(-logits_neg))

    dte_focus = np.exp(-dte / dte_scale)
    dte_max = float(focus_dte_max_days)
    if np.isfinite(dte_max) and dte_max > 0.0:
        dte_gate = (dte < dte_max).astype(np.float64)
    else:
        dte_gate = np.ones_like(dte, dtype=np.float64)

    region_focus = (x_focus_pos + neg_ratio * x_focus_neg) * dte_focus * dte_gate
    return (1.0 + max(float(focus_alpha), 0.0) * region_focus).astype(np.float64)


def _find_adaptive_focus_summary_for_stage2(run_dir: Path) -> dict[str, Any]:
    parent = run_dir.parent
    if not parent.exists():
        return {}
    target = run_dir.resolve()
    candidates = sorted(parent.glob("adaptive_focus_rerun_*.json"), key=_run_recency, reverse=True)
    for path in candidates:
        payload = _read_json(path)
        if not payload:
            continue
        stage2_resolved = _resolve_existing_path(payload.get("stage2_run_dir"), run_dir=parent)
        if stage2_resolved is None:
            continue
        if stage2_resolved.resolve() == target:
            payload["_path"] = str(path.resolve())
            return payload
    return {}


def _load_focus_distribution_for_run(run_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "mode": "none",
        "alpha": None,
        "error_power": None,
        "map_path": None,
        "summary_path": None,
        "trainval_only": None,
        "errors": [],
        "grid_df": pd.DataFrame(),
    }

    train_config = _read_json(run_dir / "train_config.json")
    train_summary = _read_json(run_dir / "train_summary.json")
    symbol_hint = (_extract_symbol_from_run_dir(run_dir) or "RUN").upper()

    density_alpha = _to_float(train_config.get("surface_focus_density_alpha"))
    if density_alpha is None:
        density_alpha = _to_float(train_summary.get("surface_focus_density_alpha"))
    density_alpha = max(float(density_alpha or 0.0), 0.0)

    density_map_path = _resolve_existing_path(train_config.get("surface_focus_density_map_path"), run_dir=run_dir)
    if density_map_path is None:
        density_map_path = _resolve_existing_path(train_summary.get("surface_focus_density_map_path"), run_dir=run_dir)

    adaptive_summary = _find_adaptive_focus_summary_for_stage2(run_dir)
    if adaptive_summary:
        out["summary_path"] = str(adaptive_summary.get("_path", ""))
        out["error_power"] = _to_float(adaptive_summary.get("adaptive_focus_error_power"))
        if out["error_power"] is None:
            meta = adaptive_summary.get("focus_map_meta")
            if isinstance(meta, dict):
                out["error_power"] = _to_float(meta.get("error_power"))
        tv = adaptive_summary.get("trainval_only_for_focus_map")
        out["trainval_only"] = bool(tv) if isinstance(tv, bool) else None
        if density_map_path is None:
            density_map_path = _resolve_existing_path(adaptive_summary.get("focus_map_path"), run_dir=run_dir.parent)

    if density_alpha > 0.0:
        out["mode"] = "adaptive_density"
        out["alpha"] = float(density_alpha)
        out["map_path"] = str(density_map_path) if density_map_path is not None else None
        if out["trainval_only"] is None and "surface_focus_density_train_only" in train_summary:
            out["trainval_only"] = bool(train_summary.get("surface_focus_density_train_only"))

        if density_map_path is None:
            out["errors"].append(
                "Run has surface_focus_density_alpha > 0 but no readable surface_focus_density_map_path."
            )
            return out

        raw = _read_json(density_map_path)
        if not raw:
            out["errors"].append(f"Unable to read focus density map JSON: {density_map_path}")
            return out
        if not isinstance(raw, dict):
            out["errors"].append(f"Focus density map is not an object: {density_map_path}")
            return out

        rows: list[dict[str, Any]] = []
        dataset_hint = train_summary.get("dataset_path")
        for asset, grid_raw in raw.items():
            asset_name = str(asset).strip().upper()
            if not asset_name:
                out["errors"].append("Focus density map contains an empty asset key.")
                continue
            try:
                density = np.asarray(grid_raw, dtype=np.float64)
            except Exception:
                out["errors"].append(f"Focus density grid for `{asset_name}` is non-numeric.")
                continue
            if density.ndim != 2:
                out["errors"].append(f"Focus density grid for `{asset_name}` must be rank-2, got {density.shape}.")
                continue
            if not np.isfinite(density).all():
                out["errors"].append(f"Focus density grid for `{asset_name}` contains non-finite values.")
                continue
            if np.any(density <= 0.0):
                out["errors"].append(f"Focus density grid for `{asset_name}` contains non-positive values.")
                continue

            nx, nt = density.shape
            x_grid, tenor_days = _load_surface_axes_from_run(
                run_dir,
                nx_hint=nx,
                nt_hint=nt,
                dataset_path_hint=dataset_hint,
            )
            if len(x_grid) != nx:
                x_grid = np.linspace(-0.30, 0.30, nx, dtype=np.float32)
            if len(tenor_days) != nt:
                tenor_days = np.arange(1, nt + 1, dtype=np.float32)

            effective = np.exp(float(density_alpha) * (density - 1.0))
            for ix, x_val in enumerate(x_grid):
                for it, dte_val in enumerate(tenor_days):
                    rows.append(
                        {
                            "symbol": asset_name,
                            "moneyness_x": float(x_val),
                            "dte": float(dte_val),
                            "distribution": float(density[ix, it]),
                            "effective_multiplier": float(effective[ix, it]),
                        }
                    )

        if rows:
            out["grid_df"] = pd.DataFrame(rows)
        return out

    static_focus_alpha = max(float(_to_float(train_config.get("surface_focus_alpha")) or 0.0), 0.0)
    if static_focus_alpha <= 0.0:
        return out

    out["mode"] = "static_focus"
    out["alpha"] = static_focus_alpha
    x_grid, tenor_days = _load_surface_axes_from_run(
        run_dir,
        dataset_path_hint=train_summary.get("dataset_path"),
    )
    multiplier = _compute_static_focus_multiplier(
        x_grid,
        tenor_days,
        focus_alpha=static_focus_alpha,
        focus_x_min=float(_to_float(train_config.get("surface_focus_x_min")) or 0.10),
        focus_x_scale=float(_to_float(train_config.get("surface_focus_x_scale")) or 0.03),
        focus_dte_scale_days=float(_to_float(train_config.get("surface_focus_dte_scale_days")) or 21.0),
        focus_dte_max_days=float(_to_float(train_config.get("surface_focus_dte_max_days")) or 30.0),
        focus_neg_x_max=float(_to_float(train_config.get("surface_focus_neg_x_max")) or -0.20),
        focus_neg_weight_ratio=float(_to_float(train_config.get("surface_focus_neg_weight_ratio")) or 0.0),
    )
    rows = []
    for ix, x_val in enumerate(x_grid):
        for it, dte_val in enumerate(tenor_days):
            rows.append(
                {
                    "symbol": symbol_hint,
                    "moneyness_x": float(x_val),
                    "dte": float(dte_val),
                    "distribution": float(multiplier[ix, it]),
                    "effective_multiplier": float(multiplier[ix, it]),
                }
            )
    out["grid_df"] = pd.DataFrame(rows)
    return out


def _format_value(v: Any, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, str):
        return v
    try:
        x = float(v)
    except Exception:
        return str(v)
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}"


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    if not np.isfinite(x):
        return None
    return x


def _edge_vs_baseline(model_value: float | None, baseline_value: float | None, *, better: str) -> tuple[float | None, float | None]:
    if model_value is None or baseline_value is None:
        return None, None
    if better == "lower":
        edge = baseline_value - model_value
    else:
        edge = model_value - baseline_value
    denom = abs(baseline_value)
    pct = (edge / denom * 100.0) if denom > 1e-12 else None
    return edge, pct


def _baseline_label_from_key(key: str) -> str:
    s = str(key).strip().lower()
    if s == "tree":
        return "Tree Baseline"
    return "Parametric Factor-HAR(1,5,22)"


def _resolve_primary_non_persistence_baseline(metrics: dict[str, Any]) -> dict[str, Any]:
    primary_raw = str(metrics.get("surface_forecast_baseline_primary", "")).strip().lower()
    specs: dict[str, dict[str, str]] = {
        "tree": {
            "key": "tree",
            "rmse_key": "surface_forecast_iv_rmse_baseline_tree",
            "skill_key": "surface_forecast_skill_mse_vs_tree",
        },
        "parametric": {
            "key": "parametric",
            "rmse_key": "surface_forecast_iv_rmse_baseline_parametric",
            "skill_key": "surface_forecast_skill_mse_vs_parametric",
        },
    }
    if "tree" in primary_raw:
        order = ["tree", "parametric"]
    elif "param" in primary_raw or "har" in primary_raw:
        order = ["parametric", "tree"]
    else:
        order = ["tree", "parametric"]

    for k in order:
        spec = specs[k]
        rmse = _to_float(metrics.get(spec["rmse_key"]))
        skill = _to_float(metrics.get(spec["skill_key"]))
        if rmse is not None or skill is not None:
            out = dict(spec)
            out["label"] = _baseline_label_from_key(k)
            return out

    # Fall back to declared primary even if values are missing.
    k = order[0]
    out = dict(specs[k])
    out["label"] = _baseline_label_from_key(k)
    return out


def _resolve_primary_skill(metrics: dict[str, Any]) -> float | None:
    primary = _resolve_primary_non_persistence_baseline(metrics)
    skill = _to_float(metrics.get(primary["skill_key"]))
    if skill is not None:
        return skill
    for key in ("surface_forecast_skill_mse_vs_tree", "surface_forecast_skill_mse_vs_parametric", "surface_forecast_skill_mse_vs_persistence"):
        skill = _to_float(metrics.get(key))
        if skill is not None:
            return skill
    return None


def _baseline_comparison_specs(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    primary = _resolve_primary_non_persistence_baseline(metrics)
    primary_label = str(primary.get("label", "Primary Baseline"))
    return [
        {
            "factor": f"Surface Forecast RMSE vs {primary_label}",
            "model_key": "surface_forecast_iv_rmse",
            "baseline_key": str(primary["rmse_key"]),
            "better": "lower",
        },
        {
            "factor": f"Surface Forecast Skill vs {primary_label}",
            "model_key": str(primary["skill_key"]),
            "baseline_value": 0.0,
            "better": "higher",
        },
        {
            "factor": "Surface Forecast RMSE vs Persistence (reference)",
            "model_key": "surface_forecast_iv_rmse",
            "baseline_key": "surface_forecast_iv_rmse_baseline_persistence",
            "better": "lower",
        },
    ]


def _build_baseline_comparison_df(metrics: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in _baseline_comparison_specs(metrics):
        model_value = _to_float(metrics.get(spec["model_key"]))
        baseline_value = _to_float(metrics.get(spec["baseline_key"])) if spec.get("baseline_key") else _to_float(spec.get("baseline_value"))
        edge, edge_pct = _edge_vs_baseline(model_value, baseline_value, better=str(spec["better"]))
        if model_value is None or baseline_value is None or edge is None:
            continue
        rows.append(
            {
                "Factor": spec["factor"],
                "Better if": str(spec["better"]),
                "Model": model_value,
                "Baseline": baseline_value,
                "Edge vs baseline": edge,
                "Edge %": edge_pct,
                "Status": "better" if edge > 0 else ("worse" if edge < 0 else "flat"),
            }
        )
    return pd.DataFrame(rows)


def _build_direct_baseline_value_df(metrics: dict[str, Any]) -> pd.DataFrame:
    primary = _resolve_primary_non_persistence_baseline(metrics)
    specs = [
        {
            "metric": "Surface Forecast RMSE",
            "model_key": "surface_forecast_iv_rmse",
            "primary_key": str(primary["rmse_key"]),
            "persistence_key": "surface_forecast_iv_rmse_baseline_persistence",
        },
    ]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        model_value = _to_float(metrics.get(spec["model_key"]))
        primary_value = _to_float(metrics.get(spec["primary_key"]))
        persistence_value = _to_float(metrics.get(spec["persistence_key"]))
        if model_value is None and primary_value is None and persistence_value is None:
            continue
        rows.append(
            {
                "Metric": str(spec["metric"]),
                "Model": model_value,
                "Primary Baseline": primary_value,
                "Persistence Baseline": persistence_value,
            }
        )
    return pd.DataFrame(rows)


def _build_noarb_alignment_df(metrics: dict[str, Any]) -> pd.DataFrame:
    specs = [
        ("Calendar violations (recon)", "calendar_violation_obs_mean", "calendar_violation_pred_mean"),
        ("Butterfly violations (recon)", "butterfly_violation_obs_mean", "butterfly_violation_pred_mean"),
        ("Calendar violations (forecast)", "calendar_violation_forecast_obs_mean", "calendar_violation_forecast_pred_mean"),
        ("Butterfly violations (forecast)", "butterfly_violation_forecast_obs_mean", "butterfly_violation_forecast_pred_mean"),
    ]
    rows: list[dict[str, Any]] = []
    for label, obs_key, pred_key in specs:
        obs = _to_float(metrics.get(obs_key))
        pred = _to_float(metrics.get(pred_key))
        if obs is None or pred is None:
            continue
        rows.append(
            {
                "Factor": label,
                "Observed": obs,
                "Predicted": pred,
                "Pred - Obs": pred - obs,
                "Abs Gap": abs(pred - obs),
            }
        )
    return pd.DataFrame(rows)


def _render_baseline_cards(metrics: dict[str, Any]) -> None:
    primary = _resolve_primary_non_persistence_baseline(metrics)
    primary_label = str(primary.get("label", "Primary Baseline"))
    cards = [
        {
            "label": f"Surface Forecast RMSE vs {primary_label}",
            "model_key": "surface_forecast_iv_rmse",
            "baseline_key": str(primary["rmse_key"]),
            "better": "lower",
        },
        {
            "label": f"Surface Forecast Skill vs {primary_label}",
            "model_key": str(primary["skill_key"]),
            "baseline_value": 0.0,
            "better": "higher",
        },
        {
            "label": "Surface Forecast RMSE vs Persistence",
            "model_key": "surface_forecast_iv_rmse",
            "baseline_key": "surface_forecast_iv_rmse_baseline_persistence",
            "better": "lower",
        },
    ]
    cols = st.columns(len(cards))
    for i, spec in enumerate(cards):
        model_value = _to_float(metrics.get(spec["model_key"]))
        baseline_value = _to_float(metrics.get(spec["baseline_key"])) if spec.get("baseline_key") else _to_float(spec.get("baseline_value"))
        edge, _ = _edge_vs_baseline(model_value, baseline_value, better=str(spec["better"]))
        label = spec["label"]
        with cols[i]:
            if model_value is None or baseline_value is None or edge is None:
                st.metric(label, "n/a")
                continue
            st.metric(
                label,
                _format_value(model_value, 6),
                delta=f"{edge:+.6f} edge",
                delta_color="normal",
            )
            st.caption(f"Baseline: {_format_value(baseline_value, 6)}")


def _render_baseline_sections(metrics: dict[str, Any]) -> None:
    if not metrics:
        return
    primary = _resolve_primary_non_persistence_baseline(metrics)
    primary_key = str(primary.get("key", "parametric"))
    st.subheader("Model vs Baselines (Primary + Persistence)")
    _render_baseline_cards(metrics)
    st.caption("Edge > 0 means the model is better than the baseline for that factor.")

    baseline_primary = str(metrics.get("surface_forecast_baseline_primary", "n/a"))
    if primary_key == "tree":
        baseline_family = str(metrics.get("surface_forecast_baseline_tree_family", "n/a"))
        baseline_trained = _format_value(metrics.get("surface_forecast_baseline_tree_trained_assets"), 0)
        baseline_min_history = _format_value(metrics.get("surface_forecast_baseline_tree_min_history"), 0)
        baseline_fallback = _format_value(metrics.get("surface_forecast_baseline_tree_fallback_days"), 0)
        st.caption(
            "Primary baseline: "
            f"`{baseline_primary}` | family=`{baseline_family}` | trained_assets={baseline_trained} | "
            f"min_history={baseline_min_history} | persistence fallback days={baseline_fallback}"
        )
    else:
        baseline_family = str(metrics.get("surface_forecast_baseline_parametric_family", "n/a"))
        baseline_windows = str(metrics.get("surface_forecast_baseline_parametric_windows", "n/a"))
        baseline_factors = _format_value(metrics.get("surface_forecast_baseline_parametric_factors"), 0)
        baseline_ridge = _format_value(metrics.get("surface_forecast_baseline_parametric_ridge"), 6)
        baseline_fallback = _format_value(metrics.get("surface_forecast_baseline_parametric_fallback_days"), 0)
        st.caption(
            "Primary baseline: "
            f"`{baseline_primary}` | family=`{baseline_family}` | windows=`{baseline_windows}` | "
            f"factors={baseline_factors} | ridge={baseline_ridge} | persistence fallback days={baseline_fallback}"
        )

    direct = _build_direct_baseline_value_df(metrics)
    if not direct.empty:
        plot_df = direct.melt(
            id_vars=["Metric"],
            value_vars=["Model", "Primary Baseline", "Persistence Baseline"],
            var_name="Series",
            value_name="Value",
        )
        plot_df = plot_df[np.isfinite(pd.to_numeric(plot_df["Value"], errors="coerce"))]
        if not plot_df.empty:
            values_chart = (
                alt.Chart(plot_df)
                .mark_bar()
                .encode(
                    x=alt.X("Metric:N", title="Metric"),
                    xOffset=alt.XOffset("Series:N"),
                    y=alt.Y("Value:Q", title="Metric value"),
                    color=alt.Color("Series:N", title="Series"),
                    tooltip=["Metric:N", "Series:N", alt.Tooltip("Value:Q", format=".6f")],
                )
                .properties(height=280)
            )
            st.altair_chart(values_chart, use_container_width=True)

    comp = _build_baseline_comparison_df(metrics)
    if not comp.empty:
        edge_chart = (
            alt.Chart(comp)
            .mark_bar()
            .encode(
                x=alt.X("Edge vs baseline:Q", title="Edge vs baseline (positive is better)"),
                y=alt.Y("Factor:N", title="Comparison", sort="-x"),
                color=alt.condition(
                    "datum['Edge vs baseline'] >= 0",
                    alt.value("#166534"),
                    alt.value("#991b1b"),
                ),
                tooltip=[
                    "Factor:N",
                    alt.Tooltip("Model:Q", format=".6f"),
                    alt.Tooltip("Baseline:Q", format=".6f"),
                    alt.Tooltip("Edge vs baseline:Q", format=".6f"),
                    alt.Tooltip("Edge %:Q", format=".2f"),
                    "Status:N",
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(edge_chart, use_container_width=True)

        show = comp.copy()
        for col in ("Model", "Baseline", "Edge vs baseline", "Edge %"):
            if col in show.columns:
                show[col] = pd.to_numeric(show[col], errors="coerce").round(6)
        st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.info("No model-vs-baseline metric pairs available in this run.")

    st.subheader("No-Arb Alignment (Predicted vs Observed)")
    noarb = _build_noarb_alignment_df(metrics)
    if noarb.empty:
        st.info("No no-arbitrage observed/predicted metrics found.")
    else:
        show = noarb.copy()
        for col in ("Observed", "Predicted", "Pred - Obs", "Abs Gap"):
            if col in show.columns:
                show[col] = pd.to_numeric(show[col], errors="coerce").round(6)
        st.dataframe(show, use_container_width=True, hide_index=True)


def _build_all_symbols_baseline_snapshot(latest_runs_by_symbol: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol, run_dir in sorted(latest_runs_by_symbol.items()):
        metrics = _read_json(run_dir / "evaluation" / "metrics.json")
        if not metrics:
            continue

        primary = _resolve_primary_non_persistence_baseline(metrics)
        surf_rmse = _to_float(metrics.get("surface_forecast_iv_rmse"))
        surf_base_primary = _to_float(metrics.get(primary["rmse_key"]))
        surf_edge_primary, _ = _edge_vs_baseline(surf_rmse, surf_base_primary, better="lower")
        surf_base_persistence = _to_float(metrics.get("surface_forecast_iv_rmse_baseline_persistence"))
        surf_edge_persistence, _ = _edge_vs_baseline(surf_rmse, surf_base_persistence, better="lower")

        cal_obs_f = _to_float(metrics.get("calendar_violation_forecast_obs_mean"))
        cal_pred_f = _to_float(metrics.get("calendar_violation_forecast_pred_mean"))
        bfly_obs_f = _to_float(metrics.get("butterfly_violation_forecast_obs_mean"))
        bfly_pred_f = _to_float(metrics.get("butterfly_violation_forecast_pred_mean"))

        rows.append(
            {
                "Symbol": symbol,
                "Run": str(run_dir),
                "Primary baseline label": str(primary.get("label", "Primary Baseline")),
                "Forecast skill vs primary": _to_float(metrics.get(primary["skill_key"])),
                "Forecast skill vs tree": _to_float(metrics.get("surface_forecast_skill_mse_vs_tree")),
                "Forecast skill vs parametric": _to_float(metrics.get("surface_forecast_skill_mse_vs_parametric")),
                "Forecast skill vs persistence": _to_float(metrics.get("surface_forecast_skill_mse_vs_persistence")),
                "Surface RMSE": surf_rmse,
                "Surface RMSE baseline (primary)": surf_base_primary,
                "Surface edge (vs primary)": surf_edge_primary,
                "Surface RMSE baseline (persistence)": surf_base_persistence,
                "Surface edge (vs persistence)": surf_edge_persistence,
                "Forecast calendar abs gap": (abs(cal_pred_f - cal_obs_f) if cal_obs_f is not None and cal_pred_f is not None else None),
                "Forecast butterfly abs gap": (abs(bfly_pred_f - bfly_obs_f) if bfly_obs_f is not None and bfly_pred_f is not None else None),
            }
        )
    return pd.DataFrame(rows)


def _render_baseline_comparison_tab(run_dir: Path, latest_runs_by_symbol: dict[str, Path]) -> None:
    st.subheader("Baseline Comparison")
    if not run_dir.exists() or not run_dir.is_dir():
        st.info("Select a valid run directory in the sidebar.")
        return

    metrics = _read_json(run_dir / "evaluation" / "metrics.json")
    if not metrics:
        st.warning("No evaluation metrics found for selected run.")
        return

    _render_baseline_sections(metrics)

    st.subheader("All Symbols: Latest Run Baseline Snapshot")
    snap = _build_all_symbols_baseline_snapshot(latest_runs_by_symbol)
    if snap.empty:
        st.info("No latest-run evaluation metrics found across symbols.")
        return

    bar_df = snap.melt(
        id_vars=["Symbol"],
        value_vars=["Surface edge (vs primary)", "Surface edge (vs persistence)"],
        var_name="Series",
        value_name="Edge",
    )
    bar_df = bar_df[np.isfinite(pd.to_numeric(bar_df["Edge"], errors="coerce"))]
    if not bar_df.empty:
        chart = (
            alt.Chart(bar_df)
            .mark_bar()
            .encode(
                x=alt.X("Symbol:N", title="Symbol"),
                xOffset=alt.XOffset("Series:N"),
                y=alt.Y("Edge:Q", title="Surface RMSE edge (positive is better)"),
                color=alt.Color("Series:N", title="Series"),
                tooltip=["Symbol:N", "Series:N", alt.Tooltip("Edge:Q", format=".6f")],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)

    show = snap.copy()
    for col in (
        "Primary baseline label",
        "Forecast skill vs primary",
        "Forecast skill vs tree",
        "Forecast skill vs parametric",
        "Forecast skill vs persistence",
        "Surface RMSE",
        "Surface RMSE baseline (primary)",
        "Surface edge (vs primary)",
        "Surface RMSE baseline (persistence)",
        "Surface edge (vs persistence)",
        "Forecast calendar abs gap",
        "Forecast butterfly abs gap",
    ):
        if col in show.columns:
            show[col] = pd.to_numeric(show[col], errors="coerce").round(6)
    st.dataframe(show, use_container_width=True, hide_index=True)


def _discover_run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    seen: set[Path] = set()
    for run_dir in root.glob("**/run_*"):
        if not run_dir.is_dir():
            continue
        p = run_dir.resolve()
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    out.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return out


def _extract_symbol_from_run_dir(run_dir: Path) -> str | None:
    resolved = run_dir.resolve()
    chain = [resolved]
    chain.extend(resolved.parents)
    for node in chain:
        name = node.name.strip()
        if not name:
            continue
        if name.startswith("03_runs_"):
            symbol = name[len("03_runs_") :].strip().upper()
            return symbol or None
        if name == "runs":
            symbol = node.parent.name.strip().upper()
            return symbol or None
    return None


def _run_recency(run_dir: Path) -> float:
    recency = run_dir.stat().st_mtime if run_dir.exists() else 0.0
    for rel in ("evaluation/metrics.json", "train_summary.json", "train_history.csv", "model.pt"):
        p = run_dir / rel
        if p.exists():
            try:
                recency = max(recency, p.stat().st_mtime)
            except Exception:
                continue
    return float(recency)


def _discover_latest_runs_by_symbol(outputs_root: Path) -> dict[str, Path]:
    by_symbol: dict[str, Path] = {}
    if not outputs_root.exists():
        return by_symbol

    candidates_by_symbol: dict[str, list[Path]] = {}
    seen: set[Path] = set()

    for p in outputs_root.glob("**/run_*"):
        if not p.is_dir():
            continue
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        symbol = _extract_symbol_from_run_dir(rp)
        if not symbol:
            continue
        candidates_by_symbol.setdefault(symbol, []).append(rp)

    for latest_path in outputs_root.glob("**/latest.txt"):
        try:
            target = Path(latest_path.read_text(encoding="utf-8").strip()).expanduser().resolve()
        except Exception:
            continue
        if not target.is_dir():
            continue
        symbol = _extract_symbol_from_run_dir(target)
        if not symbol:
            parent_name = latest_path.parent.name.strip()
            if parent_name.startswith("03_runs_"):
                symbol = parent_name[len("03_runs_") :].strip().upper() or None
            elif parent_name == "runs":
                symbol = latest_path.parent.parent.name.strip().upper() or None
        if not symbol:
            continue
        candidates_by_symbol.setdefault(symbol, []).append(target)

    for symbol, candidates in candidates_by_symbol.items():
        try:
            newest = max(candidates, key=_run_recency).resolve()
        except Exception:
            continue
        by_symbol[symbol] = newest

    return by_symbol


def _artifact_manifest(run_dir: Path) -> pd.DataFrame:
    rel_paths = [
        "model.pt",
        "train_config.json",
        "train_summary.json",
        "train_history.csv",
        "latent_states.parquet",
        "evaluation/metrics.json",
        "evaluation/contract_predictions.parquet",
        "evaluation/noarb_test_dates.parquet",
        "evaluation/noarb_forecast_test_dates.parquet",
        "evaluation/surface_predictions.npz",
    ]
    rows: list[dict[str, Any]] = []
    for rel in rel_paths:
        p = run_dir / rel
        row: dict[str, Any] = {
            "artifact": rel,
            "exists": p.exists(),
            "size_bytes": np.nan,
            "modified_utc": "",
        }
        if p.exists():
            stat = p.stat()
            row["size_bytes"] = int(stat.st_size)
            row["modified_utc"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
        rows.append(row)
    return pd.DataFrame(rows)


def _build_artifact_report(run_dir: Path) -> dict[str, Any]:
    train_summary = _read_json(run_dir / "train_summary.json")
    train_config = _read_json(run_dir / "train_config.json")
    eval_metrics = _read_json(run_dir / "evaluation" / "metrics.json")
    hist_tail = _train_history_tail_rows(run_dir)
    manifest_df = _artifact_manifest(run_dir)

    dataset_path = str(train_summary.get("dataset_path", "")) if train_summary else ""
    status = "ok" if (train_summary or eval_metrics) else "missing_artifacts"
    updated_utc = datetime.fromtimestamp(_run_recency(run_dir), tz=timezone.utc).isoformat(timespec="seconds")

    return {
        "id": f"artifact_{int(datetime.now(tz=timezone.utc).timestamp() * 1000)}",
        "timestamp_utc": _utc_now(),
        "run_type": "artifact_review",
        "status": status,
        "dataset_path": dataset_path,
        "training_parameters": _json_safe(train_config),
        "evaluation_parameters": {"source": "artifacts"},
        "run_dir": str(run_dir.resolve()),
        "run_updated_utc": updated_utc,
        "train_summary": _json_safe(train_summary),
        "evaluation_metrics": _json_safe(eval_metrics),
        "train_history_tail": _json_safe(hist_tail),
        "artifact_manifest": _json_safe(manifest_df.to_dict(orient="records")),
        "error": "",
    }


def _extract_openai_text(payload: dict[str, Any]) -> str:
    def _text_from_value(v: Any) -> str:
        if isinstance(v, str):
            return v.strip()
        if isinstance(v, dict):
            for k in ("value", "text", "output_text", "content"):
                if k in v:
                    t = _text_from_value(v.get(k))
                    if t:
                        return t
            return ""
        if isinstance(v, list):
            parts = [_text_from_value(x) for x in v]
            parts = [p for p in parts if p]
            return "\n\n".join(parts).strip()
        return ""

    chunks: list[str] = []

    raw = payload.get("output_text")
    if isinstance(raw, str) and raw.strip():
        chunks.append(raw.strip())
    elif isinstance(raw, list):
        parsed = _text_from_value(raw)
        if parsed:
            chunks.append(parsed)

    output = payload.get("output")
    if isinstance(output, list):
        for block in output:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "output_text":
                txt = _text_from_value(block.get("text"))
                if txt:
                    chunks.append(txt)
            content = block.get("content")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    txt = _text_from_value(item.get("text"))
                    if not txt:
                        txt = _text_from_value(item)
                    if txt:
                        chunks.append(txt)

    # Compatibility fallback for chat-completions shaped payloads.
    choices = payload.get("choices")
    if isinstance(choices, list):
        for ch in choices:
            if not isinstance(ch, dict):
                continue
            msg = ch.get("message")
            if isinstance(msg, dict):
                txt = _text_from_value(msg.get("content"))
                if txt:
                    chunks.append(txt)

    deduped: list[str] = []
    seen: set[str] = set()
    for c in chunks:
        cc = c.strip()
        if not cc or cc in seen:
            continue
        seen.add(cc)
        deduped.append(cc)
    return "\n\n".join(deduped).strip()


def _build_openai_feedback_prompt(report: dict[str, Any]) -> tuple[str, str]:
    system_text = (
        "You are a senior quantitative researcher reviewing an options-volatility ML system. "
        "Return specific, testable recommendations to improve model quality in training/evaluation workflows. "
        "Do not provide backtesting, portfolio allocation, or discretionary trading advice. "
        "Ground suggestions in financial microstructure and options theory (vol surface behavior, no-arbitrage, "
        "and forecast skill). "
        "Avoid generic advice; each suggestion must include rationale, expected metric impact, and tradeoff/risk."
    )

    goal_context = {
        "primary_goal": "Improve surface dynamics forecasting quality for IV-surface modeling.",
        "workflow_scope": "Training + evaluation only. Focus on surface dynamics and no-arbitrage behavior.",
        "success_signals": [
            "Lower surface_forecast_iv_rmse",
            "Higher surface_forecast_skill_mse_vs_parametric",
            "Higher surface_forecast_skill_mse_vs_persistence",
            "Lower surface_recon_iv_rmse",
            "Lower calendar/butterfly forecast violation gaps",
            "More stable validation losses in train_history",
        ],
        "constraints": [
            "Recommendations should be directly actionable in CLI-driven train/eval runs.",
            "Prefer changes that can be validated by artifacts already produced in this repo.",
        ],
    }

    financial_basis = {
        "domain": "US equity options implied-volatility dynamics",
        "state_representation": "Latent state learned from IV surface snapshots over x=ln(K/S) and tenor_days",
        "model_outputs": [
            "Surface reconstruction",
            "One-step-ahead surface forecast",
            "No-arbitrage diagnostics on reconstructed/forecast surfaces",
        ],
        "financial_principles": [
            "Smile/skew and term-structure behavior should be stable and economically plausible",
            "No-arbitrage diagnostics matter (calendar and butterfly violations)",
            "Near-ATM and short-tenor contracts can dominate risk and should be treated carefully",
            "Forecast skill should be compared to persistence baselines, not judged in isolation",
        ],
        "metric_interpretation": {
            "surface_forecast_iv_rmse": "Lower is better (forward IV surface forecast absolute error)",
            "surface_forecast_skill_mse_vs_parametric": "Higher is better (positive means beating the parametric baseline)",
            "surface_forecast_skill_mse_vs_persistence": "Higher is better (positive means beating persistence reference baseline)",
            "surface_recon_iv_rmse": "Lower is better (representation absolute error without overfitting)",
            "calendar_violation_*": "Lower is better (fewer calendar-arbitrage inconsistencies)",
            "butterfly_violation_*": "Lower is better (fewer convexity/arbitrage inconsistencies)",
        },
    }

    context = {
        "goal_context": goal_context,
        "financial_basis": financial_basis,
        "run_type": report.get("run_type"),
        "status": report.get("status"),
        "dataset_path": report.get("dataset_path"),
        "run_dir": report.get("run_dir"),
        "training_parameters": _json_safe(report.get("training_parameters", {})),
        "evaluation_parameters": _json_safe(report.get("evaluation_parameters", {})),
        "train_summary": _json_safe(report.get("train_summary", {})),
        "evaluation_metrics": _json_safe(report.get("evaluation_metrics", {})),
        "train_history_tail": _json_safe(report.get("train_history_tail", [])),
    }
    user_text = (
        "Given this run output, project goal, and financial basis, provide:\n"
        "1) Top 5 prioritized improvements (highest impact first)\n"
        "2) For each: financial/technical rationale, exact metrics expected to change, and expected direction\n"
        "3) Concrete next-run parameter changes (only fields that should be modified)\n"
        "4) Sanity checks to validate that improvements are real (including baseline checks and overfitting checks)\n"
        "5) Risks/failure-modes for each recommendation\n\n"
        "Keep the response structured and explicit. Use short sections and bullet points.\n\n"
        f"RUN_CONTEXT_JSON:\n{json.dumps(context, indent=2, sort_keys=True)}"
    )
    return system_text, user_text


def _request_openai_feedback(
    report: dict[str, Any],
    *,
    model: str,
    timeout_seconds: int,
) -> tuple[dict[str, Any], str | None, dict[str, Any] | None, str | None]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {}, None, None, "Missing OPENAI_API_KEY in environment or .env file."

    system_text, user_text = _build_openai_feedback_prompt(report)
    request_payload: dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_text}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_text}],
            },
        ],
    }

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    read_timeout = max(30, int(os.environ.get("OPENAI_TIMEOUT_SECONDS", str(timeout_seconds))))
    connect_timeout = max(5, int(os.environ.get("OPENAI_CONNECT_TIMEOUT_SECONDS", "10")))
    max_retries = max(1, int(os.environ.get("OPENAI_MAX_RETRIES", "3")))
    retry_backoff_seconds = max(1.0, float(os.environ.get("OPENAI_RETRY_BACKOFF_SECONDS", "2.0")))

    last_error: str | None = None
    last_raw: dict[str, Any] | None = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=request_payload,
                timeout=(connect_timeout, read_timeout),
            )
        except (requests.exceptions.ReadTimeout, requests.exceptions.Timeout) as exc:
            last_error = (
                f"OpenAI request timed out on attempt {attempt}/{max_retries} "
                f"(connect={connect_timeout}s, read={read_timeout}s): {exc}"
            )
            if attempt < max_retries:
                time.sleep(retry_backoff_seconds * float(attempt))
                continue
            return request_payload, None, last_raw, last_error
        except requests.exceptions.ConnectionError as exc:
            last_error = f"OpenAI connection failed on attempt {attempt}/{max_retries}: {exc}"
            if attempt < max_retries:
                time.sleep(retry_backoff_seconds * float(attempt))
                continue
            return request_payload, None, last_raw, last_error
        except Exception as exc:
            return request_payload, None, last_raw, f"OpenAI request failed: {exc}"

        if resp.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
            last_error = f"OpenAI API temporary error ({resp.status_code}) on attempt {attempt}/{max_retries}."
            time.sleep(retry_backoff_seconds * float(attempt))
            continue

        if resp.status_code >= 400:
            body_snippet = resp.text.strip()[:1200]
            return request_payload, None, last_raw, f"OpenAI API error ({resp.status_code}): {body_snippet}"

        try:
            response_json = resp.json()
        except Exception as exc:
            return request_payload, None, last_raw, f"OpenAI API returned non-JSON response: {exc}"

        last_raw = response_json
        text = _extract_openai_text(response_json)
        if not text:
            text = "OpenAI responded without extractable text. Inspect raw response JSON."
        return request_payload, text, response_json, None

    return request_payload, None, last_raw, (last_error or "OpenAI request failed for unknown reason.")


def _train_history_tail_rows(run_dir: Path, n: int = 8) -> list[dict[str, Any]]:
    hist_path = run_dir / "train_history.csv"
    if not hist_path.exists():
        return []
    try:
        hist = pd.read_csv(hist_path)
    except Exception:
        return []
    if hist.empty:
        return []
    return hist.tail(n).to_dict(orient="records")


def _append_report(report: dict[str, Any]) -> None:
    reports = st.session_state.get(RUN_REPORTS_KEY)
    if not isinstance(reports, list):
        reports = []
    reports.append(report)
    st.session_state[RUN_REPORTS_KEY] = reports


def _render_training_diagnostics(run_dir: Path) -> None:
    st.subheader("Training Diagnostics")

    hist_path = run_dir / "train_history.csv"
    if not hist_path.exists():
        st.info("No training history found for selected run.")
        return

    hist = pd.read_csv(hist_path)
    if hist.empty:
        st.info("Training history file is empty.")
        return

    numeric_cols = [
        c for c in hist.columns if c not in {"stage", "epoch"} and pd.api.types.is_numeric_dtype(hist[c])
    ]
    if not numeric_cols:
        st.dataframe(hist.tail(50), use_container_width=True)
        return

    stage_options = ["all"] + sorted(str(x) for x in hist["stage"].dropna().unique()) if "stage" in hist.columns else ["all"]
    selected_stage = st.selectbox("Stage filter", options=stage_options, index=0, key="diag_train_stage")

    chart_df = hist.copy()
    if selected_stage != "all" and "stage" in chart_df.columns:
        chart_df = chart_df[chart_df["stage"] == selected_stage].copy()

    metric_pick = st.selectbox("Metric", options=numeric_cols, index=0, key="diag_train_metric")
    if chart_df.empty:
        st.info("No rows match the selected stage.")
        return

    chart = (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y(f"{metric_pick}:Q", title=metric_pick),
            color=alt.Color("stage:N") if "stage" in chart_df.columns else alt.value("#14532d"),
            tooltip=[c for c in ["stage", "epoch", metric_pick] if c in chart_df.columns],
        )
        .properties(height=360)
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(chart_df.tail(40), use_container_width=True)


def _render_surface_forecast_diagnostics(eval_dir: Path) -> None:
    surf_path = eval_dir / "surface_predictions.npz"
    if not surf_path.exists():
        st.info("No surface_predictions.npz artifact found.")
        return
    try:
        payload = np.load(surf_path, allow_pickle=False)
    except Exception:
        st.info("Unable to read surface_predictions.npz.")
        return

    required = {"dates", "iv_surface_obs", "iv_surface_forecast", "forecast_entry_index", "forecast_target_index"}
    if not required.issubset(set(payload.files)):
        st.info("surface_predictions.npz is missing required forecast arrays.")
        return

    dates = payload["dates"].astype(str)
    obs = payload["iv_surface_obs"].astype(np.float32)
    forecast = payload["iv_surface_forecast"].astype(np.float32)
    entry_idx = payload["forecast_entry_index"].astype(np.int32)
    target_idx = payload["forecast_target_index"].astype(np.int32)

    if len(entry_idx) == 0 or len(target_idx) == 0:
        st.info("No forecast entry/target pairs available in this run.")
        return

    pred = forecast[entry_idx]
    truth = obs[target_idx]
    base_persistence = obs[entry_idx]
    metrics = _read_json(eval_dir / "metrics.json")
    primary = _resolve_primary_non_persistence_baseline(metrics)
    baseline_label = str(primary.get("label", "persistence"))
    base_tree = None
    base_param = None
    if "iv_surface_forecast_baseline_tree" in payload.files:
        raw_tree = payload["iv_surface_forecast_baseline_tree"].astype(np.float32)
        if len(raw_tree) == len(entry_idx):
            base_tree = raw_tree
    if "iv_surface_forecast_baseline_parametric" in payload.files:
        raw_param = payload["iv_surface_forecast_baseline_parametric"].astype(np.float32)
        if len(raw_param) == len(entry_idx):
            base_param = raw_param

    primary_key = str(primary.get("key", "parametric"))
    if primary_key == "tree" and base_tree is not None:
        base_primary = base_tree
    elif primary_key == "parametric" and base_param is not None:
        base_primary = base_param
    elif base_tree is not None:
        base_primary = base_tree
        baseline_label = "Tree Baseline"
    elif base_param is not None:
        base_primary = base_param
        baseline_label = "Parametric Factor-HAR(1,5,22)"
    else:
        base_primary = base_persistence
        baseline_label = "Persistence"

    model_rmse = np.sqrt(np.mean((pred - truth) ** 2, axis=(1, 2)))
    base_rmse_persistence = np.sqrt(np.mean((base_persistence - truth) ** 2, axis=(1, 2)))
    tree_rmse = None
    parametric_rmse = None
    if base_tree is not None:
        tree_rmse = np.sqrt(np.mean((base_tree - truth) ** 2, axis=(1, 2)))
    if base_param is not None:
        parametric_rmse = np.sqrt(np.mean((base_param - truth) ** 2, axis=(1, 2)))
    model_mse = np.mean((pred - truth) ** 2, axis=(1, 2))
    base_mse = np.mean((base_primary - truth) ** 2, axis=(1, 2))
    skill = np.full_like(model_mse, np.nan, dtype=np.float64)
    valid = base_mse > 1e-12
    skill[valid] = 1.0 - (model_mse[valid] / base_mse[valid])

    day_payload = {
        "date_entry": dates[entry_idx],
        "date_target": dates[target_idx],
        "model_rmse": model_rmse,
        "persistence_rmse": base_rmse_persistence,
        "skill_vs_primary_baseline": skill,
    }
    if tree_rmse is not None:
        day_payload["tree_rmse"] = tree_rmse
    if base_param is not None:
        day_payload["parametric_rmse"] = parametric_rmse
    day_df = pd.DataFrame(day_payload)

    rmse_series = ["model_rmse", "persistence_rmse"]
    if "tree_rmse" in day_df.columns:
        rmse_series.append("tree_rmse")
    if "parametric_rmse" in day_df.columns:
        rmse_series.append("parametric_rmse")
    plot_df = day_df.melt(id_vars=["date_target"], value_vars=rmse_series, var_name="series", value_name="rmse")
    rmse_chart = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("date_target:T", title="Target date"),
            y=alt.Y("rmse:Q", title="Per-day surface forecast RMSE"),
            color=alt.Color("series:N", title="Series"),
            tooltip=["date_target:T", "series:N", alt.Tooltip("rmse:Q", format=".6f")],
        )
        .properties(height=320)
    )
    st.altair_chart(rmse_chart, use_container_width=True)

    skill_chart = (
        alt.Chart(day_df)
        .mark_bar(color="#166534")
        .encode(
            x=alt.X("skill_vs_primary_baseline:Q", bin=alt.Bin(maxbins=40), title="Per-day forecast skill vs primary baseline"),
            y=alt.Y("count():Q", title="Days"),
            tooltip=[alt.Tooltip("count():Q", title="Days")],
        )
        .properties(height=240)
    )
    st.altair_chart(skill_chart, use_container_width=True)
    st.caption(f"Primary baseline for skill histogram: `{baseline_label}` (skill uses MSE).")
    st.dataframe(day_df.tail(40), use_container_width=True, hide_index=True)


def _render_surface_error_breakdown(eval_dir: Path) -> None:
    by_dte = _read_parquet_or_csv(eval_dir / "surface_forecast_error_by_dte.parquet")
    by_x = _read_parquet_or_csv(eval_dir / "surface_forecast_error_by_moneyness.parquet")
    by_grid = _read_parquet_or_csv(eval_dir / "surface_forecast_error_grid.parquet")

    if by_dte.empty and by_x.empty and by_grid.empty:
        st.info("No DTE/moneyness forecast error breakdown artifacts found.")
        return

    st.subheader("Forecast Absolute Error by DTE and Moneyness")

    if not by_dte.empty:
        dte_plot = by_dte.melt(
            id_vars=["series", "dte"],
            value_vars=["rmse"],
            var_name="metric",
            value_name="value",
        )
        chart_dte = (
            alt.Chart(dte_plot)
            .mark_line(point=True)
            .encode(
                x=alt.X("dte:Q", title="DTE"),
                y=alt.Y("value:Q", title="Absolute error"),
                color=alt.Color("series:N", title="Series"),
                tooltip=["series:N", "dte:Q", alt.Tooltip("value:Q", format=".6f")],
            )
            .properties(height=280)
        )
        st.altair_chart(chart_dte, use_container_width=True)
        show_dte = by_dte.copy()
        for c in ("rmse",):
            show_dte[c] = pd.to_numeric(show_dte[c], errors="coerce").round(6)
        st.dataframe(show_dte, use_container_width=True, hide_index=True)

    if not by_x.empty:
        x_plot = by_x.melt(
            id_vars=["series", "moneyness_x"],
            value_vars=["rmse"],
            var_name="metric",
            value_name="value",
        )
        chart_x = (
            alt.Chart(x_plot)
            .mark_line(point=True)
            .encode(
                x=alt.X("moneyness_x:Q", title="Moneyness x=ln(K/S)"),
                y=alt.Y("value:Q", title="Absolute error"),
                color=alt.Color("series:N", title="Series"),
                tooltip=["series:N", "moneyness_x:Q", alt.Tooltip("value:Q", format=".6f")],
            )
            .properties(height=280)
        )
        st.altair_chart(chart_x, use_container_width=True)
        show_x = by_x.copy()
        for c in ("rmse",):
            show_x[c] = pd.to_numeric(show_x[c], errors="coerce").round(6)
        st.dataframe(show_x, use_container_width=True, hide_index=True)

    if not by_grid.empty:
        st.markdown("**DTE x Moneyness Heatmap**")
        series_opts = sorted(str(s) for s in by_grid["series"].dropna().astype(str).unique().tolist())
        metric_opts = ["rmse"]
        if series_opts:
            sel_series = st.selectbox("Series", options=series_opts, index=0, key="diag_grid_series")
            sel_metric = st.selectbox("Grid metric", options=metric_opts, index=0, key="diag_grid_metric")
            grid = by_grid[by_grid["series"].astype(str) == sel_series].copy()
            grid[sel_metric] = pd.to_numeric(grid[sel_metric], errors="coerce")
            grid = grid[np.isfinite(grid[sel_metric])]
            if not grid.empty:
                heat = (
                    alt.Chart(grid)
                    .mark_rect()
                    .encode(
                        x=alt.X("dte:O", title="DTE"),
                        y=alt.Y("moneyness_x:Q", title="Moneyness x=ln(K/S)"),
                        color=alt.Color(f"{sel_metric}:Q", title=sel_metric.upper()),
                        tooltip=[
                            "series:N",
                            "dte:O",
                            "moneyness_x:Q",
                            alt.Tooltip(f"{sel_metric}:Q", format=".6f"),
                        ],
                    )
                    .properties(height=320)
                )
                st.altair_chart(heat, use_container_width=True)


def _render_eval_diagnostics(run_dir: Path) -> None:
    st.subheader("Evaluation Diagnostics")
    eval_dir = run_dir / "evaluation"

    metrics = _read_json(eval_dir / "metrics.json")
    if metrics:
        primary_skill = _resolve_primary_skill(metrics)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Surface Forecast RMSE", _format_value(metrics.get("surface_forecast_iv_rmse"), 6))
        c2.metric("Surface Forecast Skill", _format_value(primary_skill, 6))
        c3.metric("Surface Recon RMSE", _format_value(metrics.get("surface_recon_iv_rmse"), 6))
        c4.metric("Forecast Butterfly Pred", _format_value(metrics.get("butterfly_violation_forecast_pred_mean"), 6))

        _render_baseline_sections(metrics)
    else:
        st.warning("No evaluation metrics found.")

    st.subheader("Per-Day Surface Forecast Diagnostics")
    _render_surface_forecast_diagnostics(eval_dir)
    _render_surface_error_breakdown(eval_dir)


def _render_run_overview(run_dir: Path, latest_runs_by_symbol: dict[str, Path]) -> None:
    if not run_dir.exists() or not run_dir.is_dir():
        st.info("Select a valid run directory in the sidebar.")
        return

    report = _build_artifact_report(run_dir)
    metrics = report.get("evaluation_metrics", {})
    train_summary = report.get("train_summary", {})
    symbol = _extract_symbol_from_run_dir(run_dir) or "n/a"
    primary_skill = _resolve_primary_skill(metrics)

    st.markdown(
        (
            "<div class='block-note'>"
            f"<b>Symbol</b>: <code>{symbol}</code><br>"
            f"<b>Run directory</b>: <code>{report.get('run_dir', 'n/a')}</code><br>"
            f"<b>Last updated (UTC)</b>: <code>{report.get('run_updated_utc', 'n/a')}</code>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Surface Forecast RMSE", _format_value(metrics.get("surface_forecast_iv_rmse"), 6))
    c2.metric("Surface Forecast Skill", _format_value(primary_skill, 6))
    c3.metric("Surface Recon RMSE", _format_value(metrics.get("surface_recon_iv_rmse"), 6))
    c4.metric("Forecast Butterfly Pred", _format_value(metrics.get("butterfly_violation_forecast_pred_mean"), 6))
    c5.metric("Final Val Recon", _format_value(train_summary.get("final_val_recon"), 6))
    st.caption("For full baseline diagnostics and detailed error breakdowns, use the Evaluation Diagnostics tab.")

    if latest_runs_by_symbol:
        st.subheader("All Symbols: Latest Run Baseline Snapshot")
        snap = _build_all_symbols_baseline_snapshot(latest_runs_by_symbol)
        if snap.empty:
            st.info("No latest-run evaluation metrics found across symbols.")
        else:
            show = snap.copy()
            for col in (
                "Primary baseline label",
                "Forecast skill vs primary",
                "Forecast skill vs tree",
                "Forecast skill vs parametric",
                "Forecast skill vs persistence",
                "Surface RMSE",
                "Surface RMSE baseline (primary)",
                "Surface edge (vs primary)",
                "Surface RMSE baseline (persistence)",
                "Surface edge (vs persistence)",
                "Forecast calendar abs gap",
                "Forecast butterfly abs gap",
            ):
                if col in show.columns:
                    show[col] = pd.to_numeric(show[col], errors="coerce").round(6)
            st.caption("Positive surface edge means model beats that baseline; lower forecast no-arb abs gaps are better.")
            st.dataframe(show, use_container_width=True, hide_index=True)

    st.subheader("Training Summary")
    st.json(train_summary, expanded=False)

    tail = report.get("train_history_tail", [])
    if isinstance(tail, list) and tail:
        st.subheader("Train History Tail")
        st.dataframe(pd.DataFrame(tail), use_container_width=True, hide_index=True)

    manifest = report.get("artifact_manifest", [])
    if isinstance(manifest, list) and manifest:
        st.subheader("Artifact Manifest")
        st.dataframe(pd.DataFrame(manifest), use_container_width=True, hide_index=True)


def _render_openai_tab(run_dir: Path) -> None:
    st.subheader("OpenAI GPT-5 Improvement Suggestions")
    if not run_dir.exists() or not run_dir.is_dir():
        st.info("Select a valid run directory in the sidebar.")
        return

    timeout_seconds = int(os.environ.get("OPENAI_TIMEOUT_SECONDS", "240"))
    max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "3"))
    st.caption(f"Request settings: timeout={timeout_seconds}s, retries={max_retries}, model={OPENAI_MODEL}")

    if st.button("Generate Suggestions For Selected Run", type="primary", use_container_width=False):
        report = _build_artifact_report(run_dir)
        report["openai_model"] = OPENAI_MODEL
        with st.spinner("Requesting GPT-5 improvement suggestions..."):
            req_payload, suggestions, raw, err = _request_openai_feedback(
                report,
                model=OPENAI_MODEL,
                timeout_seconds=timeout_seconds,
            )
        report["openai_request_payload"] = req_payload
        report["openai_suggestions"] = suggestions
        report["openai_response_raw"] = raw
        report["openai_error"] = err
        _append_report(report)
        if err:
            st.error(err)
        else:
            st.success("OpenAI suggestions generated.")

    st.caption(f"Current selected run: `{run_dir.resolve()}`")

    reports = st.session_state.get(RUN_REPORTS_KEY, [])
    if not isinstance(reports, list):
        reports = []
    selected_run_reports = [r for r in reports if str(r.get("run_dir", "")) == str(run_dir.resolve())]
    if not selected_run_reports:
        st.info("No OpenAI suggestions generated yet for this run.")
        return

    for i, report in enumerate(reversed(selected_run_reports), start=1):
        label = f"{i}. {report.get('timestamp_utc', 'n/a')}"
        with st.expander(label, expanded=(i == 1)):
            st.write(f"Model selected: `{report.get('openai_model', OPENAI_MODEL)}`")

            openai_error = report.get("openai_error")
            if openai_error:
                st.error(str(openai_error))

            suggestion_text = report.get("openai_suggestions")
            if isinstance(suggestion_text, str) and suggestion_text.strip():
                st.markdown(suggestion_text)


def _render_overview_tab(run_dir: Path, latest_runs_by_symbol: dict[str, Path]) -> None:
    if not run_dir.exists() or not run_dir.is_dir():
        st.info("Select a valid run directory in the sidebar.")
        return

    eval_dir = run_dir / "evaluation"
    metrics = _read_json(eval_dir / "metrics.json")
    if not metrics:
        st.warning("No evaluation metrics found for selected run.")
        return

    primary = _resolve_primary_non_persistence_baseline(metrics)
    primary_label = str(primary.get("label", "Primary Baseline"))
    rmse_model = _to_float(metrics.get("surface_forecast_iv_rmse"))
    rmse_primary = _to_float(metrics.get(primary["rmse_key"]))
    rmse_persistence = _to_float(metrics.get("surface_forecast_iv_rmse_baseline_persistence"))
    skill_primary = _to_float(metrics.get(primary["skill_key"]))
    skill_persistence = _to_float(metrics.get("surface_forecast_skill_mse_vs_persistence"))

    edge_rmse_primary, edge_rmse_primary_pct = _edge_vs_baseline(rmse_model, rmse_primary, better="lower")
    edge_rmse_pers, edge_rmse_pers_pct = _edge_vs_baseline(rmse_model, rmse_persistence, better="lower")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        f"RMSE Edge vs {primary_label}",
        _format_value(edge_rmse_primary, 6),
        delta=(f"{edge_rmse_primary_pct:+.2f}%" if edge_rmse_primary_pct is not None else None),
    )
    c2.metric(
        "RMSE Edge vs Persistence",
        _format_value(edge_rmse_pers, 6),
        delta=(f"{edge_rmse_pers_pct:+.2f}%" if edge_rmse_pers_pct is not None else None),
    )
    c3.metric(
        f"Forecast Skill vs {primary_label}",
        _format_value(skill_primary, 6),
    )
    c4.metric(
        "Forecast Skill vs Persistence",
        _format_value(skill_persistence, 6),
    )

    values = pd.DataFrame(
        [
            {"metric": "RMSE", "series": "model", "value": rmse_model},
            {"metric": "RMSE", "series": f"primary ({primary_label})", "value": rmse_primary},
            {"metric": "RMSE", "series": "persistence", "value": rmse_persistence},
        ]
    )
    values = values[np.isfinite(pd.to_numeric(values["value"], errors="coerce"))]
    if not values.empty:
        chart_values = (
            alt.Chart(values)
            .mark_bar()
            .encode(
                x=alt.X("metric:N", title="Metric"),
                xOffset=alt.XOffset("series:N"),
                y=alt.Y("value:Q", title="Value (lower is better)"),
                color=alt.Color("series:N", title="Series"),
                tooltip=["metric:N", "series:N", alt.Tooltip("value:Q", format=".6f")],
            )
            .properties(height=280)
        )
        st.altair_chart(chart_values, use_container_width=True)

    edges = pd.DataFrame(
        [
            {"comparison": f"RMSE vs {primary_label}", "edge": edge_rmse_primary},
            {"comparison": "RMSE vs Persistence", "edge": edge_rmse_pers},
        ]
    )
    edges = edges[np.isfinite(pd.to_numeric(edges["edge"], errors="coerce"))]
    if not edges.empty:
        chart_edges = (
            alt.Chart(edges)
            .mark_bar()
            .encode(
                x=alt.X("edge:Q", title="Edge (positive = model better)"),
                y=alt.Y("comparison:N", sort="-x", title=""),
                color=alt.condition("datum.edge >= 0", alt.value("#166534"), alt.value("#991b1b")),
                tooltip=["comparison:N", alt.Tooltip("edge:Q", format=".6f")],
            )
            .properties(height=220)
        )
        st.altair_chart(chart_edges, use_container_width=True)

    st.subheader("DTE/Moneyness Absolute Error Structure")
    by_dte = _read_parquet_or_csv(eval_dir / "surface_forecast_error_by_dte.parquet")
    by_x = _read_parquet_or_csv(eval_dir / "surface_forecast_error_by_moneyness.parquet")
    by_grid = _read_parquet_or_csv(eval_dir / "surface_forecast_error_grid.parquet")

    if by_dte.empty and by_x.empty and by_grid.empty:
        st.info("No DTE/moneyness forecast error artifacts found.")
    else:
        if not by_dte.empty:
            dte_plot = by_dte.melt(
                id_vars=["series", "dte"],
                value_vars=["rmse"],
                var_name="metric",
                value_name="value",
            )
            chart_dte = (
                alt.Chart(dte_plot)
                .mark_line(point=True)
                .encode(
                    x=alt.X("dte:Q", title="DTE"),
                    y=alt.Y("value:Q", title="Absolute error"),
                    color=alt.Color("series:N", title="Series"),
                    tooltip=["series:N", "dte:Q", alt.Tooltip("value:Q", format=".6f")],
                )
                .properties(height=260)
            )
            st.altair_chart(chart_dte, use_container_width=True)

        if not by_x.empty:
            x_plot = by_x.melt(
                id_vars=["series", "moneyness_x"],
                value_vars=["rmse"],
                var_name="metric",
                value_name="value",
            )
            chart_x = (
                alt.Chart(x_plot)
                .mark_line(point=True)
                .encode(
                    x=alt.X("moneyness_x:Q", title="Moneyness x=ln(K/S)"),
                    y=alt.Y("value:Q", title="Absolute error"),
                    color=alt.Color("series:N", title="Series"),
                    tooltip=["series:N", "moneyness_x:Q", alt.Tooltip("value:Q", format=".6f")],
                )
                .properties(height=260)
            )
            st.altair_chart(chart_x, use_container_width=True)

        if not by_grid.empty:
            for metric in ("rmse",):
                grid = by_grid.copy()
                grid[metric] = pd.to_numeric(grid[metric], errors="coerce")
                grid = grid[np.isfinite(grid[metric])]
                if grid.empty:
                    continue
                heat = (
                    alt.Chart(grid)
                    .mark_rect()
                    .encode(
                        x=alt.X("dte:O", title="DTE"),
                        y=alt.Y("moneyness_x:Q", title="Moneyness x=ln(K/S)"),
                        color=alt.Color(f"{metric}:Q", title=metric.upper()),
                        tooltip=["series:N", "dte:O", "moneyness_x:Q", alt.Tooltip(f"{metric}:Q", format=".6f")],
                    )
                    .properties(height=220, width=170)
                    .facet(column=alt.Column("series:N", title=f"{metric.upper()} by Series"))
                )
                st.altair_chart(heat, use_container_width=True)

    _render_focus_distribution_section(run_dir, latest_runs_by_symbol)


def _render_focus_distribution_section(run_dir: Path, latest_runs_by_symbol: dict[str, Path]) -> None:
    st.subheader("Chosen Focus Distribution")
    runs_by_symbol = dict(latest_runs_by_symbol) if latest_runs_by_symbol else {}
    selected_symbol = (_extract_symbol_from_run_dir(run_dir) or "").upper()
    if selected_symbol:
        runs_by_symbol[selected_symbol] = run_dir
    if not runs_by_symbol:
        st.info("No runs available to compute focus density.")
        return

    blocks: list[pd.DataFrame] = []
    for symbol, symbol_run_dir in sorted(runs_by_symbol.items()):
        info = _load_focus_distribution_for_run(symbol_run_dir)
        grid = info.get("grid_df")
        if not isinstance(grid, pd.DataFrame) or grid.empty:
            continue
        part = grid.copy()
        part["symbol"] = str(symbol).upper()
        part["effective_multiplier"] = pd.to_numeric(part.get("effective_multiplier"), errors="coerce")
        part = part[np.isfinite(part["effective_multiplier"])]
        if part.empty:
            continue
        avg = float(np.mean(part["effective_multiplier"]))
        denom = avg if abs(avg) > 1e-12 else 1.0
        part["relative_density"] = part["effective_multiplier"] / denom
        blocks.append(part)

    if not blocks:
        st.info("No focus-density grids found in current runs.")
        return

    combined = pd.concat(blocks, ignore_index=True)
    by_x = (
        combined.groupby(["symbol", "moneyness_x"], as_index=False)["relative_density"]
        .mean()
        .sort_values(["symbol", "moneyness_x"])
    )
    by_dte = (
        combined.groupby(["symbol", "dte"], as_index=False)["relative_density"]
        .mean()
        .sort_values(["symbol", "dte"])
    )

    chart_x = (
        alt.Chart(by_x)
        .mark_line(point=True)
        .encode(
            x=alt.X("moneyness_x:Q", title="Moneyness x=ln(K/S)"),
            y=alt.Y("relative_density:Q", title="Relative density"),
            color=alt.Color("symbol:N", title="Symbol"),
            tooltip=[
                "symbol:N",
                alt.Tooltip("moneyness_x:Q", format=".3f"),
                alt.Tooltip("relative_density:Q", format=".4f"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(chart_x, use_container_width=True)

    chart_dte = (
        alt.Chart(by_dte)
        .mark_line(point=True)
        .encode(
            x=alt.X("dte:Q", title="DTE"),
            y=alt.Y("relative_density:Q", title="Relative density"),
            color=alt.Color("symbol:N", title="Symbol"),
            tooltip=[
                "symbol:N",
                alt.Tooltip("dte:Q", format=".0f"),
                alt.Tooltip("relative_density:Q", format=".4f"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(chart_dte, use_container_width=True)


def _render_surface_3d_tab(run_dir: Path) -> None:
    if not run_dir.exists() or not run_dir.is_dir():
        st.info("Select a valid run directory in the sidebar.")
        return
    eval_dir = run_dir / "evaluation"
    surf_path = eval_dir / "surface_predictions.npz"
    if not surf_path.exists():
        st.info("No surface predictions artifact found.")
        return
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except Exception:
        st.warning("Plotly is not available in this environment.")
        return

    try:
        payload = np.load(surf_path, allow_pickle=False)
    except Exception:
        st.info("Unable to read surface predictions artifact.")
        return

    required = {"dates", "x_grid", "tenor_days", "iv_surface_obs", "iv_surface_forecast", "forecast_entry_index", "forecast_target_index"}
    if not required.issubset(set(payload.files)):
        st.info("surface_predictions.npz is missing required arrays.")
        return

    dates = payload["dates"].astype(str)
    x_grid = payload["x_grid"].astype(np.float32)
    tenor_days = payload["tenor_days"].astype(np.int32)
    obs = payload["iv_surface_obs"].astype(np.float32)
    forecast = payload["iv_surface_forecast"].astype(np.float32)
    entry_idx = payload["forecast_entry_index"].astype(np.int32)
    target_idx = payload["forecast_target_index"].astype(np.int32)

    if len(entry_idx) == 0:
        st.info("No tomorrow forecast pairs available in this run.")
        return

    max_i = int(len(entry_idx) - 1)
    i = int(
        st.slider(
            "Forecast day index",
            min_value=0,
            max_value=max_i,
            value=max_i,
            step=1,
            help="Slide to choose which next-day IV surface comparison to display.",
        )
    )
    e = int(entry_idx[i])
    t = int(target_idx[i])
    pred_surface = forecast[e]
    actual_surface = obs[t]
    rmse_day = float(np.sqrt(np.mean((pred_surface - actual_surface) ** 2)))

    c1, c2, c3 = st.columns(3)
    c1.metric("Entry Date", dates[e])
    c2.metric("Target Date (Tomorrow)", dates[t])
    c3.metric("Daily RMSE", f"{rmse_day:.6f}")

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=("Predicted IV Surface (Tomorrow)", "Actual IV Surface (Tomorrow)"),
    )
    fig.add_trace(
        go.Surface(
            z=pred_surface,
            x=tenor_days,
            y=x_grid,
            colorscale="Viridis",
            showscale=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Surface(
            z=actual_surface,
            x=tenor_days,
            y=x_grid,
            colorscale="Viridis",
            showscale=False,
        ),
        row=1,
        col=2,
    )
    fig.update_scenes(xaxis_title="DTE", yaxis_title="Moneyness x=ln(K/S)", zaxis_title="IV")
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    _load_dotenv()
    _ensure_state()

    st.set_page_config(page_title="ivdyn: Training + Evaluation", layout="wide")
    _inject_style()

    st.title("ivdyn: Training + Evaluation Center")
    st.caption("Backtest UI removed. This view is now for training/evaluation artifact inspection.")

    outputs_root = Path("outputs")
    discovered_runs = _discover_run_dirs(outputs_root)
    run_options = [str(p) for p in discovered_runs]
    latest_runs_by_symbol = _discover_latest_runs_by_symbol(outputs_root)
    symbol_options = sorted(latest_runs_by_symbol)

    active_default = str(st.session_state.get(ACTIVE_RUN_KEY, "") or "")
    if not active_default:
        if latest_runs_by_symbol:
            newest_run = max(latest_runs_by_symbol.values(), key=_run_recency)
            active_default = str(newest_run)
        elif run_options:
            active_default = run_options[0]
    elif latest_runs_by_symbol and not Path(active_default).exists():
        newest_run = max(latest_runs_by_symbol.values(), key=_run_recency)
        active_default = str(newest_run)
    elif run_options and active_default not in run_options:
        active_default = run_options[0]

    with st.sidebar:
        st.header("Inspect Run")
        inspect_source_choices = ["Latest by symbol", "Manual run directory"]
        default_source_idx = 0 if symbol_options else 1
        inspect_source = st.radio("Inspect source", options=inspect_source_choices, index=default_source_idx)

        if inspect_source == "Latest by symbol" and symbol_options:
            default_symbol = _extract_symbol_from_run_dir(Path(active_default)) if active_default else None
            if default_symbol not in latest_runs_by_symbol:
                newest_run = max(latest_runs_by_symbol.values(), key=_run_recency)
                default_symbol = _extract_symbol_from_run_dir(newest_run) or symbol_options[0]
            default_symbol_idx = symbol_options.index(default_symbol) if default_symbol in symbol_options else 0
            selected_symbol = st.selectbox("Symbol (latest run)", options=symbol_options, index=default_symbol_idx)
            inspect_run_raw = str(latest_runs_by_symbol[selected_symbol])
            st.text_input("Resolved run directory", value=inspect_run_raw, disabled=True)
        else:
            inspect_run_raw = st.text_input("Run directory", value=active_default)
            if run_options:
                idx = run_options.index(active_default) if active_default in run_options else 0
                picked = st.selectbox("Recent discovered runs", options=run_options, index=idx)
                if picked and picked != inspect_run_raw:
                    inspect_run_raw = picked

    inspect_run = Path(inspect_run_raw).expanduser()
    if inspect_run.exists() and inspect_run.is_dir():
        st.session_state[ACTIVE_RUN_KEY] = str(inspect_run.resolve())

    tabs = st.tabs(["Overview", "Tomorrow Surface 3D", "OpenAI Suggestions"])

    with tabs[0]:
        _render_overview_tab(inspect_run, latest_runs_by_symbol)

    with tabs[1]:
        _render_surface_3d_tab(inspect_run)

    with tabs[2]:
        _render_openai_tab(inspect_run)


if __name__ == "__main__":
    main()
