"""Path helpers for run directories and artifact discovery."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def make_run_dir(base_dir: Path, prefix: str = "run") -> Path:
    run = base_dir / f"{prefix}_{utc_timestamp()}"
    run.mkdir(parents=True, exist_ok=False)
    (base_dir / "latest.txt").write_text(str(run.resolve()), encoding="utf-8")
    return run


def resolve_latest(base_dir: Path) -> Path | None:
    p = base_dir / "latest.txt"
    if p.exists():
        target = Path(p.read_text(encoding="utf-8").strip())
        if target.exists():
            return target

    # Fallback: pick newest run-like directory if latest.txt is missing/stale.
    candidates = [d for d in base_dir.glob("run_*") if d.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return candidates[0]
