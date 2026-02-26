"""Numpy/Torch compatible feature scalers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional import guard
    torch = None  # type: ignore[assignment]


@dataclass(slots=True)
class ArrayScaler:
    mean: np.ndarray
    std: np.ndarray
    clip_min: np.ndarray | None = None
    clip_max: np.ndarray | None = None
    z_clip: float | None = None

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        eps: float = 1e-6,
        winsor_quantile: float = 0.0,
        z_clip: float | None = None,
    ) -> "ArrayScaler":
        x_fit = np.asarray(x, dtype=np.float32)
        clip_min: np.ndarray | None = None
        clip_max: np.ndarray | None = None
        q = float(winsor_quantile)
        if q > 0.0:
            q = min(max(q, 1e-4), 0.2)
            lo = np.quantile(x_fit, q, axis=0, keepdims=True).astype(np.float32)
            hi = np.quantile(x_fit, 1.0 - q, axis=0, keepdims=True).astype(np.float32)
            x_fit = np.clip(x_fit, lo, hi)
            clip_min = lo
            clip_max = hi
        mean = x_fit.mean(axis=0, keepdims=True).astype(np.float32)
        std = x_fit.std(axis=0, keepdims=True).astype(np.float32)
        std = np.where(std < eps, 1.0, std)
        z_val: float | None = None
        if z_clip is not None:
            zf = float(z_clip)
            if zf > 0:
                z_val = zf
        return cls(
            mean=mean,
            std=std.astype(np.float32),
            clip_min=clip_min,
            clip_max=clip_max,
            z_clip=z_val,
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_work = np.asarray(x, dtype=np.float32)
        if self.clip_min is not None and self.clip_max is not None:
            x_work = np.clip(x_work, self.clip_min, self.clip_max)
        out = (x_work - self.mean) / self.std
        if self.z_clip is not None and self.z_clip > 0:
            out = np.clip(out, -self.z_clip, self.z_clip)
        return out.astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float32)

    def transform_torch(self, x):
        if torch is None:
            raise RuntimeError("torch is not available")
        if self.clip_min is not None and self.clip_max is not None:
            lo = torch.as_tensor(self.clip_min, dtype=x.dtype, device=x.device)
            hi = torch.as_tensor(self.clip_max, dtype=x.dtype, device=x.device)
            x = torch.clamp(x, min=lo, max=hi)
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        out = (x - mean) / std
        if self.z_clip is not None and self.z_clip > 0:
            out = torch.clamp(out, min=-float(self.z_clip), max=float(self.z_clip))
        return out

    def inverse_transform_torch(self, x):
        if torch is None:
            raise RuntimeError("torch is not available")
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return x * std + mean

    def state(self) -> dict[str, np.ndarray]:
        state: dict[str, np.ndarray | float] = {"mean": self.mean.copy(), "std": self.std.copy()}
        if self.clip_min is not None:
            state["clip_min"] = self.clip_min.copy()
        if self.clip_max is not None:
            state["clip_max"] = self.clip_max.copy()
        if self.z_clip is not None:
            state["z_clip"] = float(self.z_clip)
        return state  # type: ignore[return-value]

    @classmethod
    def from_state(cls, state: dict[str, np.ndarray]) -> "ArrayScaler":
        clip_min_raw = state.get("clip_min")
        clip_max_raw = state.get("clip_max")
        z_clip_raw = state.get("z_clip")
        z_clip_val: float | None = None
        if z_clip_raw is not None:
            z_clip_val = float(z_clip_raw)  # type: ignore[arg-type]
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            std=np.asarray(state["std"], dtype=np.float32),
            clip_min=(np.asarray(clip_min_raw, dtype=np.float32) if clip_min_raw is not None else None),
            clip_max=(np.asarray(clip_max_raw, dtype=np.float32) if clip_max_raw is not None else None),
            z_clip=z_clip_val,
        )
