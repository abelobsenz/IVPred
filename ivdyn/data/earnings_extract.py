"""Earnings calendar parsing and daily feature engineering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(slots=True)
class EarningsFeatureConfig:
    next_windows: tuple[int, ...] = (7, 14, 30)
    since_windows: tuple[int, ...] = (7, 14, 30)


def _to_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def _ann_time_to_bucket(s: pd.Series) -> pd.Series:
    txt = s.astype(str).str.strip().str.upper()
    out = pd.Series(np.array(["unknown"] * len(txt), dtype=object), index=s.index, dtype=object)

    pre_pat = txt.str.contains(r"\b(?:BMO|PRE[-_ ]?OPEN|BEFORE[-_ ]?OPEN)\b", regex=True, na=False)
    after_pat = txt.str.contains(r"\b(?:AMC|AFTER[-_ ]?CLOSE|POST[-_ ]?CLOSE)\b", regex=True, na=False)
    intra_pat = txt.str.contains(r"\b(?:INTRA|DURING[-_ ]?MARKET|MARKET[-_ ]?HOURS)\b", regex=True, na=False)
    out.loc[pre_pat] = "preopen"
    out.loc[after_pat] = "afterclose"
    out.loc[intra_pat] = "intraday"

    # Parse plain clock times (e.g. 06:54:00, 15:10, 4:20 PM).
    clock = txt.str.extract(r"^\s*(\d{1,2})(?::(\d{2}))?(?::(\d{2}))?\s*([AP]M)?\s*$")
    hh = pd.to_numeric(clock[0], errors="coerce")
    mm = pd.to_numeric(clock[1], errors="coerce").fillna(0.0)
    ampm = clock[3].fillna("")
    has_clock = hh.notna() & (out == "unknown")
    if has_clock.any():
        hour24 = hh.copy()
        pm_mask = has_clock & (ampm == "PM") & (hour24 < 12)
        am_midnight_mask = has_clock & (ampm == "AM") & (hour24 == 12)
        hour24.loc[pm_mask] = hour24.loc[pm_mask] + 12.0
        hour24.loc[am_midnight_mask] = 0.0
        tod = hour24 * 60.0 + mm
        out.loc[has_clock & (tod < 9.5 * 60.0)] = "preopen"
        out.loc[has_clock & (tod >= 9.5 * 60.0) & (tod < 16.0 * 60.0)] = "intraday"
        out.loc[has_clock & (tod >= 16.0 * 60.0)] = "afterclose"

    out.loc[out == "unknown"] = "afterclose"
    return out.astype(str)


def load_earnings_calendar(path: Path) -> pd.DataFrame:
    """Load an earnings file and normalize to [ticker, earnings_date, earnings_bucket]."""
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "earnings_date", "earnings_bucket"])

    ticker_col = None
    for c in ("OFTIC", "TICKER", "ticker"):
        if c in df.columns:
            ticker_col = c
            break
    if ticker_col is None:
        raise RuntimeError(f"Earnings file missing ticker column: {path}")

    date_col = None
    for c in ("ANNDATS", "announce_date", "date"):
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise RuntimeError(f"Earnings file missing announce date column: {path}")

    out = pd.DataFrame(
        {
            "ticker": df[ticker_col].astype(str).str.upper().str.strip(),
            "earnings_date": _to_date_series(df[date_col]),
        }
    )
    if "ANNTIMS" in df.columns:
        out["earnings_bucket"] = _ann_time_to_bucket(df["ANNTIMS"])
    elif "announce_time" in df.columns:
        out["earnings_bucket"] = _ann_time_to_bucket(df["announce_time"])
    else:
        out["earnings_bucket"] = "afterclose"
    out = out.dropna(subset=["ticker", "earnings_date"]).drop_duplicates().sort_values(["ticker", "earnings_date"])
    out.reset_index(drop=True, inplace=True)
    return out


def daily_earnings_features(
    *,
    dates: list[date],
    ticker: str,
    calendar: pd.DataFrame | None,
    cfg: EarningsFeatureConfig | None = None,
) -> pd.DataFrame:
    """Return daily earnings timing features aligned to `dates` for one ticker."""
    config = cfg or EarningsFeatureConfig()
    idx = pd.Index(dates, name="date")
    out = pd.DataFrame(index=idx)
    out["days_to_next_earnings"] = np.nan
    out["days_since_last_earnings"] = np.nan
    out["is_earnings_window"] = 0.0

    for prefix in ("next", "last"):
        out[f"{prefix}_earnings_bucket_preopen"] = 0.0
        out[f"{prefix}_earnings_bucket_intraday"] = 0.0
        out[f"{prefix}_earnings_bucket_afterclose"] = 0.0

    if calendar is None or calendar.empty:
        for w in config.next_windows:
            out[f"earnings_in_next_{int(w)}d"] = 0.0
        for w in config.since_windows:
            out[f"earnings_in_last_{int(w)}d"] = 0.0
        return out

    cal = calendar[calendar["ticker"].astype(str).str.upper() == ticker.upper()].copy()
    if cal.empty:
        for w in config.next_windows:
            out[f"earnings_in_next_{int(w)}d"] = 0.0
        for w in config.since_windows:
            out[f"earnings_in_last_{int(w)}d"] = 0.0
        return out

    cal["earnings_date"] = pd.to_datetime(cal["earnings_date"], errors="coerce").dt.date
    cal = cal.dropna(subset=["earnings_date"]).sort_values("earnings_date")
    if cal.empty:
        for w in config.next_windows:
            out[f"earnings_in_next_{int(w)}d"] = 0.0
        for w in config.since_windows:
            out[f"earnings_in_last_{int(w)}d"] = 0.0
        return out
    if "earnings_bucket" not in cal.columns:
        cal["earnings_bucket"] = "afterclose"
    cal = (
        cal.groupby("earnings_date", as_index=False)["earnings_bucket"]
        .first()
        .sort_values("earnings_date")
        .reset_index(drop=True)
    )
    e_dates = cal["earnings_date"].tolist()
    e_buckets = cal["earnings_bucket"].astype(str).str.lower().tolist()
    e_ord = np.array([d.toordinal() for d in e_dates], dtype=np.int64)
    d_ord = np.array([d.toordinal() for d in dates], dtype=np.int64)
    pos = np.searchsorted(e_ord, d_ord, side="left")

    to_next = np.full(len(d_ord), np.nan, dtype=np.float32)
    valid_next = pos < len(e_ord)
    to_next[valid_next] = (e_ord[pos[valid_next]] - d_ord[valid_next]).astype(np.float32)

    to_prev = np.full(len(d_ord), np.nan, dtype=np.float32)
    prev_pos = pos - 1
    valid_prev = prev_pos >= 0
    to_prev[valid_prev] = (d_ord[valid_prev] - e_ord[prev_pos[valid_prev]]).astype(np.float32)

    out["days_to_next_earnings"] = to_next
    out["days_since_last_earnings"] = to_prev

    for w in config.next_windows:
        w_f = float(max(int(w), 1))
        out[f"earnings_in_next_{int(w)}d"] = ((to_next >= 0.0) & (to_next <= w_f)).astype(np.float32)

    for w in config.since_windows:
        w_f = float(max(int(w), 1))
        out[f"earnings_in_last_{int(w)}d"] = ((to_prev >= 0.0) & (to_prev <= w_f)).astype(np.float32)

    # Short event window around earnings date for jump-aware dynamics.
    out["is_earnings_window"] = (
        ((to_next >= 0.0) & (to_next <= 3.0))
        | ((to_prev >= 0.0) & (to_prev <= 1.0))
    ).astype(np.float32)

    next_bucket = np.array(["unknown"] * len(d_ord), dtype=object)
    valid_next_bucket = pos < len(e_buckets)
    if np.any(valid_next_bucket):
        p = pos[valid_next_bucket].astype(np.int64)
        next_bucket[valid_next_bucket] = np.asarray(e_buckets, dtype=object)[p]
    out["next_earnings_bucket_preopen"] = (next_bucket == "preopen").astype(np.float32)
    out["next_earnings_bucket_intraday"] = (next_bucket == "intraday").astype(np.float32)
    out["next_earnings_bucket_afterclose"] = (next_bucket == "afterclose").astype(np.float32)

    last_bucket = np.array(["unknown"] * len(d_ord), dtype=object)
    valid_prev_bucket = prev_pos >= 0
    if np.any(valid_prev_bucket):
        p = prev_pos[valid_prev_bucket].astype(np.int64)
        last_bucket[valid_prev_bucket] = np.asarray(e_buckets, dtype=object)[p]
    out["last_earnings_bucket_preopen"] = (last_bucket == "preopen").astype(np.float32)
    out["last_earnings_bucket_intraday"] = (last_bucket == "intraday").astype(np.float32)
    out["last_earnings_bucket_afterclose"] = (last_bucket == "afterclose").astype(np.float32)

    return out


def earnings_before_expiry_flags(days_to_next: np.ndarray, tenor_days: np.ndarray) -> np.ndarray:
    """Feature matrix indicating if next earnings date occurs before each tenor bucket."""
    dnext = np.asarray(days_to_next, dtype=np.float32).reshape(-1, 1)
    tenor = np.asarray(tenor_days, dtype=np.float32).reshape(1, -1)
    valid = np.isfinite(dnext) & (dnext >= 0.0)
    flags = valid & (dnext <= tenor)
    return flags.astype(np.float32)
