"""Canonical schemas and validation helpers for options chain snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

CANONICAL_CHAIN_COLUMNS = [
    "date",
    "expiry",
    "dte",
    "call_put",
    "optionid",
    "symbol",
    "strike",
    "bid",
    "ask",
    "mid",
    "last",
    "volume",
    "open_interest",
    "underlying_close",
    "forward_price",
    "delta",
    "gamma",
    "theta",
    "vega",
    "iv",
]


@dataclass(slots=True)
class DaySnapshot:
    asof: date
    symbol: str
    chain: pd.DataFrame


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def normalize_chain_df(df: pd.DataFrame, asof: str | date, symbol: str) -> pd.DataFrame:
    """Convert a raw dataframe into the canonical chain format."""
    out = df.copy()

    if "date" not in out.columns:
        out["date"] = asof
    out["date"] = pd.to_datetime(out["date"]).dt.date

    if "expiry" not in out.columns:
        out["expiry"] = pd.NaT
    out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce").dt.date

    if "symbol" not in out.columns:
        out["symbol"] = symbol
    if "optionid" not in out.columns:
        out["optionid"] = np.nan

    if "call_put" in out.columns:
        out["call_put"] = out["call_put"].astype(str).str.upper().str[0]
    else:
        out["call_put"] = "C"

    _coerce_numeric(
        out,
        [
            "dte",
            "strike",
            "bid",
            "ask",
            "mid",
            "last",
            "volume",
            "open_interest",
            "underlying_close",
            "forward_price",
            "delta",
            "gamma",
            "theta",
            "vega",
            "iv",
        ],
    )

    if "dte" not in out.columns or out["dte"].isna().all():
        out["dte"] = (
            pd.to_datetime(out["expiry"], errors="coerce")
            - pd.to_datetime(out["date"], errors="coerce")
        ).dt.days

    if "mid" not in out.columns:
        out["mid"] = np.nan
    if "last" not in out.columns:
        out["last"] = np.nan
    if "bid" not in out.columns:
        out["bid"] = np.nan
    if "ask" not in out.columns:
        out["ask"] = np.nan

    out["mid"] = out["mid"].where(out["mid"].notna(), (out["bid"] + out["ask"]) / 2.0)
    out["mid"] = out["mid"].where(out["mid"].notna(), out["last"])

    if "underlying_close" not in out.columns:
        out["underlying_close"] = np.nan
    if "forward_price" not in out.columns:
        out["forward_price"] = np.nan
    # Keep usable forward proxies when provided values are missing or invalid.
    fwd = pd.to_numeric(out["forward_price"], errors="coerce")
    spot = pd.to_numeric(out["underlying_close"], errors="coerce")
    out["forward_price"] = np.where(np.isfinite(fwd) & (fwd > 0.0), fwd, spot)

    for col in CANONICAL_CHAIN_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    out = out[CANONICAL_CHAIN_COLUMNS]

    # Basic sanity filters
    out = out[out["strike"] > 0]
    out = out[out["underlying_close"] > 0]
    out = out[out["dte"] > 0]
    out = out[out["mid"] > 0]
    out = out[out["call_put"].isin(["C", "P"])]

    return out.reset_index(drop=True)
