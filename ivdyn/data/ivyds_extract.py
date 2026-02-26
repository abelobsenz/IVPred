"""WRDS IvyDB/CRSP extraction into a clean daily contract panel."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(slots=True)
class WRDSExtractConfig:
    options_path: Path
    underlying_path: Path
    out_dir: Path
    forward_path: Path | None = None
    symbols: tuple[str, ...] = ()
    start_date: str | None = None
    end_date: str | None = None
    chunksize: int = 300_000
    min_iv: float = 1e-4
    max_iv: float = 4.0


def _parse_date_opt(v: str | None) -> date | None:
    if not v:
        return None
    return date.fromisoformat(v)


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _load_underlying_panel(path: Path) -> pd.DataFrame:
    udf = pd.read_csv(path, compression="infer")
    if udf.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "underlying_close",
                "underlying_bid",
                "underlying_ask",
                "underlying_open",
                "underlying_ret",
                "underlying_volume",
            ]
        )

    date_col = "date" if "date" in udf.columns else "DATE"
    ticker_col = "TICKER" if "TICKER" in udf.columns else "ticker"

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(udf[date_col], errors="coerce").dt.date,
            "ticker": udf[ticker_col].astype(str).str.upper().str.strip(),
            "underlying_close": _safe_numeric(udf.get("PRC", udf.get("prc", np.nan))).abs(),
            "underlying_bid": _safe_numeric(udf.get("BID", udf.get("bid", np.nan))),
            "underlying_ask": _safe_numeric(udf.get("ASK", udf.get("ask", np.nan))),
            "underlying_open": _safe_numeric(udf.get("OPENPRC", udf.get("openprc", np.nan))).abs(),
            "underlying_ret": _safe_numeric(udf.get("RET", udf.get("ret", np.nan))),
            "underlying_volume": _safe_numeric(udf.get("VOL", udf.get("vol", np.nan))),
        }
    )
    out = out.dropna(subset=["date", "ticker"])
    out = out[out["underlying_close"] > 0.0]
    out = out.drop_duplicates(subset=["date", "ticker"], keep="last").sort_values(["ticker", "date"])
    out.reset_index(drop=True, inplace=True)
    return out


def _pick_col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _load_forward_panel(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["secid", "ticker", "date", "exdate", "forward_price_ext"])
    if not path.exists():
        raise FileNotFoundError(path)
    f = pd.read_csv(path, compression="infer")
    if f.empty:
        return pd.DataFrame(columns=["secid", "ticker", "date", "exdate", "forward_price_ext"])

    date_col = _pick_col(f, ("date", "DATE"))
    exdate_col = _pick_col(f, ("exdate", "expiration", "EXDATE", "EXPIRATION"))
    fwd_col = _pick_col(f, ("forward_price", "ForwardPrice", "FORWARD_PRICE"))
    if date_col is None or exdate_col is None or fwd_col is None:
        raise RuntimeError(
            f"Forward file must include date/exdate/forward columns; got columns={list(f.columns)}"
        )
    secid_col = _pick_col(f, ("secid", "SECID"))
    ticker_col = _pick_col(f, ("ticker", "TICKER", "OFTIC"))

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(f[date_col], errors="coerce").dt.date,
            "exdate": pd.to_datetime(f[exdate_col], errors="coerce").dt.date,
            "forward_price_ext": _safe_numeric(f[fwd_col]),
        }
    )
    if secid_col is not None:
        out["secid"] = pd.to_numeric(f[secid_col], errors="coerce").astype("Int64")
    else:
        out["secid"] = pd.Series([pd.NA] * len(out), dtype="Int64")
    if ticker_col is not None:
        out["ticker"] = f[ticker_col].astype(str).str.upper().str.strip()
    else:
        out["ticker"] = ""

    out = out.dropna(subset=["date", "exdate", "forward_price_ext"])
    out = out[np.isfinite(out["forward_price_ext"]) & (out["forward_price_ext"] > 0.0)]
    out = out.sort_values(["date", "exdate", "ticker"]).drop_duplicates(
        subset=["secid", "ticker", "date", "exdate"], keep="last"
    )
    out.reset_index(drop=True, inplace=True)
    return out[["secid", "ticker", "date", "exdate", "forward_price_ext"]]


def _normalize_options_chunk(
    chunk: pd.DataFrame,
    *,
    underlying: pd.DataFrame,
    forwards: pd.DataFrame,
    symbols: set[str] | None,
    start: date | None,
    end: date | None,
    min_iv: float,
    max_iv: float,
) -> pd.DataFrame:
    out = chunk.copy()

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out["exdate"] = pd.to_datetime(out["exdate"], errors="coerce").dt.date
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["secid"] = pd.to_numeric(out.get("secid"), errors="coerce").astype("Int64")
    if symbols:
        out = out[out["ticker"].isin(symbols)]
    if start is not None:
        out = out[out["date"] >= start]
    if end is not None:
        out = out[out["date"] <= end]
    if out.empty:
        return out

    out["call_put"] = out["cp_flag"].astype(str).str.upper().str[0]
    out["strike"] = _safe_numeric(out["strike_price"])
    if float(out["strike"].median(skipna=True)) > 10_000.0:
        out["strike"] = out["strike"] / 1000.0

    out["bid"] = _safe_numeric(out["best_bid"])
    out["ask"] = _safe_numeric(out["best_offer"])
    out["mid"] = 0.5 * (out["bid"] + out["ask"])
    out["spread"] = (out["ask"] - out["bid"]).clip(lower=0.0)
    out["rel_spread"] = out["spread"] / out["mid"].clip(lower=1e-6)
    out["volume"] = _safe_numeric(out.get("volume", 0.0)).fillna(0.0).clip(lower=0.0)
    out["open_interest"] = _safe_numeric(out.get("open_interest", 0.0)).fillna(0.0).clip(lower=0.0)
    out["iv"] = _safe_numeric(out["impl_volatility"])
    out["delta"] = _safe_numeric(out.get("delta", np.nan))
    out["gamma"] = _safe_numeric(out.get("gamma", np.nan))
    out["vega"] = _safe_numeric(out.get("vega", np.nan))
    out["theta"] = _safe_numeric(out.get("theta", np.nan))
    out["forward_price"] = _safe_numeric(out.get("forward_price", np.nan))

    out["dte"] = (pd.to_datetime(out["exdate"]) - pd.to_datetime(out["date"])).dt.days
    out = out.merge(underlying, on=["date", "ticker"], how="left")
    out = out[out["underlying_close"] > 0.0]
    if out.empty:
        return out

    if not forwards.empty:
        fwd_by_secid = forwards[forwards["secid"].notna()][["secid", "date", "exdate", "forward_price_ext"]]
        if not fwd_by_secid.empty:
            out = out.merge(fwd_by_secid, on=["secid", "date", "exdate"], how="left")
        else:
            out["forward_price_ext"] = np.nan

        missing_ext = ~np.isfinite(pd.to_numeric(out["forward_price_ext"], errors="coerce"))
        if missing_ext.any():
            fwd_by_ticker = forwards[["ticker", "date", "exdate", "forward_price_ext"]].drop_duplicates(
                subset=["ticker", "date", "exdate"], keep="last"
            )
            out = out.merge(
                fwd_by_ticker,
                on=["ticker", "date", "exdate"],
                how="left",
                suffixes=("", "_tk"),
            )
            out["forward_price_ext"] = np.where(
                np.isfinite(pd.to_numeric(out["forward_price_ext"], errors="coerce")),
                pd.to_numeric(out["forward_price_ext"], errors="coerce"),
                pd.to_numeric(out["forward_price_ext_tk"], errors="coerce"),
            )
            out = out.drop(columns=["forward_price_ext_tk"])
        out["forward_price"] = np.where(
            np.isfinite(pd.to_numeric(out["forward_price_ext"], errors="coerce"))
            & (pd.to_numeric(out["forward_price_ext"], errors="coerce") > 0.0),
            pd.to_numeric(out["forward_price_ext"], errors="coerce"),
            out["forward_price"],
        )
        out = out.drop(columns=["forward_price_ext"], errors="ignore")

    out["forward_price"] = np.where(
        np.isfinite(out["forward_price"]) & (out["forward_price"] > 0.0),
        out["forward_price"],
        out["underlying_close"],
    )
    out["tau"] = out["dte"] / 365.0
    out["x"] = np.log(out["strike"] / np.clip(out["forward_price"], 1e-6, None))
    out["cp_sign"] = np.where(out["call_put"] == "C", 1.0, -1.0)
    out["liquidity"] = (
        1.0
        + np.log1p(out["volume"].clip(lower=0.0))
        + 0.35 * np.log1p(out["open_interest"].clip(lower=0.0))
    )

    out = out[
        (out["bid"] > 0.0)
        & (out["ask"] >= out["bid"])
        & (out["mid"] > 0.0)
        & (out["dte"] > 0)
        & (out["call_put"].isin(["C", "P"]))
        & np.isfinite(out["x"])
        & np.isfinite(out["tau"])
        & (out["tau"] > 0.0)
        & np.isfinite(out["iv"])
        & (out["iv"] >= float(min_iv))
        & (out["iv"] <= float(max_iv))
    ]
    if out.empty:
        return out

    keep_cols = [
        "date",
        "ticker",
        "secid",
        "optionid",
        "exdate",
        "dte",
        "call_put",
        "strike",
        "bid",
        "ask",
        "mid",
        "spread",
        "rel_spread",
        "volume",
        "open_interest",
        "iv",
        "delta",
        "gamma",
        "vega",
        "theta",
        "forward_price",
        "underlying_close",
        "underlying_bid",
        "underlying_ask",
        "underlying_open",
        "underlying_ret",
        "underlying_volume",
        "tau",
        "x",
        "cp_sign",
        "liquidity",
    ]
    for col in keep_cols:
        if col not in out.columns:
            out[col] = np.nan
    out = out[keep_cols].copy()

    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["exdate"] = pd.to_datetime(out["exdate"]).dt.date
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["call_put"] = out["call_put"].astype(str)
    out["optionid"] = out["optionid"].astype(str)
    return out


def build_wrds_contract_panel(cfg: WRDSExtractConfig) -> dict[str, Path]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    contracts_out = cfg.out_dir / "contracts_daily.parquet"
    underlying_out = cfg.out_dir / "underlying_daily.parquet"
    summary_out = cfg.out_dir / "wrds_extract_summary.json"

    symbols = {s.upper().strip() for s in cfg.symbols if str(s).strip()}
    start = _parse_date_opt(cfg.start_date)
    end = _parse_date_opt(cfg.end_date)

    underlying = _load_underlying_panel(cfg.underlying_path)
    forwards = _load_forward_panel(cfg.forward_path)
    if symbols and not forwards.empty:
        forwards = forwards[forwards["ticker"].isin(symbols)].copy()
    if start is not None and not forwards.empty:
        forwards = forwards[forwards["date"] >= start]
    if end is not None and not forwards.empty:
        forwards = forwards[forwards["date"] <= end]
    if symbols:
        underlying = underlying[underlying["ticker"].isin(symbols)].copy()
    if start is not None:
        underlying = underlying[underlying["date"] >= start]
    if end is not None:
        underlying = underlying[underlying["date"] <= end]
    if underlying.empty:
        raise RuntimeError("Underlying panel is empty after filters; cannot build WRDS contract panel.")
    underlying.to_parquet(underlying_out, index=False)

    usecols = [
        "secid",
        "date",
        "exdate",
        "cp_flag",
        "strike_price",
        "best_bid",
        "best_offer",
        "volume",
        "open_interest",
        "impl_volatility",
        "delta",
        "gamma",
        "vega",
        "theta",
        "optionid",
        "forward_price",
        "ticker",
    ]

    writer: pq.ParquetWriter | None = None
    rows_raw = 0
    rows_saved = 0
    chunks_used = 0
    date_min: date | None = None
    date_max: date | None = None

    for chunk in pd.read_csv(cfg.options_path, compression="infer", usecols=usecols, chunksize=int(cfg.chunksize)):
        rows_raw += int(len(chunk))
        norm = _normalize_options_chunk(
            chunk,
            underlying=underlying,
            forwards=forwards,
            symbols=symbols if symbols else None,
            start=start,
            end=end,
            min_iv=float(cfg.min_iv),
            max_iv=float(cfg.max_iv),
        )
        if norm.empty:
            continue
        chunks_used += 1
        rows_saved += int(len(norm))
        dmin = norm["date"].min()
        dmax = norm["date"].max()
        date_min = dmin if date_min is None else min(date_min, dmin)
        date_max = dmax if date_max is None else max(date_max, dmax)

        table = pa.Table.from_pandas(norm, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(contracts_out), table.schema, compression="zstd")
        writer.write_table(table)

    if writer is not None:
        writer.close()
    else:
        pd.DataFrame().to_parquet(contracts_out, index=False)
        raise RuntimeError("No WRDS option rows survived filtering; contracts_daily.parquet is empty.")

    summary = {
        "config": {
            **asdict(cfg),
            "options_path": str(cfg.options_path),
            "underlying_path": str(cfg.underlying_path),
            "forward_path": str(cfg.forward_path) if cfg.forward_path else None,
            "out_dir": str(cfg.out_dir),
        },
        "symbols": sorted(symbols) if symbols else [],
        "rows_raw_options": int(rows_raw),
        "rows_saved_contracts": int(rows_saved),
        "chunks_used": int(chunks_used),
        "date_min": date_min.isoformat() if date_min else None,
        "date_max": date_max.isoformat() if date_max else None,
        "underlying_rows": int(len(underlying)),
        "forward_rows": int(len(forwards)),
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "contracts": contracts_out,
        "underlying": underlying_out,
        "summary": summary_out,
    }
