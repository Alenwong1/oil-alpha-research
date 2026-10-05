"""Download adjusted daily bars; cache and verify immutable local snapshots."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_bars(df: pd.DataFrame, ticker: str) -> None:
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not set(required) <= set(df):
        raise ValueError(f"{ticker}: missing OHLCV columns")
    if df.empty or df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError(f"{ticker}: empty, duplicate, or unsorted dates")
    if df.index.tz is not None:
        raise ValueError(f"{ticker}: expected timezone-naive exchange dates")
    values = df[required].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{ticker}: missing/nonfinite bars; no silent forward filling")
    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError(f"{ticker}: nonpositive prices")
    if (df.Volume < 0).any():
        raise ValueError(f"{ticker}: negative volume")
    if (df.High + 1e-6 < df[["Open", "Close", "Low"]].max(axis=1)).any():
        raise ValueError(f"{ticker}: inconsistent high")
    if (df.Low - 1e-6 > df[["Open", "Close", "High"]].min(axis=1)).any():
        raise ValueError(f"{ticker}: inconsistent low")


def download(config: dict, directory: Path, refresh: bool = False) -> dict:
    import yfinance as yf

    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists() and not refresh:
        load(config, directory)
        return json.loads((directory / "manifest.json").read_text())
    if any(directory.glob("*.csv")) and not refresh:
        raise ValueError("Incomplete cache: use --refresh to replace it explicitly")
    # Keep library cache local rather than writing into the user's home directory.
    yf.set_tz_cache_location(str(directory / ".yf-cache"))
    manifest = {
        "provider": "Yahoo Finance via yfinance",
        "yfinance_version": yf.__version__,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "start": config["data_start"],
        "end_exclusive": config["data_end_exclusive"],
        "auto_adjust": True,
        "repair": False,
        "warning": "Retrospective vendor-adjusted data, not a point-in-time archive.",
        "files": {},
    }
    for ticker in config["tickers"]:
        print(f"Downloading {ticker}", flush=True)
        df = yf.Ticker(ticker).history(
            start=config["data_start"],
            end=config["data_end_exclusive"],
            interval="1d",
            auto_adjust=True,
            actions=True,
            repair=False,
            raise_errors=True,
        )
        if df.empty:
            raise RuntimeError(f"No real market data returned for {ticker}")
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        df.index.name = "Date"
        validate_bars(df, ticker)
        path = directory / f"{ticker}.csv"
        df.to_csv(path, float_format="%.12g")
        manifest["files"][ticker] = {
            "sha256": sha256(path),
            "rows": len(df),
            "first_date": str(df.index.min().date()),
            "last_date": str(df.index.max().date()),
            "large_close_moves": int((df.Close.pct_change(fill_method=None).abs() > 0.30).sum()),
            "splits": {
                str(d.date()): float(v)
                for d, v in df.get("Stock Splits", pd.Series(dtype=float)).items()
                if v
            },
        }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    load(config, directory)
    return manifest


def load(config: dict, directory: Path) -> dict[str, pd.DataFrame]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest["start"] != config["data_start"]
        or manifest["end_exclusive"] != config["data_end_exclusive"]
    ):
        raise ValueError(
            "Cache range differs from config; use a separate cache or explicit refresh"
        )
    bars = {}
    for ticker in config["tickers"]:
        path = directory / f"{ticker}.csv"
        if sha256(path) != manifest["files"][ticker]["sha256"]:
            raise ValueError(f"Checksum mismatch: {ticker}")
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        validate_bars(df, ticker)
        bars[ticker] = df
    target_dates = bars[config["target"]].index
    for ticker, df in bars.items():
        if not df.index.equals(target_dates):
            raise ValueError(
                f"{ticker}: calendar mismatch with target; inspect missing bars before proceeding"
            )
    return bars
