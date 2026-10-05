"""Features at close t; outcome is adjusted open t+1 to adjusted open t+2."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    # Explicit simple rolling RSI variant, not Wilder's exponential smoothing.
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    result = 100 * gain / (gain + loss)
    return result.mask((gain + loss) == 0, 50)


def build_features(bars: dict[str, pd.DataFrame], target: str = "USO") -> pd.DataFrame:
    x = {}
    for ticker, b in bars.items():
        c, v = b.Close, b.Volume
        returns = c.pct_change(fill_method=None)
        for n in [1, 2, 5, 10, 21, 63]:
            x[f"{ticker}_return_{n}"] = c.pct_change(n, fill_method=None)
        for n in [5, 21, 63]:
            x[f"{ticker}_volatility_{n}"] = returns.rolling(n).std() * np.sqrt(252)
        for n in [10, 21, 63]:
            x[f"{ticker}_ma_distance_{n}"] = c / c.rolling(n).mean() - 1
        x[f"{ticker}_rsi_14"] = rsi(c) / 100
        x[f"{ticker}_intraday"] = c / b.Open - 1
        x[f"{ticker}_range"] = (b.High - b.Low) / c
        x[f"{ticker}_overnight"] = b.Open / c.shift(1) - 1
        x[f"{ticker}_relative_volume_21"] = v / v.rolling(21).mean() - 1
    target_ret = bars[target].Close.pct_change(fill_method=None)
    for ticker, b in bars.items():
        if ticker == target:
            continue
        peer_ret = b.Close.pct_change(fill_method=None)
        x[f"cross_{ticker}_correlation_63"] = target_ret.rolling(63).corr(peer_ret)
        x[f"cross_{ticker}_relative_momentum_21"] = (
            x[f"{target}_return_21"] - x[f"{ticker}_return_21"]
        )
    return pd.DataFrame(x, index=bars[target].index).replace([np.inf, -np.inf], np.nan)


def dataset(
    bars: dict[str, pd.DataFrame], target: str = "USO"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = build_features(bars, target)
    b = bars[target]
    dates = pd.Series(b.index, index=b.index)
    meta = pd.DataFrame(
        {
            "target_return": b.Open.shift(-2) / b.Open.shift(-1) - 1,
            "execution_date": dates.shift(-1),
            "label_end": dates.shift(-2),
            "annual_vol": b.Close.pct_change(fill_method=None).rolling(21).std() * np.sqrt(252),
            "momentum": b.Close.pct_change(21, fill_method=None),
            "rsi": rsi(b.Close),
        }
    )
    valid = x.notna().all(axis=1) & meta.notna().all(axis=1)
    x, meta = x.loc[valid], meta.loc[valid]
    # Dropping an interior day would silently create gaps in an open-to-open strategy.
    if len(x) and not x.index.equals(b.loc[x.index[0] : x.index[-1]].index):
        raise ValueError(
            "Interior feature gaps: inspect source data instead of skipping holding periods"
        )
    return x, meta


def feature_dictionary(columns) -> pd.DataFrame:
    rows = []
    for name in columns:
        group = "cross_asset" if name.startswith("cross_") else name.split("_")[0]
        rows.append(
            {
                "feature": name,
                "group": group,
                "available": "after signal-date close",
                "transform": "trailing-only; see features.py for exact formula",
            }
        )
    return pd.DataFrame(rows)
