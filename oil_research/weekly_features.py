"""Compact weekly features; seasonal statistics learn strictly from older releases."""

from pathlib import Path
import numpy as np
import pandas as pd
from .extension_features import cutoffs


def seasonal_features(releases):
    data = releases.copy().sort_values(["available_at", "observation_date"]).reset_index(drop=True)
    data["available_at"] = pd.to_datetime(data.available_at, utc=True)
    data["observation_date"] = pd.to_datetime(data.observation_date)
    cols = [
        "crude_change",
        "gasoline_change",
        "distillate_change",
        "refinery_inputs",
        "utilization",
        "imports",
        "exports",
    ]
    out = data[["available_at", "observation_date"]].copy()
    weeks = data.observation_date.dt.isocalendar().week.astype(int)
    for col in cols:
        values = pd.to_numeric(data[col], errors="coerce")
        z = []
        for i, row in data.iterrows():
            prior = (
                (data.available_at < row.available_at)
                & (data.observation_date < row.observation_date)
                & (data.observation_date >= row.observation_date - pd.DateOffset(years=5))
            )
            delta = (weeks - weeks.iloc[i]).abs()
            near = np.minimum(delta, 53 - delta) <= 4
            history = values[prior & near].dropna()
            if prior.sum() < 104 or len(history) < 12 or history.std() < 1e-8:
                z.append(np.nan)
            else:
                z.append(float(np.clip((values.iloc[i] - history.mean()) / history.std(), -4, 4)))
        out["inventory_" + col + "_seasonal_z"] = z
    return out


def position_features(releases):
    d = releases.copy().sort_values("available_at").reset_index(drop=True)
    d["available_at"] = pd.to_datetime(d.available_at, utc=True)
    d["observation_date"] = pd.to_datetime(d.observation_date)
    out = d[["available_at", "observation_date"]].copy()
    net = (d.money_long - d.money_short) / d.open_interest
    out["position_net"] = net
    out["position_producer_net"] = (d.producer_long - d.producer_short) / d.open_interest
    # No multi-month differences masquerading as weekly changes after an outage.
    gap = d.observation_date.diff().dt.days
    out["position_change"] = net.diff().where(gap.between(5, 9))
    percentiles = []
    for i, row in d.iterrows():
        old = net[
            (d.available_at < row.available_at)
            & (d.observation_date >= row.observation_date - pd.DateOffset(years=3))
        ]
        percentiles.append(float((old < net.iloc[i]).mean()) if len(old) >= 52 else np.nan)
    out["position_percentile"] = percentiles
    return out


def asof_features(events, index):
    left = pd.DataFrame({"signal_date": index, "cutoff": cutoffs(index)})
    right = events.sort_values(["available_at", "observation_date"]).drop_duplicates(
        "available_at", keep="last"
    )
    joined = pd.merge_asof(
        left, right, left_on="cutoff", right_on="available_at", direction="backward"
    )
    assert not (joined.available_at > joined.cutoff).any()
    stale = (joined.signal_date - joined.observation_date).dt.days > 21
    cols = [c for c in events if c not in ["available_at", "observation_date"]]
    result = joined.set_index("signal_date")[cols].copy()
    for col in cols:
        result[col] = result[col].mask(stale.to_numpy(dtype=bool, na_value=False))
    lineage = joined.set_index("signal_date")[["cutoff", "available_at", "observation_date"]]
    return result, lineage


def build(root):
    meta = pd.read_csv(
        root / "results/extension/news_refresh/model_input_outcomes.csv",
        index_col=0,
        parse_dates=True,
    )
    for c in ["execution_date", "label_end", "label_end_1", "label_end_5", "label_end_21"]:
        if c in meta:
            meta[c] = pd.to_datetime(meta[c])
    assert meta.index.max() < pd.Timestamp("2026-01-01")
    price = pd.read_csv(root / "data/raw/USO.csv", index_col=0, parse_dates=True).Close
    vol = price.pct_change().rolling(63, min_periods=40).std()
    trend = pd.DataFrame(
        {
            f"trend_{h}": (price.pct_change(h) / (vol * np.sqrt(h))).clip(-4, 4)
            for h in [21, 63, 126]
        }
    ).reindex(meta.index)
    eia = pd.read_csv(root / "data/weekly/eia_weekly.csv")
    cot = pd.read_csv(root / "data/weekly/cftc_weekly.csv")
    inv, il = asof_features(seasonal_features(eia), meta.index)
    pos, pl = asof_features(position_features(cot), meta.index)
    x = trend.join(inv).join(pos)
    groups = {"trend": list(trend), "inventory": list(inv), "positioning": list(pos)}
    lineage = il.add_prefix("eia_").join(pl.add_prefix("cftc_"))
    return x, meta, groups, lineage
