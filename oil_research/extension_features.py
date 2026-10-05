"""As-of release joins and sparse archive-news features with lineage checks."""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .data import load, sha256
from .features import dataset


def cutoffs(dates):
    # NY17:00 is after equity close; execute no earlier than the next session open.
    return (
        (pd.DatetimeIndex(dates) + pd.Timedelta(hours=17))
        .tz_localize("America/New_York")
        .tz_convert("UTC")
    )


def join_releases(events, dates, prefix):
    required = {"series", "role", "value", "available_at"}
    if not required <= set(events):
        raise ValueError(f"Release ledger lacks {required-set(events)}")
    events = events.copy()
    events["available_at"] = pd.to_datetime(events.available_at, utc=True)
    if events.duplicated(["series", "role", "available_at"]).any():
        raise ValueError("Ambiguous duplicate release values")
    clock = pd.DataFrame({"cutoff": cutoffs(dates)}, index=dates)
    values, lineage = {}, {}
    for (series, role), group in events.groupby(["series", "role"]):
        key = f"{prefix}_{series}_{role}"
        matched = pd.merge_asof(
            clock.reset_index(drop=True),
            group.sort_values("available_at")[["available_at", "value"]],
            left_on="cutoff",
            right_on="available_at",
            direction="backward",
            allow_exact_matches=True,
        )
        # Every included value must have been available by the signal clock.
        assert not (matched.available_at > matched.cutoff).any()
        age = (matched.cutoff - matched.available_at).dt.total_seconds() / 86400
        # Stale series are missing, never implicitly current.
        values[key] = pd.Series(matched.value.to_numpy(), index=dates).mask(age.to_numpy() > 100)
        values[key + "_age_days"] = pd.Series(age.to_numpy(), index=dates).clip(upper=365)
        lineage[key] = pd.Series(matched.available_at.to_numpy(), index=dates)
    return pd.DataFrame(values), pd.DataFrame(lineage)


def news_features(headlines, coverage, dates):
    """Fixed lexicon, no future-trained language model or present-day article text."""
    h = headlines.copy()
    h["available_at"] = pd.to_datetime(h.available_at, utc=True)
    h = h.sort_values("available_at").drop_duplicates("title", keep="first")
    cov = pd.to_datetime(coverage.available_at, utc=True).sort_values()
    # These are topic counts, not an asserted causal oil-price sentiment model.
    patterns = {
        "supply_disruption": r"\b(?:outage|disruption|attack|sanction|strike|shut|shutdown|cut|cuts|war)\b",
        "supply_growth": r"\b(?:output|production|supply|drilling|rigs|surplus|glut)\b",
        "demand": r"\b(?:demand|china|growth|recession|economy|consumption)\b",
        "price_up": r"\b(?:rise|rises|rally|rallies|surge|surges|soar|soars|gain|gains)\b",
        "price_down": r"\b(?:fall|falls|drop|drops|slump|slumps|crash|crashes|plunge|plunges)\b",
        "opec": r"\bopec\b",
    }
    for key, pattern in patterns.items():
        h[key] = h.title.str.lower().str.contains(pattern, regex=True).astype(float)
    rows = []
    for date, cutoff in zip(dates, cutoffs(dates)):
        prior = h.loc[h.available_at <= cutoff]
        seen = cov[cov <= cutoff]
        row = {
            "news_capture_age_days": (
                (cutoff - seen.iloc[-1]).total_seconds() / 86400 if len(seen) else np.nan
            )
        }
        for window in [7, 30, 90]:
            since = cutoff - pd.Timedelta(days=window)
            part = prior.loc[prior.available_at > since]
            n_capture = int(((cov > since) & (cov <= cutoff)).sum())
            row[f"news_captures_{window}"] = n_capture
            # No archive capture means missing coverage, not zero news.
            row[f"news_titles_per_capture_{window}"] = (
                len(part) / n_capture if n_capture else np.nan
            )
            for key in patterns:
                row[f"news_{key}_share_{window}"] = float(part[key].mean()) if len(part) else np.nan
        rows.append(row)
    return pd.DataFrame(rows, index=dates)


def build(root):
    base_config = json.loads((root / "config.json").read_text())
    config = json.loads((root / "extension_config.json").read_text())
    bars = load(base_config, root / "data/raw")
    x, meta = dataset(bars, "USO")
    base_columns = list(x)
    lineage = pd.DataFrame(index=x.index)
    source_status = []
    ext = root / "data/extension"
    for filename, prefix in [("eia_releases.csv", "eia"), ("curve_releases.csv", "curve")]:
        path = ext / filename
        if not path.exists():
            raise FileNotFoundError(f"Required release source missing: {path}")
        events = pd.read_csv(path)
        values, audit = join_releases(events, x.index, prefix)
        x = x.join(values)
        lineage = lineage.join(audit)
        source_status.append(
            {
                "source": prefix,
                "events": len(events),
                "sha256": sha256(path),
                "timing": "explicit historical availability",
            }
        )
    for a, b, name in [("DGS10", "DGS2", "10y_2y"), ("DGS10", "DGS3MO", "10y_3m")]:
        x[f"curve_slope_{name}"] = x[f"curve_{a}_level"] - x[f"curve_{b}_level"]
        x[f"curve_slope_change_{name}"] = (
            x[f"curve_{a}_change_21_observations"] - x[f"curve_{b}_change_21_observations"]
        )
    eia_names = {
        c.split("eia_", 1)[1].rsplit("_last_month_estimate", 1)[0]
        for c in x
        if c.startswith("eia_") and c.endswith("_last_month_estimate")
    }
    for name in eia_names:
        past, older = f"eia_{name}_last_month_estimate", f"eia_{name}_previous_month_estimate"
        forecast, current = f"eia_{name}_three_month_forecast", f"eia_{name}_current_month_forecast"
        if older in x:
            x[f"eia_{name}_recent_change"] = x[past] - x[older]
        if forecast in x and current in x:
            x[f"eia_{name}_forecast_change"] = x[forecast] - x[current]
    for role in ["last_month_estimate", "current_month_forecast", "three_month_forecast"]:
        a, b = f"eia_world_liquids_production_{role}", f"eia_world_liquids_consumption_{role}"
        if a in x and b in x:
            x[f"eia_global_balance_{role}"] = x[a] - x[b]
    manifest = json.loads((ext / "markets/manifest.json").read_text())
    for ticker, info in manifest.items():
        path = ext / "markets" / info["path"]
        if sha256(path) != info["sha256"]:
            raise ValueError(f"Market snapshot checksum mismatch: {ticker}")
        frame = pd.read_csv(path, index_col="Date", parse_dates=True)
        c = frame.Close.reindex(bars["USO"].index)
        if ticker.startswith("^"):
            # Index publication can trail the equity close; use a full session lag.
            c = c.shift(1)
        for n in [1, 5, 21, 63]:
            x[f"extra_{ticker}_return_{n}"] = c.pct_change(n, fill_method=None).reindex(x.index)
        x[f"extra_{ticker}_vol_21"] = (
            c.pct_change(fill_method=None).rolling(21).std().reindex(x.index)
        )
        if ticker.startswith("^"):
            x[f"extra_{ticker}_level"] = c.reindex(x.index)
    for ticker in ["USL", "BNO", "XLE", "DBC"]:
        peer = x.get(f"extra_{ticker}_return_21", x.get(f"{ticker}_return_21"))
        if peer is not None:
            # ETF relative performance proxy; do not call this a futures curve.
            x[f"extra_USO_minus_{ticker}_momentum"] = x.USO_return_21 - peer
    if (ext / "headlines.csv").exists() and (ext / "news_coverage.csv").stat().st_size > 10:
        h, cov = pd.read_csv(ext / "headlines.csv"), pd.read_csv(ext / "news_coverage.csv")
        x = x.join(news_features(h, cov, x.index))
        source_status.append(
            {
                "source": "OilPrice archived titles",
                "events": len(h),
                "captures": len(cov),
                "sha256": sha256(ext / "headlines.csv"),
                "timing": "actual archive capture + 24 hours; sparse coverage",
            }
        )
    else:
        source_status.append(
            {
                "source": "OilPrice archived titles",
                "status": "not yet available; excluded, no synthetic replacement",
            }
        )
    x = x.replace([np.inf, -np.inf], np.nan)
    cutoff = pd.Timestamp(config["selection_end"])
    if (x.index >= cutoff).any():
        raise ValueError("Reserved dates entered research feature matrix")
    groups = {
        "uso": [c for c in x if c.startswith("USO_")],
        "markets": [c for c in x if c in base_columns or c.startswith("extra_")],
        "macro": [c for c in x if c.startswith(("USO_", "eia_", "curve_"))],
        "combined": [c for c in x if not c.startswith("news_")],
        "all": list(x),
    }
    # Economic forecast horizons. End labels are purged independently per horizon.
    dates = pd.Series(bars["USO"].index, index=bars["USO"].index)
    for horizon in [1, 5, 21]:
        meta[f"target_{horizon}"] = (
            bars["USO"].Open.shift(-(horizon + 1)) / bars["USO"].Open.shift(-1) - 1
        ).reindex(x.index)
        meta[f"label_end_{horizon}"] = dates.shift(-(horizon + 1)).reindex(x.index)
    meta["cash_yield_annual"] = x.curve_DGS3MO_level / 100
    meta["cash_return"] = (
        meta.cash_yield_annual * (meta.label_end - meta.execution_date).dt.days / 365
    )
    return x, meta, groups, lineage, source_status


if __name__ == "__main__":
    root = Path.cwd()
    x, meta, groups, lineage, status = build(root)
    out = root / "data/extension"
    x.to_csv(out / "features.csv")
    meta.to_csv(out / "outcomes.csv")
    lineage.to_csv(out / "lineage.csv")
    (out / "feature_groups.json").write_text(json.dumps(groups, indent=2))
    (out / "source_status.json").write_text(json.dumps(status, indent=2))
    print(
        f"Built {len(x)} rows, {len(x.columns)} features. Groups: { {k:len(v) for k,v in groups.items()} }"
    )
