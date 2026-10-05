"""Logged chronological research search; reserved 2026 returns are never loaded."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import sha256
from .backtest import sharpe
from .extension_features import build


def simulate_cash(asset, cash, weights, cost_bps=5):
    if not asset.index.equals(cash.index) or not asset.index.equals(weights.index):
        raise ValueError("Portfolio series must align")
    if not np.isfinite(np.c_[asset, cash, weights]).all():
        raise ValueError("Missing portfolio values")
    if not weights.between(0, 1).all():
        raise ValueError("Long/cash weights must be in [0,1]")
    c, old = cost_bps / 10000, 0.0
    records = []
    for date, r in asset.items():
        rf, w = float(cash.loc[date]), float(weights.loc[date])
        diff = w - old
        q = diff / (1 + c * w) if diff >= 0 else -diff / (1 - c * w)
        gross = w * r + (1 - w) * rf
        net = (1 - c * q) * (1 + gross) - 1
        old = w * (1 + r) / (1 + gross)
        records.append([date, r, rf, w, q, net, old])
    out = pd.DataFrame(
        records,
        columns=[
            "signal_date",
            "asset_return",
            "cash_return",
            "weight",
            "turnover",
            "net_return",
            "end_weight",
        ],
    ).set_index("signal_date")
    if len(out):
        w = out.end_weight.iloc[-1]
        out.iloc[-1, out.columns.get_loc("net_return")] = (1 + out.net_return.iloc[-1]) * (
            1 - c * w
        ) - 1
        out.iloc[-1, out.columns.get_loc("turnover")] += w
    out["excess_return"] = out.net_return - out.cash_return
    out.loc[out.excess_return.abs() < 1e-14, "excess_return"] = 0.0
    out["equity"] = (1 + out.net_return).cumprod()
    return out


def score(ledger):
    equity = ledger.equity
    years = ledger.index.year.unique()
    annual = [sharpe(ledger.loc[ledger.index.year == y, "excess_return"]) for y in years]
    return {
        "sharpe_excess": sharpe(ledger.excess_return),
        "sharpe_zero_rf": sharpe(ledger.net_return),
        "cagr": float(equity.iloc[-1] ** (252 / len(ledger)) - 1),
        "max_drawdown": float((equity / equity.cummax().clip(lower=1) - 1).min()),
        "average_exposure": float(ledger.weight.mean()),
        "annual_turnover": float(ledger.turnover.sum() * 252 / len(ledger)),
        "positive_year_fraction": float(np.mean(np.array(annual) > 0)),
        "worst_year_sharpe": float(min(annual)),
        "days": len(ledger),
    }


def model(kind, seed):
    if kind == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=1000.0),
        )
    return HistGradientBoostingRegressor(
        max_iter=100,
        learning_rate=0.04,
        max_leaf_nodes=7,
        min_samples_leaf=60,
        l2_regularization=20,
        early_stopping=False,
        random_state=seed,
    )


def predict_candidate(x, meta, spec, config):
    h = spec["horizon"]
    y = (meta[f"target_{h}"] - meta.cash_yield_annual * h / 252) / (
        meta.annual_vol / np.sqrt(252) * np.sqrt(h)
    )
    # Winsorization constant is fixed, not estimated from future/test outcomes.
    y = y.clip(-4, 4)
    start, end = pd.Timestamp(config["development_start"]), pd.Timestamp(config["selection_end"])
    test_dates = x.index[(x.index >= start) & (meta.label_end < end) & meta.cash_return.notna()]
    predictions = pd.Series(np.nan, index=test_dates, name="score")
    audit = []
    # Quarterly refit schedule; all test outcomes remain unseen by that fold's model.
    quarters = test_dates.to_period("Q")
    for q in quarters.unique():
        dates = test_dates[quarters == q]
        cutoff = dates.min()
        train = (x.index < cutoff) & (meta[f"label_end_{h}"] < cutoff) & y.notna()
        if spec["window"] == "five_year":
            train &= x.index >= cutoff - pd.DateOffset(years=5)
        if train.sum() < config["minimum_train_rows"]:
            raise ValueError(f"Insufficient training rows for {q}: {train.sum()}")
        fitted = model(spec["model"], config["seed"])
        # Some library versions cannot bin all-missing columns. Learn the mask
        # from this training fold alone; never inspect test availability.
        usable = x.loc[train].notna().any(axis=0)
        with threadpool_limits(limits=1):
            fitted.fit(x.loc[train, usable], y.loc[train])
            predictions.loc[dates] = fitted.predict(x.loc[dates, usable])
        audit.append(
            {
                "quarter": str(q),
                "test_start": str(cutoff.date()),
                "train_rows": int(train.sum()),
                "train_start": str(x.index[train].min().date()),
                "last_training_outcome": str(meta.loc[train, f"label_end_{h}"].max().date()),
                "horizon": h,
                "usable_features": int(usable.sum()),
            }
        )
    if predictions.isna().any():
        raise ValueError("Unscored dates in experiment")
    return predictions, pd.DataFrame(audit)


def policy_weights(predictions, meta, policy, smoothing, config):
    risk = (
        config["annual_vol_target"] / meta.loc[predictions.index, "annual_vol"].clip(lower=0.01)
    ).clip(upper=1)
    p = predictions.ewm(span=smoothing, adjust=False).mean() if smoothing > 1 else predictions
    if policy == "sign":
        w = (p > 0).astype(float)
    elif policy == "conviction":
        w = (p / 0.5).clip(lower=0, upper=1)
    elif policy == "hysteresis":
        w = p.map(lambda v: 1.0 if v > 0.1 else (0.0 if v < -0.1 else np.nan)).ffill().fillna(0)
    else:
        raise ValueError(policy)
    return w * risk


def specifications(round_name):
    groups = ["uso", "markets", "macro", "combined", "all"]
    if round_name == "ablation":
        return [
            {"model": m, "group": g, "horizon": 1, "window": "expanding"}
            for m in ["ridge", "tree"]
            for g in groups
        ]
    if round_name == "horizons":
        return [
            {"model": m, "group": g, "horizon": h, "window": w}
            for m in ["ridge", "tree"]
            for g in ["macro", "all"]
            for h in [5, 21]
            for w in ["expanding", "five_year"]
        ]
    if round_name == "news_refresh":
        return [
            {"model": m, "group": "all", "horizon": h, "window": w}
            for m in ["ridge", "tree"]
            for h in [1, 5, 21]
            for w in ["expanding", "five_year"]
        ]
    if round_name == "innovations":
        return [
            {"model": m, "group": "innovations", "horizon": h, "window": w}
            for m in ["ridge", "tree"]
            for h in [1, 5, 21]
            for w in ["expanding", "five_year"]
        ]
    raise ValueError(round_name)


def run(root, round_name, tag=None):
    config = json.loads((root / "extension_config.json").read_text())
    x, meta, groups, lineage, status = build(root)
    if round_name == "innovations":
        from .extension_innovations import add_innovations

        x, cols, extra_lineage, revisions = add_innovations(
            x, pd.read_csv(root / "data/extension/eia_releases.csv")
        )
        groups["innovations"] = cols
        lineage = lineage.join(extra_lineage)
    output = root / "results/extension" / (tag or round_name)
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {p.name: sha256(p) for p in (root / "oil_research").glob("extension_*.py")}
    input_names = ["eia_releases.csv", "curve_releases.csv", "headlines.csv", "news_coverage.csv"]
    fingerprints = {
        name: sha256(root / "data/extension" / name)
        for name in input_names
        if (root / "data/extension" / name).exists()
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            {"sources": source_hashes, "inputs": fingerprints, "config": config}, sort_keys=True
        ).encode()
    ).hexdigest()
    plan = {
        "round": round_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "source_hashes": source_hashes,
        "input_hashes": fingerprints,
        "config": config,
        "source_status": status,
        "feature_count": len(x.columns),
        "specifications": specifications(round_name),
        "policies": ["sign", "conviction", "hysteresis"],
        "smoothing": [1, 5],
        "warning": "All scores are selection-contaminated development results, even though predictions are chronologically cross-fitted.",
    }
    plan_path = output / "plan.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        if previous["fingerprint"] != fingerprint:
            raise ValueError(
                "Sources or data changed; preserve old experiment and use a new round directory"
            )
    else:
        plan_path.write_text(json.dumps(plan, indent=2))
        snapshots = output / "source_snapshots"
        snapshots.mkdir(exist_ok=True)
        for name in fingerprints:
            shutil.copyfile(root / "data/extension" / name, snapshots / name)
        code_snapshot = output / "code_snapshot"
        code_snapshot.mkdir(exist_ok=True)
        for name in source_hashes:
            shutil.copyfile(root / "oil_research" / name, code_snapshot / name)
        x.to_csv(output / "model_input_features.csv")
        meta.to_csv(output / "model_input_outcomes.csv")
    lineage.to_csv(output / "feature_availability.csv")
    pd.DataFrame({"feature": x.columns, "missing_fraction": x.isna().mean().to_numpy()}).to_csv(
        output / "feature_audit.csv", index=False
    )
    registry = output / "registry.jsonl"
    metrics = []
    for i, spec in enumerate(plan["specifications"], 1):
        name = f"{spec['model']}_{spec['group']}_h{spec['horizon']}_{spec['window']}"
        ppath = output / f"{name}_predictions.csv"
        if ppath.exists():
            predictions = pd.read_csv(ppath, index_col=0, parse_dates=True).score
        else:
            with registry.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "event": "started",
                            "name": name,
                            "spec": spec,
                            "time": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    + "\n"
                )
            predictions, audit = predict_candidate(x[groups[spec["group"]]], meta, spec, config)
            predictions.to_csv(ppath)
            audit.to_csv(output / f"{name}_audit.csv", index=False)
            with registry.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "event": "completed",
                            "name": name,
                            "time": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    + "\n"
                )
        for policy in plan["policies"]:
            for smoothing in plan["smoothing"]:
                weights = policy_weights(predictions, meta, policy, smoothing, config)
                for cost in config["cost_bps"]:
                    ledger = simulate_cash(
                        meta.loc[predictions.index, "target_return"],
                        meta.loc[predictions.index, "cash_return"],
                        weights,
                        cost,
                    )
                    metrics.append(
                        {
                            "candidate": name,
                            "policy": policy,
                            "smoothing": smoothing,
                            "cost_bps": cost,
                            **spec,
                            **score(ledger),
                        }
                    )
        pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
        primary = (
            pd.DataFrame(metrics)
            .query("cost_bps == 5")
            .sort_values("sharpe_excess", ascending=False)
        )
        best = primary.iloc[0]
        print(
            f"{round_name}: {i}/{len(plan['specifications'])} models; best DEVELOPMENT excess Sharpe {best.sharpe_excess:.3f} ({best.candidate}, {best.policy}, smooth {best.smoothing})",
            flush=True,
        )
    primary = (
        pd.DataFrame(metrics).query("cost_bps == 5").sort_values("sharpe_excess", ascending=False)
    )
    primary.to_csv(output / "ranking.csv", index=False)
    # Save baseline ledgers on exactly the same dates, including the cash proxy.
    dates = predictions.index
    risk = (config["annual_vol_target"] / meta.loc[dates, "annual_vol"]).clip(upper=1)
    baseline_weights = {
        "buy_hold": pd.Series(1.0, index=dates),
        "vol_target": risk,
        "momentum": risk * (meta.loc[dates, "momentum"] > 0),
    }
    baselines = []
    for name, w in baseline_weights.items():
        ledger = simulate_cash(
            meta.loc[dates, "target_return"], meta.loc[dates, "cash_return"], w, 5
        )
        ledger.to_csv(output / f"baseline_{name}.csv")
        baselines.append({"candidate": name, **score(ledger)})
    pd.DataFrame(baselines).to_csv(output / "baselines.csv", index=False)
    best = primary.iloc[0]
    p = pd.read_csv(
        output / f"{best.candidate}_predictions.csv", index_col=0, parse_dates=True
    ).score
    w = policy_weights(p, meta, best.policy, int(best.smoothing), config)
    ledger = simulate_cash(
        meta.loc[p.index, "target_return"], meta.loc[p.index, "cash_return"], w, 5
    )
    ledger.to_csv(output / "selected_development_ledger.csv")
    annual = []
    for year, g in ledger.groupby(ledger.index.year):
        g = g.copy()
        g["equity"] = (1 + g.net_return).cumprod()
        annual.append({"year": int(year), **score(g)})
    pd.DataFrame(annual).to_csv(output / "selected_annual.csv", index=False)
    print(primary.head(8).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("round", choices=["ablation", "horizons", "news_refresh", "innovations"])
    parser.add_argument(
        "--tag", help="New output directory name for an explicitly revised experiment"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run(args.root.resolve(), args.round, args.tag)
