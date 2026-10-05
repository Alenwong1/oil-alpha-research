"""Predeclared small-feature weekly research with share-preserving rebalance schedules."""

from __future__ import annotations
import json, shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
from .weekly_features import build
from .extension_experiments import score
from .data import sha256


def simulate_scheduled(
    asset,
    cash,
    dt,
    targets,
    rebalance,
    cost_bps=5,
    borrow_annual=0.03,
    financing_spread=0.02,
    deadband=0.0,
):
    if not all(asset.index.equals(v.index) for v in [cash, dt, targets, rebalance]):
        raise ValueError("Inputs must align")
    if not np.isfinite(np.c_[asset, cash, dt, targets]).all() or targets.abs().max() > 2 + 1e-10:
        raise ValueError("Invalid inputs/cap")
    c = cost_bps / 10000
    old = 0.0
    halted = False
    bankrupt = False
    records = []
    for r, rf, days, target, scheduled in np.c_[asset, cash, dt, targets, rebalance]:
        safety = abs(old) > 2
        change = bool(scheduled) and (
            abs(target - old) >= deadband or target == 0 or target * old < 0
        )
        w = target if change else old
        if safety:
            w = float(np.clip(w, -2, 2))
        if halted:
            w = 0.0
        diff = w - old
        q = diff / (1 + c * w) if diff >= 0 else -diff / (1 - c * w)
        financing = max(w - 1, 0) * (rf + financing_spread * days)
        borrow = max(-w, 0) * borrow_annual * days
        gross = w * r + max(1 - max(w, 0), 0) * rf - financing - borrow
        margin = 0
        insolvent = 0
        if bankrupt:
            net = 0.0
            end = 0.0
        elif 1 + gross <= 0:
            net = -1.0
            end = 0.0
            halted = True
            bankrupt = True
            insolvent = 1
        else:
            net = (1 - c * q) * (1 + gross) - 1
            end = w * (1 + r) / (1 + gross)
            if abs(end) > 4:
                net = (1 + net) * (1 - c * abs(end)) - 1
                q += abs(end)
                end = 0.0
                halted = True
                margin = 1
                if net <= -1:
                    net = -1.0
                    bankrupt = True
                    insolvent = 1
        records.append(
            [r, rf, w, q, financing, borrow, net, end, int(change), int(safety), margin, insolvent]
        )
        old = end
    out = pd.DataFrame(
        records,
        index=asset.index,
        columns=[
            "asset_return",
            "cash_return",
            "weight",
            "turnover",
            "financing_fraction",
            "borrow_fraction",
            "net_return",
            "end_weight",
            "scheduled_trade",
            "safety_rebalance",
            "margin_call",
            "insolvent",
        ],
    )
    out.index.name = "signal_date"
    if len(out):
        exit = abs(out.end_weight.iloc[-1])
        out.iloc[-1, out.columns.get_loc("net_return")] = (1 + out.net_return.iloc[-1]) * (
            1 - c * exit
        ) - 1
        out.iloc[-1, out.columns.get_loc("turnover")] += exit
    out["excess_return"] = out.net_return - out.cash_return
    out.loc[out.excess_return.abs() < 1e-14, "excess_return"] = 0.0
    out["equity"] = (1 + out.net_return).cumprod()
    return out


def forecasts(x, meta, groups):
    index = x.index[
        (x.index >= pd.Timestamp("2015-01-01"))
        & (meta.label_end < pd.Timestamp("2026-01-01"))
        & meta.cash_return.notna()
    ]
    y = (
        (meta.target_21 - meta.cash_yield_annual * 21 / 252)
        / (meta.annual_vol / np.sqrt(252) * np.sqrt(21))
    ).clip(-4, 4)
    output = pd.DataFrame(index=index)
    audit = []
    for name, cols in groups.items():
        output[name] = np.nan
        for q in index.to_period("Q").unique():
            dates = index[index.to_period("Q") == q]
            cutoff = dates.min()
            valid = x[cols].notna().all(axis=1)
            train = (x.index < cutoff) & (meta.label_end_21 < cutoff) & y.notna() & valid
            if train.sum() < 252:
                continue
            model = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100)
            )
            with threadpool_limits(limits=1):
                model.fit(x.loc[train, cols], y.loc[train])
                usable = dates[valid.loc[dates]]
                if len(usable):
                    output.loc[usable, name] = model.predict(x.loc[usable, cols])
            audit.append(
                {
                    "sleeve": name,
                    "quarter": str(q),
                    "test_start": str(cutoff.date()),
                    "train_rows": int(train.sum()),
                    "last_training_outcome": str(meta.loc[train, "label_end_21"].max().date()),
                }
            )
        print(f"Forecasted {name}: {output[name].notna().sum()} / {len(output)} days", flush=True)
    return output, pd.DataFrame(audit)


def risk_scaled(predictions, meta):
    direction = (predictions / 0.5).clip(-1, 1).fillna(0)
    # At signal t, the daily return labeled t-2 ends at today's open.
    realized = direction.mul(meta.loc[direction.index, "target_return"], axis=0)
    risk = realized.shift(2).rolling(252, min_periods=126).std() * np.sqrt(252)
    weights = direction.mul(0.15 / risk.clip(lower=0.05)).clip(-2, 2).fillna(0)
    weights["trend_inventory"] = weights[["trend", "inventory"]].mean(axis=1)
    weights["combined"] = weights[["trend", "inventory", "positioning"]].mean(axis=1)
    expected = predictions.mul(
        meta.loc[predictions.index, "annual_vol"] / np.sqrt(252) * np.sqrt(21), axis=0
    ).fillna(0)
    expected["trend_inventory"] = expected[["trend", "inventory"]].mean(axis=1)
    expected["combined"] = expected[["trend", "inventory", "positioning"]].mean(axis=1)
    return weights, expected, risk


def gate_targets(target, expected, cash_yield, cost_bps=5, borrow_annual=0.03):
    long_cost = (
        2 * cost_bps / 10000
        + (target.abs() - 1).clip(lower=0).div(target.abs().clip(lower=1)) * 0.02 * 21 / 252
    )
    short_cost = 2 * cost_bps / 10000 + (borrow_annual + cash_yield) * 21 / 252
    keep = ((target > 0) & (expected > long_cost)) | ((target < 0) & (-expected > short_cost))
    return target.where(keep, 0.0)


def run(root):
    out = root / "results/weekly"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "plan.json").exists():
        raise ValueError("Preserve existing experiment; use a fresh output before changing inputs")
    x, meta, groups, lineage = build(root)
    inputs = [
        root / "data/weekly/eia_weekly.csv",
        root / "data/weekly/cftc_weekly.csv",
        root / "results/extension/news_refresh/model_input_outcomes.csv",
        root / "data/raw/USO.csv",
    ]
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(root / "docs/WEEKLY_PLAN.md"),
        "input_hashes": {str(p.relative_to(root)): sha256(p) for p in inputs},
        "code_hashes": {p.name: sha256(p) for p in (root / "oil_research").glob("weekly_*.py")},
        "groups": groups,
        "horizon": 21,
        "ridge_alpha": 100,
        "primary_cost_bps": 5,
        "borrow_annual": 0.03,
        "financing_spread": 0.02,
        "target_cap": 2,
        "policies": 20,
        "selection_period": "2017–2025 development; 2026 unopened",
        "cftc_status": "provisional historical archive; lag/exclusions do not certify original vintages",
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    (out / "code_snapshot").mkdir(exist_ok=True)
    for p in (root / "oil_research").glob("weekly_*.py"):
        shutil.copy2(p, out / "code_snapshot" / p.name)
    (out / "source_snapshots").mkdir(exist_ok=True)
    for p in inputs:
        shutil.copy2(p, out / "source_snapshots" / p.name)
    x.to_csv(out / "features.csv")
    lineage.to_csv(out / "lineage.csv")
    p, audit = forecasts(x, meta, groups)
    p.to_csv(out / "forecasts.csv")
    audit.to_csv(out / "training_audit.csv", index=False)
    weights, expected, risk = risk_scaled(p, meta)
    weights.to_csv(out / "targets.csv")
    risk.to_csv(out / "risk_estimates.csv")
    index = p.index[
        (p.index >= pd.Timestamp("2017-01-01"))
        & (meta.loc[p.index, "label_end"] < pd.Timestamp("2026-01-01"))
    ]
    periods = index.to_period("W-SUN")
    weekly = pd.Series(~periods.duplicated(), index=index)
    daily = pd.Series(True, index=index)
    dt = (meta.loc[index, "label_end"] - meta.loc[index, "execution_date"]).dt.days / 365
    records = []
    ledgers = {}
    candidate_inputs = {}
    for sleeve in weights:
        for frequency, mask in [("daily", daily), ("weekly", weekly)]:
            for policy in ["plain", "cost_aware"]:
                target = weights.loc[index, sleeve]
                if policy == "cost_aware":
                    target = gate_targets(
                        target, expected.loc[index, sleeve], meta.loc[index, "cash_yield_annual"]
                    )
                args = (
                    meta.loc[index, "target_return"],
                    meta.loc[index, "cash_return"],
                    dt,
                    target,
                    mask,
                )
                ledger = simulate_scheduled(*args, deadband=0.1 if policy == "cost_aware" else 0)
                name = f"{sleeve}_{frequency}_{policy}"
                ledgers[name] = ledger
                candidate_inputs[name] = (args, policy)
                ledger.to_csv(out / f"{name}_ledger.csv")
                records.append(
                    {
                        "candidate": name,
                        "sleeve": sleeve,
                        "frequency": frequency,
                        "policy": policy,
                        "cftc_dependent": sleeve in ["positioning", "combined"],
                        **score(ledger),
                        "average_gross_exposure": ledger.weight.abs().mean(),
                        "short_days": int((ledger.weight < 0).sum()),
                        "margin_calls": int(ledger.margin_call.sum()),
                        "safety_rebalances": int(ledger.safety_rebalance.sum()),
                    }
                )
    ranking = pd.DataFrame(records).sort_values("sharpe_excess", ascending=False)
    ranking.to_csv(out / "ranking.csv", index=False)
    best = ranking.iloc[0]
    selected = ledgers[best.candidate]
    selected.to_csv(out / "selected_development_ledger.csv")
    annual = []
    for name, l in ledgers.items():
        for year, g in l.groupby(l.index.year):
            g = g.copy()
            g["equity"] = (1 + g.net_return).cumprod()
            annual.append({"candidate": name, "year": year, **score(g)})
    pd.DataFrame(annual).to_csv(out / "annual.csv", index=False)
    pnl = pd.DataFrame(
        {
            name: l.excess_return
            for name, l in ledgers.items()
            if name in {s + "_daily_plain" for s in groups}
        }
    )
    pnl.corr().to_csv(out / "sleeve_correlations.csv")
    stress = []
    args, policy = candidate_inputs[best.candidate]
    for cost in [0, 5, 10, 20]:
        for borrow in [0.03, 0.10, 0.30]:
            # Freeze selected trading decisions; stress accounting without reselecting thresholds.
            l = simulate_scheduled(
                *args,
                cost_bps=cost,
                borrow_annual=borrow,
                deadband=0.1 if policy == "cost_aware" else 0,
            )
            stress.append({"cost_bps": cost, "borrow_annual": borrow, **score(l)})
    pd.DataFrame(stress).to_csv(out / "selected_cost_stress.csv", index=False)
    matrix = pd.DataFrame({name: l.excess_return for name, l in ledgers.items()}).to_numpy()
    chosen = list(ledgers).index(best.candidate)
    rng = np.random.default_rng(42)
    stats = []
    maxima = []
    centered = matrix - matrix.mean(axis=0)
    for _ in range(1000):
        starts = rng.integers(0, len(matrix), size=int(np.ceil(len(matrix) / 20)))
        ix = ((starts[:, None] + np.arange(20)) % len(matrix)).ravel()[: len(matrix)]
        sample = matrix[ix, chosen]
        stats.append(float(np.sqrt(252) * sample.mean() / sample.std(ddof=1)))
        null = centered[ix]
        sigma = null.std(axis=0, ddof=1)
        sr = np.divide(
            np.sqrt(252) * null.mean(axis=0), sigma, out=np.zeros_like(sigma), where=sigma > 1e-12
        )
        maxima.append(sr.max())
    uncertainty = {
        "selected_sharpe_interval_95": np.quantile(stats, [0.025, 0.975]).tolist(),
        "new_family_null_max_pvalue": float(
            (1 + sum(v >= best.sharpe_excess for v in maxima)) / 1001
        ),
        "bootstrap_draws": 1000,
        "block_length": 20,
        "warning": "Selected-candidate interval is not selection-adjusted. Family diagnostic covers only these 20 new policies, not the earlier 900 or all researcher choices.",
    }
    (out / "uncertainty.json").write_text(json.dumps(uncertainty, indent=2))
    coverage = {
        s: {
            "nonmissing_prediction_fraction": float(p.loc[index, s].notna().mean()),
            "first_valid_prediction": str(p[s].first_valid_index()),
        }
        for s in groups
    }
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2))
    print(ranking.to_string(index=False), flush=True)


if __name__ == "__main__":
    run(Path.cwd())
