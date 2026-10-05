"""Frozen inventory-forecast, weak-signal and risk-sizing development batch."""

from pathlib import Path
from datetime import datetime, timezone
import json, shutil
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from .weekly_features import build, asof_features
from .weekly_experiments import forecasts, simulate_scheduled, gate_targets
from .extension_leverage import simulate_signed
from .extension_experiments import score
from .data import sha256


def inventory_innovations(data):
    d = data.copy().sort_values(["available_at", "observation_date"]).reset_index(drop=True)
    d["available_at"] = pd.to_datetime(d.available_at, utc=True)
    d["observation_date"] = pd.to_datetime(d.observation_date)
    phase = 2 * np.pi * d.observation_date.dt.dayofyear / 365.25
    base = pd.DataFrame(
        {f"{f}{k}": getattr(np, f)(k * phase) for k in [1, 2] for f in ["sin", "cos"]}
    )
    for col in ["imports", "exports", "refinery_inputs"]:
        base[col] = d[col].shift(1)
    events = d[["available_at", "observation_date"]].copy()
    audit = []
    for col in ["crude_change", "gasoline_change", "distillate_change"]:
        y = d[col]
        x = base.copy()
        x["lag1"] = y.shift(1)
        x["lag4"] = y.shift(1).rolling(4).mean()
        errors = []
        z = []
        for i, row in d.iterrows():
            train = (
                (d.available_at < row.available_at)
                & (d.observation_date >= row.observation_date - pd.DateOffset(years=5))
                & x.notna().all(axis=1)
            )
            if train.sum() < 104 or x.iloc[i].isna().any():
                z.append(np.nan)
                continue
            model = make_pipeline(StandardScaler(), Ridge(alpha=10))
            with threadpool_limits(limits=1):
                model.fit(x.loc[train], y.loc[train])
                prediction = float(model.predict(x.iloc[[i]])[0])
            scale = float(np.std(errors[-104:], ddof=1)) if len(errors) >= 26 else np.nan
            error = float(y.iloc[i] - prediction)
            z.append(float(np.clip(error / scale, -4, 4)) if scale > 1e-8 else np.nan)
            weeks = d.observation_date.dt.isocalendar().week.astype(int)
            dist = (weeks - weeks.iloc[i]).abs()
            seasonal = y[train & (np.minimum(dist, 53 - dist) <= 4)]
            seasonal_prediction = float(seasonal.mean()) if len(seasonal) >= 12 else np.nan
            audit.append(
                {
                    "series": col,
                    "observation_date": str(row.observation_date.date()),
                    "available_at": str(row.available_at),
                    "last_training_release": str(d.loc[train, "available_at"].max()),
                    "prediction": prediction,
                    "seasonal_prediction": seasonal_prediction,
                    "actual": float(y.iloc[i]),
                    "error": error,
                    "prior_error_scale": scale,
                }
            )
            errors.append(error)
        events["innovation_" + col] = z
    return events, pd.DataFrame(audit)


def completed_error_rms(p, meta):
    y = (
        (meta.target_21 - meta.cash_yield_annual * 21 / 252)
        / (meta.annual_vol / np.sqrt(252) * np.sqrt(21))
    ).clip(-4, 4)
    e = (
        pd.DataFrame({"end": meta.loc[p.index, "label_end_21"], "error": (p - y.loc[p.index]) ** 2})
        .dropna()
        .sort_values("end")
    )
    e["rms"] = np.sqrt(e.error.rolling(252, min_periods=126).mean())
    left = pd.DataFrame({"date": p.index})
    r = pd.merge_asof(left, e[["end", "rms"]], left_on="date", right_on="end", direction="backward")
    assert not (r.end > r.date).any()
    return pd.Series(r.rms.to_numpy(), index=p.index)


def sized(p, meta, uncertainty, mode):
    direction = (p / 0.5).clip(-1, 1).fillna(0)
    realized = (direction * meta.loc[p.index, "target_return"]).shift(2)
    slow = realized.rolling(252, min_periods=126).std() * np.sqrt(252)
    if mode == "conservative":
        fast = realized.rolling(63, min_periods=40).std() * np.sqrt(252)
        vol = pd.concat([slow, fast], axis=1).max(axis=1).where(slow.notna())
        haircut = p.abs() / (p.abs() + uncertainty)
    else:
        vol = slow
        haircut = pd.Series(1.0, index=p.index)
    return (direction * 0.15 / vol.clip(lower=0.05) * haircut).clip(-2, 2).fillna(0)


def main(root):
    out = root / "results/refinement"
    out.mkdir(exist_ok=True)
    if (out / "plan.json").exists():
        raise ValueError("Refinement already recorded; preserve results")
    inputs = [
        root / "data/weekly/eia_weekly.csv",
        root / "results/weekly/forecasts.csv",
        root / "results/extension/news_refresh/model_input_outcomes.csv",
        root / "results/extension/horizons_repaired/selected_development_ledger.csv",
    ]
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policies": 24,
        "protocol_sha256": sha256(root / "docs/REFINEMENT_PLAN.md"),
        "code_sha256": sha256(Path(__file__)),
        "inputs": {str(p.relative_to(root)): sha256(p) for p in inputs},
        "warning": "2017–2025 development selection; 2026 untouched",
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    shutil.copy2(root / "docs/REFINEMENT_PLAN.md", out / "protocol.md")
    shutil.copy2(Path(__file__), out / "code_snapshot.py")
    x, m, groups, lineage = build(root)
    events, ia = inventory_innovations(pd.read_csv(inputs[0]))
    ia.to_csv(out / "inventory_forecast_audit.csv", index=False)
    fresh, il = asof_features(events, x.index)
    fresh.to_csv(out / "innovation_features.csv")
    il.to_csv(out / "lineage.csv")
    cols = [
        c
        for c in groups["inventory"]
        if not any(v in c for v in ["crude_change", "gasoline_change", "distillate_change"])
    ]
    xx = x[cols].join(fresh)
    new, audit = forecasts(xx, m, {"improved": list(xx)})
    audit.to_csv(out / "training_audit.csv", index=False)
    old = pd.read_csv(inputs[1], index_col=0, parse_dates=True).inventory.reindex(new.index)
    predictions = new.copy()
    predictions["original"] = old
    predictions.to_csv(out / "predictions.csv")
    ix = new.index[
        (new.index >= pd.Timestamp("2017-01-01"))
        & (m.loc[new.index, "label_end"] < pd.Timestamp("2026-01-01"))
    ]
    dt = (m.loc[ix, "label_end"] - m.loc[ix, "execution_date"]).dt.days / 365
    weekly = pd.Series(~ix.to_period("W-SUN").duplicated(), index=ix)
    prior = pd.read_csv(inputs[3], index_col=0, parse_dates=True).weight.reindex(ix)
    rows = []
    annual = []
    ledgers = {}
    stored_inputs = {}
    coverage = {}
    for model in predictions:
        p = predictions[model]
        rms = completed_error_rms(p, m)
        coverage[model] = {
            "forecast_coverage": float(p.loc[ix].notna().mean()),
            "uncertainty_coverage": float(rms.loc[ix].notna().mean()),
        }
        expected = p * m.loc[p.index, "annual_vol"] / np.sqrt(252) * np.sqrt(21)
        for mode in ["existing", "conservative"]:
            target = sized(p, m, rms, mode)
            for threshold in [0, 0.25, 0.5]:
                allowed = p.abs() >= threshold * rms
                if threshold == 0:
                    allowed = p.notna()
                t = gate_targets(
                    target.where(allowed, 0).loc[ix],
                    expected.loc[ix].fillna(0),
                    m.loc[ix, "cash_yield_annual"],
                )
                args = (m.loc[ix, "target_return"], m.loc[ix, "cash_return"], dt, t, weekly)
                inv = simulate_scheduled(*args, deadband=0.1)
                for portfolio in ["inventory", "blend"]:
                    name = f"{model}_{mode}_h{threshold}_{portfolio}"
                    if portfolio == "inventory":
                        l = inv
                        stored_inputs[name] = ("scheduled", args)
                    else:
                        w = 0.5 * prior + 0.5 * inv.weight
                        a = (m.loc[ix, "target_return"], m.loc[ix, "cash_return"], dt, w)
                        l = simulate_signed(*a)
                        stored_inputs[name] = ("signed", a)
                    ledgers[name] = l
                    l.to_csv(out / f"{name}_ledger.csv")
                    rows.append(
                        {
                            "candidate": name,
                            "model": model,
                            "risk_rule": mode,
                            "threshold": threshold,
                            "portfolio": portfolio,
                            **score(l),
                            "active_fraction": float((l.weight.abs() > 1e-8).mean()),
                            "average_gross_exposure": float(l.weight.abs().mean()),
                        }
                    )
                    for year, g in l.groupby(l.index.year):
                        g = g.copy()
                        g["equity"] = (1 + g.net_return).cumprod()
                        annual.append({"candidate": name, "year": year, **score(g)})
    r = pd.DataFrame(rows).sort_values("sharpe_excess", ascending=False)
    r.to_csv(out / "ranking.csv", index=False)
    pd.DataFrame(annual).to_csv(out / "annual.csv", index=False)
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2))
    best = r.iloc[0]
    ledgers[best.candidate].to_csv(out / "selected_development_ledger.csv")
    stress = []
    kind, args = stored_inputs[best.candidate]
    for cost in [0, 5, 10, 20]:
        for borrow in [0.03, 0.10, 0.30]:
            l = (
                simulate_scheduled(*args, cost_bps=cost, borrow_annual=borrow, deadband=0.1)
                if kind == "scheduled"
                else simulate_signed(*args, cost_bps=cost, borrow_annual=borrow)
            )
            stress.append({"cost_bps": cost, "borrow_annual": borrow, **score(l)})
    pd.DataFrame(stress).to_csv(out / "selected_cost_stress.csv", index=False)
    rmse = []
    for series, g in ia.groupby("series"):
        g = g[
            (pd.to_datetime(g.available_at, utc=True).dt.year >= 2017)
            & (pd.to_datetime(g.available_at, utc=True).dt.year <= 2025)
        ].dropna(subset=["prediction", "seasonal_prediction"])
        rmse.append(
            {
                "series": series,
                "observations": len(g),
                "improved_rmse": float(np.sqrt(np.mean((g.actual - g.prediction) ** 2))),
                "seasonal_rmse": float(np.sqrt(np.mean((g.actual - g.seasonal_prediction) ** 2))),
            }
        )
    pd.DataFrame(rmse).to_csv(out / "inventory_rmse.csv", index=False)
    # Verify original no-strength-filter controls reproduce prior results.
    controls = pd.read_csv(root / "results/weekly/ranking.csv").set_index("candidate")
    v = r.set_index("candidate").loc["original_existing_h0_inventory", "sharpe_excess"]
    assert abs(v - controls.loc["inventory_weekly_cost_aware", "sharpe_excess"]) < 1e-9
    print(r.head(10).to_string(index=False))
    print(pd.DataFrame(rmse).to_string(index=False))


if __name__ == "__main__":
    main(Path.cwd())
