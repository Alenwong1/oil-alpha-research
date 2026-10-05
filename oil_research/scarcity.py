"""Six frozen inventory scarcity/demand configurations with horizon-safe labels."""

from pathlib import Path
from datetime import datetime, timezone
import csv, io, re, json, shutil
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from .data import sha256
from .weekly_features import build, asof_features
from .weekly_experiments import simulate_scheduled
from .refinement import sized
from .extension_experiments import score
from .extension_leverage import simulate_signed
from .backtest import sharpe


def parse_products(raw, expected_date):
    rows = list(csv.reader(io.StringIO(raw.decode("cp1252"))))
    header = rows[0]
    text = header[1].strip()
    date = pd.to_datetime(text, format="%m/%d/%Y" if len(text.split("/")[-1]) == 4 else "%m/%d/%y")
    if date != pd.Timestamp(expected_date):
        raise ValueError("Release observation date mismatch")
    result = {}
    labels = {
        "Finished Motor Gasoline": "gasoline_supplied",
        "Distillate Fuel Oil": "distillate_supplied",
    }
    for row in rows:
        if len(row) > 3 and row[0].strip() == "Products Supplied":
            label = re.sub(r"^\(\d+\)\s*", "", row[1]).strip()
            if label in labels:
                result[labels[label]] = float(row[2].replace(",", ""))
    if set(result) != set(labels.values()) or not all(
        np.isfinite(v) and v > 0 for v in result.values()
    ):
        raise ValueError("Missing/invalid product supplied")
    return result


def seasonal_z(d, values):
    weeks = d.observation_date.dt.isocalendar().week.astype(int)
    out = []
    for i, row in d.iterrows():
        prior = (
            (d.available_at < row.available_at)
            & (d.observation_date < row.observation_date)
            & (d.observation_date >= row.observation_date - pd.DateOffset(years=5))
        )
        delta = (weeks - weeks.iloc[i]).abs()
        near = np.minimum(delta, 53 - delta) <= 4
        h = values[prior & near].dropna()
        if prior.sum() < 104 or len(h) < 12 or h.std() < 1e-8:
            out.append(np.nan)
        else:
            out.append(float(np.clip((values.iloc[i] - h.mean()) / h.std(), -4, 4)))
    return pd.Series(out, index=d.index)


def states(data):
    d = data.copy().sort_values(["available_at", "observation_date"]).reset_index(drop=True)
    d["available_at"] = pd.to_datetime(d.available_at, utc=True)
    d["observation_date"] = pd.to_datetime(d.observation_date)
    gap = d.observation_date.diff().dt.days
    consecutive = gap.between(5, 9).rolling(3, min_periods=3).sum() == 3
    out = d[["available_at", "observation_date"]].copy()
    out["scarcity_crude"] = -seasonal_z(d, d.crude_stocks.shift(1).where(gap <= 10))
    for product in ["gasoline", "distillate"]:
        demand = d[product + "_supplied"].rolling(4, min_periods=4).mean().where(consecutive)
        days = 1000 * d[product + "_stocks"] / demand
        out["scarcity_" + product] = -seasonal_z(d, days.shift(1).where(gap <= 10))
        out["demand_" + product] = seasonal_z(d, demand)
    return out


def target_y(m, h):
    return (
        (m[f"target_{h}"] - m.cash_yield_annual * h / 252)
        / (m.annual_vol / np.sqrt(252) * np.sqrt(h))
    ).clip(-4, 4)


def predict(x, m, h):
    ix = x.index[
        (x.index >= pd.Timestamp("2015-01-01"))
        & (m.label_end < pd.Timestamp("2026-01-01"))
        & m.cash_return.notna()
    ]
    p = pd.Series(np.nan, index=ix, name="score")
    y = target_y(m, h)
    audit = []
    valid = x.notna().all(axis=1)
    for q in ix.to_period("Q").unique():
        dates = ix[ix.to_period("Q") == q]
        cutoff = dates.min()
        train = (x.index < cutoff) & (m[f"label_end_{h}"] < cutoff) & valid & y.notna()
        if train.sum() < 252:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=100))
        with threadpool_limits(limits=1):
            model.fit(x.loc[train], y.loc[train])
            usable = dates[valid.loc[dates]]
            if len(usable):
                p.loc[usable] = model.predict(x.loc[usable])
        audit.append(
            {
                "quarter": str(q),
                "test_start": str(cutoff.date()),
                "last_training_outcome": str(m.loc[train, f"label_end_{h}"].max().date()),
                "train_rows": int(train.sum()),
                "horizon": h,
            }
        )
    return p, pd.DataFrame(audit)


def completed_rms(p, m, h):
    e = (
        pd.DataFrame(
            {
                "end": m.loc[p.index, f"label_end_{h}"],
                "squared_error": (p - target_y(m, h).loc[p.index]) ** 2,
            }
        )
        .dropna()
        .sort_values("end")
    )
    e["rms"] = np.sqrt(e.squared_error.rolling(252, min_periods=126).mean())
    joined = pd.merge_asof(
        pd.DataFrame({"date": p.index}),
        e[["end", "rms"]],
        left_on="date",
        right_on="end",
        direction="backward",
    )
    assert not (joined.end > joined.date).any()
    return pd.Series(joined.rms.to_numpy(), index=p.index)


def gate(w, expected, cash_yield, h):
    long_cost = 0.001 + (w.abs() - 1).clip(lower=0) / w.abs().clip(lower=1) * 0.02 * h / 252
    short_cost = 0.001 + (0.03 + cash_yield) * h / 252
    ok = ((w > 0) & (expected > long_cost)) | ((w < 0) & (-expected > short_cost))
    return w.where(ok, 0.0)


def main(root):
    out = root / "results/scarcity"
    out.mkdir(exist_ok=True)
    if (out / "plan.json").exists():
        raise ValueError("Preserve prior experiment")
    files = [
        "data/weekly/eia_weekly.csv",
        "results/refinement/innovation_features.csv",
        "results/extension/horizons_repaired/selected_development_ledger.csv",
        "results/diversification/source_snapshots/diversifier2_ledger.csv",
        "results/extension/news_refresh/model_input_outcomes.csv",
    ]
    deps = [
        "scarcity.py",
        "weekly_features.py",
        "weekly_experiments.py",
        "refinement.py",
        "extension_experiments.py",
        "extension_leverage.py",
        "extension_features.py",
        "backtest.py",
    ]
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "primary_configurations": 6,
        "protocol_sha256": sha256(root / "docs/SCARCITY_PLAN.md"),
        "input_hashes": {p: sha256(root / p) for p in files},
        "source_hashes": {p: sha256(root / "oil_research" / p) for p in deps},
        "warning": "2017–2025 development selection; no 2026 evaluation",
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    shutil.copy2(root / "docs/SCARCITY_PLAN.md", out / "protocol.md")
    (out / "code_snapshot").mkdir(exist_ok=True)
    for p in deps:
        shutil.copy2(root / "oil_research" / p, out / "code_snapshot" / p)
    d = pd.read_csv(root / files[0])
    products = []
    for _, r in d.iterrows():
        path = root / "data/weekly/eia" / r.release_date.replace("-", "_") / "table1.csv"
        assert sha256(path) == r.table1_sha256
        products.append(parse_products(path.read_bytes(), r.observation_date))
    d = pd.concat([d, pd.DataFrame(products)], axis=1)
    d.to_csv(out / "release_inputs.csv", index=False)
    x, m, g, _ = build(root)
    extra, lineage = asof_features(states(d), x.index)
    lineage.to_csv(out / "lineage.csv")
    innovations = pd.read_csv(root / files[1], index_col=0, parse_dates=True)
    flows = [
        c
        for c in g["inventory"]
        if not any(v in c for v in ["crude_change", "gasoline_change", "distillate_change"])
    ]
    baseline = x[flows].join(innovations)
    scarce = baseline.join(extra.filter(like="scarcity_"))
    for product in ["crude", "gasoline", "distillate"]:
        scarce["surprise_x_scarcity_" + product] = (
            -innovations["innovation_" + product + "_change"] * extra["scarcity_" + product]
        ).clip(-8, 8)
    demand = scarce.join(extra.filter(like="demand_"))
    demand["draw_x_refinery"] = (
        -innovations.innovation_crude_change * x.inventory_refinery_inputs_seasonal_z
    ).clip(-8, 8)
    demand["draw_x_imports"] = (
        -innovations.innovation_crude_change * x.inventory_imports_seasonal_z
    ).clip(-8, 8)
    demand["product_draw_x_demand"] = (
        -innovations[["innovation_gasoline_change", "innovation_distillate_change"]].mean(axis=1)
        * extra[["demand_gasoline", "demand_distillate"]].mean(axis=1)
    ).clip(-8, 8)
    demand.to_csv(out / "features.csv")
    models = {"existing": baseline, "scarcity": scarce, "demand": demand}
    macro = pd.read_csv(root / files[2], index_col=0, parse_dates=True).weight
    technical = pd.read_csv(root / files[3], index_col=0, parse_dates=True).weight
    rows = []
    annual = []
    portfolio_ledgers = {}
    weights = {}
    coverage = []
    for name, features in models.items():
        for h in [5, 21]:
            key = f"{name}_h{h}"
            p, a = predict(features, m, h)
            a.to_csv(out / f"{key}_training_audit.csv", index=False)
            p.to_csv(out / f"{key}_predictions.csv")
            rms = completed_rms(p, m, h)
            t = sized(p, m, rms, "existing").where(p.abs() >= 0.25 * rms, 0.0)
            expected = p * m.loc[p.index, "annual_vol"] / np.sqrt(252) * np.sqrt(h)
            t = gate(t, expected.fillna(0), m.loc[p.index, "cash_yield_annual"], h)
            ix = p.index[
                (p.index >= pd.Timestamp("2017-01-01"))
                & (m.loc[p.index, "label_end"] < pd.Timestamp("2026-01-01"))
            ]
            dt = (m.loc[ix, "label_end"] - m.loc[ix, "execution_date"]).dt.days / 365
            mask = pd.Series(~ix.to_period("W-SUN").duplicated(), index=ix)
            inv = simulate_scheduled(
                m.loc[ix, "target_return"],
                m.loc[ix, "cash_return"],
                dt,
                t.loc[ix],
                mask,
                deadband=0.1,
            )
            w = 0.375 * macro.loc[ix] + 0.375 * inv.weight + 0.25 * technical.loc[ix]
            port = simulate_signed(m.loc[ix, "target_return"], m.loc[ix, "cash_return"], dt, w)
            portfolio_ledgers[key] = port
            weights[key] = w
            for kind, l in [("inventory", inv), ("portfolio", port)]:
                l.to_csv(out / f"{key}_{kind}_ledger.csv")
                rows.append(
                    {
                        "candidate": key,
                        "features": features.shape[1],
                        "horizon": h,
                        "kind": kind,
                        **score(l),
                        "information_ratio_USO": sharpe(l.net_return - l.asset_return),
                        "active_fraction": float((l.weight.abs() > 1e-8).mean()),
                    }
                )
                for year, yy in l.groupby(l.index.year):
                    yy = yy.copy()
                    yy["equity"] = (1 + yy.net_return).cumprod()
                    annual.append({"candidate": key, "kind": kind, "year": year, **score(yy)})
            coverage.append(
                {
                    "candidate": key,
                    "feature_coverage": float(features.loc[ix].notna().all(axis=1).mean()),
                    "forecast_coverage": float(p.loc[ix].notna().mean()),
                    "uncertainty_coverage": float(rms.loc[ix].notna().mean()),
                }
            )
            print(key, score(port)["sharpe_excess"], flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    ranking = metrics[metrics.kind == "portfolio"].sort_values("sharpe_excess", ascending=False)
    ranking.to_csv(out / "ranking.csv", index=False)
    pd.DataFrame(annual).to_csv(out / "annual.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
    control = pd.read_csv(
        root / "results/diversification/selected_development_ledger.csv",
        index_col=0,
        parse_dates=True,
    )
    np.testing.assert_allclose(
        portfolio_ledgers["existing_h21"].net_return, control.net_return, atol=1e-11
    )
    best = ranking.iloc[0]
    portfolio_ledgers[best.candidate].to_csv(out / "selected_development_ledger.csv")
    stress = []
    for c in [0, 5, 10, 20]:
        for borrow in [0.03, 0.10, 0.30]:
            l = simulate_signed(
                m.loc[ix, "target_return"],
                m.loc[ix, "cash_return"],
                dt,
                weights[best.candidate],
                cost_bps=c,
                borrow_annual=borrow,
            )
            stress.append({"cost_bps": c, "borrow_annual": borrow, **score(l)})
    pd.DataFrame(stress).to_csv(out / "selected_cost_stress.csv", index=False)
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main(Path.cwd())
