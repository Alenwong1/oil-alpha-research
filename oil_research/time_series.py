"""Frozen six-configuration lagged boosting/adaptive ridge experiment."""

from pathlib import Path
from datetime import datetime, timezone
import json, shutil
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
from .data import sha256
from .weekly_features import asof_features
from .scarcity import target_y, completed_rms, gate
from .refinement import sized
from .weekly_experiments import simulate_scheduled
from .extension_leverage import simulate_signed
from .extension_experiments import score
from .backtest import sharpe

BASE = [
    "inventory_" + c + "_seasonal_z"
    for c in ["refinery_inputs", "utilization", "imports", "exports"]
] + ["innovation_" + c + "_change" for c in ["crude", "gasoline", "distillate"]]
SEQUENCE = (
    BASE[4:]
    + ["scarcity_" + c for c in ["crude", "gasoline", "distillate"]]
    + ["demand_gasoline", "demand_distillate"]
)


def lag_events(events):
    d = events.sort_values("available_at").reset_index(drop=True).copy()
    for k in [1, 2, 4, 8]:
        gap = (d.observation_date - d.observation_date.shift(k)).dt.days
        for c in SEQUENCE:
            d[f"{c}_lag{k}"] = d[c].shift(k).where(gap.between(7 * k - 3, 7 * k + 3))
    return d


def features_from_saved(features, lineage):
    d = features[BASE + SEQUENCE[3:]].join(lineage)
    d = d.dropna(subset=["available_at"]).sort_index().drop_duplicates("available_at", keep="first")
    assert (d.available_at <= d.cutoff).all()
    d["available_at"] = d.cutoff
    events = lag_events(d.drop(columns="cutoff").reset_index(drop=True))
    x, l = asof_features(events, features.index)
    return x, l, events


def recency_weights(index, cutoff):
    w = np.exp2(-(pd.Timestamp(cutoff) - index).days.to_numpy() / 730.5)
    return w / w.mean()


def fit_predict(x, y, test, kind, cutoff):
    with threadpool_limits(limits=1):
        if kind == "boosting":
            model = HistGradientBoostingRegressor(
                max_iter=100,
                learning_rate=0.04,
                max_leaf_nodes=7,
                min_samples_leaf=60,
                l2_regularization=20,
                early_stopping=False,
                random_state=49,
            )
            model.fit(x, y)
            return model.predict(test)
        model = make_pipeline(StandardScaler(), Ridge(alpha=100))
        kwargs = {}
        if kind == "adaptive":
            w = recency_weights(x.index, cutoff)
            kwargs = {"standardscaler__sample_weight": w, "ridge__sample_weight": w}
        model.fit(x, y, **kwargs)
        return model.predict(test)


def predict(x, m, h, kind):
    ix = x.index[
        (x.index >= pd.Timestamp("2015-01-01"))
        & (m.label_end < pd.Timestamp("2026-01-01"))
        & m.cash_return.notna()
    ]
    p = pd.Series(np.nan, index=ix, name="score")
    y = target_y(m, h)
    valid = x.notna().all(axis=1)
    audit = []
    for q in ix.to_period("Q").unique():
        dates = ix[ix.to_period("Q") == q]
        cutoff = dates.min()
        train = (x.index < cutoff) & (m[f"label_end_{h}"] < cutoff) & valid & y.notna()
        usable = dates[valid.loc[dates]]
        if train.sum() < 252:
            continue
        if len(usable):
            p.loc[usable] = fit_predict(x.loc[train], y.loc[train], x.loc[usable], kind, cutoff)
        w = recency_weights(x.index[train], cutoff) if kind == "adaptive" else np.ones(train.sum())
        audit.append(
            {
                "quarter": str(q),
                "test_start": str(cutoff.date()),
                "last_training_outcome": str(m.loc[train, f"label_end_{h}"].max().date()),
                "train_rows": int(train.sum()),
                "weight_effective_rows": float(w.sum() ** 2 / (w * w).sum()),
                "horizon": h,
            }
        )
    return p, pd.DataFrame(audit)


def main(root):
    out = root / "results/time_series"
    out.mkdir(exist_ok=True)
    if (out / "plan.json").exists():
        raise ValueError("Preserve existing experiment")
    files = [
        "results/scarcity/features.csv",
        "results/scarcity/lineage.csv",
        "results/extension/news_refresh/model_input_outcomes.csv",
        "results/extension/horizons_repaired/selected_development_ledger.csv",
        "results/diversification/source_snapshots/diversifier2_ledger.csv",
        "results/scarcity/existing_h5_portfolio_ledger.csv",
        "results/scarcity/existing_h21_portfolio_ledger.csv",
    ]
    deps = [
        "time_series.py",
        "scarcity.py",
        "weekly_features.py",
        "weekly_experiments.py",
        "refinement.py",
        "extension_experiments.py",
        "extension_leverage.py",
        "extension_features.py",
        "backtest.py",
        "data.py",
    ]
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "primary_configurations": 6,
        "protocol_sha256": sha256(root / "docs/TIME_SERIES_PLAN.md"),
        "input_hashes": {p: sha256(root / p) for p in files},
        "source_hashes": {p: sha256(root / "oil_research" / p) for p in deps},
        "warning": "Repeated 2017–2025 development selection; no 2026 evaluation",
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    shutil.copy2(root / "docs/TIME_SERIES_PLAN.md", out / "protocol.md")
    (out / "code_snapshot").mkdir(exist_ok=True)
    for p in deps:
        shutil.copy2(root / "oil_research" / p, out / "code_snapshot" / p)
    f = pd.read_csv(root / files[0], index_col=0, parse_dates=True)
    line = pd.read_csv(root / files[1], index_col=0, parse_dates=True)
    for c in ["cutoff", "available_at"]:
        line[c] = pd.to_datetime(line[c], utc=True)
    line["observation_date"] = pd.to_datetime(line.observation_date)
    lagged, line, events = features_from_saved(f, line)
    lagged.to_csv(out / "features.csv")
    line.to_csv(out / "lineage.csv")
    events.to_csv(out / "event_features.csv", index=False)
    m = pd.read_csv(root / files[2], index_col=0, parse_dates=True)
    for c in ["execution_date", "label_end", "label_end_1", "label_end_5", "label_end_21"]:
        m[c] = pd.to_datetime(m[c])
    assert m.index.max() < pd.Timestamp("2026-01-01")
    macro = pd.read_csv(root / files[3], index_col=0, parse_dates=True).weight
    technical = pd.read_csv(root / files[4], index_col=0, parse_dates=True).weight
    rows = []
    annual = []
    ledgers = {}
    weights = {}
    coverage = []
    predictions = {}
    for kind in ["ridge", "boosting", "adaptive"]:
        features = f[BASE] if kind == "ridge" else lagged
        for h in [5, 21]:
            key = f"{kind}_h{h}"
            p, a = predict(features, m, h, kind)
            predictions[key] = p
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
            ledgers[key] = port
            weights[key] = w
            for typ, l in [("inventory", inv), ("portfolio", port)]:
                l.to_csv(out / f"{key}_{typ}_ledger.csv")
                rows.append(
                    {
                        "candidate": key,
                        "features": features.shape[1],
                        "horizon": h,
                        "kind": typ,
                        **score(l),
                        "information_ratio_USO": sharpe(l.net_return - l.asset_return),
                        "active_fraction": float((l.weight.abs() > 1e-8).mean()),
                    }
                )
                for year, yy in l.groupby(l.index.year):
                    yy = yy.copy()
                    yy["equity"] = (1 + yy.net_return).cumprod()
                    annual.append({"candidate": key, "kind": typ, "year": year, **score(yy)})
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
    for h in [5, 21]:
        old = pd.read_csv(
            root / f"results/scarcity/existing_h{h}_portfolio_ledger.csv",
            index_col=0,
            parse_dates=True,
        )
        np.testing.assert_allclose(
            ledgers[f"ridge_h{h}"].net_return, old.net_return, rtol=0, atol=1e-11
        )
    accuracy = []
    for h in [5, 21]:
        pp = pd.DataFrame({k: v for k, v in predictions.items() if k.endswith(f"_h{h}")}).reindex(
            ix
        )
        y = (
            target_y(m, h)
            .reindex(ix)
            .where(m.loc[ix, f"label_end_{h}"] < pd.Timestamp("2026-01-01"))
        )
        common = pp.notna().all(axis=1) & y.notna()
        for key in pp:
            accuracy.append(
                {
                    "candidate": key,
                    "common_days": int(common.sum()),
                    "normalized_return_rmse": float(
                        np.sqrt(((pp.loc[common, key] - y[common]) ** 2).mean())
                    ),
                    "zero_forecast_rmse": float(np.sqrt((y[common] ** 2).mean())),
                    "prediction_return_correlation": float(pp.loc[common, key].corr(y[common])),
                }
            )
    pd.DataFrame(accuracy).to_csv(out / "forecast_accuracy.csv", index=False)
    best = ranking.iloc[0]
    ledgers[best.candidate].to_csv(out / "selected_development_ledger.csv")
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
