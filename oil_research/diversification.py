"""Fixed-weight low-correlation blend screen and netted-account backtest."""

from pathlib import Path
from datetime import datetime, timezone
import json, shutil
import numpy as np
import pandas as pd
from .data import sha256
from .backtest import sharpe
from .extension_experiments import policy_weights, simulate_cash, score
from .extension_leverage import signed_weights, simulate_signed


def candidates(root):
    config = json.loads((root / "extension_config.json").read_text())
    cache = {}
    for _, r in pd.read_csv(root / "results/extension/all_candidates.csv").iterrows():
        signed = r["round"] == "short_leverage"
        source = r.source_round if signed else r["round"]
        folder = root / "results/extension" / source
        key = (source, r.candidate)
        if key not in cache:
            p = pd.read_csv(
                folder / f"{r.candidate}_predictions.csv", index_col=0, parse_dates=True
            ).score
            m = pd.read_csv(
                folder / "model_input_outcomes.csv",
                index_col=0,
                parse_dates=[0, "execution_date", "label_end"],
            )
            cache[key] = (p, m)
        p, m = cache[key]
        if signed:
            w = signed_weights(p, m, r.policy, int(r.smoothing), float(r.vol_target))
            dt = (m.loc[p.index, "label_end"] - m.loc[p.index, "execution_date"]).dt.days / 365
            l = simulate_signed(
                m.loc[p.index, "target_return"], m.loc[p.index, "cash_return"], dt, w
            )
        else:
            w = policy_weights(p, m, r.policy, int(r.smoothing), config)
            l = simulate_cash(m.loc[p.index, "target_return"], m.loc[p.index, "cash_return"], w, 5)
        assert abs(sharpe(l.excess_return) - r.sharpe_excess) < 1e-8
        name = f"{source}/{r.candidate}/{r.policy}/smooth{int(r.smoothing)}" + (
            f"/vol{r.vol_target}" if signed else "/longcash"
        )
        yield name, f"{source}/{r.candidate}", l
    for folder in ["weekly", "refinement"]:
        for _, r in pd.read_csv(root / "results" / folder / "ranking.csv").iterrows():
            l = pd.read_csv(
                root / "results" / folder / f"{r.candidate}_ledger.csv",
                index_col=0,
                parse_dates=True,
            )
            group = str(r["model"]) if folder == "refinement" else str(r.sleeve)
            yield folder + "/" + r.candidate, folder + "/" + group, l
    l = pd.read_csv(
        root / "results/weekly/blend_followup/selected_development_ledger.csv",
        index_col=0,
        parse_dates=True,
    )
    yield "previous_50_50_blend", "previous_blend", l


def blend_weights(base, other, fraction):
    if not base.index.equals(other.index):
        raise ValueError("Index mismatch")
    if not 0 <= fraction <= 1:
        raise ValueError("Invalid fraction")
    w = (1 - fraction) * base + fraction * other
    if w.abs().max() > 2 + 1e-10:
        raise ValueError("Target cap exceeded")
    return w


def main(root):
    out = root / "results/diversification"
    out.mkdir(exist_ok=True)
    if (out / "plan.json").exists():
        raise ValueError("Preserve prior results")
    path = root / "results/refinement/selected_development_ledger.csv"
    inputs = [
        path,
        root / "results/extension/all_candidates.csv",
        root / "results/weekly/ranking.csv",
        root / "results/refinement/ranking.csv",
    ]
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_sha256": sha256(Path(__file__)),
        "protocol_sha256": sha256(root / "docs/DIVERSIFICATION_PLAN.md"),
        "input_hashes": {str(p.relative_to(root)): sha256(p) for p in inputs},
        "minimum_standalone_sharpe": 0.30,
        "maximum_correlation": 0.30,
        "diversifier_fractions": [0.25, 0.5],
        "maximum_diversifiers": 3,
        "warning": "Component and blend selection use full 2017–2025 development returns; 2026 remains closed.",
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    shutil.copy2(Path(__file__), out / "code_snapshot.py")
    shutil.copy2(root / "docs/DIVERSIFICATION_PLAN.md", out / "protocol.md")
    base = pd.read_csv(path, index_col=0, parse_dates=True)
    records = []
    eligible = {}
    for name, group, l in candidates(root):
        assert base.index.equals(l.index)
        sr = sharpe(l.excess_return)
        corr = float(base.excess_return.corr(l.excess_return))
        ok = np.isfinite(corr) and sr >= 0.30 and corr <= 0.30
        records.append(
            {
                "candidate": name,
                "model_group": group,
                "standalone_sharpe": sr,
                "correlation": corr,
                "eligible": ok,
            }
        )
        if ok:
            eligible[name] = l
    screen = pd.DataFrame(records).sort_values("standalone_sharpe", ascending=False)
    screen.to_csv(out / "screen.csv", index=False)
    picks = screen[screen.eligible].drop_duplicates("model_group").head(3)
    picks.to_csv(out / "selected_diversifiers.csv", index=False)
    m = pd.read_csv(
        root / "results/extension/news_refresh/model_input_outcomes.csv",
        index_col=0,
        parse_dates=[0, "execution_date", "label_end"],
    ).loc[base.index]
    dt = (m.label_end - m.execution_date).dt.days / 365
    rows = []
    annual = []
    ledgers = {}
    targets = {}

    def add(name, l, component="", fraction=0.0, corr=np.nan):
        rows.append(
            {
                "candidate": name,
                "diversifier": component,
                "diversifier_fraction": fraction,
                "component_correlation": corr,
                **score(l),
                "sharpe_2018_2025": sharpe(l.loc[l.index.year != 2017, "excess_return"]),
            }
        )
        for y, g in l.groupby(l.index.year):
            g = g.copy()
            g["equity"] = (1 + g.net_return).cumprod()
            annual.append({"candidate": name, "year": y, **score(g)})
        ledgers[name] = l
        l.to_csv(out / f"{name}_ledger.csv")

    control = simulate_signed(m.target_return, m.cash_return, dt, base.weight)
    np.testing.assert_allclose(control.net_return, base.net_return, atol=1e-12)
    add("base", base)
    targets["base"] = base.weight
    (out / "source_snapshots").mkdir(exist_ok=True)
    for j, (_, r) in enumerate(picks.iterrows(), 1):
        other = eligible[r.candidate]
        other.to_csv(out / "source_snapshots" / f"diversifier{j}_ledger.csv")
        for fraction in [0.25, 0.5]:
            w = blend_weights(base.weight, other.weight, fraction)
            name = f"diversifier{j}_weight{int(fraction*100)}"
            l = simulate_signed(m.target_return, m.cash_return, dt, w)
            targets[name] = w
            add(name, l, r.candidate, fraction, r.correlation)
    ranking = pd.DataFrame(rows).sort_values("sharpe_excess", ascending=False)
    ranking.to_csv(out / "ranking.csv", index=False)
    pd.DataFrame(annual).to_csv(out / "annual.csv", index=False)
    best = ranking.iloc[0]
    selected = ledgers[best.candidate]
    selected.to_csv(out / "selected_development_ledger.csv")
    stress = []
    for cost in [0, 5, 10, 20]:
        for borrow in [0.03, 0.10, 0.30]:
            l = simulate_signed(
                m.target_return,
                m.cash_return,
                dt,
                targets[best.candidate],
                cost_bps=cost,
                borrow_annual=borrow,
            )
            stress.append({"cost_bps": cost, "borrow_annual": borrow, **score(l)})
    pd.DataFrame(stress).to_csv(out / "selected_cost_stress.csv", index=False)
    rng = np.random.default_rng(42)
    diff = []
    for _ in range(1000):
        starts = rng.integers(0, len(base), size=int(np.ceil(len(base) / 20)))
        ix = ((starts[:, None] + np.arange(20)) % len(base)).ravel()[: len(base)]
        diff.append(sharpe(selected.excess_return.iloc[ix]) - sharpe(base.excess_return.iloc[ix]))
    uncertainty = {
        "paired_sharpe_difference_interval_95": np.quantile(diff, [0.025, 0.975]).tolist(),
        "screened_candidates": len(screen),
        "eligible_candidates": int(screen.eligible.sum()),
        "selected_diversifiers": len(picks),
        "warning": "Descriptive bootstrap only; does not account for retrospective correlation screening, model selection or prior searches.",
    }
    (out / "uncertainty.json").write_text(json.dumps(uncertainty, indent=2))
    print(picks.to_string(index=False))
    print(ranking.to_string(index=False))
    print(json.dumps(uncertainty, indent=2))


if __name__ == "__main__":
    main(Path.cwd())
