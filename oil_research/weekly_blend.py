"""Single logged, fixed-weight follow-up combining two selected development models."""

from pathlib import Path
import json, shutil
from datetime import datetime, timezone
import pandas as pd
from .extension_leverage import simulate_signed
from .extension_experiments import score
from .data import sha256


def main(root):
    out = root / "results/weekly/blend_followup"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "plan.json").exists():
        raise ValueError("Follow-up already recorded")
    files = [
        root / "results/extension/horizons_repaired/selected_development_ledger.csv",
        root / "results/weekly/selected_development_ledger.csv",
    ]
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "weights": [0.5, 0.5],
        "source_sha256": {str(p.relative_to(root)): sha256(p) for p in files},
        "code_sha256": sha256(Path(__file__)),
        "protocol_sha256": sha256(root / "docs/WEEKLY_PLAN.md"),
        "warning": "One development follow-up using two previously selected winners. No independent validation. No blend-weight optimization.",
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    shutil.copy2(Path(__file__), out / "code_snapshot.py")
    a, b = [pd.read_csv(p, index_col=0, parse_dates=True) for p in files]
    assert a.index.equals(b.index)
    target = 0.5 * a.weight + 0.5 * b.weight
    m = pd.read_csv(
        root / "results/extension/news_refresh/model_input_outcomes.csv",
        index_col=0,
        parse_dates=[0, "execution_date", "label_end"],
    ).loc[a.index]
    dt = (m.label_end - m.execution_date).dt.days / 365
    result = []
    for c in [0, 5, 10, 20]:
        for borrow in [0.03, 0.10, 0.30]:
            l = simulate_signed(
                m.target_return, m.cash_return, dt, target, cost_bps=c, borrow_annual=borrow
            )
            result.append({"cost_bps": c, "borrow_annual": borrow, **score(l)})
            if c == 5 and borrow == 0.03:
                chosen = l
    pd.DataFrame(result).to_csv(out / "selected_cost_stress.csv", index=False)
    chosen.to_csv(out / "selected_development_ledger.csv")
    annual = []
    for year, g in chosen.groupby(chosen.index.year):
        g = g.copy()
        g["equity"] = (1 + g.net_return).cumprod()
        annual.append({"year": year, **score(g)})
    pd.DataFrame(annual).to_csv(out / "selected_annual.csv", index=False)
    stats = {
        **score(chosen),
        "component_excess_return_correlation": float(a.excess_return.corr(b.excess_return)),
        "max_gross_target": float(target.abs().max()),
    }
    (out / "metrics.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main(Path.cwd())
