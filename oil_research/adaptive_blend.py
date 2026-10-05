"""Fixed mixes of the existing best portfolio and adaptive five-session portfolio."""

from pathlib import Path
from datetime import datetime, timezone
import json, shutil
import numpy as np
import pandas as pd
from .extension_leverage import simulate_signed
from .extension_experiments import score
from .data import sha256
from .report import markdown_table


def main(root):
    out = root / "results/adaptive_blend"
    out.mkdir(exist_ok=True)
    if (out / "plan.json").exists():
        raise ValueError("Preserve existing results")
    files = [
        "results/time_series/ridge_h21_portfolio_ledger.csv",
        "results/time_series/adaptive_h5_portfolio_ledger.csv",
        "results/extension/news_refresh/model_input_outcomes.csv",
    ]
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "adaptive_fractions": [0, 0.25, 0.5, 0.75, 1],
        "primary_selection": "2017–2025 cash-excess Sharpe",
        "requested_subset": "2018–2024 inclusive, report both cash-excess and zero-RF Sharpe; retrospective subset selection is separate",
        "costs": "5bps turnover, 3% borrow, cash+2% leveraged financing, no short rebate",
        "method": "Mix target USO positions, net before costs, simulate full period then slice same daily ledger. No fresh entry/exit at subset boundaries. Hold the common macro/news and technical components fixed; only inventory model mix changes. No retraining. 2026 unopened.",
        "input_hashes": {p: sha256(root / p) for p in files},
        "code_sha256": sha256(Path(__file__)),
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    shutil.copy2(Path(__file__), out / "code_snapshot.py")
    a, b = [pd.read_csv(root / p, index_col=0, parse_dates=True) for p in files[:2]]
    assert a.index.equals(b.index)
    m = pd.read_csv(root / files[2], index_col=0, parse_dates=True)
    for c in ["execution_date", "label_end"]:
        m[c] = pd.to_datetime(m[c])
    ix = a.index
    dt = (m.loc[ix, "label_end"] - m.loc[ix, "execution_date"]).dt.days / 365
    rows = []
    years = []
    ledgers = {}
    for fraction in plan["adaptive_fractions"]:
        w = (1 - fraction) * a.weight + fraction * b.weight
        l = simulate_signed(a.asset_return, a.cash_return, dt, w)
        ledgers[fraction] = l
        assert l.weight.abs().max() <= 2 + 1e-10 and l.net_return.notna().all()
        if fraction in [0, 1]:
            np.testing.assert_allclose(
                l.net_return, (a if fraction == 0 else b).net_return, rtol=0, atol=1e-11
            )
        l.to_csv(out / f"adaptive_{int(fraction*100)}_ledger.csv")
        for period, part in [
            ("2017–2025", l),
            ("2018–2024", l[(ix.year >= 2018) & (ix.year <= 2024)]),
        ]:
            part = part.copy()
            part["equity"] = (1 + part.net_return).cumprod()
            rows.append(
                {
                    "period": period,
                    "adaptive_fraction": fraction,
                    "existing_fraction": 1 - fraction,
                    **score(part),
                }
            )
        for year, part in l.groupby(l.index.year):
            part = part.copy()
            part["equity"] = (1 + part.net_return).cumprod()
            years.append({"adaptive_fraction": fraction, "year": year, **score(part)})
    r = pd.DataFrame(rows)
    r.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(years).to_csv(out / "annual.csv", index=False)
    full = r[r.period == "2017–2025"].sort_values("sharpe_excess", ascending=False)
    full.to_csv(out / "ranking.csv", index=False)
    best = full.iloc[0]
    ledgers[best.adaptive_fraction].to_csv(out / "selected_development_ledger.csv")
    # Paired descriptive bootstrap on the full period, without selection correction.
    rng = np.random.default_rng(49123)
    n = len(a)
    ind = (
        (rng.integers(0, n, size=(2000, int(np.ceil(n / 21))))[:, :, None] + np.arange(21)) % n
    ).reshape(2000, -1)[:, :n]

    def sr(v):
        return np.sqrt(252) * v.mean(axis=1) / v.std(axis=1, ddof=1)

    reference = sr((a.net_return - a.cash_return).to_numpy()[ind])
    boot = []
    for f, l in ledgers.items():
        diff = sr((l.net_return - l.cash_return).to_numpy()[ind]) - reference
        lo, hi = np.quantile(diff, [0.025, 0.975])
        boot.append({"adaptive_fraction": f, "paired_low": lo, "paired_high": hi})
    pd.DataFrame(boot).to_csv(out / "bootstrap.csv", index=False)
    subset = r[r.period == "2018–2024"]
    table = r[
        [
            "period",
            "existing_fraction",
            "adaptive_fraction",
            "sharpe_excess",
            "sharpe_zero_rf",
            "cagr",
            "max_drawdown",
        ]
    ].round(3)
    corr = float((a.net_return - a.cash_return).corr(b.net_return - b.cash_return))
    text = (
        "# Existing/adaptive strategy mixes\n\nTested three fixed interior mixes plus both endpoint controls. These are mixtures of two portfolios that already share macro/news and technical components; all trades remain in USO. Component full-period excess-return correlation: "
        + f"{corr:.3f}. "
        + "Positions are netted before modeled costs. Both endpoint controls reproduce prior returns within 1e-11.\n\n"
        + markdown_table(table)
        + "\n\nThe primary selection uses full 2017–2025 cash-excess Sharpe. The requested 2018–2024 figures are retrospective sensitivities from the same continuous ledger, not independently validated results or fresh-entry subset backtests. Zero-risk-free Sharpe includes cash income, but still deducts modeled trading/borrow/financing costs. 2026 remains unopened.\n\n## Full-period paired Sharpe-difference intervals\n\n"
        + markdown_table(pd.DataFrame(boot).round(3))
        + "\n\nIntervals use 2,000 circular 21-session block resamples and do not adjust for selecting the best mix or earlier research. No durable improvement is established by selecting the highest backtest Sharpe.\n"
    )
    (out / "report.md").write_text(text)
    print(table.to_string(index=False))
    print("Full-period selected adaptive fraction:", best.adaptive_fraction)
    print(
        "Subset best excess:",
        subset.sort_values("sharpe_excess", ascending=False).iloc[0].to_dict(),
    )
    print(
        "Subset best zero RF:",
        subset.sort_values("sharpe_zero_rf", ascending=False).iloc[0].to_dict(),
    )
    print("Correlation:", corr)
    print(pd.DataFrame(boot).to_string(index=False))


if __name__ == "__main__":
    main(Path.cwd())
