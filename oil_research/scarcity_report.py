"""Audit saved scarcity results and render the complete, unfiltered comparison."""

from pathlib import Path
import html, json
import numpy as np
import pandas as pd
from .data import sha256
from .report import markdown_table


def main(root):
    out = root / "results/scarcity"
    r = pd.read_csv(out / "ranking.csv")
    a = pd.read_csv(out / "annual.csv")
    plan = json.loads((out / "plan.json").read_text())
    for p, h in plan["input_hashes"].items():
        assert sha256(root / p) == h
    for p, h in plan["source_hashes"].items():
        assert sha256(out / "code_snapshot" / p) == h
    assert sha256(out / "protocol.md") == plan["protocol_sha256"]
    line = pd.read_csv(out / "lineage.csv")
    av = pd.to_datetime(line.available_at, utc=True)
    cut = pd.to_datetime(line.cutoff, utc=True)
    assert (av.dropna() <= cut[av.notna()]).all()
    folds = 0
    for p in out.glob("*_training_audit.csv"):
        f = pd.read_csv(p)
        assert (pd.to_datetime(f.last_training_outcome) < pd.to_datetime(f.test_start)).all()
        folds += len(f)
    ledgers = {}
    for p in out.glob("*_portfolio_ledger.csv"):
        l = pd.read_csv(p, index_col=0, parse_dates=True)
        assert len(l) == 2260 and l.index.year.min() == 2017 and l.index.year.max() == 2025
        assert l.net_return.notna().all() and l.weight.abs().max() <= 2 + 1e-10
        ledgers[p.name.replace("_portfolio_ledger.csv", "")] = l
    base = ledgers["existing_h21"]
    old = pd.read_csv(
        root / "results/diversification/selected_development_ledger.csv",
        index_col=0,
        parse_dates=True,
    )
    np.testing.assert_allclose(base.net_return, old.net_return, rtol=0, atol=1e-11)
    audit = {
        "input_hashes_verified": len(plan["input_hashes"]),
        "code_hashes_verified": len(plan["source_hashes"]),
        "availability_comparisons": int(av.notna().sum()),
        "purged_training_folds": folds,
        "primary_portfolios": len(ledgers),
        "sessions_each": len(base),
        "control_max_absolute_return_difference": float(
            (base.net_return - old.net_return).abs().max()
        ),
        "no_2026_evaluation": True,
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    # Paired circular 21-session blocks; descriptive only, not selection adjusted.
    rng = np.random.default_rng(49021)
    n = len(base)
    draws = 2000
    block = 21
    ix = (
        (rng.integers(0, n, size=(draws, int(np.ceil(n / block))))[:, :, None] + np.arange(block))
        % n
    ).reshape(draws, -1)[:, :n]

    def sr(v):
        return np.sqrt(252) * v.mean(axis=1) / v.std(axis=1, ddof=1)

    bs = sr((base.net_return - base.cash_return).to_numpy()[ix])
    boot = []
    for key, l in ledgers.items():
        assert l.index.equals(base.index)
        diff = sr((l.net_return - l.cash_return).to_numpy()[ix]) - bs
        lo, hi = np.quantile(diff, [0.025, 0.975])
        boot.append(
            {
                "candidate": key,
                "delta_sharpe": float(
                    r.set_index("candidate").loc[key, "sharpe_excess"]
                    - r.set_index("candidate").loc["existing_h21", "sharpe_excess"]
                ),
                "paired_95_low": lo,
                "paired_95_high": hi,
            }
        )
    b = pd.DataFrame(boot)
    b.to_csv(out / "bootstrap.csv", index=False)
    y = (
        a[a.kind == "portfolio"]
        .pivot(index="year", columns="candidate", values="sharpe_excess")
        .reindex(columns=r.candidate)
        .reset_index()
    )
    y.to_csv(out / "yearly_comparison.csv", index=False)
    summary = r[
        [
            "candidate",
            "features",
            "horizon",
            "sharpe_excess",
            "cagr",
            "max_drawdown",
            "information_ratio_USO",
        ]
    ].round(3)
    notes = [
        "No improvement: the existing 21-session inventory model remains best. Net cash-excess Sharpe is 1.185, CAGR 10.7%, maximum drawdown 9.0%, and information ratio versus buy-and-hold USO is 0.176. The 1.5 excess-Sharpe target is not achieved.",
        "All six configurations use fixed 37.5% macro/news, 37.5% inventory and 25% technical allocations, netted into one USO account. The new features are release-specific gasoline/distillate product supplied, constructed four-week demand and days-of-supply measures, lagged seasonal scarcity, and surprise interactions. Scarcity expands seven features to thirteen; demand confirmation expands to eighteen. These additions did not help this model on the development period; that is not proof the underlying economic hypotheses are false.",
        "All 744 retained EIA releases were read from cached original-release tables, with observation dates and file checksums checked. Availability remains conservatively delayed until the day after release, with stale observations disabled. Product supplied is an apparent-demand proxy. Constructed four-week measures require consecutive retained weeks and are not the official days-of-supply series.",
        "Primary net results charge 5 bps per traded dollar, 3% annual borrow, cash rate plus 2% financing for leveraged longs, and no short-collateral rebate. Gross targets permit long/short positions up to 2x. No new leverage or threshold search was performed.",
        "Every primary portfolio retains the same 2,260 sessions in 2017–2025. Quarterly training purges unfinished horizon outcomes. Forecast uncertainty includes completed predictions only. Coverage falls from 98.6% for the control to 96.1% for scarcity and 95.9% for demand; missing forecasts disable the inventory target under the scheduled-trading rule.",
        "The existing macro/news component still contains the previously saved sparse OilPrice headline data. This batch adds EIA features, not a new or more complete news feed. The technical component traded only in 2020–2021 and was in cash in other years, which limits the strength of the diversification claim.",
        "These are development results after many earlier experiments. The paired 21-session block-bootstrap intervals use 2,000 resamples and do not correct for the research selection history. 2026 remains unopened. Retrospective market-data adjustments, actual fills and historical borrow availability remain unverified.",
        "Next research decision: retain the existing control and reject these additions for now. A separately specified event-focused inventory strategy, using an independently constructed surprise and first tradable price after release, would test a different mechanism. A public consensus-surprise or intraday claim requires suitable historical timestamped data; this batch does not establish it.",
    ]
    tables = [
        ("Six primary configurations", summary),
        ("Annual net cash-excess Sharpe", y.round(2)),
        ("Descriptive paired Sharpe differences", b.round(3)),
        ("Feature and forecast coverage", pd.read_csv(out / "coverage.csv").round(3)),
        (
            "Selected control cost stress",
            pd.read_csv(out / "selected_cost_stress.csv")[
                ["cost_bps", "borrow_annual", "sharpe_excess", "cagr", "max_drawdown"]
            ].round(3),
        ),
    ]
    md = (
        "# Scarcity and demand alpha results\n\n"
        + "\n\n".join(notes)
        + "\n\n"
        + "\n\n".join("## " + title + "\n\n" + markdown_table(t) for title, t in tables)
        + "\n"
    )
    (out / "report.md").write_text(md)
    body = (
        "<h1>Scarcity and demand alpha results</h1>"
        + "".join("<p>" + html.escape(t) + "</p>" for t in notes)
        + "".join(
            "<h2>"
            + title
            + '</h2><div class="scroll">'
            + t.to_html(index=False, border=0)
            + "</div>"
            for title, t in tables
        )
    )
    style = "body{max-width:1200px;margin:40px auto;padding:24px;background:#fafbf9;color:#203044;font:16px/1.6 system-ui}table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:9px;text-align:left;border-bottom:1px solid #ccd6d8}th{background:#e8efed}.scroll{overflow:auto}"
    (out / "report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scarcity alpha results</title><style>'
        + style
        + "</style><body>"
        + body
        + "</body></html>"
    )
    print(json.dumps(audit, indent=2))
    print(y.round(2).to_string(index=False))


if __name__ == "__main__":
    main(Path.cwd())
