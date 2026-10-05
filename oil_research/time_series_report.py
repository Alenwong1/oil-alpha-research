"""Audit saved time-series results and render the complete, unfiltered comparison."""

from pathlib import Path
import html, json
import numpy as np
import pandas as pd
from .data import sha256
from .report import markdown_table


def main(root):
    out = root / "results/time_series"
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
    base = ledgers["ridge_h21"]
    old = pd.read_csv(
        root / "results/diversification/selected_development_ledger.csv",
        index_col=0,
        parse_dates=True,
    )
    np.testing.assert_allclose(base.net_return, old.net_return, rtol=0, atol=1e-11)
    for horizon in [5, 21]:
        prior = pd.read_csv(
            root / f"results/scarcity/existing_h{horizon}_portfolio_ledger.csv",
            index_col=0,
            parse_dates=True,
        )
        np.testing.assert_allclose(
            ledgers[f"ridge_h{horizon}"].net_return, prior.net_return, rtol=0, atol=1e-11
        )
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
                    - r.set_index("candidate").loc["ridge_h21", "sharpe_excess"]
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
        "No overall improvement: the existing 21-session ridge portfolio remains best at net cash-excess Sharpe 1.185, CAGR 10.7% and maximum drawdown 9.0%. Adaptive ridge at five sessions comes closest at Sharpe 1.159, CAGR 11.0% and maximum drawdown 7.9%. The 1.5 excess-Sharpe target remains unmet.",
        "Tested exactly six predefined configurations: existing ridge, lagged boosting and lagged adaptive ridge, each at five and 21 sessions. The seven-feature controls use the prior inventory innovations and seasonal flows. Both new models use 44 features: 12 current states plus 1/2/4/8-release lags of eight innovation/scarcity/demand states. This comparison changes both features and model treatment; it cannot isolate a pure architecture effect.",
        "Lagged features are built at release frequency, with daily copies collapsed to the first available signal row without filling missing values. Event availability is conservatively the first signal cutoff that observed it. Observation-gap checks prevent missing releases from masquerading as weekly lags. Feature coverage is 90.8% for lagged models versus 98.6% for controls; missing features disable the scheduled inventory target.",
        "Adaptive ridge uses a fixed two-year observation-weight half-life, training-only weighted scaling and quarterly coefficient updates. Boosting uses small seven-leaf trees, 100 iterations, large leaves and no randomized validation or early stopping. All parameters were fixed before scoring; no follow-up search was performed.",
        "All portfolios retain fixed 37.5% macro/news, 37.5% inventory and 25% technical allocations, netted in one USO account. Costs remain 5 bps per traded dollar, 3% annual short borrow and cash plus 2% for leveraged-long financing, with no short-collateral rebate. Long/short targets allow up to 2x gross exposure. Weekly inventory execution, risk sizing and strength thresholds are unchanged.",
        "Forecast accuracy is weak: on common valid dates, every model has worse normalized-return RMSE than a zero forecast. Adaptive five-session ridge has the highest five-session forecast/return correlation (0.073), but its RMSE is worse than the five-session ridge control. Its higher strategy Sharpe is not evidence of better calibrated forecasts. Daily outcomes overlap and weekly feature values repeat, so these daily counts are not independent samples.",
        "All 2,260 scored sessions in 2017–2025 are retained. Both ridge controls reproduce their previous daily returns within 1e-11. Training requires every outcome to finish strictly before the test quarter; forecast uncertainty waits for completed outcomes. 2026 remains unopened. The full test suite passes 51 tests.",
        "The macro/news sleeve retains the earlier sparse OilPrice headlines; this experiment does not add news coverage. The technical diversifier held USO only in 2020–2021, and cash otherwise. These are repeated development experiments, not independent validation. Retrospective market adjustments, actual open execution and historical borrow availability remain limitations.",
        "Paired circular 21-session block-bootstrap intervals use 2,000 draws with seed 49021 and compare each portfolio with the current 21-session control. They are descriptive and do not adjust for selecting models or for previous experiments. Keep the existing control; retain adaptive five-session ridge as a research candidate, not a demonstrated Sharpe improvement. A future separately specified calibration or same-feature ablation could investigate its weak forecast calibration without silently tuning this batch.",
    ]
    tables = [
        ("Six primary configurations", summary),
        ("Annual net cash-excess Sharpe", y.round(2)),
        ("Descriptive paired Sharpe differences", b.round(3)),
        ("Feature and forecast coverage", pd.read_csv(out / "coverage.csv").round(3)),
        ("Common-date forecast accuracy", pd.read_csv(out / "forecast_accuracy.csv").round(4)),
        (
            "Standalone inventory diagnostics",
            pd.read_csv(out / "metrics.csv")
            .query("kind == 'inventory'")[["candidate", "sharpe_excess", "cagr", "max_drawdown"]]
            .round(3),
        ),
        (
            "Selected control cost stress",
            pd.read_csv(out / "selected_cost_stress.csv")[
                ["cost_bps", "borrow_annual", "sharpe_excess", "cagr", "max_drawdown"]
            ].round(3),
        ),
    ]
    md = (
        "# Time-series ML results\n\n"
        + "\n\n".join(notes)
        + "\n\n"
        + "\n\n".join("## " + title + "\n\n" + markdown_table(t) for title, t in tables)
        + "\n"
    )
    (out / "report.md").write_text(md)
    body = (
        "<h1>Time-series ML results</h1>"
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
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Time-series ML results</title><style>'
        + style
        + "</style><body>"
        + body
        + "</body></html>"
    )
    print(json.dumps(audit, indent=2))
    print(y.round(2).to_string(index=False))


if __name__ == "__main__":
    main(Path.cwd())
