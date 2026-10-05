from pathlib import Path
import html, json
import pandas as pd
from .report import markdown_table


def main(root):
    out = root / "results/diversification"
    r = pd.read_csv(out / "ranking.csv")
    best = r.iloc[0]
    base = r[r.candidate == "base"].iloc[0]
    a = pd.read_csv(out / "annual.csv")
    aa = a[a.candidate == best.candidate].set_index("year")
    bb = a[a.candidate == "base"].set_index("year")
    years = pd.DataFrame(
        {
            "year": aa.index,
            "blend_excess_sharpe": aa.sharpe_excess.values,
            "base_excess_sharpe": bb.loc[aa.index, "sharpe_excess"].values,
        }
    )
    years.to_csv(out / "yearly_comparison.csv", index=False)
    uncertainty = json.loads((out / "uncertainty.json").read_text())
    lo, hi = uncertainty["paired_sharpe_difference_interval_95"]
    intro = f"Best fixed mix: 75% latest strategy plus 25% USO-only ridge long/cash strategy (daily prediction, hysteresis allocation, smoothing span 5). Full-period excess Sharpe improves from {base.sharpe_excess:.2f} to {best.sharpe_excess:.2f}; maximum drawdown improves from {abs(base.max_drawdown):.1%} to {abs(best.max_drawdown):.1%}. CAGR changes from {base.cagr:.1%} to {best.cagr:.1%}. Component excess-return correlation is {best.component_correlation:.2f}."
    notes = [
        "Important activity limitation: the selected technical diversifier held USO on 165 sessions in 2020 and 229 in 2021, and stayed in cash throughout all other evaluated years. Its low correlation partly reflects inactivity. The small blended gain should not be interpreted as a broadly established independent return source.",
        f"2018–2025 sensitivity, excluding 2017: {best.sharpe_2018_2025:.2f}, versus {base.sharpe_2018_2025:.2f} for the base. Candidate selection used the full 2017–2025 period, not this shorter window.",
        "Screened 945 saved development candidates. Required standalone excess Sharpe at least 0.30 and return correlation at most 0.30; 32 qualified. Took the three highest-Sharpe eligible candidates from distinct forecast-model/source groups, then tested only 25% and 50% allocations. All six outcomes are retained below.",
        "All strategies trade USO. This diversifies signals within one instrument, not assets. Signed target exposures are combined and netted before simulating one daily-rebalanced account; costs are charged once on net trades. Gross target exposure is capped at 2x. Forecasts and component rules were not refit for this experiment.",
        "Primary assumptions: 5 bps per traded dollar, 3% annual short borrow, cash-rate plus 2% for leveraged-long financing, no short-collateral rebate. Sharpe subtracts the cash proxy. Historical market adjustments, borrow availability and intraday margin events remain limitations.",
        f"The descriptive paired block-bootstrap 95% interval for the Sharpe change is [{lo:.2f}, {hi:.2f}]. It includes zero. Screening and selecting on the same development returns adds selection bias that this interval does not correct. The small observed gain is not established as a durable improvement.",
        f"The same winning blend has zero-risk-free-rate Sharpe {best.sharpe_zero_rf:.2f}, but that includes cash interest. The research target remains 1.5 CASH-EXCESS Sharpe, which is not achieved. No evaluation of 2026 was performed.",
    ]
    summary = pd.DataFrame(
        [
            {
                "Strategy": "Latest base",
                "Excess Sharpe": base.sharpe_excess,
                "CAGR": base.cagr,
                "Max drawdown": base.max_drawdown,
                "Annual turnover": base.annual_turnover,
            },
            {
                "Strategy": "75/25 blend",
                "Excess Sharpe": best.sharpe_excess,
                "CAGR": best.cagr,
                "Max drawdown": best.max_drawdown,
                "Annual turnover": best.annual_turnover,
            },
        ]
    )
    for c in ["CAGR", "Max drawdown"]:
        summary[c] = summary[c].map(lambda v: f"{v:.1%}")
    for c in ["Excess Sharpe", "Annual turnover"]:
        summary[c] = summary[c].map(lambda v: f"{v:.2f}")
    table = r[
        [
            "candidate",
            "diversifier_fraction",
            "component_correlation",
            "sharpe_excess",
            "cagr",
            "max_drawdown",
            "sharpe_2018_2025",
        ]
    ].round(3)
    y = years.round(2)
    md = (
        "# Low-correlation blend results\n\n"
        + intro
        + "\n\n"
        + markdown_table(summary)
        + "\n\n## Individual-year excess Sharpe\n\n"
        + markdown_table(y)
        + "\n\n## All fixed mixes\n\n"
        + markdown_table(table)
        + "\n\n## Interpretation\n\n"
        + "\n\n".join(notes)
        + "\n"
    )
    (out / "report.md").write_text(md)
    body = (
        '<h1>Low-correlation blend results</h1><p class="lead">'
        + html.escape(intro)
        + "</p>"
        + summary.to_html(index=False, border=0)
        + "<h2>Individual-year excess Sharpe</h2>"
        + y.to_html(index=False, border=0)
        + "<h2>All fixed mixes</h2>"
        + table.to_html(index=False, border=0)
        + "<h2>Interpretation</h2>"
        + "".join("<p>" + html.escape(t) + "</p>" for t in notes)
    )
    style = "body{max-width:1100px;margin:40px auto;padding:24px;background:#fafbf9;color:#203044;font:16px/1.6 system-ui}.lead{font-size:20px}table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:9px;text-align:left;border-bottom:1px solid #ccd6d8}th{background:#e8efed}"
    (out / "report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Low-correlation blend results</title><style>'
        + style
        + "</style><body>"
        + body
        + "</body></html>"
    )
    print(intro)
    print(y.to_string(index=False))


if __name__ == "__main__":
    main(Path.cwd())
