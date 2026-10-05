"""Readable weekly research report, preserving negative findings and data caveats."""

from pathlib import Path
import json, html, os
import numpy as np
import pandas as pd
from .report import markdown_table


def main(root):
    out = root / "results/weekly"
    r = pd.read_csv(out / "ranking.csv")
    best = r.iloc[0]
    independent = r[~r.cftc_dependent].iloc[0]
    uncertainty = json.loads((out / "uncertainty.json").read_text())
    coverage = json.loads((out / "coverage.json").read_text())
    corr = pd.read_csv(out / "sleeve_correlations.csv", index_col=0)
    old = pd.read_csv(root / "results/extension/all_candidates.csv").iloc[0]
    eia = pd.read_csv(root / "data/weekly/eia_weekly.csv")
    cot = pd.read_csv(root / "data/weekly/cftc_weekly.csv")
    blend = (
        json.loads((out / "blend_followup/metrics.json").read_text())
        if (out / "blend_followup/metrics.json").exists()
        else None
    )
    blend_text = (
        f'Additional fixed 50/50 follow-up: combining the new inventory winner with the prior long/cash winner gives development excess Sharpe {blend["sharpe_excess"]:.2f}, CAGR {blend["cagr"]:.1%}, and maximum drawdown {blend["max_drawdown"]:.1%}. Component return correlation is {blend["component_excess_return_correlation"]:.2f}. This single follow-up uses two selected winners and was added after the original 20-policy batch; its weights were not optimized. It is separate from the 20-policy bootstrap diagnostic.'
        if blend
        else ""
    )
    intro = f"Tested 20 predefined policies using three small models and two equal-allocation combinations. Best new development excess Sharpe: {best.sharpe_excess:.2f}, compared with {old.sharpe_excess:.2f} for the best prior experiment. The new winner is {best.candidate}; CAGR {best.cagr:.1%}, maximum drawdown {best.max_drawdown:.1%}. All results use 2017–2025 and include modeled trading, borrow and financing costs."
    independent_text = f"Best result without CFTC inputs: {independent.candidate}, excess Sharpe {independent.sharpe_excess:.2f}. CFTC-dependent results remain provisional because original historical classifications are not fully verified."
    reached = (
        max(best.sharpe_excess, old.sharpe_excess, blend["sharpe_excess"] if blend else -99) > 1.5
    )
    target = (
        "A development result above 1.5 is not independently validated."
        if reached
        else "The requested 1.5 Sharpe target has not been achieved."
    )
    t = r[
        [
            "candidate",
            "sharpe_excess",
            "cagr",
            "max_drawdown",
            "annual_turnover",
            "average_gross_exposure",
            "cftc_dependent",
        ]
    ].copy()
    for c in ["sharpe_excess", "annual_turnover", "average_gross_exposure"]:
        t[c] = t[c].map(lambda v: f"{v:.2f}")
    for c in ["cagr", "max_drawdown"]:
        t[c] = t[c].map(lambda v: f"{v:.1%}")
    lo, hi = uncertainty["selected_sharpe_interval_95"]
    notes = [
        f"EIA: {len(eia)} retained release-specific weekly observations, from {eia.release_date.min()} to {eia.release_date.max()}. Release-specific archived CSVs (unannounced later replacements cannot be ruled out), next-calendar-day availability, correction quarantine and 21-day staleness expiry. Seasonal deviations are relative to a model using previous releases, not analyst consensus.",
        f"CFTC: {len(cot)} retained WTI physical futures-only reports through 2025. Ten-day conservative observation-to-availability assumption, with shutdown/catch-up and known broad correction windows excluded. This is not an exact historical publication-time database; current annual archives may contain later revisions.",
        "Models: three expanding-window ridge regressions, 21-session outcomes, 14 total features (three trend, seven fundamentals and four positioning). Training outcomes are purged at each quarterly cutoff. Imputation/scaling uses the training fold only. Missing/stale feature sets disable the affected sleeve.",
        "Portfolio construction: each sleeve targets 15% annualized volatility using its own past completed hypothetical returns. Two combinations allocate equally to risk-scaled sleeves. All sleeves trade USO, so any benefit is diversification of signals rather than instruments. Opposing targets are netted before costs. No correlation or combination weights are optimized.",
        "Execution: daily or first signal session of each week; next-open fills. Skipped trades retain shares and allow exposure to drift. A daily 2x safety cap can override the rebalance schedule. Cost-aware rules use fixed economic hurdles plus a 0.10-equity adjustment threshold, not thresholds chosen from these outcomes.",
        "Costs: 5 bps per traded dollar, 3% annual short borrow, long financing at the cash proxy plus 2%, and no rebate on short collateral. Cash-proxy return is subtracted for Sharpe. Higher costs/borrow rates are stress-tested with the selected decision policy frozen. Intraday margin, actual historical borrow locates/rates and adjusted market-data vintages remain limitations.",
        f'New-winner descriptive 95% block-bootstrap Sharpe interval: [{lo:.2f}, {hi:.2f}]. This is not selection-adjusted. The maximum-statistic null diagnostic over only these 20 policies has p-value {uncertainty["new_family_null_max_pvalue"]:.3f}; it excludes the earlier 900 policies and other researcher choices.',
        "All scores are development results, even though forecasts were fitted chronologically. The reserved 2026 period has not been downloaded or evaluated. Negative years are retained. No live performance is claimed.",
    ]
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mpl-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fafbf9",
            "axes.facecolor": "#fafbf9",
        }
    )
    fig, ax = plt.subplots(figsize=(12, 5.5))
    r.sort_values("sharpe_excess").plot.barh(
        x="candidate", y="sharpe_excess", ax=ax, color="#238883", legend=False
    )
    ax.axvline(1.5, color="#bd4a43", ls="--")
    ax.set(
        xlabel="Development excess Sharpe after costs",
        ylabel="",
        title="All 20 predefined policies | 2017–2025",
    )
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(out / "policy_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    selected = pd.read_csv(out / "selected_development_ledger.csv", index_col=0, parse_dates=True)
    prior = pd.read_csv(
        root / "results/extension/horizons_repaired/selected_development_ledger.csv",
        index_col=0,
        parse_dates=True,
    )
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    axes[0].plot(selected.index, selected.equity, label="Best new policy", color="#238883")
    axes[0].plot(prior.index, prior.equity, label="Best prior long/cash policy", color="#735da5")
    if blend:
        combined = pd.read_csv(
            out / "blend_followup/selected_development_ledger.csv", index_col=0, parse_dates=True
        )
        axes[0].plot(
            combined.index, combined.equity, label="Fixed 50/50 follow-up", color="#d79528"
        )
    axes[0].legend(frameon=False)
    axes[0].set(
        ylabel="Growth of $1", title="Selected development performance — not an independent test"
    )
    axes[1].plot(
        selected.index,
        100 * (selected.equity / selected.equity.cummax().clip(lower=1) - 1),
        color="#238883",
    )
    axes[1].set(ylabel="New policy drawdown (%)", xlabel="Signal date")
    fig.tight_layout()
    fig.savefig(out / "selected_equity.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    annual = pd.read_csv(out / "annual.csv")
    annual = annual[annual.candidate == best.candidate][
        ["year", "sharpe_excess", "cagr", "max_drawdown"]
    ].copy()
    annual["sharpe_excess"] = annual.sharpe_excess.map(lambda v: f"{v:.2f}")
    for c in ["cagr", "max_drawdown"]:
        annual[c] = annual[c].map(lambda v: f"{v:.1%}")
    text = (
        "# Weekly fundamentals and positioning results\n\n"
        + intro
        + "\n\n"
        + blend_text
        + "\n\n"
        + independent_text
        + "\n\n**"
        + target
        + "**\n\n"
        + markdown_table(t)
        + "\n\n![Policy comparison](policy_comparison.png)\n\n![Equity and drawdown](selected_equity.png)\n\n## Winner by year\n\n"
        + markdown_table(annual)
        + "\n\n## Standalone sleeve excess-return correlations\n\n"
        + markdown_table(corr.reset_index().round(3))
        + "\n\n## Method and limitations\n\n"
        + "\n\n".join(notes)
        + "\n"
    )
    text += "\nSources: [EIA weekly archive](https://www.eia.gov/petroleum/supply/weekly/archive/), [CFTC annual archives](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm), [CFTC special announcements](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm).\n"
    (out / "report.md").write_text(text)
    body = (
        '<h1>Weekly fundamentals and positioning</h1><p class="lead">'
        + html.escape(intro)
        + "</p><p>"
        + html.escape(blend_text)
        + "</p><p>"
        + html.escape(independent_text)
        + "</p><p><b>"
        + target
        + "</b></p>"
        + t.to_html(index=False, border=0)
        + '<img src="policy_comparison.png"><img src="selected_equity.png"><h2>Winner by year</h2>'
        + annual.to_html(index=False, border=0)
        + "<h2>Standalone sleeve correlations</h2>"
        + corr.round(3).to_html(border=0)
        + "<h2>Method and limitations</h2>"
        + "".join("<p>" + html.escape(s) + "</p>" for s in notes)
    )
    body += '<p>Sources: <a href="https://www.eia.gov/petroleum/supply/weekly/archive/">EIA weekly archive</a> · <a href="https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm">CFTC annual archives</a> · <a href="https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm">CFTC delay notices</a></p>'
    style = "body{max-width:1120px;margin:40px auto;padding:24px;background:#fafbf9;color:#203044;font:16px/1.6 system-ui}.lead{font-size:20px}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:8px;text-align:left;border-bottom:1px solid #ccd6d8}th{background:#e8efed}img{width:100%;margin:24px 0}"
    (out / "report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Weekly oil research</title><style>'
        + style
        + "</style><body>"
        + body
        + "</body></html>"
    )
    print(intro)
    print(blend_text)
    print(independent_text)
    print(target)


if __name__ == "__main__":
    main(Path.cwd())
