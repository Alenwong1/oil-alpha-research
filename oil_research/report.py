"""Generate portable research reports and publication-ready PNG charts."""

from __future__ import annotations

import html
import os
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = {
    "buy_hold": "USO buy & hold",
    "vol_target_long": "USO volatility target",
    "momentum": "21-day momentum",
    "rsi": "RSI reversal",
    "ridge_uso": "Ridge / USO only",
    "ridge_full": "Ridge / all ETFs",
    "tree_uso": "Boosted trees / USO only",
    "tree_full": "Boosted trees / all ETFs",
}


def markdown_table(frame):
    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(map(str, columns)) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def generate(
    output: Path, config: dict, summary, annual, ledgers, comparisons, forecast, audit, run_info
):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mpl-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "#fafbf9",
            "axes.facecolor": "#fafbf9",
            "savefig.facecolor": "#fafbf9",
        }
    )
    primary = config["primary_cost_bps"]
    main = summary.loc[summary.cost_bps == primary].copy().set_index("strategy")
    selected = ["buy_hold", "vol_target_long", "momentum", "ridge_full", "tree_full"]
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    colors = ["#838d99", "#d79528", "#735da5", "#238883", "#163c63"]
    for name, color in zip(selected, colors):
        ledger = ledgers[(name, primary)]
        dates = pd.to_datetime(ledger.label_end)
        equity = ledger.equity.to_numpy()
        axes[0].plot(dates, equity, label=LABELS[name], color=color, linewidth=1.6)
        drawdown = equity / np.maximum.accumulate(np.r_[1.0, equity])[1:] - 1
        axes[1].plot(dates, drawdown * 100, color=color, linewidth=1.2)
    axes[0].set(
        title=f"USO research | {run_info['period']} | {primary} bps per traded dollar",
        ylabel="Growth of $1",
    )
    axes[0].legend(loc="upper left", ncol=2, frameon=False)
    axes[1].set(ylabel="Drawdown (%)", xlabel="Holding-period end date")
    for ax in axes:
        ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output / "equity_drawdown.png", dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for name in ["momentum", "ridge_uso", "ridge_full", "tree_uso", "tree_full"]:
        s = summary.loc[summary.strategy == name]
        axes[0].plot(s.cost_bps, s.sharpe_zero_rf, marker="o", label=LABELS[name])
    axes[0].set(
        title="Cost sensitivity",
        xlabel="One-way cost (basis points)",
        ylabel="Sharpe (zero cash rate)",
    )
    axes[0].legend(fontsize=8, frameon=False)
    names = ["ridge_uso", "ridge_full", "tree_uso", "tree_full"]
    axes[1].bar(
        range(4),
        [main.loc[n, "sharpe_zero_rf"] for n in names],
        color=["#81b7b0", "#238883", "#809ab7", "#163c63"],
    )
    axes[1].set_xticks(range(4), ["Ridge\nUSO", "Ridge\nAll ETFs", "Trees\nUSO", "Trees\nAll ETFs"])
    axes[1].set(title=f"Feature ablation at {primary} bps", ylabel="Sharpe (zero cash rate)")
    for ax in axes:
        ax.axhline(0, color="#777", linewidth=0.6)
        ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(output / "costs_ablation.png", dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    pivot = annual.pivot(index="year", columns="strategy", values="sharpe_zero_rf")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    pivot[["vol_target_long", "momentum", "ridge_full", "tree_full"]].rename(
        columns=LABELS
    ).plot.bar(ax=ax, color=["#d79528", "#735da5", "#238883", "#163c63"])
    ax.set(
        title="Performance across calendar years",
        ylabel="Sharpe (zero cash rate)",
        xlabel="Holding-period end year",
    )
    ax.axhline(0, color="#777", linewidth=0.6)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(output / "annual_performance.png", dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    table = pd.DataFrame(
        {
            "Strategy": [LABELS[n] for n in main.index],
            "CAGR": [f"{v:.1%}" for v in main.cagr],
            "Sharpe": [f"{v:.2f}" for v in main.sharpe_zero_rf],
            "Max drawdown": [f"{v:.1%}" for v in main.max_drawdown],
            "Turnover/year": [f"{v:.1f}x" for v in main.annual_turnover],
            "Avg exposure": [f"{v:.1%}" for v in main.average_exposure],
        }
    )
    a = main.loc["tree_full", "sharpe_zero_rf"]
    b = main.loc["momentum", "sharpe_zero_rf"]
    ci = comparisons.loc[comparisons.comparison == "tree_full - momentum"].iloc[0]
    conclusion = (
        f"At {primary} bps, boosted trees with all ETF features produced a Sharpe of {a:.2f}, "
        f"compared with {b:.2f} for momentum. The paired block-bootstrap 95% interval for "
        f"their Sharpe difference is [{ci.lower_95:.2f}, {ci.upper_95:.2f}]. "
        + (
            "The interval includes zero; this run does not establish a reliable advantage."
            if ci.lower_95 <= 0 <= ci.upper_95
            else "The interval excludes zero, but is descriptive, assumes approximate local stationarity, and is not adjusted for model search."
        )
    )
    methods = (
        "Signals use adjusted OHLCV available after close t; execution is assumed at open t+1 and exit/rebalance at open t+2. "
        "Training labels must end strictly before each fit cutoff. Development uses expanding quarterly fits; the final holdout "
        "freezes all fitted models using pre-2024 data. Each fit selects a small hyperparameter grid by MSE on its last 252 labeled "
        "training observations, with boundary labels purged. Scaling is fitted only on the inner training split. "
        "Tree early stopping is disabled to avoid random validation splits. Positive model forecasts hold USO; other forecasts hold cash. "
        "All active strategies use trailing 21-day volatility to target 15% annualized volatility, capped at 100% exposure. "
        "Buy-and-hold is unscaled; volatility-targeted long is the fairer risk-control baseline. RSI uses a simple 14-day rolling "
        "gain/loss average with 30/70 entry/exit hysteresis. Fees apply to drift-adjusted turnover, including initial purchase and final liquidation."
    )
    limits = [
        "This is a retrospective ETF study; only the measured results from the documented experiment are attributed to it.",
        "USO is a futures-holding fund, not spot crude or a directly traded futures strategy. Fund design changed around 2020 and transitioned again in 2023–2024.",
        "Yahoo data is a current adjusted historical snapshot, not an archived point-in-time feed. Adjustment revisions and data errors remain possible; hashes reproduce this snapshot only.",
        "Assumed official-open fills and fixed 0/5/10/20 bps costs do not model spreads by date, market impact, auction access, or execution capacity.",
        "Cash earns zero, and Sharpe uses a zero reference rate. Taxes are excluded; fund fees and roll effects are embedded in USO prices. Results are not excess returns over Treasury bills.",
        "The ETF universe, feature families and historical dates were selected retrospectively. The holdout is sealed within this implementation, not a prospective live test or a novelty guarantee.",
        "Multiple strategies are reported. Bootstrap intervals are descriptive and not corrected for multiple comparisons; two holdout years offer limited evidence.",
        "No macro releases, news, alternative data, shorting, live orders or brokerage integration are included.",
    ]
    source_list = [
        ("USO fund description", "https://www.uscfinvestments.com/uso"),
        ("USCF fund disclosures", "https://www.uscfinvestments.com/disclosures"),
        (
            "yfinance download and adjustment parameters",
            "https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html",
        ),
        ("yfinance data-use notice", "https://github.com/ranaroussi/yfinance"),
        (
            "scikit-learn time-series early-stopping discussion",
            "https://sklearn.org/stable/auto_examples/ensemble/plot_hgbt_regression.html",
        ),
    ]
    title = f"Oil ETF forecasting — {run_info['period']} research report"
    dates = f"Signal dates: {run_info['first_signal']} to {run_info['last_signal']}. {len(audit)} fitted model records; {run_info['features']} full-model features."
    body = f"# {title}\n\n{conclusion}\n\n{dates}\n\n## Results\n\n{markdown_table(table)}\n\n## Evaluation design\n\n{methods}\n\n"
    for filename in ["equity_drawdown.png", "costs_ablation.png", "annual_performance.png"]:
        body += f"![{filename.replace('_', ' ')}]({filename})\n\n"
    body += "## Forecast diagnostics\n\n" + markdown_table(forecast.round(6)) + "\n\n"
    body += "## Paired Sharpe comparisons\n\n" + markdown_table(comparisons.round(3)) + "\n\n"
    body += "## Limitations\n\n" + "\n".join(f"- {s}" for s in limits) + "\n\n"
    body += "## Reproducibility\n\nSee `run.json`, `config_snapshot.json`, `model_audit.csv`, `feature_dictionary.csv`, `predictions.csv`, and `ledgers/`. "
    body += "Each run records data and source hashes. Raw source bars are local research data and excluded from the shareable code archive.\n\n"
    body += "## Sources\n\n" + "\n".join(f"- [{name}]({url})" for name, url in source_list) + "\n"
    (output / "report.md").write_text(body)
    parts = [
        f"<p class='eyebrow'>REPRODUCIBLE RESEARCH · {html.escape(run_info['period'].upper())}</p>",
        f"<h1>{html.escape(title)}</h1><p class='lead'>{html.escape(conclusion)}</p><p>{html.escape(dates)}</p>",
        "<h2>Results</h2>" + table.to_html(index=False, border=0),
        f"<h2>Evaluation design</h2><p>{html.escape(methods)}</p>",
    ]
    for filename in ["equity_drawdown.png", "costs_ablation.png", "annual_performance.png"]:
        parts.append(f"<img src='{filename}' alt='{filename.replace('_', ' ')}'>")
    parts += [
        "<h2>Forecast diagnostics</h2>" + forecast.round(6).to_html(index=False, border=0),
        "<h2>Uncertainty</h2>" + comparisons.round(3).to_html(index=False, border=0),
        "<h2>Limitations</h2><ul>"
        + "".join(f"<li>{html.escape(s)}</li>" for s in limits)
        + "</ul>",
        "<h2>Sources</h2><ul>"
        + "".join(f"<li><a href='{url}'>{html.escape(name)}</a></li>" for name, url in source_list)
        + "</ul>",
        "<p class='footer'>Research artifact · all strategy returns are simulated · see run.json for provenance</p>",
    ]
    css = "body{margin:0;background:#fafbf9;color:#203044;font:16px/1.65 system-ui,sans-serif}main{max-width:1080px;margin:50px auto;padding:0 28px}h1{font-size:40px;line-height:1.15;max-width:850px}h2{margin-top:40px;font-size:23px}.eyebrow{font-size:12px;letter-spacing:2px;color:#238883}.lead{font-size:20px;max-width:950px}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}td,th{text-align:left;padding:10px;border-bottom:1px solid #d7dedf}th{background:#eaf0ee}img{display:block;width:100%;margin:34px 0}li{margin-bottom:12px}a{color:#176c72}.footer{font-size:12px;color:#65717d}@media(max-width:700px){main{padding:0 16px}h1{font-size:30px}table{display:block;overflow-x:auto}}"
    (output / "report.html").write_text(
        f"<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{css}</style><main>{''.join(parts)}</main></html>"
    )
