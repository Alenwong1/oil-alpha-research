"""Full-period refinement results and annual excess-Sharpe comparison."""

from pathlib import Path
import json, html, os
import numpy as np
import pandas as pd
from .report import markdown_table


def main(root):
    out = root / "results/refinement"
    r = pd.read_csv(out / "ranking.csv")
    best = r.iloc[0]
    annual = pd.read_csv(out / "annual.csv")
    selected = annual[annual.candidate == best.candidate].set_index("year")
    previous = pd.read_csv(root / "results/weekly/blend_followup/selected_annual.csv").set_index(
        "year"
    )
    years = pd.DataFrame(
        {
            "Year": selected.index,
            "New excess Sharpe": selected.sharpe_excess.values,
            "Prior blend excess Sharpe": previous.loc[selected.index, "sharpe_excess"].values,
            "New return": selected.cagr.values,
            "New drawdown": selected.max_drawdown.values,
        }
    )
    # CAGR in the underlying annual file is annualized using 252 sessions; show actual calendar-slice return instead.
    ledger = pd.read_csv(out / "selected_development_ledger.csv", index_col=0, parse_dates=True)
    years["New return"] = [
        float((1 + ledger.loc[ledger.index.year == y, "net_return"]).prod() - 1) for y in years.Year
    ]
    years.to_csv(out / "yearly_comparison.csv", index=False)
    matrix = pd.DataFrame(
        {
            row.candidate: pd.read_csv(
                out / f"{row.candidate}_ledger.csv", index_col=0
            ).excess_return
            for _, row in r.iterrows()
        }
    ).to_numpy()
    rng = np.random.default_rng(42)
    stats = []
    nullmax = []
    centered = matrix - matrix.mean(axis=0)
    for _ in range(1000):
        starts = rng.integers(0, len(matrix), size=int(np.ceil(len(matrix) / 20)))
        ix = ((starts[:, None] + np.arange(20)) % len(matrix)).ravel()[: len(matrix)]
        v = matrix[ix, 0]
        stats.append(np.sqrt(252) * v.mean() / v.std(ddof=1))
        n = centered[ix]
        sd = n.std(axis=0, ddof=1)
        sr = np.divide(np.sqrt(252) * n.mean(axis=0), sd, out=np.zeros_like(sd), where=sd > 1e-12)
        nullmax.append(sr.max())
    uncertainty = {
        "selected_sharpe_interval_95": np.quantile(stats, [0.025, 0.975]).tolist(),
        "batch_only_max_null_pvalue": float(
            (1 + sum(v >= best.sharpe_excess for v in nullmax)) / 1001
        ),
        "warning": "Selected interval is descriptive, not adjusted for selection. Batch diagnostic covers 24 policies only, not all prior searches.",
    }
    (out / "uncertainty.json").write_text(json.dumps(uncertainty, indent=2))
    rmse = pd.read_csv(out / "inventory_rmse.csv")
    rmse["RMSE change %"] = 100 * (rmse.improved_rmse / rmse.seasonal_rmse - 1)
    baseline = r.set_index("candidate").loc["original_existing_h0_blend"]
    invold = r.set_index("candidate").loc["original_existing_h0_inventory"]
    invnew = r.set_index("candidate").loc["improved_existing_h0.25_inventory"]
    intro = f"The best of 24 predefined variants has full-period development excess Sharpe {best.sharpe_excess:.2f}, versus {baseline.sharpe_excess:.2f} for the prior blend. CAGR is {best.cagr:.1%}; maximum drawdown is {best.max_drawdown:.1%}. Every year from 2017 through 2025 is included, with 5 bps trading cost, 3% annual short borrow and modeled financing."
    notes = [
        "Selected strategy: a fixed 50/50 exposure blend of the previous long/cash model and the autoregressive inventory-surprise model. The inventory sleeve trades weekly, requires forecast strength of at least 0.25 times past forecast-error RMS, retains the cost gate, and uses the existing trailing-volatility sizing. No blend weights were optimized.",
        "The new weekly inventory forecast uses seasonal harmonics, past changes and previously released flows. Distillate forecast RMSE improved; gasoline was nearly unchanged; crude RMSE worsened slightly. A better selected trading result is not proof of uniformly better inventory forecasts.",
        f"The new inventory sleeve is active on {invnew.active_fraction:.1%} of sessions versus {invold.active_fraction:.1%} for the prior sleeve. However, annual turnover is {invnew.annual_turnover:.2f} versus {invold.annual_turnover:.2f}; fewer exposed days did not mean lower total trading costs. The filter reduces weak positions but introduces entries/exits.",
        "The alternative conservative sizing rule uses the larger of short/long trailing risk estimates and an uncertainty haircut. It reduced drawdowns but did not win on excess Sharpe. Historical error RMS is not a calibrated confidence probability.",
        "Inventory forecasts use strictly earlier releases. Trading-model preprocessing is training-only, with 21-session outcomes purged at quarterly boundaries. Error RMS only incorporates predictions whose label end has occurred by the decision date. Risk estimates exclude unfinished daily returns.",
        "Annual Sharpe values use each year’s daily excess-return mean and sample standard deviation, annualized by sqrt(252). They are not averaged to obtain the full-period Sharpe. Signal-date years define the slices; boundary positions are carried, not independently restarted.",
        "These remain selected development results. The 2026 period is unopened, and 1.5 excess Sharpe has not been achieved. Return concentration in 2018 and 2020–2022 and weaker recent years deserve attention. Public market-data revisions and modeled financing/borrow availability remain limitations.",
        f'Descriptive selected Sharpe interval: {uncertainty["selected_sharpe_interval_95"][0]:.2f} to {uncertainty["selected_sharpe_interval_95"][1]:.2f}. The new-batch-only null-maximum diagnostic has p-value {uncertainty["batch_only_max_null_pvalue"]:.3f}. Neither accounts for all prior researcher choices.',
    ]
    display = years.copy()
    for c in ["New excess Sharpe", "Prior blend excess Sharpe"]:
        display[c] = display[c].map(lambda v: f"{v:.2f}")
    for c in ["New return", "New drawdown"]:
        display[c] = display[c].map(lambda v: f"{v:.1%}")
    table = r[["candidate", "sharpe_excess", "cagr", "max_drawdown", "annual_turnover"]].copy()
    for c in ["sharpe_excess", "annual_turnover"]:
        table[c] = table[c].map(lambda v: f"{v:.2f}")
    for c in ["cagr", "max_drawdown"]:
        table[c] = table[c].map(lambda v: f"{v:.1%}")
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mpl-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    years.plot.bar(
        x="Year",
        y=["New excess Sharpe", "Prior blend excess Sharpe"],
        ax=axes[0],
        color=["#238883", "#735da5"],
    )
    axes[0].axhline(0, color="gray", lw=0.7)
    axes[0].set(
        ylabel="Annual excess Sharpe",
        xlabel="",
        title="2017–2025 development results — every year retained",
    )
    axes[0].tick_params(axis="x", rotation=0)
    prior = pd.read_csv(
        root / "results/weekly/blend_followup/selected_development_ledger.csv",
        index_col=0,
        parse_dates=True,
    )
    axes[1].plot(ledger.index, ledger.equity, label="Selected refinement", color="#238883")
    axes[1].plot(prior.index, prior.equity, label="Prior blend", color="#735da5")
    axes[1].legend()
    axes[1].set(ylabel="Growth of $1", xlabel="Signal date")
    fig.tight_layout()
    fig.savefig(out / "annual_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    md = (
        "# Full-period inventory refinement\n\n"
        + intro
        + "\n\n## Individual years\n\n"
        + markdown_table(display)
        + "\n\n![Annual comparison](annual_comparison.png)\n\n## Interpretation\n\n"
        + "\n\n".join(notes)
        + "\n\n## Weekly inventory forecast accuracy\n\n"
        + markdown_table(rmse.round(3))
        + "\n\n## All 24 policies\n\n"
        + markdown_table(table)
        + "\n"
    )
    (out / "report.md").write_text(md)
    body = (
        '<h1>Full-period inventory refinement</h1><p class="lead">'
        + html.escape(intro)
        + "</p><h2>Individual years</h2>"
        + display.to_html(index=False, border=0)
        + '<img src="annual_comparison.png"><h2>Interpretation</h2>'
        + "".join("<p>" + html.escape(n) + "</p>" for n in notes)
        + "<h2>Weekly inventory forecast accuracy</h2>"
        + rmse.round(3).to_html(index=False, border=0)
        + "<h2>All 24 policies</h2>"
        + table.to_html(index=False, border=0)
    )
    style = "body{max-width:1120px;margin:40px auto;padding:24px;font:16px/1.6 system-ui;background:#fafbf9;color:#203044}.lead{font-size:20px}table{width:100%;border-collapse:collapse;font-size:12px}td,th{padding:8px;text-align:left;border-bottom:1px solid #ccd6d8}th{background:#e8efed}img{width:100%;margin:24px 0}"
    (out / "report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Full-period inventory refinement</title><style>'
        + style
        + "</style><body>"
        + body
        + "</body></html>"
    )
    print(intro)
    print(display.to_string(index=False))
    print(json.dumps(uncertainty, indent=2))


if __name__ == "__main__":
    main(Path.cwd())
