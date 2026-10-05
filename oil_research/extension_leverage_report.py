"""Readable comparison of the authorized signed-position experiment."""

from pathlib import Path
import json
import html
import pandas as pd
from .report import markdown_table


def main(root):
    folder = root / "results/extension/short_leverage"
    ranked = pd.read_csv(folder / "ranking.csv")
    best = ranked.iloc[0]
    match = ranked[
        (ranked.source_round == best.source_round)
        & (ranked.candidate == best.candidate)
        & (ranked.policy == best.policy)
        & (ranked.smoothing == best.smoothing)
    ]
    long = pd.read_csv(root / "results/extension/horizons_repaired/ranking.csv").iloc[0]
    records = [
        {
            "Strategy": "Best long/cash",
            "Excess Sharpe": long.sharpe_excess,
            "CAGR": long.cagr,
            "Max drawdown": long.max_drawdown,
        }
    ]
    for _, r in match.sort_values("vol_target").iterrows():
        records.append(
            {
                "Strategy": f"Long/short, {r.vol_target:.0%} volatility target",
                "Excess Sharpe": r.sharpe_excess,
                "CAGR": r.cagr,
                "Max drawdown": r.max_drawdown,
            }
        )
    table = pd.DataFrame(records)
    table["Excess Sharpe"] = table["Excess Sharpe"].map(lambda v: f"{v:.2f}")
    for col in ["CAGR", "Max drawdown"]:
        table[col] = table[col].map(lambda v: f"{v:.1%}")
    stress = pd.read_csv(folder / "selected_cost_stress.csv")
    stress = stress[stress.cost_bps == 5][
        ["borrow_annual", "sharpe_excess", "cagr", "max_drawdown"]
    ]
    notes = [
        "Signed positions are enabled from -2x to +2x equity at each rebalance. Market moves can cause exposure to drift between rebalances. Negative weights short USO; positive weights buy it.",
        "600 signed policies reuse the saved chronological forecasts from 50 completed model fits. The displayed winner is selected using 2017–2025 development outcomes. These are not independent test results. Reserved 2026 data has not been loaded.",
        "All comparisons charge 5 bps per traded dollar. Signed portfolios pay 3% annual short borrow and cash-proxy yield plus 2% on borrowed long capital. Short proceeds earn no collateral rebate. Cash return is subtracted when calculating excess Sharpe.",
        "A 25% maintenance-equity convention is checked at daily endpoints, with liquidation and a halt after a margin call; insolvency is recorded. Real intraday calls, historical borrow availability and actual broker rates remain unverified.",
        "The best signed strategy shorts on about 57% of days. Higher borrow costs materially reduce its result: excess Sharpe falls from 0.60 at 3% borrow to 0.46 at 10% and 0.09 at 30%, at the same 5 bps trading cost.",
        "The requested Sharpe above 1.5 has not been achieved. Leverage permission is implemented, but the research does not support claiming a successful high-Sharpe strategy.",
    ]
    title = "Shorting and leverage: enabled"
    md = (
        "# "
        + title
        + "\n\n"
        + markdown_table(table)
        + "\n\n"
        + "\n\n".join(notes)
        + "\n\n[Expanded search report](../report.html) · [Cost stress results](selected_cost_stress.csv) · [All signed candidates](ranking.csv)\n"
    )
    (folder / "report.md").write_text(md)
    page = (
        "<h1>"
        + title
        + "</h1>"
        + table.to_html(index=False, border=0)
        + "".join("<p>" + html.escape(s) + "</p>" for s in notes)
        + '<p><a href="../report.html">Expanded search report</a> · <a href="selected_cost_stress.csv">Cost stress results</a> · <a href="ranking.csv">All signed candidates</a></p>'
    )
    (folder / "report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'
        + title
        + "</title><style>body{max-width:960px;margin:48px auto;padding:24px;background:#fafbf9;color:#203044;font:17px/1.6 system-ui}table{width:100%;border-collapse:collapse}th,td{padding:12px;text-align:left;border-bottom:1px solid #ccd6d8}th{background:#e8efed}</style><body>"
        + page
        + "</body></html>"
    )


if __name__ == "__main__":
    main(Path.cwd())
