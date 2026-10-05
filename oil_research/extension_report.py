"""Aggregate all attempted candidates; explicitly account for research selection."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .extension_features import build
from .extension_experiments import policy_weights, simulate_cash
from .report import markdown_table
from .extension_leverage import signed_weights, simulate_signed


def main(root):
    x, meta, groups, lineage, status = build(root)
    config = json.loads((root / "extension_config.json").read_text())
    output = root / "results/extension"
    tables = []
    for folder in output.iterdir():
        if folder.is_dir() and (folder / "ranking.csv").exists():
            table = pd.read_csv(folder / "ranking.csv")
            table["round"] = folder.name
            tables.append(table)
    if not tables:
        raise ValueError("No complete rounds")
    ranking = pd.concat(tables).sort_values("sharpe_excess", ascending=False).reset_index(drop=True)
    ranking.to_csv(output / "all_candidates.csv", index=False)
    chosen = ranking.iloc[0]
    chosen_source = (
        chosen["source_round"] if chosen["round"] == "short_leverage" else chosen["round"]
    )
    selected_plan = json.loads((output / chosen_source / "plan.json").read_text())
    status = selected_plan["source_status"]
    selected = pd.read_csv(
        output / chosen["round"] / "selected_development_ledger.csv", index_col=0, parse_dates=True
    )
    # Rebuild ALL policy outcomes, not only winners, for a search-aware null diagnostic.
    returns = []
    cached = {}
    for _, row in ranking.iterrows():
        signed = row["round"] == "short_leverage"
        source = row["source_round"] if signed else row["round"]
        key = (source, row.candidate)
        if key not in cached:
            p = pd.read_csv(
                output / source / f"{row.candidate}_predictions.csv", index_col=0, parse_dates=True
            ).score
            m = pd.read_csv(
                output / source / "model_input_outcomes.csv",
                index_col=0,
                parse_dates=[0, "execution_date", "label_end"],
            )
            cached[key] = (p, m)
        p, m = cached[key]
        if signed:
            w = signed_weights(p, m, row.policy, int(row.smoothing), float(row.vol_target))
            dt = (m.loc[p.index, "label_end"] - m.loc[p.index, "execution_date"]).dt.days / 365
            ledger = simulate_signed(
                m.loc[p.index, "target_return"], m.loc[p.index, "cash_return"], dt, w
            )
        else:
            w = policy_weights(p, m, row.policy, int(row.smoothing), config)
            ledger = simulate_cash(
                m.loc[p.index, "target_return"], m.loc[p.index, "cash_return"], w, 5
            )
        returns.append(ledger.excess_return.to_numpy())
    matrix = np.column_stack(returns)
    rng = np.random.default_rng(config["seed"])
    centered = matrix - matrix.mean(axis=0)
    maxima = []
    # This approximates a family-wise, centered block-bootstrap diagnostic, not proof.
    for _ in range(500):
        starts = rng.integers(0, len(matrix), size=int(np.ceil(len(matrix) / 20)))
        indices = ((starts[:, None] + np.arange(20)) % len(matrix)).ravel()[: len(matrix)]
        sample = centered[indices]
        sigma = sample.std(axis=0, ddof=1)
        stat = np.divide(
            np.sqrt(252) * sample.mean(axis=0), sigma, out=np.zeros_like(sigma), where=sigma > 1e-12
        )
        maxima.append(float(stat.max()))
    search = {
        "policy_candidates": len(ranking),
        "bootstrap_draws": 500,
        "block_length": 20,
        "best_development_sharpe_excess": float(chosen.sharpe_excess),
        "max_statistic_null_pvalue": float(
            (1 + np.sum(np.array(maxima) >= chosen.sharpe_excess)) / 501
        ),
        "null_max_sharpe_95pct": float(np.quantile(maxima, 0.95)),
        "warning": "Exploratory centered block-bootstrap across searched candidates; stationarity assumptions and data/research choices are not fully accounted for.",
    }
    (output / "search_uncertainty.json").write_text(json.dumps(search, indent=2))
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mpl-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "figure.facecolor": "#fafbf9",
            "axes.facecolor": "#fafbf9",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ablation = pd.read_csv(output / "ablation/ranking.csv")
    chart = ablation.groupby(["group", "model"]).sharpe_excess.max().unstack()
    chart.plot.bar(ax=axes[0], color=["#238883", "#163c63"])
    axes[0].set(
        title="Best searched policy in each feature group",
        ylabel="Development excess Sharpe",
        xlabel="Feature group",
    )
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].axhline(1.5, color="#bd4a43", linestyle="--", label="Target 1.5")
    axes[1].hist(ranking.sharpe_excess, bins=20, color="#238883", alpha=0.8)
    axes[1].axvline(1.5, color="#bd4a43", linestyle="--")
    axes[1].set(
        title=f"All {len(ranking)} searched policies at 5 bps",
        xlabel="Development excess Sharpe",
        ylabel="Candidates",
    )
    fig.tight_layout()
    fig.savefig(output / "search_results.png", dpi=170, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    axes[0].plot(
        selected.index, selected.equity, color="#238883", label="Selected development candidate"
    )
    for name, color in [
        ("momentum", "#735da5"),
        ("vol_target", "#d79528"),
        ("buy_hold", "#8d969e"),
    ]:
        b = pd.read_csv(
            output / chosen_source / f"baseline_{name}.csv", index_col=0, parse_dates=True
        )
        axes[0].plot(b.index, b.equity, label=name, color=color, alpha=0.8)
    axes[0].set(
        title="Selection-contaminated development performance | 5 bps", ylabel="Growth of $1"
    )
    axes[0].legend(frameon=False)
    axes[1].plot(
        selected.index,
        100 * (selected.equity / selected.equity.cummax().clip(lower=1) - 1),
        color="#238883",
    )
    axes[1].set(ylabel="Selected drawdown (%)", xlabel="Signal date")
    fig.tight_layout()
    fig.savefig(output / "selected_equity.png", dpi=170, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    table = ranking.head(12)[
        [
            "round",
            "candidate",
            "policy",
            "smoothing",
            "sharpe_excess",
            "sharpe_zero_rf",
            "cagr",
            "max_drawdown",
            "positive_year_fraction",
        ]
    ].copy()
    for c in ["sharpe_excess", "sharpe_zero_rf"]:
        table[c] = table[c].map(lambda v: f"{v:.2f}")
    for c in ["cagr", "max_drawdown", "positive_year_fraction"]:
        table[c] = table[c].map(lambda v: f"{v:.1%}")
    reached = float(chosen.sharpe_excess) > 1.5
    intro = f"Evaluated {len(ranking)} strategy policies across {sum(len(json.loads((output/r/'plan.json').read_text())['specifications']) for r in ranking['round'].unique() if r!='short_leverage')} model specifications using {selected_plan['feature_count']} features. The best selected development excess Sharpe is {chosen.sharpe_excess:.2f} at 5 bps per traded dollar. "
    intro += (
        "The 1.5 research threshold was exceeded in development, but has not been independently validated."
        if reached
        else "The requested 1.5 Sharpe target has not been reached."
    )
    reasons = [
        "All displayed scores use 2017–2025 development data. Chronological fitting prevents direct training on future labels, but selecting the best of many policies still overfits this development record.",
        "The 2026 reserved period remains unopened. It has not been used to select the model or to claim success.",
        "ALFRED snapshots are truly historical requested vintages, with per-series header validation, but update monthly rather than daily. This trades freshness for auditable free historical availability.",
        "EIA production inputs are archived EIA estimates of OPEC output, not a separately sourced OPEC Secretariat feed. Membership changes can affect the aggregate. Workbooks with correction notices were excluded.",
        "OilPrice headlines are a sparse archive sample, timestamped by capture plus 24 hours. Coverage indicators and missing values prevent absent archives from being treated as zero news. This is not a complete daily news feed.",
        "ETF price inputs remain retrospectively adjusted vendor history. Published-source timestamps and tests greatly reduce look-ahead risk but do not prove an institutional-quality point-in-time market database.",
        "Cash yield uses a lagged 3-month Treasury yield proxy and a simple calendar-day accrual convention. Excess Sharpe subtracts that proxy so cash interest is not counted as alpha.",
        "Hundreds of strategy trials can generate lucky results. The centered block-bootstrap diagnostic is exploratory and cannot fully correct for all researcher choices.",
    ]
    reasons.append(
        "Signed strategies permit long and short USO positions, with a 2x gross target cap. They assume 3% annual short borrow, borrowing at the cash proxy plus 2%, no short collateral rebate, and 5 bps per traded dollar. Borrow availability and intraday margin calls are not verified; the 25% maintenance convention is checked only at daily endpoints. The original long/cash experiments remain separately identified."
    )
    notes = f"Across the searched family, a centered 20-day-block bootstrap gives an approximate maximum-statistic p-value of {search['max_statistic_null_pvalue']:.3f}; its 95th-percentile null maximum Sharpe is {search['null_max_sharpe_95pct']:.2f}. This is a diagnostic under strong stationarity assumptions, not a formal validation of a chosen strategy."
    source_table = pd.DataFrame(
        [
            {
                "source": s["source"],
                "records": s.get("events", "not available"),
                "timing": s.get("timing", s.get("status", "")),
            }
            for s in status
        ]
    )
    md = (
        "# Expanded oil research results\n\n"
        + intro
        + "\n\n## Data included\n\n"
        + markdown_table(source_table)
        + "\n\n## All attempts and selected results\n\n"
        + markdown_table(table)
        + "\n\n"
        + notes
        + "\n\n![Search results](search_results.png)\n\n![Selected development performance](selected_equity.png)\n\n## Interpretation\n\n"
        + "\n".join("- " + s for s in reasons)
        + "\n\nSee `all_candidates.csv`, each round's `plan.json`, `registry.jsonl`, training audits and feature-availability ledger. The original experiment is preserved.\n"
    )
    (output / "report.md").write_text(md)
    page = (
        f"<h1>Expanded oil research</h1><p class='lead'>{html.escape(intro)}</p><h2>Data included</h2>{source_table.to_html(index=False,border=0)}<h2>Search results</h2><img src='search_results.png'><p>{html.escape(notes)}</p>{table.to_html(index=False,border=0)}<img src='selected_equity.png'><h2>Interpretation</h2><ul>"
        + "".join("<li>" + html.escape(s) + "</li>" for s in reasons)
        + "</ul>"
    )
    style = "body{background:#fafbf9;color:#203044;font:16px/1.6 system-ui;margin:40px auto;max-width:1150px;padding:24px}h1{font-size:38px}.lead{font-size:21px}table{font-size:12px;width:100%;border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #ccd6d8;text-align:left}th{background:#e8efed}img{width:100%;margin:30px 0}li{margin:12px 0}"
    (output / "report.html").write_text(
        f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Expanded oil research</title><style>{style}</style><body>{page}</body></html>'
    )
    print(intro)
    print(notes)


if __name__ == "__main__":
    main(Path.cwd())
