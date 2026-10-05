"""Publish an explicit allowlist of aggregate results; never export raw/daily data."""

from pathlib import Path
import hashlib
import json
import os
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROUNDS = ["refinement", "diversification", "scarcity", "time_series", "adaptive_blend"]
FILES = [
    "ranking.csv",
    "metrics.csv",
    "annual.csv",
    "yearly_comparison.csv",
    "forecast_accuracy.csv",
    "coverage.csv",
    "bootstrap.csv",
    "selected_cost_stress.csv",
    "audit.json",
    "plan.json",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(frame):
    columns = list(frame.columns)
    return "\n".join(
        [
            "| " + " | ".join(map(str, columns)) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        + [
            "| " + " | ".join(map(str, row)) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
    )


def main():
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    manifest = {}
    for name in ROUNDS:
        destination = out / name
        destination.mkdir(exist_ok=True)
        for filename in FILES:
            source = ROOT / "results" / name / filename
            if not source.exists():
                continue
            # Refuse to publish accidental local paths, credentials or raw payloads.
            text = source.read_text()
            if "/Users/" in text or "BEGIN PRIVATE KEY" in text:
                raise ValueError(f"Unexpected private content in {source.name}")
            target = destination / filename
            shutil.copyfile(source, target)
            manifest[str(target.relative_to(out))] = {
                "source": str(source.relative_to(ROOT)),
                "sha256": digest(target),
            }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "README.md").write_text(
        "# Published research artifacts\n\nThese are aggregate exports of preserved local development experiments. `manifest.json` records each source path and SHA-256 checksum. No raw market history, headline text, daily forecasts or daily ledgers are included.\n\nStart with [the results summary](../docs/RESULTS.md). The final mix is in `adaptive_blend`; `time_series` includes all six model configurations, including failures. Older rounds document how the research evolved. Metrics with `sharpe_excess` subtract the cash proxy; `sharpe_zero_rf` use a zero benchmark but retain cash income and costs.\n\nOriginal experiment plan hashes refer to source snapshots at run time. The maintained source was subsequently formatted for publication. Public aggregates verify the reported tables, but exact forecast reproduction requires the original local input snapshots; see [reproduction limits](../docs/REPRODUCING.md).\n"
    )
    r = pd.read_csv(out / "adaptive_blend/ranking.csv")
    best = r.iloc[0]
    annual = pd.read_csv(out / "adaptive_blend/annual.csv")
    annual = annual[annual.adaptive_fraction == best.adaptive_fraction].copy()
    display = annual[["year", "sharpe_zero_rf", "sharpe_excess", "cagr", "max_drawdown"]].copy()
    display.columns = [
        "Year",
        "Sharpe (zero RF)",
        "Cash-excess Sharpe",
        "Annualized return",
        "Max drawdown",
    ]
    for col in ["Sharpe (zero RF)", "Cash-excess Sharpe"]:
        display[col] = display[col].map(lambda v: f"{v:.2f}")
    for col in ["Annualized return", "Max drawdown"]:
        display[col] = display[col].map(lambda v: f"{v:.1%}")
    accuracy = pd.read_csv(out / "time_series/forecast_accuracy.csv")
    ranking = pd.read_csv(out / "time_series/ranking.csv")[
        ["candidate", "sharpe_zero_rf", "sharpe_excess", "cagr", "max_drawdown"]
    ].round(3)
    mix = pd.read_csv(out / "adaptive_blend/metrics.csv")[
        ["period", "existing_fraction", "adaptive_fraction", "sharpe_zero_rf", "sharpe_excess"]
    ].round(3)
    summary = {
        "period": "2017–2025",
        "strategy": "50/50 existing and adaptive portfolio mix",
        "adaptive_fraction": float(best.adaptive_fraction),
        "sharpe_zero_rf": float(best.sharpe_zero_rf),
        "sharpe_excess": float(best.sharpe_excess),
        "cagr": float(best.cagr),
        "max_drawdown": float(best.max_drawdown),
        "selection": "Full-period development cash-excess Sharpe; five fixed allocations including endpoints",
        "independent_validation": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "docs/RESULTS.md").write_text(
        "# Research results\n\nThe selected 50/50 mix achieved **1.731 zero-risk-free Sharpe**, **1.330 cash-excess Sharpe**, **11.0% CAGR** and **8.4% maximum drawdown** over all of **2017–2025**, after modeled transaction, borrow and financing costs. All these dates are development data; no independent performance claim is made.\n\n## Annual results for the full-period selected mix\n\n"
        + table(display)
        + "\n\nAnnual returns use the project's 252-session annualization convention. Calendar groups use signal dates. The same portfolio weights are used in every year.\n\n## All six time-series configurations\n\n"
        + table(ranking)
        + "\n\nThese are portfolios with shared macro/news and technical components. Existing ridge uses seven features; boosting and adaptive ridge use 44. More complex features and models did not automatically improve performance.\n\n## Forecast accuracy on common dates\n\n"
        + table(accuracy.round(4))
        + "\n\nEvery candidate has worse normalized-return RMSE than a zero forecast. Overlapping targets and repeated weekly features make daily counts dependent.\n\n## All fixed portfolio mixes and requested subset\n\n"
        + table(mix)
        + "\n\nThe 2018–2024 subset was requested after viewing full-period results; its 75/25 winner is a retrospective sensitivity, not the primary selection. The full-period winner is 50/50. A zero benchmark is not a transaction-cost-only simulation.\n\n## Uncertainty and research history\n\nThe full-period 50/50 mix's descriptive paired 95% Sharpe-change interval versus the existing portfolio is approximately **[−0.201, 0.472]** and includes zero. It uses 2,000 paired circular 21-session block resamples and does not correct repeated model selection. Scarcity/demand additions failed to improve the control; boosted trees also underperformed. The technical component was active only in 2020–2021.\n\nPreserved aggregate tables, original plan metadata and checksums are in [reports](../reports/README.md). See [methodology](RESEARCH_DESIGN.md) and [reproduction limits](REPRODUCING.md) before interpreting these results. 2026 returns remain unevaluated.\n"
    )
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    colors = ["#187f80" if v >= 0 else "#b34c43" for v in annual.cagr]
    axes[0].bar(annual.year.astype(str), annual.cagr * 100, color=colors)
    axes[0].axhline(0, color="#64748b", linewidth=0.7)
    axes[0].set(
        title="Selected mix: annualized net returns", ylabel="Return (%)", xlabel="Signal year"
    )
    selected = r.set_index("adaptive_fraction").loc[[0.0, 1.0, 0.5]]
    x = list(range(3))
    axes[1].bar(
        [v - 0.18 for v in x],
        selected.sharpe_zero_rf,
        width=0.36,
        label="Zero-RF Sharpe",
        color="#187f80",
    )
    axes[1].bar(
        [v + 0.18 for v in x],
        selected.sharpe_excess,
        width=0.36,
        label="Cash-excess Sharpe",
        color="#64748b",
    )
    axes[1].set_xticks(x, ["Existing", "Adaptive", "50/50 mix"])
    axes[1].set(title="Full 2017–2025 development period", ylabel="Annualized Sharpe")
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle("Oil Alpha Research | After modeled costs | Development results", fontsize=13)
    fig.savefig(out / "research_summary.png", dpi=160)
    plt.close(fig)
    print(f"Exported {len(manifest)} allowlisted artifacts. No daily data exported.")


if __name__ == "__main__":
    main()
