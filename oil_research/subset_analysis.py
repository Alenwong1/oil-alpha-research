"""Retrospective 2018–2024 sensitivity of frozen forecasts and allocations."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from .backtest import sharpe
from .extension_experiments import policy_weights, simulate_cash
from .extension_leverage import signed_weights, simulate_signed


def main(root):
    out = root / "results/subset_2018_2024"
    out.mkdir(exist_ok=True)
    config = json.loads((root / "extension_config.json").read_text())
    records = []
    cache = {}

    def record(name, family, l, original=None):
        if original is not None:
            assert abs(sharpe(l.excess_return) - original) < 1e-8, (
                name,
                original,
                sharpe(l.excess_return),
            )
        s = l.loc[~l.index.year.isin([2017, 2025])]
        assert s.index.year.min() == 2018 and s.index.year.max() == 2024
        row = {
            "candidate": name,
            "family": family,
            "subset_sharpe_excess": sharpe(s.excess_return),
            "subset_sharpe_zero_rf": sharpe(s.net_return),
            "full_period_sharpe_excess": sharpe(l.excess_return),
            "sessions": len(s),
            "first_signal": str(s.index.min().date()),
            "last_signal": str(s.index.max().date()),
        }
        records.append(row)

    ranking = pd.read_csv(root / "results/extension/all_candidates.csv")
    for _, r in ranking.iterrows():
        signed = r["round"] == "short_leverage"
        source = r.source_round if signed else r["round"]
        folder = root / "results/extension" / source
        key = (source, r.candidate)
        if key not in cache:
            p = pd.read_csv(
                folder / f"{r.candidate}_predictions.csv", index_col=0, parse_dates=True
            ).score
            m = pd.read_csv(
                folder / "model_input_outcomes.csv",
                index_col=0,
                parse_dates=[0, "execution_date", "label_end"],
            )
            cache[key] = (p, m)
        p, m = cache[key]
        if signed:
            w = signed_weights(p, m, r.policy, int(r.smoothing), float(r.vol_target))
            dt = (m.loc[p.index, "label_end"] - m.loc[p.index, "execution_date"]).dt.days / 365
            l = simulate_signed(
                m.loc[p.index, "target_return"], m.loc[p.index, "cash_return"], dt, w
            )
        else:
            w = policy_weights(p, m, r.policy, int(r.smoothing), config)
            l = simulate_cash(m.loc[p.index, "target_return"], m.loc[p.index, "cash_return"], w, 5)
        name = f"{source}/{r.candidate}/{r.policy}/smooth{int(r.smoothing)}" + (
            f"/vol{r.vol_target}" if signed else ""
        )
        record(name, "signed" if signed else "long_cash", l, r.sharpe_excess)
    weekly = pd.read_csv(root / "results/weekly/ranking.csv")
    for _, r in weekly.iterrows():
        l = pd.read_csv(
            root / "results/weekly" / f"{r.candidate}_ledger.csv", index_col=0, parse_dates=True
        )
        record(r.candidate, "weekly", l, r.sharpe_excess)
    l = pd.read_csv(
        root / "results/weekly/blend_followup/selected_development_ledger.csv",
        index_col=0,
        parse_dates=True,
    )
    record("fixed_50_50_blend", "blend", l)
    results = pd.DataFrame(records).sort_values("subset_sharpe_excess", ascending=False)
    results.to_csv(out / "ranking.csv", index=False)
    chosen = results.iloc[0].to_dict()
    blend = results[results.family == "blend"].iloc[0].to_dict()
    result = {
        "best": chosen,
        "current_blend": blend,
        "candidates": len(results),
        "method": "Exclude signal years 2017 and 2025 from original net daily returns. No refitting or changed allocations; retain original cost/financing assumptions. Positions carry into the subset; no artificial boundary entry/exit fees are added. Sharpe is annualized sqrt(252) times daily cash-excess mean/sample standard deviation. This is retrospective window and candidate selection, not independent validation.",
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    (out / "report.md").write_text(
        "# 2018–2024 retrospective sensitivity\n\n"
        + result["method"]
        + "\n\nBest across "
        + str(len(results))
        + " saved candidates: **"
        + f'{chosen["subset_sharpe_excess"]:.3f}'
        + "** excess Sharpe.\n\nCurrent fixed 50/50 blend: **"
        + f'{blend["subset_sharpe_excess"]:.3f}'
        + "**, compared with "
        + f'{blend["full_period_sharpe_excess"]:.3f}'
        + " for 2017–2025.\n\nWinner: `"
        + chosen["candidate"]
        + "`.\n\nSee [all subset results](ranking.csv).\n"
    )
    print(json.dumps(result, indent=2))
    print(results.head(6).to_string(index=False))


if __name__ == "__main__":
    main(Path.cwd())
