"""Signed ETF positions, financing, borrow costs and explicit leverage stress tests."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .data import sha256
from .extension_experiments import score


def simulate_signed(
    asset,
    cash,
    day_fraction,
    weights,
    cost_bps=5,
    borrow_annual=0.03,
    financing_spread=0.02,
    max_leverage=2,
):
    if not all(asset.index.equals(v.index) for v in [cash, day_fraction, weights]):
        raise ValueError("All signed-ledger inputs must align")
    arrays = np.c_[asset, cash, day_fraction, weights]
    if not np.isfinite(arrays).all() or (np.abs(arrays[:, 3]) > max_leverage + 1e-10).any():
        raise ValueError("Invalid values or leverage cap exceeded")
    if min(cost_bps, borrow_annual, financing_spread) < 0:
        raise ValueError("Costs must be nonnegative")
    c = cost_bps / 10000
    if c * max_leverage >= 1:
        raise ValueError("Fee/leverage combination makes trading infeasible")
    old = 0.0
    halted = False
    records = []
    for r, rf, dt, desired in arrays:
        w = 0.0 if halted else desired
        diff = w - old
        q = diff / (1 + c * w) if diff >= 0 else -diff / (1 - c * w)
        financing = max(w - 1, 0) * (rf + financing_spread * dt)
        borrow = max(-w, 0) * borrow_annual * dt
        # Short sale collateral receives no rebate; own cash may earn the cash proxy.
        cash_income = max(1 - max(w, 0), 0) * rf
        gross = w * r + cash_income - financing - borrow
        margin_call = 0
        insolvent = 0
        if 1 + gross <= 0:
            net = -1.0
            end_weight = 0.0
            halted = True
            insolvent = 1
        else:
            net = (1 - c * q) * (1 + gross) - 1
            end_weight = w * (1 + r) / (1 + gross)
            # 25% maintenance-equity convention, checked only at daily endpoints.
            if abs(end_weight) > 4:
                net = (1 + net) * (1 - c * abs(end_weight)) - 1
                q += abs(end_weight)
                end_weight = 0.0
                halted = True
                margin_call = 1
                if net <= -1:
                    net = -1.0
                    insolvent = 1
        records.append([r, rf, w, q, financing, borrow, net, end_weight, margin_call, insolvent])
        old = end_weight
    out = pd.DataFrame(
        records,
        index=asset.index,
        columns=[
            "asset_return",
            "cash_return",
            "weight",
            "turnover",
            "financing_fraction",
            "borrow_fraction",
            "net_return",
            "end_weight",
            "margin_call",
            "insolvent",
        ],
    )
    out.index.name = "signal_date"
    if len(out):
        exit_notional = abs(out.end_weight.iloc[-1])
        out.iloc[-1, out.columns.get_loc("net_return")] = (1 + out.net_return.iloc[-1]) * (
            1 - c * exit_notional
        ) - 1
        out.iloc[-1, out.columns.get_loc("turnover")] += exit_notional
    out["excess_return"] = out.net_return - out.cash_return
    out.loc[out.excess_return.abs() < 1e-14, "excess_return"] = 0.0
    out["equity"] = (1 + out.net_return).cumprod()
    return out


def signed_weights(p, meta, policy, smoothing, vol_target):
    p = p.ewm(span=smoothing, adjust=False).mean() if smoothing > 1 else p
    if policy == "sign":
        direction = np.sign(p)
    elif policy == "conviction":
        direction = (p / 0.5).clip(-1, 1)
    elif policy == "hysteresis":
        direction = (
            p.map(lambda v: 1.0 if v > 0.1 else (-1.0 if v < -0.1 else np.nan)).ffill().fillna(0)
        )
    else:
        raise ValueError(policy)
    risk = (vol_target / meta.loc[p.index, "annual_vol"].clip(lower=0.01)).clip(upper=2)
    return direction * risk


def run(root):
    out = root / "results/extension/short_leverage"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.csv").exists():
        raise ValueError(
            "Signed experiment already exists; preserve prior output before a new search"
        )
    sources = []
    for folder in (root / "results/extension").iterdir():
        if (
            folder.is_dir()
            and (folder / "ranking.csv").exists()
            and (folder / "model_input_outcomes.csv").exists()
        ):
            for path in folder.glob("*_predictions.csv"):
                sources.append((folder, path))
    if not sources:
        raise ValueError("No completed model rounds")
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "primary_cost_bps": 5,
        "borrow_annual": 0.03,
        "financing_spread": 0.02,
        "short_collateral_rebate": 0,
        "max_gross_exposure": 2,
        "vol_targets": [0.15, 0.30],
        "policies": ["sign", "conviction", "hysteresis"],
        "smoothing": [1, 5],
        "source_predictions": {str(p.relative_to(root)): sha256(p) for _, p in sources},
        "code_sha256": sha256(Path(__file__)),
        "maintenance_equity_fraction": 0.25,
        "warning": "Daily endpoint margin checks only. Actual intraday calls, historical locate availability and borrow rates are not verified. Development search, not independent validation.",
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    records = []
    best = None
    best_row = None
    best_inputs = None
    for i, (folder, path) in enumerate(sources, 1):
        p = pd.read_csv(path, index_col=0, parse_dates=True).score
        meta = pd.read_csv(
            folder / "model_input_outcomes.csv",
            index_col=0,
            parse_dates=[0, "execution_date", "label_end"],
        )
        dt = (meta.loc[p.index, "label_end"] - meta.loc[p.index, "execution_date"]).dt.days / 365
        name = path.name.replace("_predictions.csv", "")
        for vol in plan["vol_targets"]:
            for policy in plan["policies"]:
                for smoothing in plan["smoothing"]:
                    w = signed_weights(p, meta, policy, smoothing, vol)
                    ledger = simulate_signed(
                        meta.loc[p.index, "target_return"], meta.loc[p.index, "cash_return"], dt, w
                    )
                    row = {
                        "source_round": folder.name,
                        "candidate": name,
                        "policy": policy,
                        "smoothing": smoothing,
                        "vol_target": vol,
                        "gross_cap": 2,
                        "cost_bps": 5,
                        "borrow_annual": 0.03,
                        **score(ledger),
                        "average_gross_exposure": float(ledger.weight.abs().mean()),
                        "maximum_gross_exposure": float(ledger.weight.abs().max()),
                        "short_day_fraction": float((ledger.weight < 0).mean()),
                        "margin_calls": int(ledger.margin_call.sum()),
                        "insolvent": int(ledger.insolvent.sum()),
                    }
                    records.append(row)
                    if best_row is None or row["sharpe_excess"] > best_row["sharpe_excess"]:
                        best = ledger
                        best_row = row
                        best_inputs = (
                            meta.loc[p.index, "target_return"],
                            meta.loc[p.index, "cash_return"],
                            dt,
                            w,
                        )
        pd.DataFrame(records).to_csv(out / "metrics.csv", index=False)
        if i % 5 == 0:
            print(
                f'Short/leverage: {i}/{len(sources)} models, {len(records)} policies; best development excess Sharpe {best_row["sharpe_excess"]:.3f}',
                flush=True,
            )
    ranking = pd.DataFrame(records).sort_values("sharpe_excess", ascending=False)
    ranking.to_csv(out / "ranking.csv", index=False)
    best.to_csv(out / "selected_development_ledger.csv")
    stress = []
    for cost in [0, 5, 10, 20]:
        for borrow in [0.03, 0.10, 0.30]:
            l = simulate_signed(*best_inputs, cost_bps=cost, borrow_annual=borrow)
            stress.append({"cost_bps": cost, "borrow_annual": borrow, **score(l)})
    pd.DataFrame(stress).to_csv(out / "selected_cost_stress.csv", index=False)
    annual = []
    for year, g in best.groupby(best.index.year):
        g = g.copy()
        g["equity"] = (1 + g.net_return).cumprod()
        annual.append({"year": year, **score(g)})
    pd.DataFrame(annual).to_csv(out / "selected_annual.csv", index=False)
    (out / "selected.json").write_text(json.dumps(best_row, indent=2))
    print(ranking.head(8).to_string(index=False), flush=True)


if __name__ == "__main__":
    run(Path.cwd())
