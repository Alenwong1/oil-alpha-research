"""Long/cash daily allocation with drift-aware turnover and proportional fees."""

from __future__ import annotations

import numpy as np
import pandas as pd


def allocations(predictions, meta, config):
    m = meta.loc[predictions.index]
    scale = (config["annual_vol_target"] / m.annual_vol.clip(lower=0.01)).clip(
        upper=config["max_weight"]
    )
    weights = pd.DataFrame(index=predictions.index)
    weights["buy_hold"] = 1.0
    weights["vol_target_long"] = scale
    weights["momentum"] = (m.momentum > 0).astype(float) * scale
    # Hysteresis: enter below 30, exit above 70, otherwise retain the prior state.
    state = (
        meta.rsi.map(lambda value: 1.0 if value < 30 else (0.0 if value > 70 else np.nan))
        .ffill()
        .fillna(0)
    )
    weights["rsi"] = state.loc[predictions.index] * scale
    for name in predictions:
        weights[name] = (predictions[name] > 0).astype(float) * scale
    return weights


def simulate(
    returns: pd.Series, weights: pd.Series, cost_bps: float, liquidate=True
) -> pd.DataFrame:
    if not returns.index.equals(weights.index):
        raise ValueError("Returns and weights must align exactly")
    if not np.isfinite(returns).all() or not np.isfinite(weights).all():
        raise ValueError("Nonfinite returns or weights")
    if (returns <= -1).any() or ((weights < 0) | (weights > 1)).any():
        raise ValueError("Only positive-price assets and unlevered long/cash supported")
    if cost_bps < 0 or cost_bps >= 10000:
        raise ValueError("Cost must be in [0, 10000) bps")
    fee_rate = cost_bps / 10000
    previous_weight = 0.0
    records = []
    for date, r in returns.items():
        target = float(weights.loc[date])
        # Solve q = |w * (1 - fee_rate*q) - previous_weight|.
        # q is traded notional / pre-trade equity; target is weight AFTER fees.
        difference = target - previous_weight
        turnover = (
            difference / (1 + target * fee_rate)
            if difference >= 0
            else -difference / (1 - target * fee_rate)
        )
        entry_fee = fee_rate * turnover
        growth = (1 - entry_fee) * (1 + target * r)
        gross = target * r
        end_weight = target * (1 + r) / (1 + gross)
        records.append([date, r, target, turnover, entry_fee, gross, growth - 1, end_weight])
        previous_weight = end_weight
    out = pd.DataFrame(
        records,
        columns=[
            "signal_date",
            "asset_return",
            "weight",
            "turnover",
            "fee_fraction",
            "gross_return",
            "net_return",
            "end_weight",
        ],
    ).set_index("signal_date")
    if liquidate and len(out):
        # Terminal sale occurs at the final label-end open.
        terminal = float(out.end_weight.iloc[-1])
        out.iloc[-1, out.columns.get_loc("net_return")] = (1 + out.net_return.iloc[-1]) * (
            1 - fee_rate * terminal
        ) - 1
        out.iloc[-1, out.columns.get_loc("turnover")] += terminal
        out.iloc[-1, out.columns.get_loc("fee_fraction")] += fee_rate * terminal
    out["equity"] = (1 + out.net_return).cumprod()
    return out


def sharpe(values) -> float:
    a = np.asarray(values, dtype=float)
    std = a.std(ddof=1) if len(a) > 1 else 0
    return float(np.sqrt(252) * a.mean() / std) if std > 0 else 0.0


def metrics(ledger):
    r = ledger.net_return
    equity = (1 + r).cumprod()
    peaks = equity.cummax().clip(lower=1.0)
    return {
        "days": len(r),
        "total_return": float(equity.iloc[-1] - 1),
        "cagr": float(equity.iloc[-1] ** (252 / len(r)) - 1),
        "annual_vol": float(r.std(ddof=1) * np.sqrt(252)),
        "sharpe_zero_rf": sharpe(r),
        "max_drawdown": float((equity / peaks - 1).min()),
        "annual_turnover": float(ledger.turnover.sum() * 252 / len(r)),
        "average_exposure": float(ledger.weight.mean()),
    }


def paired_bootstrap(a, b, seed=42, repetitions=1000, block_length=20):
    """Circular moving-block interval, descriptive and not search-adjusted."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b) or len(a) < block_length:
        raise ValueError("Aligned samples at least one block long are required")
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(repetitions):
        starts = rng.integers(0, len(a), size=int(np.ceil(len(a) / block_length)))
        indices = ((starts[:, None] + np.arange(block_length)) % len(a)).ravel()[: len(a)]
        diffs.append(sharpe(a[indices]) - sharpe(b[indices]))
    lower, upper = np.quantile(diffs, [0.025, 0.975])
    return {"difference": sharpe(a) - sharpe(b), "lower_95": float(lower), "upper_95": float(upper)}
