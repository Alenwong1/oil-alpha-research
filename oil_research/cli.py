from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .features import dataset, feature_dictionary
from .models import predict_period
from .backtest import allocations, simulate, metrics, paired_bootstrap


def run(config, root, period, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Output exists: choose a new --output path to preserve the previous experiment"
        )
    bars = data.load(config, root / "data/raw")
    x, meta = dataset(bars, config["target"])
    if len(x) == 0:
        raise ValueError("No complete feature rows")
    output.mkdir(parents=True, exist_ok=True)
    # Write the plan/source fingerprints BEFORE running predictions or viewing metrics.
    info = {
        "period": period,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "pandas", "scikit-learn", "matplotlib", "yfinance"]
        },
        "source_sha256": {
            str(p.relative_to(root)): data.sha256(p)
            for p in sorted((root / "oil_research").glob("*.py"))
        },
        "config_sha256": data.sha256(root / "config.json"),
        "data_manifest": json.loads((root / "data/raw/manifest.json").read_text()),
        "features": len(x.columns),
    }
    (output / "run.json").write_text(json.dumps(info, indent=2))
    (output / "config_snapshot.json").write_text(json.dumps(config, indent=2))
    feature_dictionary(x.columns).to_csv(output / "feature_dictionary.csv", index=False)
    predictions, audit = predict_period(x, meta, config, period)
    info.update(
        first_signal=str(predictions.index.min().date()),
        last_signal=str(predictions.index.max().date()),
    )
    predictions.join(meta).to_csv(output / "predictions.csv")
    audit.to_csv(output / "model_audit.csv", index=False)
    weights = allocations(predictions, meta, config)
    weights.to_csv(output / "allocations.csv")
    summary, annual, ledgers = [], [], {}
    (output / "ledgers").mkdir()
    for cost in config["cost_bps"]:
        for name in weights:
            ledger = simulate(meta.loc[weights.index, "target_return"], weights[name], cost)
            ledger = ledger.join(meta[["execution_date", "label_end"]])
            ledgers[(name, cost)] = ledger
            ledger.to_csv(output / "ledgers" / f"{name}_{cost}bps.csv")
            summary.append({"strategy": name, "cost_bps": cost, **metrics(ledger)})
            if cost == config["primary_cost_bps"]:
                for year, group in ledger.groupby(ledger.label_end.dt.year):
                    annual.append({"strategy": name, "year": year, **metrics(group)})
    summary, annual = pd.DataFrame(summary), pd.DataFrame(annual)
    summary.to_csv(output / "metrics.csv", index=False)
    annual.to_csv(output / "annual_metrics.csv", index=False)
    comparisons = []
    for a, b in [
        ("tree_full", "momentum"),
        ("ridge_full", "momentum"),
        ("tree_full", "tree_uso"),
        ("ridge_full", "ridge_uso"),
    ]:
        cost = config["primary_cost_bps"]
        comparisons.append(
            {
                "comparison": f"{a} - {b}",
                **paired_bootstrap(
                    ledgers[(a, cost)].net_return,
                    ledgers[(b, cost)].net_return,
                    seed=config["seed"],
                    repetitions=config["bootstrap_repetitions"],
                    block_length=config["bootstrap_block_length"],
                ),
            }
        )
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(output / "bootstrap.csv", index=False)
    forecast = []
    truth = meta.loc[predictions.index, "target_return"]
    for name in predictions:
        p = predictions[name]
        mse = float(np.mean((p - truth) ** 2))
        forecast.append(
            {
                "model": name,
                "mse": mse,
                "zero_forecast_mse": float(np.mean(truth**2)),
                "r2_vs_zero": 1 - mse / float(np.mean(truth**2)),
                "sign_accuracy": float(((p > 0) == (truth > 0)).mean()),
                "prediction_return_correlation": float(p.corr(truth)),
            }
        )
    forecast = pd.DataFrame(forecast)
    forecast.to_csv(output / "forecast_metrics.csv", index=False)
    from .report import generate

    generate(output, config, summary, annual, ledgers, comparisons, forecast, audit, info)
    info["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "run.json").write_text(json.dumps(info, indent=2))
    print(
        summary.loc[
            summary.cost_bps == config["primary_cost_bps"],
            ["strategy", "cagr", "sharpe_zero_rf", "max_drawdown"],
        ].to_string(index=False)
    )
    print(f"Report: {output / 'report.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Reproducible USO research")
    parser.add_argument("command", choices=["download", "run"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--period", choices=["development", "holdout"], default="development")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--refresh", action="store_true", help="Explicitly replace the local market-data snapshot"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / "config.json").read_text())
    if args.command == "download":
        data.download(config, root / "data/raw", args.refresh)
    else:
        output = args.output.resolve() if args.output else root / "results" / args.period
        run(config, root, args.period, output)


if __name__ == "__main__":
    main()
