"""Offline integrity checks for the public aggregate package."""

from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "reports"
    manifest = json.loads((out / "manifest.json").read_text())
    forbidden = (
        "ledger",
        "predictions",
        "headlines",
        "model_input",
        "release_inputs",
        "source_snapshots",
    )
    for name, record in manifest.items():
        assert not any(word in name for word in forbidden), name
        path = out / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], name
        assert "/Users/" not in path.read_text(), name
    summary = json.loads((out / "summary.json").read_text())
    r = pd.read_csv(out / "adaptive_blend/ranking.csv").iloc[0]
    for key in ["sharpe_zero_rf", "sharpe_excess", "cagr", "max_drawdown", "adaptive_fraction"]:
        np.testing.assert_allclose(summary[key], r[key], rtol=0, atol=1e-12)
    assert summary["independent_validation"] is False
    assert len(pd.read_csv(out / "time_series/ranking.csv")) == 6
    annual = pd.read_csv(out / "adaptive_blend/annual.csv")
    assert set(annual.year) == set(range(2017, 2026))
    maintained = [
        "README.md",
        "CONTRIBUTING.md",
        "docs/RESULTS.md",
        "docs/DATA.md",
        "docs/REPRODUCING.md",
        "docs/RESEARCH_DESIGN.md",
        "reports/README.md",
    ]
    for name in maintained:
        p = ROOT / name
        for target in re.findall(r"\]\(([^)]+)\)", p.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            assert (p.parent / target.split("#")[0]).exists(), (name, target)
    print(
        f"PASS: {len(manifest)} aggregate hashes, headline metrics, date coverage and maintained documentation links."
    )


if __name__ == "__main__":
    main()
