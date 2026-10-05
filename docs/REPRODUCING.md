# Reproducing the research

## 1. Code and aggregate checks without external data

From the repository root, install the package in a virtual environment:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m black --check oil_research tests scripts
python scripts/verify_repository.py
```

The tests use synthetic fixtures to exercise causality, purging, availability and accounting. Aggregate checks compare the committed summary with underlying published tables and verify manifest hashes. They do not independently reproduce the original forecasts.

`requirements-lock.txt` records the original research environment on Python 3.13/macOS ARM, not a cross-platform lock. Use `pyproject.toml` for a compatible fresh environment. The public CI workflow tests the package on Python 3.11 and 3.13; its status is determined by actual GitHub runs.

## 2. Original baseline on fresh public data

```bash
python -m oil_research.cli download
python -m oil_research.cli run --period development
```

For another run without overwriting earlier output:

```bash
python -m oil_research.cli run --period development --output results/development-reproduction
```

`download --refresh` explicitly replaces a cache. Do not use it if preserving a historical input snapshot. The baseline's `holdout` command refers to the original 2024–2025 experiment; those dates have since been inspected and are no longer an independent holdout.

## 3. Expanded research dependency chain

The final blend is the end of several preserved experiments, not a single standalone estimator. Run from the repository root. Public collectors need network access and may encounter archive gaps/rate limits.

```bash
python -m oil_research.extension_sources curve
python -m oil_research.extension_sources eia
python -m oil_research.extension_sources markets
python -m oil_research.extension_sources news
python -m oil_research.extension_features
python -m oil_research.extension_experiments ablation
python -m oil_research.extension_experiments horizons --tag horizons_repaired
python -m oil_research.extension_experiments innovations
python -m oil_research.extension_experiments news_refresh
python -m oil_research.extension_leverage
python -m oil_research.extension_report
python -m oil_research.weekly_sources eia
python -m oil_research.weekly_sources cftc
python -m oil_research.weekly_errata
python -m oil_research.weekly_experiments
python -m oil_research.weekly_blend
python -m oil_research.refinement
python -m oil_research.diversification
python -m oil_research.scarcity
python -m oil_research.time_series
python -m oil_research.adaptive_blend
```

The tag `horizons_repaired` is an existing downstream dependency name from the historical experiment. Some pipelines intentionally assert that controls reproduce prior results; they may fail with changed provider history or a different selected candidate. Do not weaken these assertions or relabel a new result as the historical run.

## 4. Historical reproduction boundaries

Exact historical reproduction requires the original local data and intermediate prediction snapshots. They are excluded from Git for size and data-use reasons. Earlier macro/news experiments used a smaller headline archive than later experiments. A fresh collection cannot guarantee the same archive membership, classifications or adjusted market prices. Re-running every command against today's sources is therefore a **new experiment**, not proof of exact reproduction of the published 1.73 Sharpe.

Immutable output guards preserve previous runs. For a new development run, use a separate clean checkout/cache or supported output tags rather than deleting the original evidence. The complete chain above is documented from module dependencies; it has not been re-downloaded end-to-end as part of publication. Tests, existing-data controls and aggregate export checks were run locally.

## 5. Exporting aggregate results

With completed local experiments present:

```bash
python scripts/export_results.py
python scripts/verify_repository.py
```

The exporter writes only allowlisted aggregate files under `reports/`, plus a manifest and aggregate figure. Do not commit `data/` or `results/` to make a missing snapshot check pass.
