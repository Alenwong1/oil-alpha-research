# Contributing

Use Python 3.11+ and install `pip install -e '.[dev]'`. Run `pytest -q`, `python -m black oil_research tests scripts` and `python scripts/verify_repository.py` before proposing a change.

For a research extension, record the hypothesis, candidate count, dates, feature availability, costs and selection rule before scoring. Preserve failed experiments and control comparisons. Use chronological, horizon-purged training and synthetic causality/accounting tests. Do not overwrite completed experiments, replace unavailable historical inputs with future vintages, or tune on reserved dates.

Document whether a change alters economic logic, metrics or only presentation. Aggregate published metrics must identify their period, benchmark and costs. Keep provider data, headline text, credentials and personal files outside Git. Dependency additions should have a concrete purpose; the existing models run on CPU.
