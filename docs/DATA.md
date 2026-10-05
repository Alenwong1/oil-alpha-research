# Data and provenance

| Source | Inputs | Availability treatment | Limitation |
|---|---|---|---|
| Yahoo Finance through yfinance | Adjusted ETF OHLCV and volatility indices | Trailing features; next-open convention | Downloaded history is retrospectively adjusted |
| ALFRED | Historical yield-curve vintages | Monthly vintage plus conservative delay; validate vintage headers | Monthly snapshots do not capture every daily information change |
| EIA weekly archives | Inventory, refinery and trade flows | Release-specific files; availability next day; 21-day expiry | Seven retained-source records quarantined after errata review |
| EIA STEO archives | Supply, production and forecast releases | Archived release plus conservative delay | Corrected vintages excluded when original state is unclear |
| OilPrice via archived captures | Sparse historical news headlines | Capture time plus 24 hours | Coverage is incomplete; not a real-time news feed |
| CFTC annual archives | WTI trader positioning | Conservative lag and outage exclusions | Historical revisions/classifications are not certified |

Sources: [EIA weekly archive](https://www.eia.gov/petroleum/supply/weekly/archive/), [EIA STEO](https://www.eia.gov/outlooks/steo/), [ALFRED](https://alfred.stlouisfed.org/), [CFTC historical files](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm), [OilPrice](https://oilprice.com/), [yfinance](https://github.com/ranaroussi/yfinance).

The latest weekly feature set retains 744 EIA releases. Downloads record file hashes and retrieval metadata; models record input/source fingerprints. Aggregate publication manifests retain the original experiment metadata where available. Formatting the maintained source for publication changes its hashes; original run source snapshots remain in the local results cache and are not silently rewritten.

## Repository contents

Committed: source code, configurations, tests, experiment plans, aggregate performance/annual/accuracy tables, aggregate charts and selected provenance manifests.

Excluded: provider payloads, price histories, headline text, daily model features/predictions, daily ledgers, personal documents, credentials, local environment files and caches. `scripts/export_results.py` uses an explicit file allowlist rather than copying the entire results tree.

Public accessibility is not a blanket redistribution license. Retrieve data directly under the relevant provider's terms; this repository does not grant rights to third-party data. No license to raw provider data or headline text is implied. The source repository currently has no additional open-source license grant.
