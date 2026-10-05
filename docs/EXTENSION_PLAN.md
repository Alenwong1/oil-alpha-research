# Public-data extension and search protocol

Created October 5, 2026 before evaluating the expanded feature set. Public/free sources only.

## Objective and evidence standard

Research target: net annualized Sharpe above 1.5, preferably measured on excess returns over a cash-yield proxy. This is not a promised outcome. No change of evaluation dates, removal of losing years, selection of favorable transaction costs, or presentation of a selected in-sample winner as independent evidence is allowed.

The previously viewed 2024–2025 sample is now development data. This explicitly supersedes the earlier plan's prohibition on modifying models after opening that sample: the user requested a new experiment. Preserve all old results. Reserve January 1–October 2, 2026 for a later locked-candidate evaluation; do not download, score or select on it during development. That reserved period is short and will not prove a durable 1.5 Sharpe by itself.

## Inputs

- Base eight ETFs and additional oil/bond/commodity ETFs; VIX and OVX lagged one full session. Vendor-adjusted market histories remain a retrospective-data limitation.
- ALFRED 3-month, 2-year and 10-year Treasury yields at monthly historical vintages, with the returned vintage suffix asserted for every series. A snapshot becomes usable one day after its vintage date. This deliberately uses monthly rather than daily curve updates to keep free archived retrieval tractable and verifiable.
- EIA archived STEO monthly workbooks, mapped to the archive's release calendar. Include OPEC production/capacity, world production/demand, inventories, US output and published price forecasts. Workbooks flagged by correction notices are quarantined rather than assigned the original release date. All releases wait two UTC calendar days after their listed date; monthly forecasts are marked as forecasts and may refer to future economic periods without using realized future outcomes.
- OilPrice World News archive captures: one earliest capture per calendar month. Use title text embedded in that capture, not present-day article text. Availability is actual resolved archive timestamp plus 24 hours. Deduplicate titles by first capture. Coverage is sparse: missing captures do not mean no news. Retain capture counts and age as explicit features. This is an archived news sample, not the full publisher feed.

General release joins use `available_at <= signal_cutoff`, with cutoffs at 17:00 New York time and execution at next open. Values older than 100 days become missing. The complete source ledger and per-feature availability lineage are saved. No current-vintage fallback. Later data revisions enter only as part of later snapshots. Public archive modifications and incomplete source history remain limitations; exact historical availability cannot be certified beyond the preserved evidence.

## Research rounds

1. Feature-group ablations: USO, all market features, macro plus USO, market plus macro, and all including archived news. Compare regularized linear and shallow boosted-tree regressors on a daily horizon.
2. Economic-horizon and adaptation tests: 5- and 21-session outcomes, expanding versus trailing five-year training, and lower-turnover allocation policies. Longer-horizon labels are purged using their actual ending dates. Report all attempts in an append-only registry.
3. Review stability, costs and forecast quality. Test any follow-up only against development periods; record its rationale before fitting. Use multiple chronological folds and report the full 2017–2025 development record rather than just its best years.

Source-retrieval amendment before the first fit: a second attempt at failed archive connections is running. Preserve each round's exact release/news inputs and model-ready matrices. If the retry adds usable captures, run a separate `news_refresh` round over both model families, 1/5/21-day horizons and expanding/five-year windows. This is motivated by improved input coverage, not by any observed model score. Keep both attempts in the registry.

Development follow-up after round-one results: test same-observation-month revisions against the latest previously available forecast/estimate, with fixed 10-calendar-day exponential decay after release. Use a smaller set of USO price/volume, yield-slope and lagged volatility-index controls. Test the same two model families, three horizons and two training windows. This is an additional development search, not independent validation.

Implementation repair: the first horizons run stopped when the installed histogram-tree library encountered an entirely missing training column. Preserve that partial run. Exclude entirely missing columns using only each training fold's observations, and rerun as `horizons_repaired`. This changes data handling, not the target or scoring criterion. Include the failed attempt in the research history.

Selection on these folds still makes them development results. A selected high Sharpe must not be treated as a fresh out-of-sample validation. Leave the reserved period closed until there is a defensible candidate worth testing.

## Costs and interpretation

Long/cash only, maximum 100% exposure; maintain the 15% volatility-targeting convention. Report fixed 0/5/10/20 bps costs, with 5 bps primary. Model cash accrual using the latest available 3-month yield as an explicitly approximate cash proxy, and report Sharpe on both total and cash-excess returns. Do not boost reported performance merely by adding risk-free interest. No live trading or paid data purchases.

## User-authorized shorting and leverage amendment

The user subsequently requested shorting and leverage. Preserve the long/cash studies, then evaluate signed predictions with a 2× gross cap and 15%/30% volatility targets. Model primary short borrow at 3% annualized, with 10% and 30% stress cases; model leveraged-long financing at the contemporaneously available cash proxy plus 2% annually. Short-sale collateral earns no rebate. Borrow availability and historical fees are assumptions, not verified facts. Include transaction costs on signed turnover and on terminal liquidation. A 25% maintenance-equity rule is checked at daily endpoints and halts trading after a modeled call; intraday margin calls cannot be assessed from this daily execution model. Leverage is not treated as a source of increased pre-cost Sharpe.
