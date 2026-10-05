# Weekly fundamentals research plan — frozen before scoring

Authorized October 5, 2026. Public sources only. Keep 2026 returns unopened. This is a new development search following the previous 900-policy search; no independent validation is implied.

## Data and timing

Download original EIA weekly archive pages and their CSV tables 1 and 2, 2011–2025 (earliest modern archive). Use the release date in the archive URL plus one calendar day at 00:00 New York time. Validate the observation date precedes release. Read only the latest and prior-week values printed in that vintage. Inspect errata and quarantine affected vintages rather than backdating corrections. No current-series fallback.

Download CFTC disaggregated futures-only annual archives through 2025, WTI physical contract code 067651. Keep managed-money long/short, producer long/short and total open interest. Use a deliberately conservative 10-calendar-day lag after the report's observation date in normal weeks. Quarantine observations during delayed-publication intervals (2013 shutdown/catch-up, 2018–2019 shutdown/catch-up, 2023 ION outage/catch-up, and 2025 shutdown/catch-up) instead of guessing release dates. Also exclude documented broad correction episodes. Ten days is an availability assumption supported by the normal weekly schedule, not the actual release timestamp. Current historical archives cannot certify original classifications; report CFTC-dependent results as provisional and separately show results excluding CFTC. Save official special-announcement evidence and source checksums. Missing sources do not become zero-valued observations.

## Small, fixed experiment set

Three separate ridge forecasters with a fixed alpha of 100, predicting 21-session USO excess return normalized by known volatility. Features: (1) trend: trailing 21/63/126-session normalized returns; (2) inventories: seasonal unexpected changes in crude, gasoline and distillate stocks plus seasonal deviations in refinery inputs, utilization, imports and exports; (3) positioning: managed-money net/open-interest level, trailing percentile and weekly change, plus producer net/open interest. All seasonal expectations use earlier releases only, at least two years of observations, and a fixed ±4-week seasonal window. No analyst consensus is claimed. Feature preprocessing is fit inside each chronological training fold.

Quarterly expanding-window refits; purge labels ending on/after the first test signal. Forecast from 2015 where sufficient training data exists to allow portfolio-risk warmup, but score the same 2017–2025 period as prior experiments. No selection of model alpha, feature sign, lookback or outcome horizon inside this batch.

Scale each sleeve using its own trailing 252-session hypothetical return volatility, shifted two signal rows so every risk observation has ended by the decision cutoff. Target 15% annual volatility per standalone sleeve, cap absolute target exposure at 2. Combine by equal capital allocation to these independently risk-scaled sleeves, netting exposures and applying portfolio costs once. Also include a trend+inventory combination to isolate CFTC limitations. Correlations are measured, not optimized. All sleeves trade USO, so this is signal diversification, not cross-asset diversification.

Evaluate five sleeves/portfolios × daily versus weekly (first signal session of each week) × always-rebalance versus fixed cost-aware no-trade policy = 20 policies. Cost-aware policy requires predicted 21-session excess return to exceed estimated round-trip transaction cost plus borrow/financing over that horizon; small target changes below 0.10 equity are skipped, except exits/sign changes or exposure-limit enforcement. Thresholds are fixed economically, not optimized on outcomes. For combined portfolios use the average constituent expected return for the gate.

Use the signed accounting engine, 5 bps primary trading cost, 3% annual short borrow, cash proxy +2% financing spread, no short collateral rebate, 2× target cap. Weekly/no-trade holds preserve shares (allow weight drift), rather than secretly rebalancing daily. Daily safety cap enforcement may override the rebalance schedule. Daily maintenance/insolvency limitations remain. Stress the selected candidate at 0/10/20 bps and 10%/30% borrow. Report all 20 policies, annual results, sleeve correlations and a block-bootstrap interval. A selected confidence interval does not correct the broader research selection.

## Acceptance

A larger development Sharpe alone does not qualify for a 1.5 claim. Assess year-to-year stability, costs, data coverage and whether CFTC-independent evidence supports the result. Preserve every attempt. No autonomous live trading or external publication.

Implementation detail recorded before scoring: normalized forecasts are converted to preliminary directions with `clip(score / 0.5, -1, 1)`. Sleeve volatility uses a 5% annual floor and at least 126 prior completed observations. Model fitting requires at least 252 complete-feature training rows. CFTC gaps disable its signal after observation age exceeds 21 days; no differences across long outage gaps are treated as weekly changes. EIA observations also expire after 21 days. Equal allocation leaves unavailable sleeves in cash. Cost gates use a 21/252-year holding approximation and add the foregone asset cash yield to the short hurdle, because the model predicts asset excess returns.

## Follow-up recorded after the 20-policy results, before blend scoring

The inventory winner and the prior long/cash winner have different losing-year patterns. Test exactly one additional portfolio: a fixed 50/50 capital allocation to their saved target exposures, netted and rebalanced daily with the same costs, borrowing and 2× cap. Do not optimize the blend weight, volatility scale or component selection further. This is explicitly a development follow-up using two already-selected winners; it is not a predefined member of the original 20-policy family and does not create independent evidence. Keep its results in a separate subdirectory and preserve the original plan hash and 20-policy results.
