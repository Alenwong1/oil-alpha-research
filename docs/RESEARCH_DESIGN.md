# Research design

## Information and execution timing

A signal at day t's 17:00 New York cutoff uses only eligible releases and trailing market features. The execution convention is open t+1; a one-day label ends at open t+2. Inventory targets update on the first signal session of a week. Combined portfolios net component target positions and rebalance the resulting account daily. A weekly decision model is not an intraday announcement-event strategy.

EIA observation dates differ from publication dates. Releases are conservatively usable the following calendar day. Backward as-of joins require availability no later than the signal cutoff; observations older than 21 days are disabled. Extracted time-series events are delayed to the first actual daily cutoff that observed them. Missing features do not get backfilled.

## Features and models

**Seasonal anomalies:** z-scores against strictly earlier observations from the previous five years within ±4 seasonal weeks, requiring 104 prior observations and 12 comparable seasonal observations; clipped at ±4.

**Inventory surprises:** separate ridge regressions forecast crude, gasoline and distillate weekly changes using prior changes, the preceding four-report mean, previous imports/exports/refinery inputs and annual/semiannual sine/cosine terms. Each release is fitted using up to five years of prior data with at least 104 usable rows and alpha 10. Actual minus predicted change is divided by the standard deviation of up to 104 previous forecast errors (minimum 26), clipped at ±4. This is a model-implied surprise, not an analyst-consensus surprise. The original autoregressive stage uses preceding retained reports, which may not be exactly consecutive weeks.

**Lagged model features:** 12 current features and 32 lags produce 44 columns. Lags are at 1/2/4/8 retained releases, with observation gaps required to be 7k ±3 days. They are constructed before daily alignment. Lagged histories are intentional; a stale current report is a separate condition.

**Return ridge:** quarterly expanding training, minimum 252 complete daily rows, alpha 100 and training-only standardization. Targets are 5/21-session returns minus the contemporaneous cash-yield approximation, divided by trailing volatility scaled to the horizon and clipped at ±4. Reporting zero-RF Sharpe does not change this training target.

**Adaptive ridge:** the same objective with observation weights proportional to 2^(-age_days/730.5), normalized to mean one. Weighted training-only scaling and quarterly refits. The lagged adaptive package also changes features relative to the seven-feature control; it is not a pure architecture ablation.

**Boosting:** histogram gradient boosting, 100 iterations, learning rate .04, seven leaves, minimum 60 rows per leaf, L2 regularization 20, no random early-stopping split. Full specifications remain in the frozen plans.

All training labels must finish strictly before the first signal of the test quarter. Historical forecast-error filters include only completed outcomes. Daily labels overlap and weekly features repeat, so training rows are not independent observations.

## Risk and portfolio construction

The inventory sleeve converts normalized predictions to signed conviction, estimates trailing 252-session strategy volatility using completed returns, targets 15% annual volatility with a 5% floor and clips targets to ±2. Signals require magnitude at least 0.25 times historical forecast-error RMS and must clear a horizon-specific modeled cost hurdle. Weekly execution uses a 0.10-equity deadband.

The final 50/50 mix retains 37.5% macro/news and 25% technical components; each of the existing and adaptive inventory components receives 18.75%. Component USO positions are netted before one account's turnover, borrow and financing costs are calculated. Strategy allocations are not fixed asset exposures. The simulator also models drift, terminal liquidation, daily endpoint maintenance checks and insolvency; intraday broker behavior is not established.

## Metrics and costs

For daily net return r and daily cash proxy c:

- Zero-risk-free Sharpe: sqrt(252) × mean(r) / sample_std(r).
- Cash-excess Sharpe: sqrt(252) × mean(r − c) / sample_std(r − c).
- CAGR: annualized compounded daily net returns using 252 sessions per year.
- Drawdown: decline of cumulative net equity from its running high-water mark.

Primary costs: 5 bps per traded dollar, 3% annual borrow for short exposure and cash rate plus 2% for leveraged-long financing. Own uninvested cash earns the modeled cash proxy; short proceeds receive no rebate. Zero-RF Sharpe retains this cash income and all these costs. It is not a transaction-cost-only simulation.

Subset statistics slice the continuous full-period ledger; they do not create new positions or liquidations at subset boundaries. Annual labels follow signal-date calendar years in the later experiments.

## Evidence and limits

Predictions are walk-forward, but the configuration and portfolio selection used the same 2017–2025 development period repeatedly. The original baseline holdout is now part of development. 2026 remains unevaluated. Paired 21-session circular block-bootstrap intervals (2,000 samples in the latest mix) are descriptive and do not correct the full search history.

Archive availability checks reduce look-ahead risk; they do not establish perfect point-in-time market prices, original CFTC classifications, tradeable news latency or verified historical borrow rates. Additional features and models are retained even when they fail. Successful accounting/timing tests are not proof of a durable economic edge.
