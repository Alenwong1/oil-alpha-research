# Experiment plan

Written before the first model evaluation in this rebuild, October 5, 2026. This is a local design freeze, not a public preregistration.

## Question and primary comparison

Does a boosted-tree forecast using related ETF features improve a USO long/cash strategy relative to 21-day momentum after assumed transaction costs? The primary comparison is `tree_full - momentum` at 5 basis points per traded dollar. Report all candidates regardless of their performance. Secondary comparisons examine ridge, USO-only features, volatility-targeted long, RSI reversal and buy-and-hold.

## Data and boundaries

Use daily adjusted USO, XLE, XOP, SPY, TLT, HYG, UUP and GLD bars from 2010 through 2025. The selection reflects an ex-ante economic narrative within this experiment: oil fund, energy equities, exploration equities, broad equities, long Treasuries, credit, dollar and gold. It remains a retrospectively chosen universe. The vendor is Yahoo Finance via yfinance. No macro data whose release vintage is uncertain is used. No forward fill or silent skipping of interior bars.

2010–2015 supplies initial training. Development signals begin in 2016, with quarterly expanding fits through 2023. Holding periods ending on or after 2024-01-01 are excluded from development. Final evaluation uses signals in 2024–2025, keeping only labels ending before 2026-01-01. All final model fitting and selection uses labels ending before 2024-01-01; fitted models remain frozen for both holdout years. Daily features may use trailing observations from the holdout, as they would become available sequentially.

At each fit, the last 252 available labeled observations are validation. Inner training labels must end before that validation starts. Ridge alpha is selected from 1, 100, 10000. Boosted-tree maximum leaf count is selected from 7, 15, with 100 iterations, learning rate 0.05, minimum leaf sample count 40 and L2 regularization 10. Select by validation MSE, not strategy Sharpe. Refit the winning configuration on eligible training data. No automatic random validation or early stopping. Seeds are 42.

## Timing and strategy

Feature date t means after that day's close. Trade at next session's adjusted open, then earn the return to the following session's adjusted open. Outcomes ending at or after the fit cutoff are purged. Features include trailing returns, volatility, moving-average distance, simple RSI, intraday range/return, overnight return, relative volume, trailing cross-correlations and relative momentum. No raw price levels.

Model forecast above zero means long; otherwise cash. All active strategies target 15% annualized volatility using 21-day close-return volatility, with a 1% volatility floor and maximum 100% exposure. Momentum is long when the 21-day return is positive. RSI uses simple rolling gains/losses, enters below 30, exits above 70 and retains its state between thresholds. Standard buy-and-hold stays fully invested. A separate volatility-targeted long baseline isolates the risk-scaling effect.

Rebalance notional accounts for the previous weight's return-driven drift and the reduction of equity by trading fees. Include initial and terminal trades. Cash yield is zero. Evaluate costs of 0, 5, 10 and 20 bps without choosing a favorable cost after seeing results.

## Evidence and stopping rules

Report CAGR, zero-reference-rate Sharpe, realized volatility, maximum drawdown, exposure, turnover, annual results, forecast MSE against zero forecasts and feature ablations. Use a paired circular moving-block bootstrap (20-day blocks, 1,000 replicates) for Sharpe differences. These are descriptive intervals, not corrected for research search or multiple comparisons.

Do not alter model or signal specifications after opening holdout results. Fix genuine implementation defects with a written record and rerun under a new output directory. If data downloads fail, stop the real-data experiment; synthetic data is only permitted in tests. Failure to beat the baselines is an acceptable outcome and must be reported.

## Scope

This project studies an ETF, not spot oil or a futures roll strategy. USO's changing structure is a material interpretation limit. It does not inherit any result from the user's lost historical project. No live trading, paid data or account connections are required.
