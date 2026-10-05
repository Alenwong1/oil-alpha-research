# Methodology and audit notes

## Information clock

An index date denotes a signal computed after that session's close. For a Monday signal, the default holding period is Tuesday open to Wednesday open (assuming no holidays). Monday's close and volume are therefore available; Tuesday's close and Wednesday's open are not model inputs. The target is `Open.shift(-2) / Open.shift(-1) - 1`.

Every fitted model uses only rows with feature date before its cutoff and label-end date strictly before the cutoff. This is conservative: even an outcome observed at the cutoff day's open is excluded. For each inner validation split the same outcome-boundary rule is applied. Purging is based on actual label-end dates, rather than guessing a number of calendar days.

The feature matrix is computed once using backward-looking operations only. A test perturbs all future prices and volumes and verifies that earlier features remain identical. StandardScaler is fit within the training pipeline, independently for each model fit; validation and test features do not set its mean or variance.

## Portfolio accounting

All strategies are long/cash with weight between zero and one. Suppose the old weight has drifted to `p`, the desired post-fee weight is `w`, and the proportional fee is `c`. With pre-trade equity normalized to one, traded notional `q` solves:

```text
q = abs(w * (1 - c*q) - p)
q = (w-p)/(1+c*w)   when buying
q = (p-w)/(1-c*w)   when selling
net growth factor = (1-c*q) * (1+w*r)
next pre-trade weight = w*(1+r)/(1+w*r)
```

This handles the small effect of trading costs on available equity and avoids treating the previous target weight as the current weight. At the final exit, remaining asset weight is sold at the same fee. A full buy-and-hold position consequently earns the adjusted-open price ratio less its entry and exit fees. Equity starts at one; drawdown includes the initial capital level, so a first-day loss is not hidden.

`fee_fraction` in each ledger records fee fractions at their respective transaction times. On the terminal row it includes an exit fee measured against terminal equity; it is not a currency total and should not be summed as an exact dollar P&L cost. Gross minus net return is the appropriate row-level return comparison. Annual turnover is a conventional sum of locally normalized traded notionals, annualized at 252 sessions.

Development quarters are continuous holdings with no forced liquidation between quarterly fits. Development and holdout are separate experiments, each starting in cash and ending with a liquidation. Holding periods crossing the development/holdout boundary are omitted. Annual tables group by holding-period end year and preserve the strategy's existing holdings across year boundaries; they do not restart a portfolio each January.

## Features

Each ETF contributes 17 features: returns over 1/2/5/10/21/63 sessions; volatility over 5/21/63; price distance from 10/21/63-session means; simple 14-session RSI; intraday return; high-low range divided by close; overnight return; and volume relative to its 21-session mean. Seven peer ETFs each add trailing 63-session correlation with USO and relative 21-session momentum. Total: 8 × 17 + 7 × 2 = 150 features. USO-only models use 17.

RSI here uses simple rolling gains and losses, not Wilder smoothing. This choice is explicit and constant. Relative volume can still be distorted by split-related vendor conventions; inspect split dates. All prices are vendor-adjusted. Economic adjustments are retrospective and can be revised; the strict information clock does not turn them into a historical point-in-time archive.

## Metrics and uncertainty

Annualization uses 252 trading sessions. Sharpe is the mean net daily return divided by its sample standard deviation times sqrt(252), with a zero reference rate. Cash pays no interest. CAGR is geometric; maximum drawdown is peak-to-trough loss from an initial equity of one. Daily model MSE is compared with forecasting zero, with `r2_vs_zero = 1 - model_mse / zero_mse`; it is not sklearn's mean-centered R².

The paired circular moving-block bootstrap resamples the same consecutive-return blocks for both strategies before taking their Sharpe difference. It preserves some local dependence, but assumes enough stability for those blocks to be informative. It is not a multiple-testing correction or a guarantee of generalization. Report 5/20/60-day sensitivity as a future robustness exercise rather than selecting a block length that makes a preferred result significant.

## Data audit

The downloader checks date uniqueness/order, finite values, positive prices, nonnegative volume and OHLC consistency. Every peer ETF must have exactly the USO calendar. Cached checksums are checked on every load. Interior feature gaps fail rather than silently shortening a holding interval. The manifest records split events and counts close moves larger than 30%; these flags require interpretation rather than automatic deletion. Tests use synthetic data, while delivered experimental metrics must use the downloaded snapshot.

## Known exclusions

There is no point-in-time macroeconomic data, futures term structure, transaction-level data, market impact, variable spread model, tax model, cash-interest model, live broker connection or production deployment. No conclusion of profitable deployable alpha follows from this project alone.
