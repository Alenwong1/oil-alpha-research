# Research results

The selected 50/50 mix achieved **1.731 zero-risk-free Sharpe**, **1.330 cash-excess Sharpe**, **11.0% CAGR** and **8.4% maximum drawdown** over all of **2017–2025**, after modeled transaction, borrow and financing costs. All these dates are development data; no independent performance claim is made.

## Annual results for the full-period selected mix

| Year | Sharpe (zero RF) | Cash-excess Sharpe | Annualized return | Max drawdown |
| --- | --- | --- | --- | --- |
| 2017 | 0.69 | 0.50 | 3.1% | -3.5% |
| 2018 | 3.26 | 2.92 | 20.7% | -4.2% |
| 2019 | 0.21 | -0.36 | 0.7% | -3.7% |
| 2020 | 2.16 | 2.09 | 15.5% | -3.1% |
| 2021 | 2.46 | 2.46 | 26.5% | -5.2% |
| 2022 | 2.39 | 2.13 | 19.2% | -4.8% |
| 2023 | 0.88 | -0.29 | 3.9% | -4.0% |
| 2024 | 0.66 | -0.27 | 3.6% | -8.4% |
| 2025 | 1.91 | 0.95 | 8.8% | -4.9% |

Annual returns use the project's 252-session annualization convention. Calendar groups use signal dates. The same portfolio weights are used in every year.

## All six time-series configurations

| candidate | sharpe_zero_rf | sharpe_excess | cagr | max_drawdown |
| --- | --- | --- | --- | --- |
| ridge_h21 | 1.551 | 1.185 | 0.107 | -0.09 |
| adaptive_h5 | 1.505 | 1.159 | 0.11 | -0.079 |
| adaptive_h21 | 1.274 | 0.892 | 0.083 | -0.14 |
| ridge_h5 | 1.214 | 0.803 | 0.073 | -0.053 |
| boosting_h5 | 1.087 | 0.674 | 0.065 | -0.079 |
| boosting_h21 | 0.753 | 0.339 | 0.044 | -0.13 |

These are portfolios with shared macro/news and technical components. Existing ridge uses seven features; boosting and adaptive ridge use 44. More complex features and models did not automatically improve performance.

## Forecast accuracy on common dates

| candidate | common_days | normalized_return_rmse | zero_forecast_rmse | prediction_return_correlation |
| --- | --- | --- | --- | --- |
| ridge_h5 | 2047 | 1.1485 | 1.1375 | 0.0301 |
| boosting_h5 | 2047 | 1.1875 | 1.1375 | 0.0209 |
| adaptive_h5 | 2047 | 1.2036 | 1.1375 | 0.0731 |
| ridge_h21 | 2031 | 1.2205 | 1.174 | 0.0264 |
| boosting_h21 | 2031 | 1.3588 | 1.174 | -0.0449 |
| adaptive_h21 | 2031 | 1.4663 | 1.174 | -0.0255 |

Every candidate has worse normalized-return RMSE than a zero forecast. Overlapping targets and repeated weekly features make daily counts dependent.

## All fixed portfolio mixes and requested subset

| period | existing_fraction | adaptive_fraction | sharpe_zero_rf | sharpe_excess |
| --- | --- | --- | --- | --- |
| 2017–2025 | 1.0 | 0.0 | 1.551 | 1.185 |
| 2018–2024 | 1.0 | 0.0 | 1.785 | 1.44 |
| 2017–2025 | 0.75 | 0.25 | 1.689 | 1.295 |
| 2018–2024 | 0.75 | 0.25 | 1.866 | 1.497 |
| 2017–2025 | 0.5 | 0.5 | 1.731 | 1.33 |
| 2018–2024 | 0.5 | 0.5 | 1.837 | 1.464 |
| 2017–2025 | 0.25 | 0.75 | 1.655 | 1.274 |
| 2018–2024 | 0.25 | 0.75 | 1.694 | 1.34 |
| 2017–2025 | 0.0 | 1.0 | 1.505 | 1.159 |
| 2018–2024 | 0.0 | 1.0 | 1.49 | 1.169 |

The 2018–2024 subset was requested after viewing full-period results; its 75/25 winner is a retrospective sensitivity, not the primary selection. The full-period winner is 50/50. A zero benchmark is not a transaction-cost-only simulation.

## Uncertainty and research history

The full-period 50/50 mix's descriptive paired 95% Sharpe-change interval versus the existing portfolio is approximately **[−0.201, 0.472]** and includes zero. It uses 2,000 paired circular 21-session block resamples and does not correct repeated model selection. Scarcity/demand additions failed to improve the control; boosted trees also underperformed. The technical component was active only in 2020–2021.

Preserved aggregate tables, original plan metadata and checksums are in [reports](../reports/README.md). See [methodology](RESEARCH_DESIGN.md) and [reproduction limits](REPRODUCING.md) before interpreting these results. 2026 returns remain unevaluated.
