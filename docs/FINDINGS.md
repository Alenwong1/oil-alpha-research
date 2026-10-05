# Findings from the first completed experiment

The original hypothesis was not supported: adding the full related-ETF feature set to boosted trees did not improve the USO strategy relative to momentum in this run. A simpler ridge model produced a favorable holdout trading result, but no model improved return-prediction MSE over forecasting zero. These observations warrant investigation, not a claim of established alpha.

All numbers below are simulated, net of an assumed 5 bps per traded dollar, with zero cash interest and zero-reference-rate Sharpe. Model choices were unchanged between the development run and opening the final evaluation.

| Strategy | 2016–2023 development Sharpe | 2024–2025 holdout Sharpe | Holdout CAGR | Holdout maximum drawdown |
| --- | --- | --- | --- | --- |
| USO buy-and-hold | 0.13 | 0.23 | 2.39% | -26.06% |
| Volatility-targeted long | 0.27 | 0.16 | 1.30% | -17.25% |
| Momentum | 0.34 | -0.23 | -3.48% | -18.48% |
| RSI reversal | 0.14 | 0.42 | 4.37% | -8.37% |
| Ridge, USO only | 0.09 | 0.72 | 7.09% | -9.08% |
| Ridge, all ETFs | 0.09 | 0.63 | 4.32% | -6.22% |
| Boosted trees, USO only | 0.37 | 0.27 | 2.26% | -12.62% |
| Boosted trees, all ETFs | -0.05 | -0.73 | -8.04% | -20.06% |

## What the evidence says

The primary holdout comparison, full-feature trees minus momentum, is a Sharpe difference of -0.50. Its descriptive 95% paired block-bootstrap interval is [-1.75, 0.72]. Full-feature ridge minus momentum is +0.85 with interval [-0.64, 2.45]. Neither interval excludes zero. The intervals are not adjusted for multiple strategy comparisons.

All four ML variants have negative holdout R² relative to a zero forecast. USO-only ridge's value is approximately -0.0064, and full-feature trees' is approximately -0.0670. Positive strategy returns can coexist with weak MSE because a sign-based allocation uses a different part of the prediction than squared-error scoring, and realized samples can be noisy. The holdout result is not sufficient to select a deployment strategy.

The larger feature set did not help the holdout trading results of either model family. The contrast is particularly large for trees. This is compatible with overfitting or distribution shift, but the experiment does not isolate a causal explanation. More features are not evidence of better research.

Transaction costs matter. Refer to the cost sensitivity chart for all four assumptions rather than reporting only the gross Sharpe. The two-year holdout is short, and all strategies depend on simplifying execution and data assumptions.

## What to say in an application

Lead with the reproducible methodology: 150 trailing features across eight ETFs; chronological model selection; outcome-boundary purging; a frozen evaluation; fees and drift; and transparent negative findings. If including performance, show the period and cost assumption beside the result and acknowledge its uncertainty. Do not describe the 0.72 holdout Sharpe as a confirmed persistent advantage.

## Next useful experiment

Study why the full-feature model degrades: create a prespecified feature-group ablation on development data, using energy equities, risk assets and defensive assets as separate groups. Keep the existing holdout results visible as historical reference; they are no longer fresh evidence for future design choices. Obtain new unseen dates before treating an extension as independently confirmed.

## Links

- [Development report](../results/development/report.html)
- [Holdout report](../results/holdout/report.html)
- [Experiment plan](EXPERIMENT_PLAN.md)
- [Validation record](VALIDATION.md)
- [Learning guide](LEARNING_GUIDE.md)
