# Make the project yours in one week

The code is an assisted starting point. Your interview advantage comes from understanding its choices, checking its results and making a defensible extension yourself.

## Day 1 — Understand the instrument and reproduce the run

Read README, the fund description and the experiment plan. Create the environment and run tests. Explain why USO is not spot oil and why the 2020 period warrants particular care. Reproduce a report in a new output directory without changing any settings. Verify that metric CSVs match the supplied run to numerical precision on the same platform.

## Day 2 — Trace one prediction and trade

Choose a row in predictions.csv. Write down the signal date, execution date, label-end date, forecast and actual outcome. Find the corresponding raw opens. Recompute the target return manually. Open that strategy's ledger and reproduce its target weight, drift, turnover, fee and net return. Explain why trading on the same day's closing price would introduce an execution assumption this study avoids.

## Day 3 — Understand the models and validation

Read features.py and models.py. Explain what standardization does to ridge and why it is fitted inside the pipeline. Identify the selected hyperparameter and the latest permitted training-label date in model_audit.csv. Explain why random train/test splits and random early-stopping validation are unsuitable here. Identify which features contribute information rather than simply multiplying the feature count.

## Day 4 — Interpret the results without cherry-picking

Compare full features to USO-only features. Compare buy-and-hold to volatility-targeted long before attributing lower drawdown to prediction. Check whether the main forecast beats a zero forecast on MSE, and whether a good trading result could arise despite poor MSE. Explain what the bootstrap interval does and does not establish. A model that loses to momentum is a valid finding.

## Day 5 — Design one extension

Choose a single extension: assess forecast calibration; compare models using identical feature sets; examine stability under a predefined regime split; or replace a feature group with a smaller economically justified subset. Write the hypothesis and scoring rule before running it. Use development data for this work. Since the provided holdout is now visible, it is no longer a fresh holdout for your new ideas; reserve future unseen dates or explicitly call the new analysis exploratory.

## Day 6 — Write a two-minute explanation

Use this sequence: research question; information timing; evaluation design; primary result; most important limitation; next experiment. Include one result figure and one example trade. Be prepared to explain a disappointing result without changing the subject to the best-looking alternative.

## Day 7 — Applications and interview preparation

Keep the project dated October 2026. Link the code repository and a short report once you choose to publish them. Do not wait for a positive result to apply. Use only skills and contributions you can explain and reproduce.

Possible resume wording after completing this work:

> Developed release-aligned inventory features and purged walk-forward USO return models; compared ridge, boosted trees and adaptive ridge under drift-aware transaction, borrowing and financing costs.

Adapt the wording to your actual contribution and add a measured finding from the report rather than a promised performance improvement. If asked about tools or assistance, describe them accurately. Historical results must refer to the matching experiment and period.

## Questions you should be able to answer

1. Why use next-open execution instead of same-close execution?
2. What exactly is purged from training, and why?
3. Why does a 50% position require rebalancing after an asset move even if its target stays 50%?
4. Does the model beat a zero return forecast? Does that imply profitable trading?
5. Which improvements come from volatility scaling versus forecasting?
6. Why might adding 133 features reduce out-of-sample performance?
7. What does a block-bootstrap confidence interval assume?
8. How would fund structural changes and revised adjusted prices affect the interpretation?
9. What happens when cash earns a nonzero rate?
10. After seeing this holdout, how would you obtain genuinely new evidence?
