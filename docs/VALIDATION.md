# Validation

12 unit tests passed.

development: source hashes, all fit boundaries, actual-price labels, 32 ledgers, and closed-form buy/hold fee identities verified.

holdout: source hashes, all fit boundaries, actual-price labels, 32 ledgers, and closed-form buy/hold fee identities verified.

All six report charts were visually inspected. No strategy changes were made after viewing holdout results.


## Expanded data and signed-portfolio validation

- Full test suite: 29 passed. Includes short payoff and borrow fees, long financing, no short-collateral rebate, exact signed turnover fees, exposure cap rejection, insolvency visibility, daily maintenance liquidation, and agreement with the original long-only ledger.
- Latest macro feature lineage: 274,838 populated availability timestamps checked against the signal cutoff; no violations. Training audit files, including preserved partial attempts: 2,268 fold records checked; every last training outcome precedes the test start.
- Expanded feature cache: 3,959 rows and 405 columns, latest signal date December 29, 2025. Latest raw extension sources include 10,704 deduplicated EIA observations and 1,331 unique archived headlines. Each model round preserves its own actual input snapshot; earlier rounds used a smaller headline archive.
- Four completed model rounds contain 50 specifications and 300 long/cash policies. The signed experiment adds 600 policies. An earlier interrupted horizon run is retained and excluded from completed rankings.
- Best long/cash development excess Sharpe: 0.738. Best signed development excess Sharpe: 0.596. All primary results include 5 bps turnover costs; signed strategies also include borrow/financing assumptions. No 1.5 result or independent validation is claimed.
- The centered block-bootstrap family diagnostic has approximate p-value 0.661 and 95th-percentile maximum-null Sharpe 1.28 across the 900 scored policies. It does not account for every research choice.
- The reserved 2026 period remains unopened. Market-data vendor adjustments and actual borrow/locate availability remain unverified.


## Weekly fundamentals batch and fixed-blend follow-up

38 tests passed. New checks cover archived date parsing, CFTC outage quarantine, stale/future as-of behavior, seasonal feature future isolation, share-preserving holds, daily ledger equivalence, asymmetric short-cost hurdles and completed-return-only risk estimates. The actual data audit passed 7,581 availability comparisons, 132 purged training folds, 20 equally dated policies and 2,260 scored sessions per policy. All target weights obey the 2x cap. The EIA collector retrieved 751 release-specific records; seven were conservatively quarantined after errata review. 793 CFTC observations survived explicit exclusions. CFTC availability is conservatively assumed, not certified by original publication snapshots.

The new inventory winner achieved 0.739 development excess Sharpe; trend and positioning models were negative after costs. A single separately logged 50/50 follow-up with the prior winner achieved 0.942, with 11.5% CAGR and 11.8% maximum drawdown. At 20 bps trading cost and 3% borrow, blend Sharpe is 0.776; at 20 bps and 30% borrow it is 0.582. The blend is not independently validated and does not meet the 1.5 goal. Reserved 2026 returns remain untouched.

A report-only correction removed the combined sleeve from the standalone correlation table. No forecasts, allocations or scores changed; original and repaired source hashes are recorded in `results/weekly/report_repairs.json`.


## Full-period refinement

Three new tests verify inventory predictions cannot use current/future actuals, forecast uncertainty waits for label completion, and the conservative risk rule never increases exposure relative to its control. Actual-data audits verify 1908 weekly forecast training cutoffs, 43 quarterly training boundaries, as-of availability and all 24 policy ledgers. The original inventory control reproduces its earlier result within 1e-9. Source hashes and imported dependency snapshots are retained. Every scored policy contains 2,260 sessions spanning 2017–2025, with no target above 2x gross. 2026 is not evaluated.

Full suite after refinement: **41 tests passed**.


## Diversification follow-up

43 tests passed. New tests verify opposing exposures net before transaction/borrow costs and reject misaligned inputs or invalid blend fractions. The unmodified base reproduces daily net returns within 1e-12. All seven portfolios contain the same 2,260 sessions, retain 2017–2025, and obey the 2x target cap. Input hashes verified. The selected diversifier traded only in 2020–2021, so low correlation partly reflects inactivity; the descriptive paired Sharpe-difference interval [-0.106, 0.203] includes zero and does not correct the full screening history.


## Scarcity and demand batch

47 tests passed, including new checks for product-supplied parsing/date validation, causal scarcity states, gap handling and completed-error timing at both forecast horizons. All 744 retained EIA table checksums and observation dates were validated. The saved-result audit verified five input hashes, eight code hashes, 3,622 as-of availability comparisons and 258 purged quarterly training folds. Six equally dated portfolios contain 2,260 sessions each in 2017–2025; no target exceeds 2x. The existing 21-session control reproduces prior daily returns with maximum absolute difference 3.01e-16. All six outcomes and individual-year metrics are retained. No 2026 evaluation was performed.

New scarcity/demand features did not improve the 1.185 control. Paired 21-session circular block-bootstrap intervals (2,000 draws, seed 49021) are descriptive and do not correct repeated development selection. The latest feature, audit and result report is in `results/scarcity`.


## Time-series ML batch

51 tests passed. New tests verify release-frequency lagging, no future feature influence, missing-week rejection, first-row extraction without backfilling nulls, the adaptive half-life and normalization, and unfinished-label isolation for all three models. Seven input hashes, ten source snapshots, 3,622 availability comparisons and 254 purged training folds passed audit. Six equally dated portfolios each contain 2,260 sessions in 2017–2025 with targets bounded by 2x; both ridge controls reproduce prior daily returns within 1e-11. The only test warning concerned unavailable physical-core detection; execution used threadpool limits.

The existing 21-session control remains best at 1.185 cash-excess Sharpe. Adaptive five-session ridge achieves 1.159, while boosting does not improve either horizon control. Every model has worse common-date normalized-return RMSE than a zero forecast. The new model packages change both features and estimation; this is not a pure architecture ablation. All six outcomes, annual results and descriptive paired bootstrap intervals are retained. No 2026 evaluation or parameter tuning followed scoring.


## Public repository preparation

Maintained Python sources were formatted with Black; the original run snapshots in local results were preserved. All 51 tests pass after formatting. Publication includes 34 allowlisted aggregate artifacts verified by SHA-256, plus a generated headline summary and aggregate chart. The offline repository checker validates headline numbers, date coverage and maintained documentation links. Raw data, daily derived series, local credentials and personal files are excluded. The complete public-data pipeline was not re-downloaded; exact historical reproduction still requires original input snapshots. GitHub CI is configured separately and must be checked for its actual run status.
