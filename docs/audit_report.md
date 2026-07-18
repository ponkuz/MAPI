# MAPI v0.2 targeted source audit

This source audit does not establish profitability. No parameter was tuned against the final test set. Each item below records the confirmed finding, changed files, regression coverage, behavioral change, compatibility effect, and remaining methodological uncertainty.

## 1. Novelty and recurrence

- **Confirmed:** `rolling_percentile_rank` included the current value and assigned a constant series rank 1.0. It did not model recurrence.
- **Files:** `mapi/normalization.py`, all five enabled component modules, `mapi/models.py`, `mapi/scoring.py`, `mapi/explainability.py`, `tests/test_novelty.py`.
- **Regression:** constant, repeated near-identical, gradually increasing, and one-off anomaly sequences.
- **Before/after:** novelty was current-inclusive extremeness; it is now prior-only mid-rank extremeness multiplied by one minus near-identical recurrence. Constant events converge to novelty 0.
- **Compatibility:** `rolling_percentile_rank` remains for descriptive current-inclusive ranks. Component JSON adds `historical_extremeness` and `recurrence_rate`; `novelty` semantics changed.
- **Uncertainty:** the absolute recurrence tolerance is heuristic and may need scale-aware calibration using training data only.

## 2. Explicit research score

- **Confirmed:** the event study dynamically preferred actionability while score buckets defaulted to `mapi_score`.
- **Files:** `mapi/config.py`, `configs/mapi_v0_2.yaml`, `examples/run_backtest.py`, `mapi/research/backtest.py`, `mapi/research/baselines.py`, `mapi/research/ablation.py`, `tests/test_cli.py`.
- **Regression:** CLI output asserts one `score_column_used`; the value is passed to event study, buckets, frequency matching, controls, and ablation.
- **Before/after:** report sections could evaluate different scores; one configured or CLI-selected column now governs the report and is serialized in JSON.
- **Compatibility:** `run_event_study` now has a stable `mapi_score` default instead of dynamic column selection. The CLI default is explicitly `mapi_actionability_score` through configuration.
- **Uncertainty:** choosing raw anomaly versus actionability is a research-design decision, not evidence that either has predictive value.

## 3. Stock-sector correlation breakdown

- **Confirmed:** `1 - abs(current correlation)` measured low correlation, not decline from prior correlation; covariance and variance used inconsistent ddof values.
- **Files:** `mapi/components/stock_sector.py`, `tests/test_stock_sector_correlation.py`.
- **Regression:** persistently low correlation produces no sustained breakdown; a transition from high to low correlation does. Covariance and variance both use `ddof=0`.
- **Before/after:** low correlation was always anomalous; only a positive decline from a lagged rolling median absolute-correlation baseline contributes now.
- **Compatibility:** component strength and reason composition can change materially; metrics add baseline and decline values.
- **Uncertainty:** rolling median baseline/window selection is robust but heuristic and can react slowly to genuine structural changes.

## 4. Frequency-matched negative controls

- **Confirmed:** unrelated baseline score scales used the same numeric threshold.
- **Files:** `mapi/research/matching.py`, `mapi/research/baselines.py`, `examples/run_backtest.py`, `tests/test_baselines.py`.
- **Regression:** every event baseline reports fitted threshold, target/fitted/test frequency, selected count, and excluded overlaps; names verify both buy-and-hold concepts.
- **Before/after:** each control fits a threshold on the chronological fit partition to match MAPI frequency and freezes it for test. `buy_and_hold` became `always_long_fixed_horizon`; `full_period_buy_and_hold` is separate.
- **Compatibility:** baseline names and score column (`baseline_score`) changed; result rows gained fit metadata.
- **Uncertainty:** discrete/tied scores may only approximately match frequency, and frequency matching does not equalize turnover, exposure, or information content.

## 5. OHLC path labels

- **Confirmed:** MFE, MAE, and breakout used close only despite path terminology.
- **Files:** `mapi/research/labels.py`, `mapi/research/backtest.py`, `mapi/models.py`, `tests/test_backtest.py`.
- **Regression:** hand-calculated long and short OHLC paths verify intrabar MFE/MAE while realized return remains close-to-close.
- **Before/after:** canonical fields are `intrabar_mfe`, `intrabar_mae`, and `intrabar_breakout`; future high/low after entry drive them.
- **Compatibility:** legacy `mfe`, `mae`, and `breakout` aliases remain but now carry intrabar semantics and emit compatibility warnings in metrics output.
- **Uncertainty:** OHLC bars do not reveal intrabar ordering, fillability, gaps, or whether both favorable and adverse extremes occurred before exit.

## 6. Event-return statistics

- **Confirmed:** `sqrt(252 / horizon_bars)` annualization ignored event frequency and timestamp spacing.
- **Files:** `mapi/research/backtest.py`, `mapi/models.py`, `mapi/config.py`, `configs/mapi_v0_2.yaml`, `tests/test_backtest.py`.
- **Regression:** event ratios equal unannualized mean/std values and deterministic bootstrap intervals contain the sample mean.
- **Before/after:** reports use unannualized `event_return_mean_to_std`, downside equivalent, and a 95% bootstrap mean-return interval.
- **Compatibility:** legacy Sharpe/Sortino fields alias unannualized event ratios and carry an explicit warning; they are no longer annualized.
- **Uncertainty:** iid bootstrap resampling does not model clustered regimes or serial dependence; a portfolio simulator is required for annualized risk metrics.

## 7. Descriptive classification names

- **Confirmed:** `precision` meant gross directional accuracy and `recall` used a sample-median opportunity definition.
- **Files:** `mapi/research/backtest.py`, `mapi/models.py`, `mapi/config.py`, `tests/test_backtest.py`.
- **Regression:** canonical fields are present and the legacy aliases are warning-labeled.
- **Before/after:** reports expose `gross_directional_accuracy`, `net_profitable_event_rate`, and `large_move_capture_rate` over a fixed configured large-move threshold.
- **Compatibility:** `precision`, `recall`, and `hit_rate` remain aliases for serialized consumers but are deprecated semantically.
- **Uncertainty:** the fixed large-move threshold is not volatility-normalized and may not be comparable across symbols or regimes.

## 8. Regime source integrity

- **Confirmed:** an invalid benchmark bar silently substituted stock close into the benchmark-derived regime series.
- **Files:** `mapi/regimes.py`, `mapi/scoring.py`, `mapi/models.py`, `tests/test_regime_frequency_realization.py`.
- **Regression:** stale benchmark rows return `unknown`, `benchmark_stale`, confidence 0; no-benchmark runs explicitly return `stock_fallback`.
- **Before/after:** benchmark and stock sources no longer switch within one regime series. Unknown regimes receive neutral dynamic-weight behavior.
- **Compatibility:** regime detection now returns a three-column frame and signal JSON adds `regime_source` and `regime_confidence`.
- **Uncertainty:** stock fallback confidence 0.6 is a transparent prior, not an empirically calibrated probability.

## 9. Horizon frequency semantics

- **Confirmed:** named horizons were bar counts without validating input frequency.
- **Files:** `mapi/data/frequency.py`, `mapi/models.py`, `mapi/config.py`, `configs/mapi_v0_2.yaml`, `mapi/scoring.py`, `tests/test_regime_frequency_realization.py`.
- **Regression:** daily-as-intraday and intraday-as-position conflicts are detected; warn mode zeros aggregate confidence and error mode fails.
- **Before/after:** default horizons declare `intraday` or `daily` expectations based on median interval and output compatibility metadata.
- **Compatibility:** custom horizons default to `expected_frequency="any"`; configured defaults may now warn or raise under the selected policy.
- **Uncertainty:** median interval classification does not replace an exchange calendar and cannot fully characterize mixed sessions or overnight bars.

## 10. Realization adjustment

- **Confirmed:** `already_realized_score` was only a percentile of absolute trailing return and ignored anomaly direction and first detection.
- **Files:** `mapi/realization.py`, `mapi/scoring.py`, `mapi/models.py`, `mapi/config.py`, `configs/mapi_v0_2.yaml`, `tests/test_regime_frequency_realization.py`, `tests/test_no_lookahead.py`.
- **Regression:** bullish reversal, bearish reversal, bullish/bearish continuation, neutral direction, and future-mutation invariance.
- **Before/after:** `recent_move_extremeness` is descriptive; actionability is reduced only by aligned movement since `anomaly_first_detected_at`, scaled by lagged historical movement.
- **Compatibility:** `already_realized_score` remains an alias for `recent_move_extremeness`; old config key maps to `directional_realization_penalty`.
- **Uncertainty:** direction may evolve after first detection, and the 90th-percentile movement scale is heuristic.

## 11. Configuration validation and reliability

- **Confirmed:** invalid enums, unknown components, weights, reliability, windows, thresholds, redundancy, and staleness values could pass silently; zero reliability retained half weight.
- **Files:** `mapi/config.py`, `mapi/weights.py`, `mapi/scoring.py`, `mapi/components/__init__.py`, `mapi/components/base.py`, `tests/test_config_validation.py`.
- **Regression:** invalid categories fail fast; unknown enabled components cannot be skipped; zero reliability produces zero dynamic weight.
- **Before/after:** load and scoring validate the full contract. Reliability is now a direct `[0,1]` multiplier.
- **Compatibility:** configurations relying on misspellings, zero total weight, out-of-range values, or the former 0.5 reliability floor now fail or score differently.
- **Uncertainty:** configured reliability remains a subjective prior until estimated with walk-forward data.

## 12. Ablation horizon and matching

- **Confirmed:** ablation defaulted every horizon to five bars and compared variants at one numeric threshold.
- **Files:** `mapi/research/ablation.py`, `mapi/research/matching.py`, `examples/run_backtest.py`, `tests/test_ablation.py`.
- **Regression:** swing and position derive 20- and 60-bar defaults; rows include matched thresholds, frequencies, sample-count changes, and confidence intervals.
- **Before/after:** each ablated variant matches full-MAPI fit frequency, freezes its threshold for test, and uses the configured horizon return window.
- **Compatibility:** `backtest_horizon_bars=None` now derives from configuration; explicit positive overrides remain supported.
- **Uncertainty:** removing a component can change direction and event identity even when aggregate frequency is matched.

## 13. Python CI coverage

- **Confirmed:** prior evidence covered local Python 3.13.1 only.
- **Files:** `.github/workflows/ci.yml`, `pyproject.toml`, `tests/test_cli.py`, `tests/test_config_validation.py`.
- **Regression:** CI matrix declares Python 3.12 and 3.13 and runs full `unittest`, full `pytest`, both CLI paths, and configuration validation.
- **Before/after:** cross-version verification is automated on push and pull request instead of inferred from one local runtime.
- **Compatibility:** CI installs the existing `.[dev]` extra; package runtime requirements are unchanged.
- **Uncertainty:** CI results exist only after the branch is pushed and GitHub Actions runs; local verification remains Python 3.13.1 in this environment.

## Remaining system-level limits

- The event study is not a capital-aware portfolio simulator and does not model sizing, exposure, borrow, capacity, order-book spreads, or intrabar execution order.
- UTC date checks are not an exchange calendar.
- Optional news, fundamental, options, and sentiment components still require point-in-time data providers.
- None of these source corrections demonstrates out-of-sample economic value.
