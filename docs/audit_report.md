# MAPI v0.2 to v0.3 targeted source audit

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
- **Before/after:** each control fits on the chronological fit partition and freezes its frequency match for test. The constant always-long event control was replaced by `same_event_times_always_long`; `full_period_buy_and_hold` is separate.
- **Compatibility:** baseline names and score column (`baseline_score`) changed; result rows gained fit metadata.
- **Uncertainty:** deterministic tie selection matches event count but can introduce timestamp-order dependence; frequency matching still does not equalize turnover, exposure, or information content.

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
- **Regression:** daily-as-intraday, intraday-as-position, weekly-as-daily, and monthly-as-daily conflicts are detected; research eligibility blocks incompatible events and error mode fails calculation.
- **Before/after:** default horizons declare interval expectations and output compatibility metadata. Warn mode preserves diagnostic signal confidence while the research gate prevents incompatible trades.
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
- **Uncertainty:** commit `0db7bbe` completed a green 3.12/3.13 matrix. The current v0.3 revision has only local Python 3.13 verification until its own pushed CI matrix completes.

## 14. One chronological CLI partition

- **Confirmed:** main metrics and score buckets used the full frame while controls and ablation used an internally created test partition.
- **Files/lines:** `examples/run_backtest.py:79-89,195-212`; `mapi/research/baselines.py:106-132,236-262`; `mapi/research/ablation.py:16-75`; `mapi/research/matching.py:24-65`; `tests/test_cli.py:93-116`.
- **Regression:** the CLI serializes fit/test counts, fractions, and timestamp bounds; it asserts main MAPI `sample_count` equals the `full_mapi` ablation count.
- **Before/after:** report sections could evaluate different universes; the orchestration layer now creates one split and passes the same test mask to main study, buckets, controls, random controls, and ablation. Fit and test candidate frequencies are both serialized.
- **Compatibility:** programmatic research functions may still create a chronological split when masks are omitted, but supplied masks are authoritative. CLI metric sample counts now cover test only.
- **Uncertainty:** one holdout split avoids report inconsistency but does not replace walk-forward or multiple-period validation.

## 15. Evidence-quality event eligibility

- **Confirmed:** event selection ignored confidence, data quality, and horizon-frequency compatibility.
- **Files/lines:** `mapi/config.py:61-64,134-141`; `mapi/research/matching.py:68-199`; `mapi/research/backtest.py:16-88,147-150`; `examples/run_backtest.py:85-93`; `tests/test_backtest.py:192`; `tests/test_matching.py:53`.
- **Regression:** low confidence, low quality, and frequency mismatch are excluded and counted separately; matching and event selection use the same eligibility definition.
- **Before/after:** score and direction alone could create an event; configured evidence thresholds now gate candidates, and an incompatible horizon cannot create a research trade in the default v0.3 configuration.
- **Compatibility:** direct event-study calls retain permissive zero/false gate defaults; CLI and ablation use configured defaults of 0.25 confidence, 0.25 quality, and required frequency compatibility.
- **Uncertainty:** quality thresholds are transparent research priors, not calibrated probabilities, and sequential exclusion counts depend on the documented gate order.

## 16. Intensity, novelty, alert, and actionability

- **Confirmed:** `mapi_raw_score` included novelty, allowing recurrence alone to erase anomaly intensity and invalidate state.
- **Files/lines:** `mapi/models.py:43-82,114-125`; `mapi/scoring.py:189-273,298-336,382-402`; `tests/test_scoring.py:49-65,81-127`.
- **Regression:** score ordering and bounds are checked, and a persistent 0.8-strength component with novelty decayed to zero remains a confirmed high-intensity anomaly while alert/actionability become zero.
- **Before/after:** intensity excludes novelty; novelty is an aggregate retention percentage; alert includes novelty; actionability applies directional realization to alert. State, age, trend, persistence, and confirmation are intensity-based.
- **Compatibility:** `mapi_raw_score` now aliases intensity. `mapi_score` remains config-selected; old `pure_anomaly` and `legacy_actionability` selectors are accepted without changing algorithm identity.
- **Uncertainty:** aggregate novelty is weighted by active intensity and remains a heuristic alert-suppression factor, not an event probability.

## 17. Version and data-contract identity

- **Confirmed:** materially changed formulas were still labeled package and signal v0.2, and old YAML could relabel current calculations as v0.1.
- **Files/lines:** `mapi/version.py:1-5`; `pyproject.toml:3`; `mapi/__init__.py:9-12`; `mapi/config.py:57-64,99-104`; `mapi/models.py:132-136`; `mapi/scoring.py:399-403`; `configs/mapi_v0_3.yaml:1-7`; `tests/test_scoring.py:17-35,66-79`.
- **Regression:** serialized output asserts algorithm revision and data-contract version, a 64-character config fingerprint, and unchanged algorithm identity when legacy public-score selection is requested.
- **Before/after:** output now exposes implementation `0.3.0`, algorithm `mapi_v0.3_source_audit_r2`, data contract `mapi_signal_v0.3`, and a canonical SHA-256 config fingerprint.
- **Compatibility:** package version and default config path changed to v0.3. Old YAML files still load as behavior selectors, but `signal_version` is now a compatibility alias for the actual algorithm revision.
- **Uncertainty:** a config fingerprint identifies settings, not input data, provider revision, or execution environment.

## 18. MFE and MAE entry bounds

- **Confirmed:** future highs entirely below entry produced negative MFE, and future lows entirely above entry produced positive MAE.
- **Files/lines:** `mapi/research/labels.py:68-69`; `tests/test_backtest.py:161-190`.
- **Regression:** hand-built falling and rising OHLC paths assert long MFE is clamped to zero and long MAE is clamped to zero respectively; short metrics inherit the same bounds through sign conversion.
- **Before/after:** entry is treated as zero excursion, so long/short MFE is non-negative and MAE is non-positive.
- **Compatibility:** pathological excursion values change at the boundary; realized close-to-close return is unchanged.
- **Uncertainty:** bar OHLC still cannot reveal whether favorable or adverse extremes occurred first.

## 19. Repaired negative controls

- **Confirmed:** constant always-long scores could frequency-match to zero; shuffled signals crossed fit/test boundaries; previous-day score used current return while direction used lagged return.
- **Files/lines:** `mapi/research/baselines.py:40-45,65-103,456-491,566-576`; `mapi/research/matching.py:201-274`; `tests/test_baselines.py:98-181`; `tests/test_matching.py:16-51`.
- **Regression:** same-event always-long and random controls select events when feasible; score/direction shuffles preserve each partition's value set; previous-day score is invariant to the current bar; constant-score matching selects exact deterministic counts.
- **Before/after:** always-long uses MAPI event timestamps, shuffling occurs independently inside fit and test, previous-day score and direction are both lagged, and tied scores use a frozen tie fraction.
- **Compatibility:** `always_long_fixed_horizon` is replaced by `same_event_times_always_long`; matched selections now include deterministic tie metadata internally.
- **Uncertainty:** same-event always-long isolates directional mapping but shares MAPI timing; deterministic tie order may interact with regime clustering.

## 20. Momentum divergence and direction contract

- **Confirmed:** momentum at a new price extreme was compared with the global rolling momentum maximum/minimum, and contemporaneous pressure was used as forecast direction.
- **Files/lines:** `mapi/components/base.py:55-91`; `mapi/components/momentum_disagreement.py:35-72,119-153`; `mapi/scoring.py:223-251,389-394`; `examples/run_backtest.py:69,112-113`; `tests/test_components.py:49-84`; `tests/test_cli.py:118-121`.
- **Regression:** a weaker-momentum new high maps bearish and a weaker-selling new low maps bullish; CLI event study records `mapi_forecast_direction` as the direction column used.
- **Before/after:** current momentum is compared with momentum at the prior relevant price extreme. Components expose observed pressure, forecast direction, and direction semantics; the aggregate legacy direction aliases forecast direction.
- **Compatibility:** momentum component direction can reverse relative to v0.2 on divergence bars. Consumers treating `mapi_direction` as contemporaneous pressure should migrate to `mapi_observed_pressure`.
- **Uncertainty:** mapping divergence to reversal is a hypothesis and has not been established as predictive out of sample.

## 21. Directional price-volume flow

- **Confirmed:** the sign of price z-score was compared with the sign of volume z-score even though positive/negative volume surprise has no bullish/bearish meaning.
- **Files/lines:** `mapi/components/price_volume.py:54-72,84-125`; `tests/test_components.py:86-108`.
- **Regression:** a high-volume decline with a close near the bar low has negative price and directional-flow z-scores and no mismatch flag.
- **Before/after:** mismatch now compares price movement with a close-location-weighted relative-volume flow proxy. Volume magnitude remains available for weak/strong participation features.
- **Compatibility:** price-volume strengths, reasons, and forecast directions can change materially; metrics add `directional_flow_z` and `directional_flow_mismatch`.
- **Uncertainty:** close location is a coarse bar-level flow proxy and is not a substitute for signed trades or order-book imbalance.

## 22. Secondary semantics and verification claims

- **Confirmed:** weekly/monthly bars were labeled daily, regime multipliers ignored regime confidence, zero-reliability components inflated evidence, and selected-event correlation was called a generic information coefficient.
- **Files/lines:** `mapi/data/frequency.py:27-35`; `mapi/weights.py:51-55`; `mapi/scoring.py:176-181,276-303`; `mapi/models.py:245,296-322`; `.github/workflows/ci.yml:16,31-40`; `tests/test_regime_frequency_realization.py:72-84`; `tests/test_confidence.py:173-196`; `tests/test_config_validation.py:93-120`.
- **Regression:** interval-range classification distinguishes four frequencies; zero regime confidence produces a neutral multiplier; zero reliability adds no coverage or confirmation; output contains canonical `active_event_ic`.
- **Before/after:** regime effects interpolate toward neutral by confidence, reliable component weight defines coverage capacity, and the selected-event correlation name states its evaluation universe.
- **Compatibility:** `information_coefficient` remains a warning-labeled alias. Weekly/monthly inputs can now fail daily expectations. Current v0.3 results are reported as local Python 3.13 verification only.
- **Uncertainty:** interval buckets remain calendar-agnostic, regime confidence is heuristic, and selected-event IC remains selection-conditioned.

## Remaining system-level limits

- The event study is not a capital-aware portfolio simulator and does not model sizing, exposure, borrow, capacity, order-book spreads, or intrabar execution order.
- UTC date checks are not an exchange calendar.
- Optional news, fundamental, options, and sentiment components still require point-in-time data providers.
- None of these source corrections demonstrates out-of-sample economic value.
