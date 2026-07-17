# Targeted pre-Phase-2 audit

This audit does not establish profitability. The pre-audit suite had 13 passing tests; the post-audit suite has 44 passing tests on Python 3.13.1.

## Confirmed findings and fixes

1. **Anomaly intensity and actionability were conflated.** The v0.1 scorer multiplied the normalized anomaly score by `1 - penalty * already_realized_score` before publishing `mapi_score`. Version 0.2 publishes pure intensity as both `mapi_score` and `mapi_raw_score`, and publishes the adjusted value as `mapi_actionability_score`. The v0.1 YAML config retains legacy semantics.
2. **Novelty was double-counted.** It was both the explicit `novelty` multiplier and the dynamic-weight `rarity` factor. The rarity factor was removed; novelty now appears only in the explicit score formula.
3. **Availability was over-counted.** Component confidence also modified dynamic weight. Availability was removed from dynamic weighting; it now affects the explicit confidence multiplier, independent evidence coverage, aggregate confidence, and composite data quality for distinct documented purposes.
4. **Cross-asset observations were forward-filled indefinitely.** Sector, benchmark, and regime code now use backward-only as-of alignment, same-UTC-date session protection, maximum staleness, frequency compatibility, and zero component confidence for invalid matches.
5. **Redundancy correlation included the current event.** Rolling correlation is now shifted by one bar, configurable minimum observations are enforced, and tests cover symmetry, order invariance, negative correlation, zero variance, and unavailable components.
6. **Backtest terminology implied a portfolio simulation.** The engine is now explicitly `run_event_study`, non-overlapping by default, and returns warnings and event-metric aliases. `run_backtest` remains a compatibility wrapper. Portfolio metrics have a separate reserved namespace.
7. **Evidence coverage was not independently visible.** Outputs now separate `ohlcv_quality_score`, `evidence_coverage_score`, composite `data_quality_score`, and aggregate `mapi_confidence`.
8. **Corporate-action assumptions were implicit.** Adjusted-price requirements and split-like discontinuity warnings were added.
9. **Negative controls were incomplete.** Sector-relative strength, equal-weight components, shuffled MAPI, isolated stock-sector residual, isolated volatility anomaly, and ten-seed random distributions were added.
10. **Calibration fitting discipline was unenforced.** The calibrator records its fit period and refuses `dataset_role="test"`.

## Scoring concept audit after fixes

| Concept | Component confidence | Dynamic weight | Explicit score multiplier | Aggregate confidence / quality |
| --- | --- | --- | --- | --- |
| Data availability | Yes | No | Through confidence | Evidence coverage and confidence |
| Data freshness | Cross-asset validity | Yes for stock bar spacing | No | Through invalid-source confidence |
| Reliability | No | Yes, static configured prior | No | No |
| Liquidity | No | Yes, trailing relative proxy | No | No |
| Persistence | No | Yes | No | State/age metadata only |
| Novelty | No | No | Yes | No |
| Cross-sectional rarity | Not available in Phase 1 | No | No | No |
| Redundancy | No | No | Yes, lagged penalty | Independent evidence coverage |

Redundancy appears in both effective score and capacity so it changes relative component contribution rather than mechanically shrinking every normalized score. It also lowers evidence coverage because correlated evidence is less independent. Family-level caps were considered but not added: Phase 1 currently has one enabled detector per information family. Caps should be added before multiple detectors from one family are enabled.

## Checked non-issues

- Historical z-score means and standard deviations are shifted by one bar.
- Rolling beta, stock mean, and sector mean are shifted before residual calculation.
- Prior highs/lows and prior momentum extrema exclude the current bar.
- Novelty percentiles include only the current and earlier observations; they do not use future rows.
- Liquidity and regime features are point-in-time rolling features.
- `already_realized_score` is the rolling percentile rank of the absolute trailing return ending at the signal bar. It contains no forward return.
- Missing components have zero confidence and contribute no scoring capacity; zero denominator returns zero score and confidence.
- Direction can cancel to neutral while anomaly intensity remains high.
- Score and confidence ranges remain finite under constant prices, zero volume, short history, NaN, and infinity inputs.
- Fixed score and calibration buckets are not estimated from test outcomes. Fitted calibration explicitly rejects a test role.

Exact realization formula for horizon return window `h` and rolling window `w`:

```text
trailing_abs_return[t] = abs(close[t] / close[t-h] - 1)
already_realized_score[t] = percentile_rank(
    trailing_abs_return[t],
    trailing_abs_return[t-w+1 : t+1]
)
actionability[t] = raw_anomaly[t] * (1 - penalty * already_realized_score[t])
```

Every term ends at `t`; forward labels are stored only in the research namespace.

## Compatibility changes

- Default code and CLI configuration moved to `mapi_v0.2`.
- v0.2 changes `mapi_score` semantics from realization-adjusted opportunity to pure anomaly intensity.
- Additive fields: `mapi_raw_score`, `mapi_actionability_score`, `ohlcv_quality_score`, and `evidence_coverage_score`.
- `configs/mapi_v0_1.yaml` preserves the old `mapi_score` semantics while still exposing the additive fields.
- Legacy event metric names remain, but warnings and explicit event aliases are now returned.
- `market_breadth.py` is a compatibility import; the implementation lives in `market_regime.py` because Phase 1 uses a broad index, not true breadth data.

## Remaining unproven assumptions

- No Python 3.12 runtime is installed in the current environment; verification was completed on Python 3.13.1 only.
- UTC dates are an approximation for trading sessions, not an exchange calendar.
- Relative liquidity is a within-symbol proxy, not cross-sectional capacity.
- Reliability priors remain neutral and are not learned from out-of-sample results.
- Online novelty may use earlier observations from the chronological test stream, which is point-in-time valid but differs from a frozen train-only normalization policy.
- No portfolio capital, sizing, borrow, capacity, exposure, or realistic order-book spread model exists.
- Split detection is heuristic and cannot replace vendor corporate-action metadata.
- Family-level caps are deferred until more than one detector per family is enabled.
- The additional controls and metrics have not demonstrated economic value on an untouched real-market dataset.
