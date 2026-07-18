# Stock AI Scout: Market Anomaly Pressure Index

MAPI is an experimental, modular stock-market anomaly signal. Version 0.3.4 separates pure anomaly intensity (`mapi_intensity_score`, 0-100), recurrence retention (`mapi_novelty_score`, 0-100), recurrence-adjusted alerting (`mapi_alert_score`, 0-100), realization-adjusted opportunity (`mapi_actionability_score`, 0-100), forecast direction (`mapi_forecast_direction`, -1 to 1), observed pressure, and evidence quality. It is a research instrument, not a profitability claim or trading recommendation.

This workspace did not contain an existing Stock AI Scout codebase or git history, so the implementation is a standalone Python package with explicit provider interfaces. It can be integrated behind the project's eventual data and signal APIs without coupling the indicator to a vendor.

## What is implemented

- Multi-horizon signals: `intraday`, `short_term`, `swing`, and `position`.
- Phase 1 components: price-volume divergence, stock-sector divergence, momentum disagreement, volatility anomaly, and market-regime divergence.
- Optional placeholders for news reaction, fundamentals, options, and sentiment. Missing feeds return zero confidence instead of fabricated values.
- Dynamic regime, freshness, reliability, liquidity, and persistence weights; prior-only event novelty with recurrence decay; correlation-based redundancy penalties; anomaly state/age/trend; confirmation count; and a directional realization adjustment measured from first detection.
- Deterministic machine reasons and human summaries.
- CSV providers, OHLCV validation, backtesting with delayed entry and costs, score buckets, baselines, ablation, calibration, and chronological splits.

## Signal formula

For component `i`:

```text
historical_extremeness_i = prior_only_percentile(strength_i)
novelty_i   = historical_extremeness_i * (1 - near_identical_recurrence_i)
intensity_i = strength_i * confidence_i * weight_i * redundancy_penalty_i
alert_i     = intensity_i * novelty_i
capacity_i  = confidence_i * weight_i * redundancy_penalty_i
directional_alert_i = alert_i * directional_evidence_strength_i
intensity_score = 100 * sum(intensity_i) / sum(capacity_i)
novelty_score   = 100 * sum(alert_i) / sum(intensity_i)
alert_score     = 100 * sum(alert_i) / sum(capacity_i)
forecast_direction = sum(forecast_direction_i * directional_alert_i) / sum(directional_alert_i)
actionability_score = alert_score * (1 - directional_realization_penalty * directional_realization_score)
```

`mapi_raw_score` is a compatibility alias for intensity. `mapi_score` is selected by `score_semantics`; the v0.3.4 default is intensity. Anomaly state, age, trend, persistence, and confirmation use intensity and therefore are not invalidated solely by recurrence. `dominant_intensity_anomalies` explains that state, while `dominant_alert_anomalies` contains only novelty-retaining contributors; legacy `dominant_anomalies` aliases intensity reasons. `mapi_direction` aliases the explicit forecast-direction aggregate; `mapi_observed_pressure` remains separate. Because direction is weighted by recurrence-adjusted `alert_i`, `mapi_forecast_direction` is a novelty-adjusted alert direction rather than an intensity direction. Each enabled component declares a deterministic subtype semantic and `directional_evidence_strength`. Forecast direction and directional evidence must be jointly zero or jointly nonzero beyond a technical `1e-12` epsilon. Zero means no directional view, so that component does not enter the direction denominator but still contributes to anomaly intensity and alerting. MAPI v0.3.4 does not model positive evidence for a near-zero future return as a separate forecast class. Near-zero subtype forecasts are explicitly converted to no-view directional evidence. `recent_move_extremeness` is descriptive only. `directional_realization_score` is positive only when movement since `anomaly_first_detected_at` aligns with forecast direction. Old YAML files may select legacy public-score behavior, but they cannot relabel the implementation as an older algorithm revision. Output includes implementation, algorithm, data-contract, and config-fingerprint identifiers.

## Data flow

```text
OHLCV CSV/provider
  -> normalization and quality checks
  -> horizon and market regime
  -> independent anomaly components
  -> redundancy and dynamic-weight adjustment
  -> score, direction, confidence, state, reasons
  -> JSON output or non-overlapping event study
```

Stock OHLCV is required. Sector ETF and broad benchmark OHLCV are optional, but without them stock-sector and market-regime evidence is reduced or inactive. Cross-asset bars are backward-matched with staleness, session-date, and frequency checks. See [the CSV contract](docs/example_csv_schema.md) and [data requirements](docs/data_requirements.md).

## Run it

Python 3.12+ with NumPy and pandas is required. PyYAML is optional because the bundled configuration uses the built-in simple YAML reader.

```powershell
python examples\run_mapi.py --symbol AAPL --prices data\AAPL.csv --sector data\XLK.csv --benchmark data\SPY.csv --config configs\mapi_v0_3.yaml --output out\mapi_aapl.json
```

```powershell
python examples\run_backtest.py --symbol AAPL --prices data\AAPL.csv --sector data\XLK.csv --benchmark data\SPY.csv --config configs\mapi_v0_3.yaml --horizon short_term --output out\event_study_aapl.json
```

The event-study CLI uses `research_score_column` from configuration (`mapi_actionability_score` by default). Override it with `--score-column`; JSON records the score and direction columns. One chronological split is created for the entire report. All metrics use the test mask, while baseline and ablation thresholds are fitted only on the fit mask. Configured confidence, data-quality, and frequency-compatibility gates apply to both matching and event selection.

For backward compatibility, direct `run_event_study()` calls remain permissive unless evidence gates are passed explicitly. `run_configured_event_study(signals, prices, config, horizon_name, ...)` applies the CLI-equivalent event eligibility and configured score, horizon, cost, bootstrap, quality, frequency, and forecast-direction policy. It does not create the CLI chronological holdout split; programmatic callers must pass `evaluation_mask` for holdout-only evaluation.

Programmatic use:

```python
from mapi.config import load_config
from mapi.scoring import calculate_latest_mapi

signals = calculate_latest_mapi(
    symbol="AAPL",
    price_frame=stock_ohlcv,
    sector_frame=sector_ohlcv,
    benchmark_frame=benchmark_ohlcv,
    config=load_config("configs/mapi_v0_3.yaml"),
)
```

## No-lookahead policy

Component baselines use lagged rolling statistics through `historical_zscore`; novelty compares the current event with prior-only history; beta, expected-return estimates, and prior breakouts are shifted before the current bar is scored. Future returns exist only in `mapi.research.labels` and are consumed after signal generation. The no-lookahead test changes all rows after a cutoff and asserts that earlier signals remain identical.

Signals are assumed known at bar close. The event-study engine defaults to entry one bar later, rejects overlapping events, and applies one round-trip transaction cost, spread, and slippage estimate. It reports unannualized event-return ratios and bootstrap confidence intervals. Intrabar MFE, MAE, and breakout use future high/low while realized return stays close-to-close. Legacy Sharpe, precision/recall, MFE/MAE, and breakout names are compatibility aliases with warnings. See [execution semantics](docs/execution_semantics.md).

## Minimum history

The defaults begin partial estimates at 8, 20, 40, and 80 bars by horizon. Full rolling context needs 20, 60, 120, and 252 bars respectively. Use substantially more history for research; short warm-up output carries lower confidence.

## Verification

```powershell
python -m unittest discover -s tests -v
```

The tests cover novelty recurrence, score separation, normalization, stale and mixed-frequency cross-assets, regime source, horizon validation, confidence coverage, component-level future mutation, directional realization, redundancy invariants, hand-calculated long/short OHLC paths, transaction costs, score-bucket isolation, matched-frequency controls, ablation horizons, configuration validation, JSON schema, and end-to-end CLI reproducibility. The CI workflow is configured to run both `unittest` and `pytest` on Python 3.12 and 3.13. A revision is described as cross-version verified only after that revision's complete matrix is green.

## Known limitations

- Phase 1 features are heuristic and have not demonstrated out-of-sample edge.
- Session protection currently uses UTC calendar dates, not an exchange-calendar service; overnight markets need a calendar-aware Phase 2 adapter.
- The event-study engine is not a capital-aware portfolio simulator, despite preserving legacy metric field names for compatibility.
- Horizon windows remain bar-count based, but each default now declares an expected bar frequency and warns or fails on a mismatch. Session boundaries and overnight gaps still need a calendar-aware provider.
- Optional news, fundamental, options, and sentiment components are interfaces only until point-in-time feeds are connected.

Before parameter tuning, freeze an untouched test period and use [the research report template](docs/research_report_template.md) to record assumptions and negative controls.

The targeted pre-Phase-2 findings and compatibility notes are in [the audit report](docs/audit_report.md).

## Phase 2 candidates

- Connect point-in-time news, estimates, fundamentals, options, and sentiment providers through the existing protocols.
- Add exchange-calendar-aware intraday alignment, data freshness limits, corporate actions, and delisted-symbol universes.
- Estimate component reliability only from walk-forward out-of-sample results, then compare the learned policy with the transparent rule-based policy.
- Add portfolio-level exposure, capacity, borrow cost, spread, and overlapping-position simulation.
- Expand breadth inputs with advance/decline, volatility indices, rates, credit, currency, and liquidity series.
