# Stock AI Scout: Market Anomaly Pressure Index

MAPI is an experimental, modular stock-market anomaly signal. Version 0.2 separates pure anomaly intensity (`mapi_score` and `mapi_raw_score`, 0-100), realization-adjusted opportunity (`mapi_actionability_score`, 0-100), directional pressure (`mapi_direction`, -1 to 1), and evidence quality (`mapi_confidence`, 0-1). It is a research instrument, not a profitability claim or trading recommendation.

This workspace did not contain an existing Stock AI Scout codebase or git history, so the implementation is a standalone Python package with explicit provider interfaces. It can be integrated behind the project's eventual data and signal APIs without coupling the indicator to a vendor.

## What is implemented

- Multi-horizon signals: `intraday`, `short_term`, `swing`, and `position`.
- Phase 1 components: price-volume divergence, stock-sector divergence, momentum disagreement, volatility anomaly, and market-regime divergence.
- Optional placeholders for news reaction, fundamentals, options, and sentiment. Missing feeds return zero confidence instead of fabricated values.
- Dynamic regime, freshness, reliability, liquidity, and persistence weights; explicit novelty and correlation-based redundancy penalties; anomaly state/age/trend; confirmation count; and a separate already-realized actionability adjustment.
- Deterministic machine reasons and human summaries.
- CSV providers, OHLCV validation, backtesting with delayed entry and costs, score buckets, baselines, ablation, calibration, and chronological splits.

## Signal formula

For component `i`:

```text
effective_i = strength_i * confidence_i * weight_i * novelty_i * redundancy_penalty_i
capacity_i  = confidence_i * weight_i * redundancy_penalty_i
raw_score   = 100 * sum(effective_i) / sum(capacity_i)
actionability_score = raw_score * (1 - already_realized_penalty * already_realized_score)
```

In v0.2, `mapi_score == mapi_raw_score`; `mapi_actionability_score` carries the realization adjustment. The legacy `configs/mapi_v0_1.yaml` keeps the old `mapi_score == mapi_actionability_score` behavior. `mapi_direction` is the effective-score-weighted mean of component directions. `mapi_confidence` combines conditional component confidence, independent evidence coverage, and OHLCV quality. All public outputs are clamped to their documented ranges.

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
python examples\run_mapi.py --symbol AAPL --prices data\AAPL.csv --sector data\XLK.csv --benchmark data\SPY.csv --config configs\mapi_v0_2.yaml --output out\mapi_aapl.json
```

```powershell
python examples\run_backtest.py --symbol AAPL --prices data\AAPL.csv --sector data\XLK.csv --benchmark data\SPY.csv --config configs\mapi_v0_2.yaml --horizon short_term --output out\event_study_aapl.json
```

Programmatic use:

```python
from mapi.config import load_config
from mapi.scoring import calculate_latest_mapi

signals = calculate_latest_mapi(
    symbol="AAPL",
    price_frame=stock_ohlcv,
    sector_frame=sector_ohlcv,
    benchmark_frame=benchmark_ohlcv,
    config=load_config("configs/mapi_v0_2.yaml"),
)
```

## No-lookahead policy

Component baselines use lagged rolling statistics through `historical_zscore`; beta, expected-return estimates, and prior breakouts are shifted before the current bar is scored. Future returns exist only in `mapi.research.labels` and are consumed after signal generation. The no-lookahead test changes all rows after a cutoff and asserts that earlier signals remain identical.

Signals are assumed known at bar close. The event-study engine defaults to entry one bar later, rejects overlapping events, and applies one round-trip transaction cost, spread, and slippage estimate. Its Sharpe, drawdown, and profit factor fields describe event-return sequences and are not realizable portfolio metrics. See [execution semantics](docs/execution_semantics.md).

## Minimum history

The defaults begin partial estimates at 8, 20, 40, and 80 bars by horizon. Full rolling context needs 20, 60, 120, and 252 bars respectively. Use substantially more history for research; short warm-up output carries lower confidence.

## Verification

```powershell
python -m unittest discover -s tests -v
```

The tests cover normalization, stale and mixed-frequency cross-assets, confidence coverage, component-level future mutation, already-realized leakage, redundancy invariants, hand-calculated long/short execution, transaction costs, score-bucket isolation, negative controls, DST, corporate-action warnings, JSON schema, and end-to-end CLI reproducibility.

## Known limitations

- Phase 1 features are heuristic and have not demonstrated out-of-sample edge.
- Session protection currently uses UTC calendar dates, not an exchange-calendar service; overnight markets need a calendar-aware Phase 2 adapter.
- The event-study engine is not a capital-aware portfolio simulator, despite preserving legacy metric field names for compatibility.
- Intraday defaults are bar-count based; session boundaries and overnight gaps need a calendar-aware provider.
- Optional news, fundamental, options, and sentiment components are interfaces only until point-in-time feeds are connected.

Before parameter tuning, freeze an untouched test period and use [the research report template](docs/research_report_template.md) to record assumptions and negative controls.

The targeted pre-Phase-2 findings and compatibility notes are in [the audit report](docs/audit_report.md).

## Phase 2 candidates

- Connect point-in-time news, estimates, fundamentals, options, and sentiment providers through the existing protocols.
- Add exchange-calendar-aware intraday alignment, data freshness limits, corporate actions, and delisted-symbol universes.
- Estimate component reliability only from walk-forward out-of-sample results, then compare the learned policy with the transparent rule-based policy.
- Add portfolio-level exposure, capacity, borrow cost, spread, and overlapping-position simulation.
- Expand breadth inputs with advance/decline, volatility indices, rates, credit, currency, and liquidity series.
