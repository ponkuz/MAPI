# OHLCV CSV contract

MAPI accepts one CSV file per symbol. These columns are required:

| Column | Type | Meaning |
| --- | --- | --- |
| `timestamp` | ISO-8601 string | Bar close timestamp; converted to UTC |
| `open` | number | Bar open price |
| `high` | number | Bar high price |
| `low` | number | Bar low price |
| `close` | number | Bar close price |
| `volume` | number | Non-negative traded volume |

Example:

```csv
timestamp,open,high,low,close,volume
2025-01-02T21:00:00Z,100.00,102.10,99.70,101.40,1250000
2025-01-03T21:00:00Z,101.50,103.00,100.90,102.20,1175000
```

Rows may arrive unsorted. The loader sorts by timestamp, keeps the last duplicate, and converts numeric fields with invalid values to missing data. Stock, sector ETF, and broad benchmark files use the same schema and should have matching bar frequency.

All OHLC prices must share one split- and dividend-adjusted basis, and volume should be split-adjusted where the vendor supports it. The validator emits a strong warning for close-to-close jumps of 1.8x or greater in either direction. This is a diagnostic, not a complete corporate-action engine.

Cross-asset data is matched backward only. A source observation is rejected when it comes from another UTC date, exceeds the configured staleness limit, or has a lower sampling frequency than the target beyond the configured tolerance.

For daily default settings, provide at least 80 observations for partial output and at least 252 for the full position horizon. Longer histories produce more stable rolling estimates.
