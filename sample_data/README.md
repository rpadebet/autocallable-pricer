# sample_data/ — SPX Options Snapshot Library

## Purpose
Pre-collected SPX options chain snapshots for the AutoCallable Analytics Platform demo.
No live API calls happen at runtime — the app loads from these files.

## File Naming Convention
`spx_options_YYYYMMDD_HHMM.csv`

Example: `spx_options_20260609_1200.csv` = June 9, 2026, collected at 12:00 PM ET.

## Session Labels (UI Display)
| Hour (ET) | Label |
|-----------|-------|
| < 11:00   | Market Open |
| 11:00–14:00 | Midday |
| > 14:00   | Near Close |

## Snapshot Library

| File | Date | Time (ET) | Session | Quality Score | Status |
|------|------|-----------|---------|---------------|--------|
| *(none yet)* | — | — | — | — | 🔲 Collect |

## Collection Log

### Friday Jun 6, 2026 (retroactive check)
- yfinance retroactive: [TBD — run collect_snapshot.py on Sat Jun 7 morning]
- CBOE DataShop check: [TBD]
- Outcome: [TBD]

### Monday Jun 9, 2026
- 09:45 ET (open): [TBD]
- 12:00 ET (midday): [TBD]
- 15:45 ET (close): [TBD]
- Best kept: [TBD]

### Tuesday Jun 10, 2026
- [TBD]

### Wednesday Jun 11, 2026
- [TBD]

## CSV Column Schema

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| contractSymbol | str | raw | yfinance option ticker |
| strike | float | raw | Strike price |
| expiry | date | raw | Expiration date |
| lastPrice | float | raw | Last traded price |
| bid | float | raw | Best bid |
| ask | float | raw | Best offer |
| volume | int | raw | Day's volume |
| openInterest | int | raw | Open interest |
| impliedVolatility | float | raw | yfinance IV (Black-Scholes) |
| inTheMoney | bool | raw | Whether option is ITM |
| spot | float | raw | SPX spot at collection time |
| rfr | float | raw | Risk-free rate (^IRX / 100) |
| optionType | str | added | "call" or "put" |
| moneyness | float | computed | strike / spot |
| ttm_years | float | computed | (expiry - collection_date).days / 365 |
| mid | float | computed | (bid + ask) / 2 |
