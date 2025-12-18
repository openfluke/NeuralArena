# Stock Price Diffusion Model

Trains a diffusion model to predict stock price movements using Alpha Vantage API data.

## What It Does

- **Downloads stock data** — Fetches daily OHLCV data from Alpha Vantage API
- **Discretizes movements** — Converts price changes to tokens (up/down/flat)
- **Trains diffusion model** — Learns patterns from historical data
- **Predicts future** — Generates 7-day and 90-day predictions

## Token Classes

| Token | Meaning |
|-------|---------|
| 0 | Price down |
| 1 | Price flat |
| 2 | Price up |

## Data Source

- **API**: Alpha Vantage
- **Endpoint**: TIME_SERIES_DAILY
- **Fields**: Open, High, Low, Close, Volume

## Predictions

1. **Weekly** — 7-day prediction
2. **Quarterly** — 90-day prediction

## Running

```bash
# Set your Alpha Vantage API key
export ALPHA_VANTAGE_KEY=your_key_here
go run .
```

Requires valid Alpha Vantage API key.
