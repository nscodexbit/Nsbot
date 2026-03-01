from __future__ import annotations

import pandas as pd
import yfinance as yf


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def load_market_data(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """Download and clean OHLCV data from yfinance."""
    try:
        raw = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise ValueError(f"Failed to download data for {ticker}: {exc}") from exc

    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    df = raw[REQUIRED_COLUMNS].copy()
    df = df.sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="last")]

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = df["Volume"].fillna(0.0)

    if len(df) < 250:
        raise ValueError("Not enough candles after cleaning; please widen date range.")

    return df
