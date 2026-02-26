from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import requests
import websockets

LOGGER = logging.getLogger(__name__)

BINANCE_REST = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_WS = "wss://fstream.binance.com/ws"


@dataclass
class DataEngine:
    parquet_dir: Path
    zscore_threshold: float = 6.0
    rolling_norm_window: int = 288

    def __post_init__(self) -> None:
        self.parquet_dir.mkdir(parents=True, exist_ok=True)

    def fetch_binance_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        limit: int = 1500,
    ) -> pd.DataFrame:
        rows = []
        cursor = start_ms
        while cursor < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": limit,
            }
            response = requests.get(BINANCE_REST, params=params, timeout=30)
            response.raise_for_status()
            chunk = response.json()
            if not chunk:
                break
            rows.extend(chunk)
            cursor = int(chunk[-1][0]) + 1
            if len(chunk) < limit:
                break

        df = pd.DataFrame(
            rows,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )
        if df.empty:
            return df
        keep = ["open_time", "open", "high", "low", "close", "volume"]
        df = df[keep]
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.drop(columns=["open_time"]).set_index("timestamp").sort_index()
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype("float32")

        return self._clean(df)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[~df.index.duplicated(keep="last")].copy()
        returns = df["close"].pct_change()
        z = (returns - returns.rolling(500).mean()) / (returns.rolling(500).std() + 1e-9)
        df = df.loc[z.abs().fillna(0) < self.zscore_threshold]
        df = df.ffill().dropna()
        return df

    def resample_align(self, raw_5m: pd.DataFrame, tfs: Iterable[str]) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {"5m": raw_5m}
        for tf in tfs:
            if tf == "5m":
                continue
            agg = (
                raw_5m.resample(tf)
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
                .ffill()
                .dropna()
            )
            out[tf] = agg.astype("float32")

        common_index = out["5m"].index
        for tf, frame in out.items():
            if tf == "5m":
                continue
            aligned = frame.reindex(common_index, method="ffill")
            aligned.columns = [f"{c}_{tf}" for c in aligned.columns]
            out[tf] = aligned.astype("float32")
        return out

    def rolling_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        mean = df.rolling(self.rolling_norm_window).mean()
        std = df.rolling(self.rolling_norm_window).std().replace(0, np.nan)
        norm = (df - mean) / std
        return norm.astype("float32")

    def save_parquet(self, symbol: str, interval: str, df: pd.DataFrame) -> Path:
        path = self.parquet_dir / f"{symbol}_{interval}.parquet"
        df.to_parquet(path)
        return path

    async def stream_klines(self, symbol: str, interval: str, queue: asyncio.Queue) -> None:
        stream_name = f"{symbol.lower()}@kline_{interval}"
        payload = json.dumps({"method": "SUBSCRIBE", "params": [stream_name], "id": 1})

        while True:
            try:
                async with websockets.connect(BINANCE_WS, ping_interval=10, ping_timeout=30) as ws:
                    await ws.send(payload)
                    LOGGER.info("Connected to Binance WS stream: %s", stream_name)
                    async for message in ws:
                        data = json.loads(message)
                        if "k" in data:
                            await queue.put(data["k"])
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("WS disconnected (%s), reconnecting in 5s", exc)
                await asyncio.sleep(5)
