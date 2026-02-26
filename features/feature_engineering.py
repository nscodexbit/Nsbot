from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.ewm(alpha=1 / period, adjust=False).mean() / (down.ewm(alpha=1 / period, adjust=False).mean() + 1e-9)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = atr(df, period=1)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / (tr.ewm(alpha=1 / period, adjust=False).mean() + 1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / (tr.ewm(alpha=1 / period, adjust=False).mean() + 1e-9)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def linear_regression_slope(series: pd.Series, window: int = 20) -> pd.Series:
    x = np.arange(window)

    def _slope(values: np.ndarray) -> float:
        y = values
        x_mean = x.mean()
        y_mean = y.mean()
        return float(np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-9))

    return series.rolling(window).apply(_slope, raw=True)


def market_structure(close: pd.Series, window: int = 20) -> pd.Series:
    rolling_high = close.rolling(window).max()
    rolling_low = close.rolling(window).min()
    out = pd.Series(0, index=close.index, dtype="float32")
    out[close >= rolling_high.shift(1)] = 1
    out[close <= rolling_low.shift(1)] = -1
    return out


def build_features(df: pd.DataFrame, btc_close: pd.Series | None = None) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["ret_1"] = df["close"].pct_change()
    feat["ema20"] = ema(df["close"], 20)
    feat["ema50"] = ema(df["close"], 50)
    feat["ema200"] = ema(df["close"], 200)
    feat["ema20_50_spread"] = (feat["ema20"] - feat["ema50"]) / (df["close"] + 1e-9)
    feat["rsi14"] = rsi(df["close"], 14)
    feat["atr14"] = atr(df, 14)
    feat["adx14"] = adx(df, 14)

    vwap_num = (df["close"] * df["volume"]).rolling(96).sum()
    vwap_den = df["volume"].rolling(96).sum() + 1e-9
    feat["vwap"] = vwap_num / vwap_den

    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    feat["bb_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-9)
    feat["bb_squeeze"] = (feat["bb_width"] < feat["bb_width"].rolling(100).quantile(0.2)).astype("int8")

    feat["lr_slope_20"] = linear_regression_slope(df["close"], 20)
    momentum = df["close"].pct_change(5)
    feat["mom_accel"] = momentum.diff()
    feat["atr_pctile"] = feat["atr14"].rolling(500, min_periods=100).rank(pct=True)
    feat["regime_high_vol"] = (feat["atr_pctile"] > 0.7).astype("int8")
    feat["market_structure"] = market_structure(df["close"])

    rolling_ret = df["close"].pct_change()
    feat["rolling_sharpe"] = np.sqrt(252 * 24 * 12) * rolling_ret.rolling(288, min_periods=50).mean() / (rolling_ret.rolling(288, min_periods=50).std() + 1e-9)

    if btc_close is not None:
        asset_ret = df["close"].pct_change()
        btc_ret = btc_close.reindex(df.index).pct_change()
        feat["btc_corr"] = asset_ret.rolling(288).corr(btc_ret)

    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.shift(1)  # strict lag to prevent leakage
    return feat.dropna().astype("float32")


def correlation_filter(df: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if (upper[col] > threshold).any()]
    return [c for c in df.columns if c not in drop_cols]
