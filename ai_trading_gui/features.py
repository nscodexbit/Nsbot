from __future__ import annotations

import numpy as np
import pandas as pd

from ai_trading_gui.utils import safe_div


def _wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def _hma(series: pd.Series, period: int) -> pd.Series:
    half = max(period // 2, 1)
    sqrt_p = max(int(np.sqrt(period)), 1)
    return _wma(2 * _wma(series, half) - _wma(series, period), sqrt_p)


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = safe_div(up.ewm(alpha=1 / period, adjust=False).mean(), down.ewm(alpha=1 / period, adjust=False).mean())
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _zscore(x: pd.Series, window: int) -> pd.Series:
    roll = x.rolling(window)
    return safe_div(x - roll.mean(), roll.std())


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    feat["log_return_1"] = np.log(close / close.shift(1))
    feat["hl_spread"] = safe_div(high - low, close)
    feat["oc_spread"] = safe_div(df["Close"] - df["Open"], df["Open"])

    ma_periods = list(range(5, 205, 5))

    for p in ma_periods:
        sma = close.rolling(p).mean()
        ema = _ema(close, p)
        wma = _wma(close, p)
        dema = 2 * ema - _ema(ema, p)
        tema = 3 * (ema - _ema(ema, p)) + _ema(_ema(ema, p), p)
        hma = _hma(close, p)

        feat[f"sma_{p}"] = sma
        feat[f"ema_{p}"] = ema
        feat[f"wma_{p}"] = wma
        feat[f"dema_{p}"] = dema
        feat[f"tema_{p}"] = tema
        feat[f"hma_{p}"] = hma
        feat[f"ma_dist_sma_{p}"] = safe_div(close - sma, sma)
        feat[f"ma_dist_ema_{p}"] = safe_div(close - ema, ema)
        feat[f"slope_ema_{p}"] = safe_div(ema - ema.shift(3), close.shift(3))

    for fast in [5, 10, 20, 30]:
        for slow in [40, 60, 100, 150, 200]:
            feat[f"cross_strength_{fast}_{slow}"] = safe_div(
                feat[f"ema_{fast}"] - feat[f"ema_{slow}"], feat[f"ema_{slow}"]
            )

    for p in [5, 7, 10, 14, 21, 30, 50, 100]:
        feat[f"rsi_{p}"] = _rsi(close, p)
        roll_min = low.rolling(p).min()
        roll_max = high.rolling(p).max()
        feat[f"stoch_k_{p}"] = 100 * safe_div(close - roll_min, roll_max - roll_min)
        feat[f"roc_{p}"] = close.pct_change(p)
        tp = (high + low + close) / 3
        md = (tp - tp.rolling(p).mean()).abs().rolling(p).mean()
        feat[f"cci_{p}"] = safe_div(tp - tp.rolling(p).mean(), 0.015 * md)
        feat[f"williamsr_{p}"] = -100 * safe_div(roll_max - close, roll_max - roll_min)
        feat[f"mom_{p}"] = close.diff(p)
        feat[f"mom_diff_{p}"] = feat[f"mom_{p}"].diff()

    for p in [10, 14, 20, 30, 50, 100]:
        atr = _atr(df, p)
        feat[f"atr_{p}"] = atr
        mean = close.rolling(p).mean()
        std = close.rolling(p).std()
        upper = mean + 2 * std
        lower = mean - 2 * std
        feat[f"bb_width_{p}"] = safe_div(upper - lower, mean)
        feat[f"bb_pctb_{p}"] = safe_div(close - lower, upper - lower)
        feat[f"rolling_std_{p}"] = std
        feat[f"vol_rank_{p}"] = atr.rolling(p).rank(pct=True)
        feat[f"donchian_{p}"] = safe_div(close - low.rolling(p).min(), high.rolling(p).max() - low.rolling(p).min())
        ema_mid = _ema(close, p)
        feat[f"keltner_pos_{p}"] = safe_div(close - (ema_mid - 2 * atr), (ema_mid + 2 * atr) - (ema_mid - 2 * atr))

    obv = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    feat["obv"] = obv
    typical_price = (high + low + close) / 3
    feat["vwap"] = safe_div((typical_price * volume).cumsum(), volume.cumsum())

    for p in [5, 10, 20, 30, 50, 100, 150, 200]:
        vol_ma = volume.rolling(p).mean()
        feat[f"vol_ma_{p}"] = vol_ma
        feat[f"vol_spike_{p}"] = safe_div(volume, vol_ma)
        feat[f"vol_roc_{p}"] = volume.pct_change(p)

    for p in [5, 10, 20, 30, 50, 100]:
        ret = close.pct_change()
        feat[f"lag_ret_{p}"] = ret.shift(p)
        feat[f"roll_skew_{p}"] = ret.rolling(p).skew()
        feat[f"roll_kurt_{p}"] = ret.rolling(p).kurt()
        feat[f"zscore_{p}"] = _zscore(close, p)
        feat[f"autocorr_{p}"] = ret.rolling(p).apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False)
        feat[f"entropy_{p}"] = ret.rolling(p).apply(
            lambda x: -np.sum(np.histogram(x, bins=5, density=True)[0] * np.log(np.histogram(x, bins=5, density=True)[0] + 1e-9)),
            raw=False,
        )
        feat[f"fractal_proxy_{p}"] = safe_div((high.rolling(p).max() - low.rolling(p).min()), close.rolling(p).std())

    feat["rsi_slope_14"] = feat["rsi_14"].diff()
    feat["vol_adj_ret"] = safe_div(close.pct_change(), feat["atr_14"])
    feat["mom_vol_spread"] = feat["roc_10"] - feat["rolling_std_10"]
    feat["ema_sma_ratio_20"] = safe_div(feat["ema_20"], feat["sma_20"])
    feat["obv_slope"] = obv.diff(5)

    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.shift(1)  # anti-leakage
    return feat


def build_dataset(df: pd.DataFrame, horizon: int, return_threshold: float) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    features = generate_features(df)
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1.0
    y = (future_return > return_threshold).astype(int)

    dataset = features.join(y.rename("target")).dropna()
    x = dataset.drop(columns=["target"])
    y = dataset["target"]

    aligned_close = df.loc[dataset.index, "Close"]
    return x, y, aligned_close
