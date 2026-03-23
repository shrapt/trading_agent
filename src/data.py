"""
Data fetching and technical-indicator computation for XAUUSD.

Indicators computed per candle:
  open, high, low, close, volume  (raw OHLCV)
  rsi_14                          (momentum)
  macd, macd_signal               (trend)
  bb_upper, bb_lower              (volatility bands)
  atr_14                          (volatility)

That gives 10 features × WINDOW_SIZE = STATE_DIM.
"""

import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Indicator helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append 10 feature columns to an OHLCV DataFrame."""
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # RSI-14
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD (12/26/9)
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = _ema(df["macd"], 9)

    # Bollinger Bands (20, 2σ)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20

    # ATR-14
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.ewm(com=13, adjust=False).mean()

    return df


def fetch_data(symbol: str, interval: str, lookback_days: int,
               cache_path: str) -> pd.DataFrame:
    """
    Download OHLCV data via yfinance with local CSV cache.
    Falls back to synthetic data generation if yfinance is unavailable.
    """
    if os.path.exists(cache_path):
        logger.info("Loading cached data from %s", cache_path)
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        df = _download(symbol, interval, lookback_days)
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        df.to_csv(cache_path)
        logger.info("Saved data cache to %s", cache_path)

    df = add_indicators(df)
    df.dropna(inplace=True)
    logger.info("Dataset: %d candles, %d features", len(df), len(df.columns))
    return df


def _download(symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
    try:
        import yfinance as yf
        from datetime import datetime, timedelta
        end   = datetime.utcnow()
        start = end - timedelta(days=lookback_days)
        ticker = yf.Ticker(symbol)
        raw = ticker.history(start=start, end=end, interval=interval)
        if raw.empty:
            raise ValueError("yfinance returned empty DataFrame")
        raw.columns = [c.lower() for c in raw.columns]
        raw = raw[["open", "high", "low", "close", "volume"]]
        logger.info("Downloaded %d rows from yfinance", len(raw))
        return raw
    except Exception as exc:
        logger.warning("yfinance failed (%s) – using synthetic data", exc)
        return _synthetic_data(lookback_days)


def _synthetic_data(days: int) -> pd.DataFrame:
    """
    Generate realistic-looking synthetic XAUUSD price data using a
    geometric Brownian motion model seeded for reproducibility.
    """
    rng   = np.random.default_rng(42)
    n     = days * 24          # hourly candles
    dt    = 1 / (365 * 24)
    mu    = 0.05               # annual drift  (gold tends upward long-term)
    sigma = 0.15               # annual volatility

    # GBM log-returns
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    closes = 1950.0 * np.exp(np.cumsum(log_returns))  # start near ~$1950/oz

    # Synthesise OHLV from close
    noise = rng.uniform(0.0005, 0.003, size=n)
    opens  = closes * rng.uniform(0.998, 1.002, size=n)
    highs  = np.maximum(opens, closes) * (1 + noise)
    lows   = np.minimum(opens, closes) * (1 - noise)
    volume = rng.integers(1_000, 10_000, size=n).astype(float)

    idx = pd.date_range("2022-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows,
         "close": closes, "volume": volume},
        index=idx,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation helper used by the environment
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = ["open", "high", "low", "close", "volume",
                "rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower", "atr_14"]


def normalise_window(window: pd.DataFrame) -> np.ndarray:
    """
    Return a 1-D float32 array of shape (WINDOW_SIZE * len(FEATURE_COLS),).

    Price/band columns are normalised by the first close in the window.
    RSI is divided by 100; MACD values by close; ATR by close; volume by max.
    """
    arr = window[FEATURE_COLS].values.astype(np.float32)
    base_price = arr[0, 3] + 1e-9   # first close in the window

    # price-like columns: open, high, low, close, bb_upper, bb_lower
    for col in [0, 1, 2, 3, 8, 9]:
        arr[:, col] /= base_price

    # volume: normalise by its own max in the window
    vol_max = arr[:, 4].max() + 1e-9
    arr[:, 4] /= vol_max

    # RSI → [0, 1]
    arr[:, 5] /= 100.0

    # MACD, MACD signal → relative to price
    arr[:, 6]  /= base_price
    arr[:, 7]  /= base_price

    # ATR → relative to price
    arr[:, 10] /= base_price

    return arr.flatten()
