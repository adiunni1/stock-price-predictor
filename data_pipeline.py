"""
data_pipeline.py
----------------
Fetches AAPL historical data via yfinance, engineers technical indicator
features, creates a binary next-day movement label, and uploads the clean
dataset to S3 as a CSV.

Usage:
    python data_pipeline.py --ticker AAPL --bucket YOUR_S3_BUCKET
"""

import argparse
import io
import time
from datetime import datetime

import boto3
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


# ── Feature Engineering ────────────────────────────────────────────────────────

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Simple and exponential moving averages + price-to-MA ratios."""
    for window in [5, 10, 20, 50]:
        df[f"sma_{window}"] = df["Close"].rolling(window).mean()
        df[f"ema_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()
    # Price relative to moving averages (normalised signal)
    df["price_to_sma20"] = df["Close"] / df["sma_20"]
    df["price_to_sma50"] = df["Close"] / df["sma_50"]
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index — momentum oscillator (0–100)."""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD line, signal line, and histogram — trend/momentum indicator."""
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Bollinger Bands — price relative to volatility envelope."""
    sma = df["Close"].rolling(window).mean()
    std = df["Close"].rolling(window).std()
    df["bb_upper"] = sma + 2 * std
    df["bb_lower"] = sma - 2 * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma
    df["bb_position"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume ratio vs. 20-day average — distinguishes conviction moves."""
    df["volume_sma_20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_sma_20"]
    return df


def add_lag_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Daily returns and lagged returns for the past 1–5 days."""
    df["daily_return"] = df["Close"].pct_change()
    for lag in range(1, 6):
        df[f"return_lag_{lag}"] = df["daily_return"].shift(lag)
    return df


def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling 10-day realised volatility."""
    df["volatility_10"] = df["daily_return"].rolling(10).std()
    return df


def add_market_features(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Market-level context: VIX close (fear gauge) and SPY daily return (broad market move).

    Fetched separately and merged by date rather than engineered from the
    ticker's own OHLCV, since these describe the overall market environment.
    """
    market = yf.download(["^VIX", "SPY"], start=start, end=end, auto_adjust=True, progress=False)

    vix_close = market["Close"]["^VIX"]
    spy_return = market["Close"]["SPY"].pct_change()

    df["vix_close"] = vix_close.reindex(df.index)
    df["spy_return"] = spy_return.reindex(df.index)
    df[["vix_close", "spy_return"]] = df[["vix_close", "spy_return"]].ffill()
    return df


_vader = None


def _get_vader() -> SentimentIntensityAnalyzer:
    """Lazily load the VADER lexicon (downloads it once if not already cached)."""
    global _vader
    if _vader is None:
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        _vader = SentimentIntensityAnalyzer()
    return _vader


def add_sentiment_features(df: pd.DataFrame, ticker: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """Daily news sentiment: fetch headlines from Finnhub, score each with VADER,
    and merge the daily mean compound score onto df as `sentiment_score`.

    Finnhub's company-news endpoint only returns roughly the trailing year of
    headlines even on paid tiers for older history, and requests are chunked
    into ~90-day windows to stay under the response size/rate limits — so
    older rows in a multi-year `df` will often have no matching news and fall
    back to 0.0 (neutral) via the forward-fill below.
    """
    analyzer = _get_vader()

    scores_by_date = {}
    chunk_start = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    while chunk_start <= end_ts:
        chunk_end = min(chunk_start + pd.Timedelta(days=90), end_ts)
        params = {
            "symbol": ticker,
            "from": chunk_start.strftime("%Y-%m-%d"),
            "to": chunk_end.strftime("%Y-%m-%d"),
            "token": api_key,
        }
        try:
            resp = requests.get("https://finnhub.io/api/v1/company-news", params=params, timeout=30)
            resp.raise_for_status()
            for item in resp.json():
                headline = item.get("headline")
                ts = item.get("datetime")
                if not headline or not ts:
                    continue
                date = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
                compound = analyzer.polarity_scores(headline)["compound"]
                scores_by_date.setdefault(date, []).append(compound)
        except requests.RequestException as e:
            print(f"    Warning: Finnhub request failed for {chunk_start.date()}–{chunk_end.date()} ({e}); skipping")

        chunk_start = chunk_end + pd.Timedelta(days=1)
        if chunk_start <= end_ts:
            time.sleep(1.1)  # stay under Finnhub's free-tier rate limit

    daily_mean = {date: sum(scores) / len(scores) for date, scores in scores_by_date.items()}
    sentiment = pd.Series(daily_mean)
    sentiment.index = pd.to_datetime(sentiment.index)
    sentiment = sentiment.sort_index()

    df["sentiment_score"] = sentiment.reindex(df.index)
    df["sentiment_score"] = df["sentiment_score"].ffill().fillna(0.0)
    return df


def engineer_features(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Apply all feature engineering steps in order."""
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_volume_features(df)
    df = add_lag_returns(df)
    df = add_volatility(df)
    df = add_market_features(df, start, end)
    return df


def create_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary target: 1 if next trading day's close is HIGHER than today's, else 0.
    We shift by -1 so each row's label reflects tomorrow's outcome.
    """
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(ticker: str, start: str, end: str, bucket: str, finnhub_api_key: str = None) -> pd.DataFrame:
    print(f"[1/4] Fetching {ticker} data from Yahoo Finance ({start} → {end})...")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker symbol and date range.")

    # yfinance returns multi-level columns when downloading; flatten them
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Keep only OHLCV columns
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    print(f"    Raw rows: {len(df):,}")

    print("[2/4] Engineering features...")
    df = engineer_features(df, start, end)

    if finnhub_api_key:
        print("    Fetching news sentiment from Finnhub...")
        df = add_sentiment_features(df, ticker, start, end, finnhub_api_key)
    else:
        print("    No Finnhub API key provided — sentiment_score defaulting to 0.0")
        df["sentiment_score"] = 0.0

    df = create_label(df)

    # Drop rows with NaN from rolling windows and the last row (no tomorrow label)
    df.dropna(inplace=True)
    print(f"    Rows after feature engineering: {len(df):,}")
    print(f"    Features: {[c for c in df.columns if c not in ['target','Open','High','Low','Close','Volume']]}")
    print(f"    Label distribution: UP={df['target'].sum()} ({df['target'].mean():.1%})  "
          f"DOWN={len(df)-df['target'].sum()} ({1-df['target'].mean():.1%})")

    print("[3/4] Saving to S3...")
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer)
    s3 = boto3.client("s3")
    s3_key = f"stock-predictor/{ticker}_features.csv"
    s3.put_object(Bucket=bucket, Key=s3_key, Body=csv_buffer.getvalue())
    print(f"    Uploaded → s3://{bucket}/{s3_key}")

    print("[4/4] Done!\n")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--start",  default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",    default="2025-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--finnhub-key", default=None, help="Finnhub API key for news sentiment (optional)")
    args = parser.parse_args()

    run_pipeline(args.ticker, args.start, args.end, args.bucket, args.finnhub_key)
