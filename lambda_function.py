"""
lambda_function.py
------------------
AWS Lambda handler for live stock movement predictions.
Deployed behind API Gateway — accepts a POST request with a stock ticker,
fetches the latest market data via yfinance, engineers the same features
used during training, loads the XGBoost model from S3, and returns a
prediction with a confidence score.

Expected request body:
    { "ticker": "AAPL" }

Response:
    {
        "ticker": "AAPL",
        "prediction": "UP",
        "confidence": 0.63,
        "latest_close": 189.42,
        "latest_date": "2025-01-10",
        "model_version": "xgb_v1"
    }

Lambda environment variables:
    S3_BUCKET        — the S3 bucket where the model is stored
    MODEL_KEY        — e.g. stock-predictor/AAPL_xgb_model.joblib
    FINNHUB_API_KEY  — optional; enables live news-sentiment scoring. If unset
                        (or the Finnhub request fails), sentiment_score defaults
                        to 0.0, matching the offline pipeline's fallback.
"""

import os
import io
import json
import logging
import tempfile
from datetime import datetime, timedelta

import boto3
import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "sma_5", "sma_10", "sma_20", "sma_50",
    "ema_5", "ema_10", "ema_20", "ema_50",
    "price_to_sma20", "price_to_sma50",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_position",
    "volume_ratio",
    "daily_return",
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_4", "return_lag_5",
    "volatility_10",
    "vix_close", "spy_return",
    "sentiment_score",
]

# Cache the model in Lambda's memory between warm invocations
_model_cache = {}


# ── Feature Engineering (mirrors data_pipeline.py exactly) ───────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Moving averages
    for window in [5, 10, 20, 50]:
        df[f"sma_{window}"] = df["Close"].rolling(window).mean()
        df[f"ema_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()
    df["price_to_sma20"] = df["Close"] / df["sma_20"]
    df["price_to_sma50"] = df["Close"] / df["sma_50"]

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    sma = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["bb_upper"] = sma + 2 * std
    df["bb_lower"] = sma - 2 * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma
    df["bb_position"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # Volume
    df["volume_sma_20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_sma_20"]

    # Lag returns
    df["daily_return"] = df["Close"].pct_change()
    for lag in range(1, 6):
        df[f"return_lag_{lag}"] = df["daily_return"].shift(lag)

    # Volatility
    df["volatility_10"] = df["daily_return"].rolling(10).std()

    return df


def add_market_features(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Market-level context: VIX close (fear gauge) and SPY daily return (broad market move)."""
    market = yf.download(["^VIX", "SPY"], start=start, end=end, auto_adjust=True, progress=False)

    vix_close = market["Close"]["^VIX"]
    spy_return = market["Close"]["SPY"].pct_change()

    df["vix_close"] = vix_close.reindex(df.index)
    df["spy_return"] = spy_return.reindex(df.index)
    df[["vix_close", "spy_return"]] = df[["vix_close", "spy_return"]].ffill()
    return df


_vader = None


def _get_vader() -> SentimentIntensityAnalyzer:
    """Lazily load the VADER lexicon into /tmp — the only writable path in Lambda."""
    global _vader
    if _vader is None:
        nltk_data_dir = "/tmp/nltk_data"
        nltk.data.path.append(nltk_data_dir)
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", download_dir=nltk_data_dir, quiet=True)
        _vader = SentimentIntensityAnalyzer()
    return _vader


def add_sentiment_features(df: pd.DataFrame, ticker: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """Same approach as data_pipeline.py's add_sentiment_features, sized for
    predict()'s short (~90 day) lookback window — one Finnhub request covers
    the whole range, no chunking needed. Defaults to 0.0 if no key is
    configured or the request fails, so a live prediction never crashes on
    sentiment being unavailable.
    """
    df["sentiment_score"] = 0.0
    if not api_key:
        return df

    try:
        analyzer = _get_vader()
        params = {"symbol": ticker, "from": start, "to": end, "token": api_key}
        resp = requests.get("https://finnhub.io/api/v1/company-news", params=params, timeout=10)
        resp.raise_for_status()

        scores_by_date = {}
        for item in resp.json():
            headline = item.get("headline")
            ts = item.get("datetime")
            if not headline or not ts:
                continue
            date = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            compound = analyzer.polarity_scores(headline)["compound"]
            scores_by_date.setdefault(date, []).append(compound)

        daily_mean = {date: sum(scores) / len(scores) for date, scores in scores_by_date.items()}
        sentiment = pd.Series(daily_mean)
        sentiment.index = pd.to_datetime(sentiment.index)
        df["sentiment_score"] = sentiment.sort_index().reindex(df.index).ffill().fillna(0.0)
    except requests.RequestException as e:
        logger.warning(f"Finnhub sentiment fetch failed ({e}); defaulting sentiment_score to 0.0")

    return df


# ── Model Loading ─────────────────────────────────────────────────────────────

DEFAULT_THRESHOLD = 0.5


def load_threshold(s3, bucket: str, ticker: str) -> float:
    metrics_key = f"stock-predictor/{ticker}_metrics.csv"
    try:
        obj = s3.get_object(Bucket=bucket, Key=metrics_key)
        metrics = pd.read_csv(io.BytesIO(obj["Body"].read()))
        return float(metrics["threshold"].iloc[0])
    except Exception as e:
        logger.warning(f"Could not load tuned threshold from s3://{bucket}/{metrics_key} "
                        f"({e}); falling back to default {DEFAULT_THRESHOLD}")
        return DEFAULT_THRESHOLD


def load_model(bucket: str, model_key: str, ticker: str):
    cache_key = f"{bucket}/{model_key}"
    if cache_key in _model_cache:
        logger.info("Using cached model + threshold")
        return _model_cache[cache_key]

    logger.info(f"Loading model from s3://{bucket}/{model_key}")
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=model_key)
    model = joblib.load(io.BytesIO(obj["Body"].read()))
    threshold = load_threshold(s3, bucket, ticker)

    _model_cache[cache_key] = (model, threshold)
    return model, threshold


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(ticker: str, bucket: str, model_key: str, finnhub_api_key: str = None) -> dict:
    # Fetch 90 days of data — enough for all rolling windows (max=50 days)
    end = datetime.today()
    start = end - timedelta(days=90)
    raw = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = engineer_features(df)
    df = add_market_features(df, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    df = add_sentiment_features(df, ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), finnhub_api_key)
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError("Not enough data to engineer features after dropping NaNs")

    # Use the LATEST row for prediction
    latest = df.iloc[-1]
    latest_date = df.index[-1].strftime("%Y-%m-%d")
    latest_close = round(float(latest["Close"]), 2)

    X = pd.DataFrame([latest[FEATURE_COLS]])

    model, threshold = load_model(bucket, model_key, ticker)
    up_prob = float(model.predict_proba(X)[0][1])
    prediction_int = int(up_prob >= threshold)
    confidence = round(up_prob if prediction_int == 1 else 1 - up_prob, 4)

    return {
        "ticker": ticker.upper(),
        "prediction": "UP" if prediction_int == 1 else "DOWN",
        "confidence": confidence,
        "latest_close": latest_close,
        "latest_date": latest_date,
        "model_version": "xgb_v1",
    }


# ── Lambda Handler ────────────────────────────────────────────────────────────

def lambda_handler(event, context):
    try:
        # Parse request body
        if isinstance(event.get("body"), str):
            body = json.loads(event["body"])
        else:
            body = event.get("body", event)

        ticker = body.get("ticker", "AAPL").upper().strip()
        bucket = os.environ["S3_BUCKET"]
        model_key = os.environ.get("MODEL_KEY", f"stock-predictor/{ticker}_xgb_model.joblib")
        finnhub_api_key = os.environ.get("FINNHUB_API_KEY")

        logger.info(f"Predicting for {ticker} using model {model_key}")
        result = predict(ticker, bucket, model_key, finnhub_api_key)

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(result),
        }

    except KeyError as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": f"Missing field: {e}"}),
        }
    except ValueError as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(e)}),
        }
    except Exception as e:
        logger.exception("Unexpected error")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal server error", "detail": str(e)}),
        }
