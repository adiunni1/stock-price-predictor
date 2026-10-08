"""
run_all.py
----------
Runs the full pipeline (data_pipeline.py → train_model.py) for a list of
tickers, one at a time. A failure on one ticker is logged and skipped so
the rest of the batch still completes.

Usage:
    python run_all.py --bucket YOUR_S3_BUCKET
    python run_all.py --bucket YOUR_S3_BUCKET --tickers AAPL MSFT
    python run_all.py --bucket YOUR_S3_BUCKET --start 2015-01-01 --end 2025-01-01
    python run_all.py --bucket YOUR_S3_BUCKET --finnhub-key YOUR_FINNHUB_KEY
"""

import argparse

from data_pipeline import run_pipeline
from train_model import run_training

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]


def run_all(tickers: list, start: str, end: str, bucket: str, finnhub_key: str = None):
    succeeded, failed = [], []

    for i, ticker in enumerate(tickers, 1):
        print(f"\n{'='*70}\n[{i}/{len(tickers)}] {ticker}\n{'='*70}")
        try:
            run_pipeline(ticker, start, end, bucket, finnhub_key)
            run_training(ticker, bucket)
            succeeded.append(ticker)
        except Exception as e:
            print(f"  !! {ticker} failed: {e}")
            failed.append(ticker)

    print(f"\n{'='*70}\nSummary\n{'='*70}")
    print(f"  Succeeded ({len(succeeded)}): {succeeded}")
    print(f"  Failed    ({len(failed)}): {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="Stock ticker symbols")
    parser.add_argument("--start",  default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",    default="2025-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--finnhub-key", default=None, help="Finnhub API key for news sentiment (optional)")
    args = parser.parse_args()

    run_all(args.tickers, args.start, args.end, args.bucket, args.finnhub_key)
