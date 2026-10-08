"""
backtest.py
-----------
Simulates a simple trading strategy driven by the trained model's daily
UP/DOWN predictions on the existing held-out test set, and compares the
resulting portfolio value against a SPY buy-and-hold benchmark.

Strategy: starting with $10,000, whenever the model predicts UP for a given
day, hold the stock from that day's close to the next day's close (realising
that day's return); whenever it predicts DOWN, sit in cash (0% return) for
that period instead. No transaction costs are modeled.

Usage:
    python backtest.py --ticker AAPL --bucket YOUR_S3_BUCKET
"""

import argparse
import io
import os

import boto3
import joblib
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_model import FEATURE_COLS, load_from_s3, time_split

STARTING_CASH = 10_000.0


def load_model_and_threshold(bucket: str, ticker: str):
    s3 = boto3.client("s3")

    model_key = f"stock-predictor/{ticker}_xgb_model.joblib"
    model_obj = s3.get_object(Bucket=bucket, Key=model_key)
    model = joblib.load(io.BytesIO(model_obj["Body"].read()))

    metrics_key = f"stock-predictor/{ticker}_metrics.csv"
    metrics_obj = s3.get_object(Bucket=bucket, Key=metrics_key)
    metrics = pd.read_csv(io.BytesIO(metrics_obj["Body"].read()))
    threshold = float(metrics["threshold"].iloc[0])

    return model, threshold


def simulate_strategy(test_df: pd.DataFrame, predictions: pd.Series,
                       starting_cash: float = STARTING_CASH) -> pd.Series:
    """Long the stock from close[t-1] to close[t] whenever the model predicted
    UP at close[t-1]; otherwise stay in cash (no return) for that period."""
    closes = test_df["Close"]
    returns = closes.pct_change()

    value = starting_cash
    values = [starting_cash]
    for i in range(1, len(test_df)):
        prev_date, curr_date = test_df.index[i - 1], test_df.index[i]
        if predictions.loc[prev_date] == 1:
            value *= (1 + returns.loc[curr_date])
        values.append(value)

    return pd.Series(values, index=test_df.index)


def spy_buy_and_hold(start: str, end: str, target_index: pd.DatetimeIndex,
                      starting_cash: float = STARTING_CASH) -> pd.Series:
    """$10,000 invested in SPY at the first close of the test period, held throughout."""
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    closes = spy["Close"].reindex(target_index).ffill().bfill()
    shares = starting_cash / closes.iloc[0]
    return shares * closes


def plot_backtest(strategy_values: pd.Series, spy_values: pd.Series, ticker: str,
                   output_dir: str = ".") -> str:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(strategy_values.index, strategy_values.values, label=f"{ticker} Model Strategy", linewidth=2)
    ax.plot(spy_values.index, spy_values.values, label="SPY Buy & Hold", linewidth=2, linestyle="--")
    ax.axhline(STARTING_CASH, color="gray", linewidth=1, linestyle=":", label="Starting Capital")

    ax.set_title(f"{ticker} Model Strategy vs. SPY Buy & Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"{ticker}_backtest.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


def save_plot_to_s3(plot_path: str, bucket: str, ticker: str):
    s3 = boto3.client("s3")
    key = f"stock-predictor/{ticker}_backtest.png"
    with open(plot_path, "rb") as f:
        s3.put_object(Bucket=bucket, Key=key, Body=f.read(), ContentType="image/png")
    print(f"  Plot → s3://{bucket}/{key}")


def run_backtest(ticker: str, bucket: str):
    print(f"\n=== Backtest: {ticker} model strategy vs. SPY buy & hold ===\n")

    df = load_from_s3(bucket, ticker)
    _, test_df = time_split(df)

    print("\nLoading model + tuned threshold from S3...")
    model, threshold = load_model_and_threshold(bucket, ticker)

    X_test = test_df[FEATURE_COLS]
    probs = model.predict_proba(X_test)[:, 1]
    predictions = pd.Series((probs >= threshold).astype(int), index=test_df.index)

    print("\nSimulating model strategy...")
    strategy_values = simulate_strategy(test_df, predictions)

    print("Fetching SPY buy-and-hold benchmark...")
    start = test_df.index[0].strftime("%Y-%m-%d")
    end = (test_df.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    spy_values = spy_buy_and_hold(start, end, test_df.index)

    print("\nPlotting results...")
    plot_path = plot_backtest(strategy_values, spy_values, ticker)
    print(f"  Plot saved → {plot_path}")

    print("\nUploading plot to S3...")
    save_plot_to_s3(plot_path, bucket, ticker)

    strategy_final = strategy_values.iloc[-1]
    spy_final = spy_values.iloc[-1]
    strategy_return = (strategy_final / STARTING_CASH - 1) * 100
    spy_return = (spy_final / STARTING_CASH - 1) * 100

    print("\n── Backtest Results ────────────────────────────────────────")
    print(f"  Period: {test_df.index[0].date()} → {test_df.index[-1].date()}")
    print(f"  Starting capital:       ${STARTING_CASH:,.2f}")
    print(f"  {ticker} Model Strategy: ${strategy_final:,.2f}  ({strategy_return:+.2f}%)")
    print(f"  SPY Buy & Hold:          ${spy_final:,.2f}  ({spy_return:+.2f}%)")
    print("\n=== Done! ===")

    return {
        "strategy_final_value": strategy_final,
        "strategy_return_pct": strategy_return,
        "spy_final_value": spy_final,
        "spy_return_pct": spy_return,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--bucket", required=True)
    args = parser.parse_args()

    run_backtest(args.ticker, args.bucket)
