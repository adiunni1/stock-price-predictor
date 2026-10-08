"""
train_model.py
--------------
Loads the engineered feature CSV from S3, performs a time-aware train/test
split, trains an XGBoost classifier, evaluates it, plots feature importance,
and saves the serialised model back to S3.

Usage:
    python train_model.py --ticker AAPL --bucket YOUR_S3_BUCKET

Why time-aware split?
    Stock data is sequential. Randomly shuffling and splitting would let
    the model "see the future" during training (data leakage), producing
    artificially inflated accuracy on test data. We always train on the
    PAST and test on the FUTURE.
"""

import argparse
import io
import os
import joblib
import boto3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import RandomOverSampler


# ── Feature columns (everything except raw OHLCV and the label) ───────────────
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
TARGET_COL = "target"
TRAIN_RATIO = 0.80   # 80% train, 20% test — always chronological

# ── Walk-forward validation settings ───────────────────────────────────────
WF_TRAIN_YEARS = 3
WF_TEST_MONTHS = 6
WF_STEP_MONTHS = 6
WF_START = pd.Timestamp("2018-01-01")
WF_END = pd.Timestamp("2025-12-31")


def load_from_s3(bucket: str, ticker: str) -> pd.DataFrame:
    s3 = boto3.client("s3")
    key = f"stock-predictor/{ticker}_features.csv"
    print(f"Loading s3://{bucket}/{key} ...")
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()), index_col=0, parse_dates=True)
    return df


def time_split(df: pd.DataFrame):
    """Chronological 80/20 split — no shuffling."""
    split_idx = int(len(df) * TRAIN_RATIO)
    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]
    print(f"Train: {train.index[0].date()} → {train.index[-1].date()}  ({len(train):,} rows)")
    print(f"Test:  {test.index[0].date()} → {test.index[-1].date()}  ({len(test):,} rows)")
    return train, test


def oversample(X_train, y_train):
    """Randomly duplicate minority-class rows so the train split is 50/50."""
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_train, y_train)
    print(f"  Oversampled train set: {len(X_train):,} → {len(X_res):,} rows "
          f"(class balance: {y_res.mean():.1%} UP)")
    return X_res, y_res


def train(X_train, y_train) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False,
    )
    return model


def find_optimal_threshold(model, X_train, y_train) -> float:
    """Sweep candidate thresholds on the train set and pick the one maximising F1.

    The default 0.5 cutoff assumes balanced classes; on imbalanced stock
    data it biases predictions toward the majority (DOWN) class, so we
    tune it instead of hardcoding it.
    """
    probs = model.predict_proba(X_train)[:, 1]
    candidates = [round(0.30 + 0.05 * i, 2) for i in range(5)]  # 0.30 .. 0.50

    best_threshold, best_f1 = 0.5, -1.0
    for t in candidates:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_train, preds)
        if f1 > best_f1:
            best_threshold, best_f1 = t, f1

    return best_threshold


def evaluate(model, X_test, y_test, ticker: str, threshold: float = 0.5, output_dir: str = "."):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    print("\n── Evaluation Results ──────────────────────────────")
    print(f"  Threshold: {threshold:.2f}  (tuned on train set to maximise F1)")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}  (of predicted UPs, how many were correct?)")
    print(f"  Recall   : {rec:.4f}  (of actual UPs, how many did we catch?)")
    print(f"  F1 Score : {f1:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))

    # ── Confusion matrix plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{ticker} — XGBoost Stock Movement Predictor", fontsize=14, fontweight="bold")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"])
    axes[0].set_title("Confusion Matrix (Test Set)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # ── Feature importance plot ───────────────────────────────────────────────
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importance_sorted = importance.sort_values(ascending=True).tail(15)
    importance_sorted.plot(kind="barh", ax=axes[1], color="steelblue")
    axes[1].set_title("Top 15 Feature Importances")
    axes[1].set_xlabel("Importance Score")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{ticker}_evaluation.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved → {plot_path}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "threshold": threshold}, plot_path


def walkforward_windows(df: pd.DataFrame):
    """Yield (train_df, test_df) pairs for rolling 3yr-train / 6mo-test windows,
    stepping forward 6 months at a time from WF_START to WF_END."""
    train_start = WF_START
    while True:
        train_end = train_start + pd.DateOffset(years=WF_TRAIN_YEARS)
        test_end = train_end + pd.DateOffset(months=WF_TEST_MONTHS)
        if test_end > WF_END:
            break

        train_df = df.loc[(df.index >= train_start) & (df.index < train_end)]
        test_df = df.loc[(df.index >= train_end) & (df.index < test_end)]

        yield train_start, train_end, test_end, train_df, test_df

        train_start = train_start + pd.DateOffset(months=WF_STEP_MONTHS)


def evaluate_metrics_only(model, X_test, y_test, threshold: float) -> dict:
    """Same metrics as evaluate(), without plotting — used per-window in walk-forward."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }


def save_walkforward_to_s3(summary_df: pd.DataFrame, bucket: str, ticker: str):
    s3 = boto3.client("s3")
    buffer = io.StringIO()
    summary_df.to_csv(buffer, index=False)
    key = f"stock-predictor/{ticker}_walkforward.csv"
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    print(f"\n  Walk-forward results → s3://{bucket}/{key}")


def run_walkforward(ticker: str, bucket: str):
    """Rolling walk-forward validation: train on 3 years, test on the following
    6 months, stepping forward 6 months at a time from 2018 to 2025.

    Separate from run_training() so the single-split flow used by run_all.py
    is unaffected.
    """
    print(f"\n=== Walk-forward validation for {ticker} ===\n")

    df = load_from_s3(bucket, ticker)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}. Re-run data_pipeline.py first.")

    rows = []
    for i, (train_start, train_end, test_end, train_df, test_df) in enumerate(walkforward_windows(df), start=1):
        if len(train_df) == 0 or len(test_df) == 0:
            print(f"  Window {i}: {train_start.date()} → {test_end.date()} — skipped (no data in range)")
            continue

        print(f"  Window {i}: train {train_start.date()} → {train_end.date()} "
              f"({len(train_df):,} rows)  |  test {train_end.date()} → {test_end.date()} "
              f"({len(test_df):,} rows)")

        X_train = train_df[FEATURE_COLS]
        y_train = train_df[TARGET_COL]
        X_test = test_df[FEATURE_COLS]
        y_test = test_df[TARGET_COL]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            print(f"    skipped — window doesn't contain both classes")
            continue

        X_train_res, y_train_res = oversample(X_train, y_train)
        model = train(X_train_res, y_train_res)
        threshold = find_optimal_threshold(model, X_train_res, y_train_res)
        metrics = evaluate_metrics_only(model, X_test, y_test, threshold)

        rows.append({
            "window": i,
            "train_start": train_start.date(),
            "train_end": train_end.date(),
            "test_start": train_end.date(),
            "test_end": test_end.date(),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "threshold": threshold,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        })

    summary_df = pd.DataFrame(rows)

    print("\n── Walk-Forward Summary ────────────────────────────────────────────")
    if summary_df.empty:
        print("  No windows produced results.")
    else:
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\n  Averages across windows:")
        print(f"    Accuracy : {summary_df['accuracy'].mean():.4f}")
        print(f"    Precision: {summary_df['precision'].mean():.4f}")
        print(f"    Recall   : {summary_df['recall'].mean():.4f}")
        print(f"    F1 Score : {summary_df['f1'].mean():.4f}")

        print("\nSaving walk-forward results to S3...")
        save_walkforward_to_s3(summary_df, bucket, ticker)

    print("\n=== Done! ===")
    return summary_df


def save_to_s3(model, metrics: dict, plot_path: str, bucket: str, ticker: str):
    s3 = boto3.client("s3")

    # Serialise model with joblib
    model_buffer = io.BytesIO()
    joblib.dump(model, model_buffer)
    model_buffer.seek(0)
    model_key = f"stock-predictor/{ticker}_xgb_model.joblib"
    s3.put_object(Bucket=bucket, Key=model_key, Body=model_buffer.getvalue())
    print(f"\n  Model → s3://{bucket}/{model_key}")

    # Upload evaluation plot
    plot_key = f"stock-predictor/{ticker}_evaluation.png"
    with open(plot_path, "rb") as f:
        s3.put_object(Bucket=bucket, Key=plot_key, Body=f.read(), ContentType="image/png")
    print(f"  Plot  → s3://{bucket}/{plot_key}")

    # Save metrics as CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_buffer = io.StringIO()
    metrics_df.to_csv(metrics_buffer, index=False)
    metrics_key = f"stock-predictor/{ticker}_metrics.csv"
    s3.put_object(Bucket=bucket, Key=metrics_key, Body=metrics_buffer.getvalue())
    print(f"  Metrics → s3://{bucket}/{metrics_key}")


def run_training(ticker: str, bucket: str):
    print(f"\n=== Training XGBoost classifier for {ticker} ===\n")

    df = load_from_s3(bucket, ticker)

    # Validate all feature columns exist
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}. Re-run data_pipeline.py first.")

    train_df, test_df = time_split(df)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df[TARGET_COL]

    print("\nOversampling minority class in train split...")
    X_train, y_train = oversample(X_train, y_train)

    print("\nTraining XGBoost...")
    model = train(X_train, y_train)

    print("\nTuning decision threshold on train set...")
    threshold = find_optimal_threshold(model, X_train, y_train)

    metrics, plot_path = evaluate(model, X_test, y_test, ticker, threshold=threshold)

    print("\nSaving model + artifacts to S3...")
    save_to_s3(model, metrics, plot_path, bucket, ticker)

    print("\n=== Done! ===")
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--walkforward", action="store_true",
                         help="Run rolling walk-forward validation instead of a single train/test split")
    args = parser.parse_args()

    if args.walkforward:
        run_walkforward(args.ticker, args.bucket)
    else:
        run_training(args.ticker, args.bucket)
