from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphacast.data_loader import TIME_COL, infer_target_column
from alphacast.eval import mae, mse, smape


PREDICTION_COLUMNS = ("prediction", "predicted_ans", "forecast", "value")


def _prediction_column(pred_df: pd.DataFrame) -> str:
    for col in PREDICTION_COLUMNS:
        if col in pred_df.columns:
            return col
    raise ValueError(f"Prediction CSV must include one of {PREDICTION_COLUMNS}.")


def _aligned_frame(
    test_csv: str,
    pred_csv: str,
    dataset_name: Optional[str],
) -> pd.DataFrame:
    test_df = pd.read_csv(test_csv)
    pred_df = pd.read_csv(pred_csv)

    if TIME_COL not in test_df.columns:
        raise ValueError(f"Test CSV must include '{TIME_COL}'.")
    if "time_stamp" not in pred_df.columns:
        raise ValueError("Prediction CSV must include 'time_stamp'.")

    target_col = infer_target_column(test_df, dataset_name)
    pred_col = _prediction_column(pred_df)

    actual = test_df[[TIME_COL, target_col]].copy()
    actual[TIME_COL] = pd.to_datetime(actual[TIME_COL])
    actual[target_col] = pd.to_numeric(actual[target_col], errors="coerce")
    actual = actual.dropna(subset=[TIME_COL, target_col])

    pred = pred_df[["time_stamp", pred_col]].copy()
    pred["time_stamp"] = pd.to_datetime(pred["time_stamp"])
    pred[pred_col] = pd.to_numeric(pred[pred_col], errors="coerce")
    pred = pred.dropna(subset=["time_stamp", pred_col])

    if "emission_index" in pred_df.columns:
        pred["_order"] = pd.to_numeric(pred_df["emission_index"], errors="coerce")
    elif {"window_offset", "horizon_index"}.issubset(pred_df.columns):
        window_vals = pd.to_numeric(pred_df["window_offset"], errors="coerce")
        horizon_vals = pd.to_numeric(pred_df["horizon_index"], errors="coerce")
        pred["_order"] = window_vals.fillna(0) * 1_000_000 + horizon_vals.fillna(0)

    if "_order" in pred.columns:
        pred = pred.sort_values("_order", kind="mergesort").drop(columns="_order")

    pred = pred.drop_duplicates(subset=["time_stamp"], keep="last")
    merged = actual.merge(pred, left_on=TIME_COL, right_on="time_stamp", how="inner")
    merged = merged.rename(columns={target_col: "actual", pred_col: "prediction"})
    merged = merged[[TIME_COL, "actual", "prediction"]].sort_values(TIME_COL)
    return merged.reset_index(drop=True)


def plot_predictions(
    frame: pd.DataFrame,
    output: str,
    title: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> None:
    view = frame.iloc[start:end].copy()
    if view.empty:
        raise ValueError("No aligned rows to plot after applying start/end slice.")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(view[TIME_COL], view["actual"], label="Actual", linewidth=1.6)
    ax.plot(view[TIME_COL], view["prediction"], label="Prediction", linewidth=1.2, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot prediction CSV against actual test values.")
    parser.add_argument("--test-csv", default="data/raw/ETTh1/test.csv")
    parser.add_argument("--pred-csv", default="outputs_deepseek/ETTh1/predictions.csv")
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--output", default="outputs_deepseek/ETTh1/predictions_vs_actual.png")
    parser.add_argument("--start", type=int, default=None, help="Optional aligned-row start index.")
    parser.add_argument("--end", type=int, default=None, help="Optional aligned-row end index.")
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    frame = _aligned_frame(args.test_csv, args.pred_csv, args.dataset)
    if frame.empty:
        raise ValueError("No overlapping timestamps between predictions and test data.")

    y_true = frame["actual"].to_numpy(dtype=float)
    y_pred = frame["prediction"].to_numpy(dtype=float)
    print(f"Aligned rows: {len(frame)}")
    print(f"MSE: {mse(y_true, y_pred):.6f}")
    print(f"MAE: {mae(y_true, y_pred):.6f}")
    print(f"sMAPE: {smape(y_true, y_pred):.6f}")

    title = f"{args.dataset}: Prediction vs Actual"
    plot_predictions(frame, args.output, title, args.start, args.end)
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
