from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .data_loader import TIME_COL, infer_target_column

_PREDICTION_COLUMNS = ("predicted_ans", "prediction", "forecast", "value")


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def align_predictions(
    test_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    dataset_name: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if TIME_COL not in test_df.columns:
        raise ValueError(f"Ground-truth frame must contain '{TIME_COL}'.")
    if "time_stamp" not in pred_df.columns:
        raise ValueError("Prediction frame must contain 'time_stamp'.")

    candidate_col = next((col for col in _PREDICTION_COLUMNS if col in pred_df.columns), None)
    if candidate_col is None:
        raise ValueError(
            f"Prediction frame must include one of the columns {_PREDICTION_COLUMNS} containing forecast values."
        )

    gt_df = test_df.copy()
    gt_df[TIME_COL] = pd.to_datetime(gt_df[TIME_COL])
    target_col = infer_target_column(gt_df, dataset_name)
    gt_df = gt_df.sort_values(TIME_COL, kind="mergesort")
    target_series = gt_df.set_index(TIME_COL)[target_col]

    preds_df = pred_df.copy()
    preds_df["time_stamp"] = pd.to_datetime(preds_df["time_stamp"])

    if "emission_index" in preds_df.columns:
        order_values = pd.to_numeric(preds_df["emission_index"], errors="coerce")
        preds_df = preds_df.assign(_order=order_values).sort_values("_order", kind="mergesort")
    elif {"window_offset", "horizon_index"}.issubset(preds_df.columns):
        window_vals = pd.to_numeric(preds_df["window_offset"], errors="coerce")
        horizon_vals = pd.to_numeric(preds_df["horizon_index"], errors="coerce")
        preds_df = preds_df.assign(
            _order=window_vals.fillna(0) * 1_000_000 + horizon_vals.fillna(0)
        ).sort_values("_order", kind="mergesort")
    if "_order" in preds_df.columns:
        preds_df = preds_df.drop(columns="_order")

    y_true_list: list[float] = []
    y_pred_list: list[float] = []

    for _, row in preds_df.iterrows():
        ts = row["time_stamp"]
        if pd.isna(ts):
            continue
        try:
            pred_value = float(row[candidate_col])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(pred_value):
            continue
        try:
            actual = target_series.loc[ts]
        except KeyError:
            continue
        if isinstance(actual, pd.Series):
            if actual.empty:
                continue
            actual_value = actual.iloc[0]
        else:
            actual_value = actual
        try:
            actual_value = float(actual_value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(actual_value):
            continue
        y_true_list.append(actual_value)
        y_pred_list.append(pred_value)

    return np.asarray(y_true_list, dtype=float), np.asarray(y_pred_list, dtype=float)
