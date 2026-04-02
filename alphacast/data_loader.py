from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


TARGET_COL = "real_power"
TIME_COL = "date"


@dataclass
class LoadedData:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    frequency: Optional[str]


def _infer_frequency(train_df: pd.DataFrame, explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    try:
        freq = pd.infer_freq(train_df[TIME_COL])
        return freq
    except Exception:
        return None


def load_dataset(training_csv: str, test_csv: str, frequency_hint: Optional[str] = None) -> LoadedData:
    def _read(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if TIME_COL not in df.columns:
            raise ValueError(f"CSV must contain '{TIME_COL}' column: {path}")
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        df = df.sort_values(TIME_COL).reset_index(drop=True)
        return df

    train_df = _read(training_csv)
    test_df = _read(test_csv)

    # Basic cleaning: drop duplicates by timestamp (keep last)
    train_df = train_df.drop_duplicates(subset=[TIME_COL], keep="last")
    test_df = test_df.drop_duplicates(subset=[TIME_COL], keep="last")

    frequency = _infer_frequency(train_df, frequency_hint)
    return LoadedData(train_df=train_df, test_df=test_df, frequency=frequency)


def infer_target_column(df: pd.DataFrame, dataset_name: Optional[str] = None) -> str:
    """
    Always select the last data column as the target variable, ignoring dataset-specific
    naming. Excludes known non-target columns like the time column and prediction metadata.
    """
    exclude_cols = {TIME_COL, "time_stamp", "predicted_ans", "features_used"}
    cols = [c for c in df.columns if c not in exclude_cols]
    if not cols:
        raise ValueError("Unable to infer target column: no usable columns available")
    return cols[-1]


def get_series_arrays(train_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    ts = train_df[TIME_COL].to_numpy()
    target_col = infer_target_column(train_df)
    y = train_df[target_col].to_numpy(dtype=float)
    return ts, y