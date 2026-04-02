from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from ..data_loader import TIME_COL, TARGET_COL
from ..models.base import (
    ForecastModel,
    configure_deep_learning_runtime,
    get_default_models,
)
if TYPE_CHECKING:
    from castmind.config import DatasetConfig
from ..utils.time import generate_future_timestamps


def forecast_with_model(
    model_name: str,
    last_window: np.ndarray,
    h: int,
    season_length: Optional[int],
    dataset: Optional["DatasetConfig"] = None,
    **kwargs
) -> np.ndarray:
    if dataset is not None:
        configure_deep_learning_runtime(dataset.checkpoints, dataset.predicted_window)

    models = {m.alias: m for m in get_default_models()}
    if model_name not in models:
        model_name = "SeasonalNaive"
    model = models[model_name]
    
    timestamps = kwargs.get("timestamps", None)
    future_timestamps = generate_future_timestamps(timestamps.iloc[-1], h, pd.infer_freq(pd.to_datetime(timestamps)))

    # Pass timestamps through to fit; keep predict unchanged
    model.fit(last_window, season_length=season_length, timestamps=timestamps)
    return model.predict(h, future_timestamps=future_timestamps)


def save_predictions_csv(out_path: str, timestamps: List[pd.Timestamp], preds: np.ndarray) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.DataFrame({"time_stamp": timestamps, "predicted_ans": preds})
    df.to_csv(out_path, index=False)
