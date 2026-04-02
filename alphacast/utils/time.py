from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd

from ..data_loader import TIME_COL, TARGET_COL


class AnalysisMemory(TypedDict):
    max: float
    min: float
    mean: float
    variance: float
    periodicity_lag: int
    series_length: int
    frequency: Optional[str]


@dataclass
class CaseEntry:
    window: List[float]  # z-scored L-length vector
    best_model: str
    
@dataclass
class ClusterEntry:
    window: List[float]  # z-scored L-length vector
    best_model: Dict[str, int]  # model_name -> weight
    total_weight: int

@dataclass
class CaseNeighbor:
    look_back_window: List[float]  # z-scored L-length vector
    pred_window: List[float]

def estimate_periodicity(y: np.ndarray, max_lag: Optional[int] = None) -> int:
    if len(y) < 3:
        return 1
    if max_lag is None:
        max_lag = max(2, len(y) // 2)
    y = y - np.mean(y)
    autocorr = np.correlate(y, y, mode="full")[len(y)-1:len(y)-1+max_lag]
    if len(autocorr) < 2:
        return 1
    lag = int(np.argmax(autocorr[1:]) + 1)
    return max(1, lag)


def generate_future_timestamps(last_ts: pd.Timestamp, h: int, freq: Optional[str]) -> List[pd.Timestamp]:
    if freq is None:
        # Fallback: assume uniform daily spacing and increment by i
        return [last_ts + pd.Timedelta(days=i) for i in range(1, h + 1)]
    return list(pd.date_range(start=last_ts, periods=h + 1, freq=freq)[1:])