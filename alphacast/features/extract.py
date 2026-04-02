from __future__ import annotations

from typing import Dict, Optional, List, Callable, Any

import numpy as np
import pandas as pd


def _map_pandas_freq_to_period(freq: Optional[str]) -> int:
    if not freq:
        return 1
    f = str(freq).upper()
    # Very lightweight mapping; extend as needed
    if f.startswith("H"):  # hourly
        return 24
    if f.startswith("T") or "MIN" in f:  # minutely
        return 60
    if f.startswith("S"):  # secondly
        return 60
    if f.startswith("D"):  # daily
        return 7
    if f.startswith("W"):  # weekly
        return 52
    if f.startswith("M"):  # monthly
        return 12
    if f.startswith("Q"):  # quarterly
        return 4
    if f.startswith("A") or f.startswith("Y"):  # yearly/annual
        return 1
    return 1


def _basic_statistics(y: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    if not finite.any():
        return {
            "count": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "skew": float("nan"),
            "kurt": float("nan"),
        }
    z = y[finite]
    n = float(z.size)
    mean = float(np.mean(z))
    std = float(np.std(z, ddof=1)) if z.size > 1 else 0.0
    min_v = float(np.min(z))
    max_v = float(np.max(z))
    # sample skewness / kurtosis (Fisher definition for excess kurtosis)
    if z.size > 2 and std > 0:
        m3 = float(np.mean((z - mean) ** 3))
        m4 = float(np.mean((z - mean) ** 4))
        skew = m3 / (std ** 3 + 1e-12)
        kurt = m4 / (std ** 4 + 1e-12) - 3.0
    else:
        skew = 0.0
        kurt = 0.0
    return {
        "count": n,
        "mean": mean,
        "std": std,
        "min": min_v,
        "max": max_v,
        "skew": float(skew),
        "kurt": float(kurt),
    }


def _spectral_entropy(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return float("nan")
    # Welch periodogram
    try:
        from scipy.signal import welch  # type: ignore
    except Exception:
        return float("nan")
    y = y - np.nanmean(y)
    if not np.any(np.isfinite(y)):
        return float("nan")
    f, pxx = welch(y[np.isfinite(y)], nperseg=min(256, max(8, y.size)))
    p = pxx / (np.sum(pxx) + 1e-12)
    p = p[p > 0]
    if p.size == 0:
        return float("nan")
    ent = -np.sum(p * np.log(p)) / np.log(float(p.size))
    return float(ent)


def _call_tsfeatures(y: np.ndarray, pandas_freq: Optional[str]) -> Dict[str, float]:
    # Defer imports and be resilient to missing functions
    try:
        import tsfeatures as tsf  # type: ignore
        from tsfeatures import tsfeatures as tsf_main  # type: ignore
    except Exception:
        return {}

    # Assemble available feature callables
    candidate_names: List[str] = [
        "acf_features",
        "entropy",
        "lumpiness",
        "flat_spots",
        "crossing_points",
        # Add more if available in the installed version
        # "stability", "arch_stat", "stl_features" ...
    ]
    fns: List[Callable[..., Dict[str, float]]] = []
    for name in candidate_names:
        fn = getattr(tsf, name, None)
        if callable(fn):
            fns.append(fn)

    if not fns:
        return {}

    period = _map_pandas_freq_to_period(pandas_freq)
    # Build a minimal panel with synthetic ds to satisfy interface
    n = int(len(y))
    if n <= 0:
        return {}
    # Use daily frequency for ds spacing; does not affect functions using only y & freq
    ds = pd.date_range(start="2000-01-01", periods=n, freq="D")
    panel = pd.DataFrame({
        "unique_id": ["series_0"] * n,
        "ds": ds,
        "y": np.asarray(y, dtype=float),
    })

    try:
        df = tsf_main(panel, freq=period, features=fns)
        if df is None or len(df) == 0:
            return {}
        row = df.iloc[0].to_dict()
        # Convert numpy types to native floats
        clean: Dict[str, float] = {}
        for k, v in row.items():
            try:
                clean[k] = float(v)
            except Exception:
                # Skip non-scalar values
                pass
        return clean
    except Exception:
        return {}


def extract_target_features(y: np.ndarray, pandas_freq: Optional[str] = None) -> Dict[str, float]:
    """
    Compute a comprehensive feature set for a univariate target series using tsfeatures with
    graceful degradation. Only the target variable is processed.

    Returns a flat dict of feature_name -> value.
    """
    y = np.asarray(y, dtype=float)
    out: Dict[str, float] = {}

    # Basic statistics
    out.update({f"basic_{k}": v for k, v in _basic_statistics(y).items()})

    # Nonlinear: spectral entropy
    out["spectral_entropy"] = _spectral_entropy(y)

    # tsfeatures-based features
    tf = _call_tsfeatures(y, pandas_freq)
    out.update(tf)

    # A derived seasonal_strength proxy if not present
    if "seasonal_strength" not in out:
        # Heuristic: use difference between var and rolling mean var as a proxy via lumpiness/stability if present
        seas = 0.0
        try:
            seas = float(out.get("seas_acf1", 0.0))  # if provided by some versions
        except Exception:
            seas = 0.0
        out.setdefault("seasonal_strength", float(seas))

    # Ensure JSON-serializable
    clean_out: Dict[str, float] = {}
    for k, v in out.items():
        try:
            clean_out[str(k)] = float(v)
        except Exception:
            pass
    return clean_out

