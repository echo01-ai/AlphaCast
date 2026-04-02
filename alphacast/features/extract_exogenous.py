from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .extract import extract_target_features


EXOGENOUS_BASES: Dict[str, List[str]] = {
    "wind_speed_80m": ["wind_speed_80m", "ws_80m", "Wind speed 80m"],
    "wind_direction_80m": ["wind_direction_80m", "wd_80m", "Wind direction 80m"],
    "temperature_2m": ["temperature_2m", "temp_2m", "Temperature 2m"],
    "relative_humidity_2m": ["relative_humidity_2m", "rh_2m", "Relative humidity 2m"],
    "precipitation": ["precipitation", "rain", "Precipitation"],
    "direct_radiation": ["direct_radiation", "shortwave_radiation", "Direct radiation"],
    # Added for EPF_NP dataset: treat wind power forecast as an exogenous variable relative to grid load
    "wind_power_forecast": [
        "Wind power forecast",
        "wind_power_forecast",
        "wind power",
        "wind forecast",
        "windpower",
        "wind_power",
    ],
    # Added for EPF_BE dataset: generation and system load forecasts
    "generation_forecast": [
        "Generation forecast",
        "generation_forecast",
        "generation forecast",
        "gen_forecast",
        "generation",
    ],
    "system_load_forecast": [
        "System load forecast",
        "system_load_forecast",
        "system load forecast",
        "load_forecast",
        "system_load",
    ],
    # Added for EPF_DE dataset: wind power and zonal load forecasts
    "ampirion_zonal_load_forecast": [
        "Ampirion zonal load forecast",
        "ampirion_zonal_load_forecast",
        "zonal load forecast",
        "ampirion_load",
        "zonal_load",
    ],
    # Added for ETTh1 and ETTm1 datasets: transformer load variables
    "hufl": ["HUFL", "hufl", "High UseFul Load"],
    "hull": ["HULL", "hull", "High UseLess Load"],
    "mufl": ["MUFL", "mufl", "Middle UseFul Load"],
    "mull": ["MULL", "mull", "Middle UseLess Load"],
    "lufl": ["LUFL", "lufl", "Low UseFul Load"],
    "lull": ["LULL", "lull", "Low UseLess Load"],
}


EXOGENOUS_DESCRIPTIONS: Dict[str, str] = {
    "wind_speed_80m": "80-meter wind speed meteorological series (m/s).",
    "wind_direction_80m": "80-meter wind direction (degrees from north).",
    "temperature_2m": "Near-surface (2 m) air temperature.",
    "relative_humidity_2m": "Relative humidity measured at 2 meters above ground.",
    "precipitation": "Accumulated precipitation or rainfall intensity.",
    "direct_radiation": "Direct shortwave solar radiation impacting the site.",
    "wind_power_forecast": "Day-ahead wind power generation forecast aggregated across the bidding zone.",
    "generation_forecast": "Day-ahead power generation forecast for the market/system.",
    "system_load_forecast": "System-wide day-ahead load forecast.",
    "ampirion_zonal_load_forecast": "Day-ahead load forecast for the Amprion transmission system operator (TSO) region.",
    "hufl": "Transformer High UseFul Load (HUFL).",
    "hull": "Transformer High UseLess Load (HULL).",
    "mufl": "Transformer Middle UseFul Load (MUFL).",
    "mull": "Transformer Middle UseLess Load (MULL).",
    "lufl": "Transformer Low UseFul Load (LUFL).",
    "lull": "Transformer Low UseLess Load (LULL).",
}


def _normalize_name(s: str) -> str:
    # Normalize for matching: strip spaces, lower-case, convert spaces to underscores
    return s.strip().lower().replace(" ", "_")


def _find_station_columns(df: pd.DataFrame, base_names: List[str]) -> List[str]:
    # Build normalized lookup for dataframe columns
    cols_orig = list(df.columns)
    cols_norm = [_normalize_name(c) for c in cols_orig]
    norm_to_orig: Dict[str, str] = {n: o for n, o in zip(cols_norm, cols_orig)}

    # Normalize base names
    base_norms = [_normalize_name(b) for b in base_names]

    found: List[str] = []
    # direct matches (case/space-insensitive)
    for bn in base_norms:
        if bn in norm_to_orig:
            found.append(norm_to_orig[bn])

    # station patterns: base_station1, base_station2, base_station3 or base_1, base_2, base_3
    station_suffixes = ["station1", "station2", "station3", "1", "2", "3"]
    for bn in base_norms:
        for suffix in station_suffixes:
            cand_norm = f"{bn}_{suffix}"
            if cand_norm in norm_to_orig:
                found.append(norm_to_orig[cand_norm])

    # deduplicate preserving order
    seen = set()
    uniq = []
    for c in found:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def _safe_mean(series_list: List[pd.Series]) -> pd.Series:
    if not series_list:
        return pd.Series(dtype=float)
    stacked = pd.concat(series_list, axis=1)
    return stacked.mean(axis=1, skipna=True)


def _pearson_corr(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    mask = np.isfinite(a.to_numpy(dtype=float)) & np.isfinite(b.to_numpy(dtype=float))
    if not np.any(mask):
        return float("nan")
    aa = a.to_numpy(dtype=float)[mask]
    bb = b.to_numpy(dtype=float)[mask]
    if aa.size < 2 or bb.size < 2:
        return float("nan")
    try:
        # prefer scipy if available
        from scipy.stats import pearsonr  # type: ignore
        r, _ = pearsonr(aa, bb)
        return float(r)
    except Exception:
        # fallback
        c = np.corrcoef(aa, bb)
        return float(c[0, 1])


def extract_exogenous_features(
    df: pd.DataFrame,
    target_col: str,
    dataset_name: str,
    pandas_freq: str | None = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], List[str], Dict[str, List[str]]]:
    """
    Identify exogenous variables for the dataset, compute station-level averages per indicator
    to produce the final exogenous variables, compute Pearson correlation with the target,
    and compute per-variable feature dictionaries.

    Returns (features_by_var, correlations_by_var, top3_var_names_by_abs_corr, source_columns_by_var).
    """
    # Normalize column names (handle headers with leading spaces like EPF_NP)
    try:
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
    except Exception:
        pass

    # Normalize date ordering
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    # Determine available base variables
    features_by_var: Dict[str, Dict[str, float]] = {}
    corr_by_var: Dict[str, float] = {}
    columns_by_var: Dict[str, List[str]] = {}

    # Target series
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in dataframe")
    y = pd.to_numeric(df[target_col], errors="coerce")

    for var_name, names in EXOGENOUS_BASES.items():
        cols = _find_station_columns(df, names)
        if not cols:
            # not available in this dataset
            continue
        columns_by_var[var_name] = cols
        # If multiple station columns found, average; otherwise use the single column
        series_list = [pd.to_numeric(df[c], errors="coerce") for c in cols]
        exo_series = _safe_mean(series_list) if len(series_list) > 1 else series_list[0]
        # Features for this exogenous variable
        feats = extract_target_features(exo_series.to_numpy(dtype=float), pandas_freq)
        features_by_var[var_name] = feats
        # Pearson corr with target
        corr = _pearson_corr(exo_series, y)
        corr_by_var[var_name] = float(corr)

    # Select Top-3 by absolute correlation (or fewer if not enough variables available)
    ordered = sorted(
        [(k, v) for k, v in corr_by_var.items() if np.isfinite(v)],
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    top3 = [k for k, _ in ordered[:3]]

    return features_by_var, corr_by_var, top3, columns_by_var
