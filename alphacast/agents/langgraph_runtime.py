from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alphacast.config import DatasetConfig, ExperimentConfig
from alphacast.data_loader import TIME_COL
from alphacast.eval import align_predictions, mae, mse, smape
from .common import assess_forecast, json_default, prepare_investor_packet
from .reflector_agent import scan_chain_of_thought


_RESUME_STATE_FILE = "llm_resume_state.json"


class _LangGraphMissing(RuntimeError):
    pass


class WindowState(TypedDict, total=False):
    dataset_name: str
    ds_out_dir: str
    total_needed: int
    total_len: int
    horizon: int
    stride: int
    look_back: int
    current_collected: int
    current_len: int
    step_index: int
    window_offset: int
    step_horizon: int
    investor_packet: Dict[str, Any]
    raw_generation: Dict[str, Any]
    predictions: List[float]
    reasoning_summary: str
    selected_features: List[str]
    feature_weights: Dict[str, float]
    exogenous_vars: List[str]
    exogenous_feature_selection: Dict[str, List[str]]
    exogenous_correlations: Dict[str, float]
    reflection: Dict[str, Any]
    approved: bool
    retry_count: int
    max_retries: int
    max_network_retries: int
    failed: bool
    resume_required: bool
    done: bool
    failure_stage: str
    failure_error: str
    early_stop_due_to_bounds: bool


@dataclass
class LangGraphDatasetResult:
    status: str
    metrics: Optional[dict[str, Any]] = None
    resume_required: bool = False
    error: Optional[str] = None


def _resume_state_path(ds_out_dir: str) -> str:
    return os.path.join(ds_out_dir, _RESUME_STATE_FILE)


def _load_resume_state(ds_out_dir: str) -> Optional[dict[str, Any]]:
    path = _resume_state_path(ds_out_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _save_resume_state(ds_out_dir: str, state: dict[str, Any]) -> None:
    try:
        os.makedirs(ds_out_dir, exist_ok=True)
        with open(_resume_state_path(ds_out_dir), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass


def _clear_resume_state(ds_out_dir: str) -> None:
    path = _resume_state_path(ds_out_dir)
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def _append_llm_failure(
    ds_out_dir: str,
    window_offset: int,
    step_index: int,
    stage: str,
    exc: Exception | str,
    payload: Optional[dict[str, Any]] = None,
) -> None:
    try:
        os.makedirs(ds_out_dir, exist_ok=True)
        path = os.path.join(ds_out_dir, "llm_failures.jsonl")
        entry: dict[str, Any] = {
            "window_offset": int(window_offset),
            "step_index": int(step_index),
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "stage": stage,
            "error_type": type(exc).__name__ if isinstance(exc, Exception) else "Error",
            "error": str(exc),
        }
        if payload is not None:
            entry["payload"] = payload
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=json_default) + "\n")
    except Exception:
        pass


def _resume_position_from_coverage(
    coverage: int,
    horizon: int,
    stride: int,
    total_needed: int,
) -> tuple[int, int]:
    coverage = max(0, min(int(coverage), int(total_needed)))
    horizon = max(1, int(horizon))
    stride = max(1, int(stride))
    if coverage <= 0:
        return 0, 0
    if coverage < horizon and coverage < total_needed:
        return 0, 0
    completed_steps = 1 + int(math.ceil(max(0, coverage - horizon) / stride))
    return completed_steps * stride, completed_steps


def _clean_predictions_for_resume(
    out_csv: str,
    target_df: pd.DataFrame,
    total_needed: int,
) -> tuple[int, bool]:
    if not os.path.exists(out_csv):
        return 0, False

    try:
        pred_df = pd.read_csv(out_csv)
    except Exception:
        os.remove(out_csv)
        return 0, True

    original_len = len(pred_df)
    prediction_col = "prediction" if "prediction" in pred_df.columns else None
    if prediction_col is None and "predicted_ans" in pred_df.columns:
        prediction_col = "predicted_ans"
    if (
        original_len == 0
        or "time_stamp" not in pred_df.columns
        or prediction_col is None
        or TIME_COL not in target_df.columns
    ):
        os.remove(out_csv)
        return 0, True

    cleaned = pred_df.copy()
    cleaned["time_stamp"] = pd.to_datetime(cleaned["time_stamp"], errors="coerce")
    cleaned[prediction_col] = pd.to_numeric(cleaned[prediction_col], errors="coerce")
    cleaned = cleaned.dropna(subset=["time_stamp", prediction_col])
    cleaned = cleaned[np.isfinite(cleaned[prediction_col].to_numpy(dtype=float))]

    expected_ts = pd.to_datetime(target_df[TIME_COL].iloc[:total_needed]).reset_index(drop=True)
    expected_index = {ts: idx for idx, ts in enumerate(expected_ts)}
    cleaned = cleaned[cleaned["time_stamp"].isin(expected_index)]

    if cleaned.empty:
        os.remove(out_csv)
        return 0, True

    if "emission_index" in cleaned.columns:
        order_values = pd.to_numeric(cleaned["emission_index"], errors="coerce")
        cleaned = cleaned.assign(_order=order_values).sort_values("_order", kind="mergesort")
    elif {"window_offset", "horizon_index"}.issubset(cleaned.columns):
        window_vals = pd.to_numeric(cleaned["window_offset"], errors="coerce")
        horizon_vals = pd.to_numeric(cleaned["horizon_index"], errors="coerce")
        cleaned = cleaned.assign(
            _order=window_vals.fillna(0) * 1_000_000 + horizon_vals.fillna(0)
        ).sort_values("_order", kind="mergesort")
    else:
        cleaned = cleaned.sort_values("time_stamp", kind="mergesort")
    if "_order" in cleaned.columns:
        cleaned = cleaned.drop(columns="_order")

    cleaned = cleaned.drop_duplicates(subset=["time_stamp"], keep="last")
    available_ts = set(cleaned["time_stamp"])
    contiguous_count = 0
    for ts in expected_ts:
        if ts not in available_ts:
            break
        contiguous_count += 1

    if contiguous_count <= 0:
        os.remove(out_csv)
        return 0, True

    keep_order = {ts: idx for idx, ts in enumerate(expected_ts.iloc[:contiguous_count])}
    cleaned = cleaned[cleaned["time_stamp"].isin(keep_order)]
    cleaned = cleaned.assign(_target_order=cleaned["time_stamp"].map(keep_order))
    cleaned = cleaned.sort_values("_target_order", kind="mergesort").drop(columns="_target_order")

    if prediction_col != "prediction":
        cleaned = cleaned.rename(columns={prediction_col: "prediction"})

    changed = len(cleaned) != original_len or prediction_col != "prediction"
    if changed:
        cleaned.to_csv(out_csv, index=False)
    return int(contiguous_count), changed


def _is_network_error(exc: Exception) -> bool:
    txt = str(exc).lower()
    return any(
        marker in txt
        for marker in [
            "timeout",
            "timed out",
            "connection",
            "network",
            "temporarily unavailable",
            "connection reset",
            "dns",
            "host unreachable",
            "429",
            "502",
            "503",
            "504",
        ]
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(stripped[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _coerce_prediction_list(raw: Any) -> List[float]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        for key in ("predictions", "values", "forecast", "prediction"):
            if key in raw:
                raw = raw[key]
                break
    if isinstance(raw, str):
        text = raw.strip()
        try:
            parsed = json.loads(text)
            return _coerce_prediction_list(parsed)
        except Exception:
            raw = re.findall(r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text)
    if hasattr(raw, "tolist") and not isinstance(raw, (str, bytes, bytearray)):
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)):
        raise ValueError("predictions must be a list, JSON list, or numeric string")
    values: List[float] = []
    for item in raw:
        if isinstance(item, dict):
            for key in ("value", "prediction", "predicted_ans", "forecast"):
                if key in item:
                    item = item[key]
                    break
        try:
            values.append(float(item))
        except Exception:
            continue
    return values


def _normalize_feature_selection(
    ds_out_dir: str,
    selected_features: Any,
    feature_weights: Any,
) -> tuple[List[str], Dict[str, float]]:
    feat_path = os.path.join(ds_out_dir, "features.json")
    if not os.path.exists(feat_path):
        names = [str(v) for v in selected_features] if isinstance(selected_features, list) else []
        weights: Dict[str, float] = {}
        if isinstance(feature_weights, dict):
            for key, value in feature_weights.items():
                try:
                    weights[str(key)] = float(value)
                except Exception:
                    continue
        return names, weights
    try:
        with open(feat_path, "r", encoding="utf-8") as f:
            features = json.load(f)
    except Exception:
        features = {}
    if not isinstance(features, dict) or not features:
        return [], {}

    available = [str(k) for k in features.keys()]
    desired_count = min(3, len(available))
    provided = (
        [str(name) for name in selected_features if str(name) in available]
        if isinstance(selected_features, list)
        else []
    )
    if len(provided) < desired_count:
        for name in available:
            if name not in provided:
                provided.append(name)
            if len(provided) >= desired_count:
                break

    cleaned_weights: Dict[str, float] = {}
    if isinstance(feature_weights, dict):
        for key, value in feature_weights.items():
            if str(key) in provided:
                try:
                    val = float(value)
                except Exception:
                    continue
                if np.isfinite(val) and val >= 0:
                    cleaned_weights[str(key)] = val

    if len(cleaned_weights) != len(provided) or sum(cleaned_weights.values()) <= 0:
        weight = 1.0 / len(provided) if provided else 0.0
        cleaned_weights = {name: weight for name in provided} if provided else {}
    else:
        total = float(sum(cleaned_weights.values())) or 1.0
        cleaned_weights = {name: float(val) / total for name, val in cleaned_weights.items()}
    return provided, cleaned_weights


def _normalize_exogenous_selection(
    ds_out_dir: str,
    exogenous_vars: Any,
    exogenous_feature_selection: Any,
    exogenous_correlations: Any,
    use_exogenous: bool,
) -> tuple[List[str], Dict[str, List[str]], Dict[str, float]]:
    if not use_exogenous:
        return [], {}, {}

    top3_path = os.path.join(ds_out_dir, "exogenous_top3.json")
    feat_path = os.path.join(ds_out_dir, "exogenous_features.json")
    corr_path = os.path.join(ds_out_dir, "exogenous_correlations.json")
    if not (os.path.exists(top3_path) and os.path.exists(feat_path)):
        return [], {}, {}
    try:
        with open(top3_path, "r", encoding="utf-8") as f:
            top3 = json.load(f)
        with open(feat_path, "r", encoding="utf-8") as f:
            exo_feats = json.load(f)
    except Exception:
        return [], {}, {}

    if not isinstance(top3, list):
        top3 = []
    top3 = [str(v) for v in top3]
    if not isinstance(exo_feats, dict):
        exo_feats = {}

    vars_clean = (
        [str(v) for v in exogenous_vars if str(v) in top3]
        if isinstance(exogenous_vars, list)
        else []
    )
    if len(vars_clean) != len(top3):
        vars_clean = list(top3)

    dims_clean: Dict[str, List[str]] = {}
    if isinstance(exogenous_feature_selection, dict):
        for var, dims in exogenous_feature_selection.items():
            if str(var) in top3 and isinstance(dims, list):
                allowed = set((exo_feats.get(str(var)) or {}).keys())
                dims_clean[str(var)] = [str(d) for d in dims if str(d) in allowed]

    for var in top3:
        allowed = list((exo_feats.get(var) or {}).keys())
        selected = list(dims_clean.get(var, []))
        for name in allowed:
            if name not in selected:
                selected.append(name)
            if len(selected) >= min(3, len(allowed)):
                break
        dims_clean[var] = selected

    corr_source: dict[str, Any] = {}
    if os.path.exists(corr_path):
        try:
            with open(corr_path, "r", encoding="utf-8") as f:
                corr_source = json.load(f)
        except Exception:
            corr_source = {}
    corr_clean: Dict[str, float] = {}
    if isinstance(exogenous_correlations, dict):
        for var, val in exogenous_correlations.items():
            if str(var) in top3:
                try:
                    corr_clean[str(var)] = float(val)
                except Exception:
                    continue
    for var in top3:
        if var not in corr_clean:
            try:
                corr_clean[var] = float(corr_source.get(var, 0.0))
            except Exception:
                corr_clean[var] = 0.0
    return vars_clean, dims_clean, corr_clean


class LangGraphOrchestrator:
    def __init__(
        self,
        cfg: ExperimentConfig,
        dataset_briefings: Optional[Dict[str, str]],
        llm: Any,
    ) -> None:
        self.cfg = cfg
        self.dataset_briefings = dataset_briefings or {}
        self.llm = llm

    def run_dataset(
        self,
        ds: DatasetConfig,
        test_df: pd.DataFrame,
        target_df: pd.DataFrame,
        frequency: Optional[str],
    ) -> LangGraphDatasetResult:
        try:
            graph = self._build_window_graph(ds, test_df, target_df)
        except _LangGraphMissing as exc:
            return LangGraphDatasetResult(status="unavailable", error=str(exc))

        ds_out_dir = os.path.join(self.cfg.output_dir, ds.name)
        out_csv = os.path.join(ds_out_dir, "predictions.csv")
        meta_path = os.path.join(ds_out_dir, "metadata.json")
        if os.path.exists(out_csv) and os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    existing_meta = json.load(f) or {}
                if existing_meta.get("chosen_model") != "LLM":
                    os.remove(out_csv)
                    _clear_resume_state(ds_out_dir)
                    print(
                        f"[info] Removed non-LLM predictions for dataset '{ds.name}' before starting LangGraph orchestration."
                    )
            except Exception:
                pass

        total_needed = int(len(target_df))
        if total_needed == 0:
            return LangGraphDatasetResult(
                status="success",
                metrics={"dataset": ds.name, "MSE": np.nan, "MAE": np.nan, "sMAPE": np.nan, "model": "LLM"},
            )

        file_coverage, file_cleaned = _clean_predictions_for_resume(out_csv, target_df, total_needed)
        if file_cleaned:
            print(
                f"[warn] Cleaned inconsistent LLM predictions for dataset '{ds.name}'; contiguous usable coverage is {file_coverage}/{total_needed}."
            )

        horizon = int(ds.predicted_window)
        stride = int(ds.sliding_window)
        resume_state = _load_resume_state(ds_out_dir)
        resume_state_valid = False
        if resume_state is not None:
            try:
                resume_state_valid = (
                    int(resume_state.get("total_len", total_needed)) == total_needed
                    and int(resume_state.get("stride", stride)) == stride
                    and int(resume_state.get("horizon", horizon)) == horizon
                )
            except Exception:
                resume_state_valid = False
            if not resume_state_valid:
                print(
                    f"[warn] Ignoring stored resume state for dataset '{ds.name}' due to configuration mismatch; starting fresh."
                )

        if resume_state_valid and file_coverage <= 0:
            _clear_resume_state(ds_out_dir)
            resume_state_valid = False

        if resume_state_valid or file_coverage > 0:
            current_collected = min(file_coverage, total_needed)
            current_len, step_index = _resume_position_from_coverage(
                current_collected,
                horizon,
                stride,
                total_needed,
            )
            print(
                f"[info] Resuming LangGraph orchestration for dataset '{ds.name}' from step {step_index} with {current_collected}/{total_needed} predictions already collected."
            )
        else:
            if resume_state is not None:
                _clear_resume_state(ds_out_dir)
            current_collected = 0
            current_len = 0
            step_index = 0
            if os.path.exists(out_csv):
                os.remove(out_csv)

        initial_state: WindowState = {
            "dataset_name": ds.name,
            "ds_out_dir": ds_out_dir,
            "total_needed": total_needed,
            "total_len": total_needed,
            "horizon": horizon,
            "stride": stride,
            "look_back": int(ds.look_back),
            "current_collected": int(current_collected),
            "current_len": int(current_len),
            "step_index": int(step_index),
            "retry_count": 0,
            "max_retries": int(os.getenv("LANGGRAPH_MAX_REFLECTION_RETRIES", "2")),
            "max_network_retries": int(os.getenv("LANGGRAPH_MAX_NETWORK_RETRIES", "3")),
            "failed": False,
            "resume_required": False,
            "done": current_collected >= total_needed,
            "early_stop_due_to_bounds": False,
        }

        max_steps = max(100, 12 * (int(math.ceil(total_needed / max(stride, 1))) + 5))
        try:
            final_state = graph.invoke(initial_state, config={"recursion_limit": max_steps})
        except Exception as exc:
            _append_llm_failure(ds_out_dir, current_len, step_index, "langgraph_failed", exc)
            _save_resume_state(
                ds_out_dir,
                {
                    "current_collected": int(current_collected),
                    "current_len": int(current_len),
                    "step_index": int(step_index),
                    "total_needed": int(total_needed),
                    "total_len": int(total_needed),
                    "stride": int(stride),
                    "horizon": int(horizon),
                },
            )
            return LangGraphDatasetResult(status="resume_required", resume_required=True, error=str(exc))

        if final_state.get("failed") or final_state.get("resume_required"):
            _save_resume_state(
                ds_out_dir,
                {
                    "current_collected": int(final_state.get("current_collected", current_collected)),
                    "current_len": int(final_state.get("current_len", current_len)),
                    "step_index": int(final_state.get("step_index", step_index)),
                    "total_needed": int(total_needed),
                    "total_len": int(total_needed),
                    "stride": int(stride),
                    "horizon": int(horizon),
                },
            )
            print(
                f"[info] Saved partial LangGraph results for dataset '{ds.name}'. Re-run the experiment to resume forecasting from step {final_state.get('step_index', step_index)}."
            )
            return LangGraphDatasetResult(
                status="resume_required",
                resume_required=True,
                error=str(final_state.get("failure_error") or final_state.get("failure_stage") or ""),
            )

        _clear_resume_state(ds_out_dir)
        if not os.path.exists(out_csv):
            return LangGraphDatasetResult(status="failed", error="LangGraph completed without predictions.csv")

        pred_df = pd.read_csv(out_csv)
        if "time_stamp" in pred_df.columns:
            pred_df["time_stamp"] = pd.to_datetime(pred_df["time_stamp"])
        y_true, y_pred = align_predictions(target_df, pred_df, ds.name)
        metrics = {
            "dataset": ds.name,
            "MSE": mse(y_true, y_pred),
            "MAE": mae(y_true, y_pred),
            "sMAPE": smape(y_true, y_pred),
            "model": "LLM",
        }
        return LangGraphDatasetResult(status="success", metrics=metrics)

    def _build_window_graph(
        self,
        ds: DatasetConfig,
        test_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> Any:
        try:
            from langgraph.graph import END, StateGraph
        except Exception as exc:
            raise _LangGraphMissing(str(exc)) from exc

        ds_out_dir = os.path.join(self.cfg.output_dir, ds.name)
        out_csv = os.path.join(ds_out_dir, "predictions.csv")

        def prepare_window(state: WindowState) -> WindowState:
            current_collected = int(state.get("current_collected", 0))
            total_needed = int(state["total_needed"])
            if current_collected >= total_needed:
                return {**state, "done": True}
            step_horizon = min(int(state["horizon"]), total_needed - current_collected)
            return {
                **state,
                "window_offset": int(state.get("current_len", 0)),
                "step_horizon": int(step_horizon),
                "retry_count": 0,
                "approved": False,
                "done": False,
            }

        def build_investigator_packet(state: WindowState) -> WindowState:
            try:
                packet = prepare_investor_packet(
                    self.cfg,
                    ds,
                    self.dataset_briefings,
                    int(state["window_offset"]),
                    int(state["step_horizon"]),
                )
                return {**state, "investor_packet": packet}
            except Exception as exc:
                _append_llm_failure(
                    ds_out_dir,
                    int(state.get("window_offset", 0)),
                    int(state.get("step_index", 0)),
                    "investigator_packet_failed",
                    exc,
                )
                return {
                    **state,
                    "failed": True,
                    "resume_required": True,
                    "failure_stage": "investigator_packet_failed",
                    "failure_error": str(exc),
                }

        def generate_forecast(state: WindowState) -> WindowState:
            packet = state.get("investor_packet") or {}
            prompt = self._generation_prompt(ds, state, packet, revision=False)
            max_network_retries = max(0, int(state.get("max_network_retries", 3)))
            network_attempt = 0
            while True:
                try:
                    response = self.llm.invoke(prompt)
                    content = getattr(response, "content", response)
                    raw = _extract_json_object(str(content))
                    if not raw:
                        raise ValueError("LLM did not return a JSON object.")
                    return {**state, "raw_generation": raw}
                except Exception as exc:
                    if _is_network_error(exc) and network_attempt < max_network_retries:
                        network_attempt += 1
                        _append_llm_failure(
                            ds_out_dir,
                            int(state.get("window_offset", 0)),
                            int(state.get("step_index", 0)),
                            "generation_network_retry",
                            exc,
                            {"attempt": network_attempt, "max_attempts": max_network_retries},
                        )
                        print(
                            f"[warn] Network error during LangGraph generation for dataset '{ds.name}' "
                            f"step {state.get('step_index', 0)} "
                            f"(attempt {network_attempt}/{max_network_retries}). Retrying..."
                        )
                        time.sleep(1.0)
                        continue
                    _append_llm_failure(
                        ds_out_dir,
                        int(state.get("window_offset", 0)),
                        int(state.get("step_index", 0)),
                        "generation_failed",
                        exc,
                    )
                    return {
                        **state,
                        "failed": True,
                        "resume_required": True,
                        "failure_stage": "generation_failed",
                        "failure_error": str(exc),
                    }

        def normalize_predictions(state: WindowState) -> WindowState:
            raw = state.get("raw_generation") or {}
            h = int(state["step_horizon"])
            try:
                predictions = _coerce_prediction_list(raw.get("predictions"))
                if not predictions:
                    reference = (state.get("investor_packet") or {}).get("reference_prediction")
                    predictions = _coerce_prediction_list(reference)
                    _append_llm_failure(
                        ds_out_dir,
                        int(state.get("window_offset", 0)),
                        int(state.get("step_index", 0)),
                        "prediction_fallback",
                        "LLM emitted no parseable predictions; used reference_prediction as fallback.",
                    )
                if not predictions:
                    raise ValueError("No predictions available after normalization.")
                if len(predictions) > h:
                    predictions = predictions[:h]
                elif len(predictions) < h:
                    predictions = predictions + [float(predictions[-1])] * (h - len(predictions))
                arr = np.asarray(predictions, dtype=float)
                if not np.all(np.isfinite(arr)):
                    raise ValueError("predictions must be finite numbers")

                selected_features, feature_weights = _normalize_feature_selection(
                    ds_out_dir,
                    raw.get("selected_features"),
                    raw.get("feature_weights"),
                )
                exo_vars, exo_dims, exo_corrs = _normalize_exogenous_selection(
                    ds_out_dir,
                    raw.get("exogenous_vars"),
                    raw.get("exogenous_feature_selection"),
                    raw.get("exogenous_correlations"),
                    bool(getattr(self.cfg, "use_exogenous", False)),
                )
                reasoning = str(raw.get("reasoning_summary") or raw.get("summary") or "").strip()
                if not reasoning:
                    reasoning = self._fallback_reasoning_summary(state.get("investor_packet") or {})
                return {
                    **state,
                    "predictions": arr.tolist(),
                    "reasoning_summary": reasoning,
                    "selected_features": selected_features,
                    "feature_weights": feature_weights,
                    "exogenous_vars": exo_vars,
                    "exogenous_feature_selection": exo_dims,
                    "exogenous_correlations": exo_corrs,
                }
            except Exception as exc:
                _append_llm_failure(
                    ds_out_dir,
                    int(state.get("window_offset", 0)),
                    int(state.get("step_index", 0)),
                    "normalization_failed",
                    exc,
                    raw,
                )
                return {
                    **state,
                    "failed": True,
                    "resume_required": True,
                    "failure_stage": "normalization_failed",
                    "failure_error": str(exc),
                }

        def reflect_forecast(state: WindowState) -> WindowState:
            predictions = [float(v) for v in state.get("predictions", [])]
            packet = state.get("investor_packet") or {}
            reasoning = state.get("reasoning_summary") or ""
            report = assess_forecast(predictions, int(state["step_horizon"]), packet, reasoning)
            chain_report = scan_chain_of_thought(
                predictions,
                int(state["step_horizon"]),
                packet,
                reasoning,
                int(state.get("window_offset", 0)),
            )
            diagnostics = report.setdefault("diagnostics", {})
            diagnostics["chain_of_thought"] = chain_report
            issues = list(report.get("issues") or [])
            numeric_warnings: List[str] = []
            if chain_report.get("unsupported_numbers"):
                samples = ", ".join(
                    str(item.get("raw")) for item in chain_report["unsupported_numbers"][:3]
                )
                numeric_warnings.append(f"Chain-of-thought numeric claims lack grounding: {samples}")
            if chain_report.get("window_claim_mismatches"):
                numeric_warnings.append("Chain-of-thought horizon references contradict request.")
            if chain_report.get("baseline_claim_mismatches"):
                numeric_warnings.append("Chain-of-thought baseline stats contradict investor packet.")
            if numeric_warnings:
                diagnostics["numeric_audit_warnings"] = numeric_warnings
            report["issues"] = issues
            report["approved"] = not issues
            report["window_offset"] = int(state.get("window_offset", 0))
            return {**state, "reflection": report, "approved": bool(report.get("approved"))}

        def revise_forecast(state: WindowState) -> WindowState:
            prompt = self._generation_prompt(
                ds,
                state,
                state.get("investor_packet") or {},
                revision=True,
            )
            max_network_retries = max(0, int(state.get("max_network_retries", 3)))
            network_attempt = 0
            while True:
                try:
                    response = self.llm.invoke(prompt)
                    content = getattr(response, "content", response)
                    raw = _extract_json_object(str(content))
                    if not raw:
                        raise ValueError("LLM did not return a JSON object during revision.")
                    return {
                        **state,
                        "raw_generation": raw,
                        "retry_count": int(state.get("retry_count", 0)) + 1,
                    }
                except Exception as exc:
                    if _is_network_error(exc) and network_attempt < max_network_retries:
                        network_attempt += 1
                        _append_llm_failure(
                            ds_out_dir,
                            int(state.get("window_offset", 0)),
                            int(state.get("step_index", 0)),
                            "revision_network_retry",
                            exc,
                            {"attempt": network_attempt, "max_attempts": max_network_retries},
                        )
                        print(
                            f"[warn] Network error during LangGraph revision for dataset '{ds.name}' "
                            f"step {state.get('step_index', 0)} "
                            f"(attempt {network_attempt}/{max_network_retries}). Retrying..."
                        )
                        time.sleep(1.0)
                        continue
                    _append_llm_failure(
                        ds_out_dir,
                        int(state.get("window_offset", 0)),
                        int(state.get("step_index", 0)),
                        "revision_failed",
                        exc,
                    )
                    return {
                        **state,
                        "failed": True,
                        "resume_required": True,
                        "failure_stage": "revision_failed",
                        "failure_error": str(exc),
                    }

        def reference_prediction_fallback(state: WindowState) -> WindowState:
            packet = state.get("investor_packet") or {}
            h = int(state["step_horizon"])
            try:
                predictions = _coerce_prediction_list(packet.get("reference_prediction"))
                if not predictions:
                    raise ValueError("reference_prediction is unavailable.")
                if len(predictions) > h:
                    predictions = predictions[:h]
                elif len(predictions) < h:
                    predictions = predictions + [float(predictions[-1])] * (h - len(predictions))
                reasoning = self._fallback_reasoning_summary(packet)
                _append_llm_failure(
                    ds_out_dir,
                    int(state.get("window_offset", 0)),
                    int(state.get("step_index", 0)),
                    "reference_prediction_fallback",
                    "Reflector rejected LLM forecast; persisted reference_prediction fallback.",
                    state.get("reflection") or {},
                )
                return {
                    **state,
                    "predictions": predictions,
                    "reasoning_summary": reasoning,
                    "approved": True,
                    "reflection": {
                        "approved": True,
                        "issues": [],
                        "notes": "Persisted reference_prediction fallback after reflection rejection.",
                        "fallback_from_reflection": state.get("reflection") or {},
                    },
                }
            except Exception as exc:
                _append_llm_failure(
                    ds_out_dir,
                    int(state.get("window_offset", 0)),
                    int(state.get("step_index", 0)),
                    "reference_prediction_fallback_failed",
                    exc,
                )
                return {
                    **state,
                    "failed": True,
                    "resume_required": True,
                    "failure_stage": "reference_prediction_fallback_failed",
                    "failure_error": str(exc),
                }

        def persist_predictions(state: WindowState) -> WindowState:
            try:
                os.makedirs(ds_out_dir, exist_ok=True)
                h = int(state["step_horizon"])
                start_idx = int(state["look_back"]) + int(state["window_offset"])
                end_idx = min(start_idx + h, len(test_df))
                if end_idx <= start_idx:
                    return {**state, "early_stop_due_to_bounds": True, "done": True}
                timestamps = pd.to_datetime(test_df[TIME_COL].iloc[start_idx:end_idx]).tolist()
                predictions = [float(v) for v in state.get("predictions", [])][: len(timestamps)]
                if len(predictions) < len(timestamps):
                    raise ValueError("Prediction length is shorter than available timestamps.")

                existing_df: Optional[pd.DataFrame] = None
                if os.path.exists(out_csv):
                    existing_df = pd.read_csv(out_csv)
                if existing_df is not None and "emission_index" in existing_df.columns:
                    try:
                        start_sequence = (
                            int(pd.to_numeric(existing_df["emission_index"], errors="coerce").max(skipna=True) or -1)
                            + 1
                        )
                    except Exception:
                        start_sequence = int(existing_df.shape[0])
                elif existing_df is not None:
                    start_sequence = int(existing_df.shape[0])
                else:
                    start_sequence = 0

                new_chunk = pd.DataFrame(
                    {
                        "time_stamp": pd.to_datetime(timestamps),
                        "prediction": predictions,
                        "window_offset": int(state["window_offset"]),
                        "horizon_index": list(range(len(predictions))),
                        "emission_index": np.arange(
                            start_sequence,
                            start_sequence + len(predictions),
                            dtype=int,
                        ),
                    }
                )
                combined = (
                    pd.concat([existing_df, new_chunk], ignore_index=True)
                    if existing_df is not None and len(existing_df) > 0
                    else new_chunk
                )
                combined.to_csv(out_csv, index=False)

                with open(os.path.join(ds_out_dir, "chain_of_thought.log"), "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "window_offset": int(state["window_offset"]),
                                "timestamp": pd.Timestamp.utcnow().isoformat(),
                                "content": state.get("reasoning_summary") or "",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                with open(os.path.join(ds_out_dir, "reflector_report.jsonl"), "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "window_offset": int(state["window_offset"]),
                                **(state.get("reflection") or {"approved": True, "issues": []}),
                            },
                            ensure_ascii=False,
                            default=json_default,
                        )
                        + "\n"
                    )

                selected_features = state.get("selected_features") or []
                feature_weights = state.get("feature_weights") or {}
                with open(os.path.join(ds_out_dir, "selected_features.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "selected_features": selected_features,
                            "feature_weights": feature_weights,
                        },
                        f,
                        indent=2,
                    )

                meta: Dict[str, Any] = {
                    "dataset": ds.name,
                    "chosen_model": "LLM",
                    "frequency": (state.get("investor_packet") or {}).get("frequency"),
                    "look_back": int(ds.look_back),
                    "predicted_window": int(state["step_horizon"]),
                    "orchestration_backend": "langgraph",
                    "features_used": {
                        "selected_features": selected_features,
                        "feature_weights": feature_weights,
                    },
                }
                exo_vars = state.get("exogenous_vars") or []
                exo_dims = state.get("exogenous_feature_selection") or {}
                exo_corrs = state.get("exogenous_correlations") or {}
                if exo_vars or exo_dims or exo_corrs:
                    exo_sel = {
                        "exogenous_vars": exo_vars,
                        "exogenous_feature_selection": exo_dims,
                        "exogenous_correlations": exo_corrs,
                    }
                    with open(os.path.join(ds_out_dir, "exogenous_selected_features.json"), "w", encoding="utf-8") as f:
                        json.dump(exo_sel, f, indent=2)
                    meta["exogenous_used"] = exo_sel
                with open(os.path.join(ds_out_dir, "metadata.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                return state
            except Exception as exc:
                _append_llm_failure(
                    ds_out_dir,
                    int(state.get("window_offset", 0)),
                    int(state.get("step_index", 0)),
                    "persist_predictions_failed",
                    exc,
                )
                return {
                    **state,
                    "failed": True,
                    "resume_required": True,
                    "failure_stage": "persist_predictions_failed",
                    "failure_error": str(exc),
                }

        def update_coverage(state: WindowState) -> WindowState:
            total_needed = int(state["total_needed"])
            new_collected, file_cleaned = _clean_predictions_for_resume(out_csv, target_df, total_needed)
            if file_cleaned:
                print(
                    f"[warn] Cleaned inconsistent LangGraph predictions for dataset '{ds.name}' after step {state.get('step_index', 0)}; contiguous usable coverage is {new_collected}/{total_needed}."
                )
            added = int(new_collected) - int(state.get("current_collected", 0))
            if added <= 0 and int(new_collected) < total_needed:
                _append_llm_failure(
                    ds_out_dir,
                    int(state.get("window_offset", 0)),
                    int(state.get("step_index", 0)),
                    "no_new_predictions",
                    "No additional predictions were added.",
                )
                return {
                    **state,
                    "failed": True,
                    "resume_required": True,
                    "failure_stage": "no_new_predictions",
                    "failure_error": "No additional predictions were added.",
                }

            next_state: WindowState = {
                **state,
                "current_collected": int(new_collected),
                "current_len": int(state.get("current_len", 0)) + int(state["stride"]),
                "step_index": int(state.get("step_index", 0)) + 1,
                "done": int(new_collected) >= total_needed,
            }
            _save_resume_state(
                ds_out_dir,
                {
                    "current_collected": int(next_state["current_collected"]),
                    "current_len": int(next_state["current_len"]),
                    "step_index": int(next_state["step_index"]),
                    "total_needed": int(total_needed),
                    "total_len": int(total_needed),
                    "stride": int(next_state["stride"]),
                    "horizon": int(next_state["horizon"]),
                },
            )
            print(
                f"[info] Dataset '{ds.name}': collected {next_state['current_collected']}/{total_needed} predictions via LangGraph."
            )
            return next_state

        def route_after_prepare(state: WindowState) -> str:
            if state.get("done"):
                return "done"
            return "continue"

        def route_after_packet(state: WindowState) -> str:
            if state.get("failed"):
                return "done"
            return "continue"

        def route_after_generation(state: WindowState) -> str:
            if state.get("failed"):
                return "done"
            return "continue"

        def route_reflection(state: WindowState) -> str:
            if state.get("failed"):
                return "done"
            if state.get("approved"):
                return "approved"
            if int(state.get("retry_count", 0)) < int(state.get("max_retries", 2)):
                return "revise"
            return "fallback"

        def route_next_window(state: WindowState) -> str:
            if state.get("failed") or state.get("done"):
                return "done"
            return "next"

        graph = StateGraph(WindowState)
        graph.add_node("prepare_window", prepare_window)
        graph.add_node("build_investigator_packet", build_investigator_packet)
        graph.add_node("generate_forecast", generate_forecast)
        graph.add_node("normalize_predictions", normalize_predictions)
        graph.add_node("reflect_forecast", reflect_forecast)
        graph.add_node("revise_forecast", revise_forecast)
        graph.add_node("reference_prediction_fallback", reference_prediction_fallback)
        graph.add_node("persist_predictions", persist_predictions)
        graph.add_node("update_coverage", update_coverage)
        graph.set_entry_point("prepare_window")
        graph.add_conditional_edges(
            "prepare_window",
            route_after_prepare,
            {"continue": "build_investigator_packet", "done": END},
        )
        graph.add_conditional_edges(
            "build_investigator_packet",
            route_after_packet,
            {"continue": "generate_forecast", "done": END},
        )
        graph.add_conditional_edges(
            "generate_forecast",
            route_after_generation,
            {"continue": "normalize_predictions", "done": END},
        )
        graph.add_conditional_edges(
            "normalize_predictions",
            route_after_generation,
            {"continue": "reflect_forecast", "done": END},
        )
        graph.add_conditional_edges(
            "reflect_forecast",
            route_reflection,
            {
                "approved": "persist_predictions",
                "revise": "revise_forecast",
                "fallback": "reference_prediction_fallback",
                "done": END,
            },
        )
        graph.add_edge("revise_forecast", "normalize_predictions")
        graph.add_edge("reference_prediction_fallback", "persist_predictions")
        graph.add_edge("persist_predictions", "update_coverage")
        graph.add_conditional_edges(
            "update_coverage",
            route_next_window,
            {"next": "prepare_window", "done": END},
        )
        return graph.compile()

    def _generation_prompt(
        self,
        ds: DatasetConfig,
        state: WindowState,
        packet: Dict[str, Any],
        revision: bool,
    ) -> str:
        reflection = state.get("reflection") or {}
        prior = state.get("raw_generation") or {}
        briefing = self.dataset_briefings.get(ds.name, "")
        mode = "Revise the rejected forecast" if revision else "Generate a forecast"
        revision_block = ""
        if revision:
            revision_block = (
                "\nReflection report:\n"
                + json.dumps(reflection, ensure_ascii=False, default=json_default)
                + "\nPrior forecast draft:\n"
                + json.dumps(prior, ensure_ascii=False, default=json_default)
            )
        return (
            "You are GeneratorAgent in the AlphaCast LangGraph workflow. "
            "Forecast using the supplied evidence packet and return strict JSON only.\n\n"
            f"Task: {mode} for dataset {ds.name}, window_offset={state.get('window_offset')}, "
            f"horizon={state.get('step_horizon')}.\n"
            "Use reference_prediction as the anchor. Adjust only when features, neighbor behavior, "
            "or exogenous outlook clearly justify the change.\n"
            "Return exactly this JSON object shape:\n"
            "{\n"
            '  "predictions": [float, ...],\n'
            '  "reasoning_summary": "concise evidence-grounded summary, no hidden chain of thought",\n'
            '  "selected_features": ["feature_name", ...],\n'
            '  "feature_weights": {"feature_name": 0.0},\n'
            '  "exogenous_vars": ["var", ...],\n'
            '  "exogenous_feature_selection": {"var": ["dimension", ...]},\n'
            '  "exogenous_correlations": {"var": 0.0}\n'
            "}\n"
            f"The predictions array must contain exactly {state.get('step_horizon')} finite numbers. "
            "Feature weights must be non-negative and sum to 1 when features are present. "
            "If no features or exogenous evidence are available, use [] and {}.\n"
            + (f"\nDataset briefing:\n{briefing}\n" if briefing else "")
            + revision_block
            + "\nInvestigator packet:\n"
            + json.dumps(packet, ensure_ascii=False, default=json_default)
        )

    @staticmethod
    def _fallback_reasoning_summary(packet: Dict[str, Any]) -> str:
        reference = packet.get("reference_prediction")
        ref_note = "Reference prediction and available context were used as guidance."
        if isinstance(reference, list) and reference:
            try:
                clean = [float(v) for v in reference if np.isfinite(float(v))]
                if clean:
                    ref_note = (
                        f"Reference prediction length {len(clean)}; first={clean[0]:.6g}, "
                        f"last={clean[-1]:.6g}, mean={float(np.mean(clean)):.6g}."
                    )
            except Exception:
                pass
        top_vars = packet.get("exogenous_top3") or packet.get("top_exogenous_vars")
        exo_note = ""
        if isinstance(top_vars, list) and top_vars:
            exo_note = f" Exogenous variables considered: {', '.join(str(v) for v in top_vars[:3])}."
        return "LangGraph fallback summary: " + ref_note + exo_note


def build_langgraph_or_none(
    cfg: ExperimentConfig | None = None,
    dataset_briefings: Optional[Dict[str, str]] = None,
) -> Optional[LangGraphOrchestrator]:
    load_dotenv(override=True)
    if cfg is None:
        return None

    model_name = os.getenv("MODEL")
    pya_model = os.getenv("PYA_MODEL")
    if not model_name and pya_model:
        model_name = pya_model.split(":", 1)[1] if ":" in pya_model else pya_model
    if not model_name:
        return None

    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        return None

    kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0")),
    }
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    if base_url:
        kwargs["base_url"] = base_url
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    try:
        llm = ChatOpenAI(**kwargs)
    except Exception:
        return None
    if base_url:
        print(f"[info] Using OpenAI base URL: {base_url}")
    return LangGraphOrchestrator(cfg, dataset_briefings, llm)


__all__ = [
    "LangGraphDatasetResult",
    "LangGraphOrchestrator",
    "build_langgraph_or_none",
]
