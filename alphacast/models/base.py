from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol, Tuple, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from ..DeepLearningModels.Autoformer import Model as Autoformer 
from ..DeepLearningModels.DLinear import Model as DLinear
from ..DeepLearningModels.iTransformer import Model as iTransformer
from ..DeepLearningModels.PatchTST import Model as PatchTST
from ..DeepLearningModels.TimesNet import Model as TimesNet
from ..DeepLearningModels.TimeXer import Model as TimeXer
from ..utils.timefeatures import time_features
import torch
import os
import logging
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel as StatsmodelsTheta
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoCES as SFAutoCES,
    AutoETS as SFAutoETS,
    CrostonClassic as SFCrostonClassic,
    DynamicOptimizedTheta as SFDynamicOptimizedTheta,
    ZeroModel as SFZeroModel,
)


@dataclass
class _DeepLearningRuntimeContext:
    checkpoints: Dict[str, str]
    pred_len: Optional[int] = None


_ACTIVE_DL_CONTEXT: Optional[_DeepLearningRuntimeContext] = None


def configure_deep_learning_runtime(
    checkpoints: Optional[Dict[str, str]],
    pred_len: Optional[int],
) -> None:
    """Register dataset-specific resources for deep learning models."""
    global _ACTIVE_DL_CONTEXT
    if not checkpoints and pred_len is None:
        _ACTIVE_DL_CONTEXT = None
        return

    normalized: Dict[str, str] = {}
    for alias, path in (checkpoints or {}).items():
        if not path:
            continue
        normalized[str(alias).lower()] = path

    _ACTIVE_DL_CONTEXT = _DeepLearningRuntimeContext(
        checkpoints=normalized,
        pred_len=int(pred_len) if pred_len is not None else None,
    )


def _active_dl_context() -> Optional[_DeepLearningRuntimeContext]:
    return _ACTIVE_DL_CONTEXT


def _resolve_checkpoint(alias: str) -> Optional[str]:
    ctx = _active_dl_context()
    if ctx is None:
        return None
    return ctx.checkpoints.get(alias.lower())


def _resolve_pred_len(default: int) -> int:
    ctx = _active_dl_context()
    if ctx is None or ctx.pred_len is None:
        return int(default)
    return int(ctx.pred_len)

class ForecastModel(Protocol):
    alias: str

    def fit(self, y: np.ndarray, season_length: Optional[int] = None, **kwargs) -> None: ...
    def predict(self, h: int,**kwargs) -> np.ndarray: ...



@dataclass
class ArimaModel(ForecastModel):
    alias: str = "AutoARIMA"
    order: Optional[Tuple[int, int, int]] = None
    seasonal_order: Optional[Tuple[int, int, int, int]] = None

    _fitted: Optional[any] = None

    def fit(self, y: np.ndarray, season_length: Optional[int] = None, **kwargs) -> None:
        # Simple heuristic ARIMA: try (p,d,q)=(1,1,1) and seasonal if season_length provided
        order = self.order or (1, 1, 1)
        if season_length and season_length > 1:
            seasonal_order = self.seasonal_order or (1, 1, 1, int(season_length))
        else:
            seasonal_order = (0, 0, 0, 0)
        self._fitted = ARIMA(y, order=order, seasonal_order=seasonal_order).fit()

    def predict(self, h: int, **kwargs) -> np.ndarray:
        assert self._fitted is not None
        fcst = self._fitted.forecast(steps=h)
        return np.asarray(fcst, dtype=float)


@dataclass
class EtsModel(ForecastModel):
    alias: str = "AutoETS"
    seasonal: Optional[str] = "add"

    _fitted: Optional[any] = None

    def fit(self, y: np.ndarray, season_length: Optional[int] = None, **kwargs) -> None:
        slen = int(season_length) if season_length and season_length > 1 else None
        self._fitted = ExponentialSmoothing(
            y,
            trend="add",
            seasonal=self.seasonal if slen else None,
            seasonal_periods=slen,
        ).fit()

    def predict(self, h: int,**kwargs) -> np.ndarray:
        assert self._fitted is not None
        fcst = self._fitted.forecast(h)
        return np.asarray(fcst, dtype=float)


@dataclass
class SeasonalNaiveModel(ForecastModel):    
    alias: str = "SeasonalNaive"

    def fit(self, y: np.ndarray, season_length: Optional[int] = None, **kwargs) -> None:
        self._y = y
        self._s = int(season_length) if season_length and season_length > 1 else 1

    def predict(self, h: int, **kwargs) -> np.ndarray:
        s = max(1, getattr(self, "_s", 1))
        history = getattr(self, "_y", np.array([], dtype=float))
        if len(history) == 0:
            return np.zeros(h, dtype=float)
        reps = int(np.ceil(h / s))
        pattern = history[-s:]
        out = np.tile(pattern, reps)[:h]
        return np.asarray(out, dtype=float)


@dataclass
class HistoricAverageModel(ForecastModel):
    alias: str = "HistoricAverage"

    def fit(self, y: np.ndarray, season_length: Optional[int] = None,**kwargs) -> None:
        self._mean = float(np.mean(y)) if len(y) else 0.0

    def predict(self, h: int, **kwargs) -> np.ndarray:
        return np.full(h, getattr(self, "_mean", 0.0), dtype=float)

@dataclass
class AutoformerModel(ForecastModel):
    alias: str = "Autoformer"

    # === Required: checkpoint path consistent with training ===
    model_path: str = ""

    # Runtime device and decoder history length
    label_len: int = 48
    timefeat_freq: str = "min"   # Match DLinearModel: encode using minute frequency

    # Runtime cache
    _model: Optional[torch.nn.Module] = None
    _args: Optional[object] = None
    _x_enc: Optional[torch.Tensor] = None
    _x_mark_enc: Optional[torch.Tensor] = None
    _enc_len: int = 0
    _device: str = "cpu"

    def _build_x_mark(self, ts_list) -> torch.Tensor:
        """Generate encoder time features from historical timestamps, matching the DLinearModel format."""
        df = pd.DataFrame({"date": pd.to_datetime(list(ts_list))})
        stamp = time_features(pd.to_datetime(df["date"].values), freq=self.timefeat_freq)  # [F, L]
        stamp = stamp.transpose(1, 0)                                                      # [L, F]
        return torch.from_numpy(np.asarray(stamp)).float().unsqueeze(0)                    # [1, L, F]

    def fit(self, y: np.ndarray, season_length: Optional[int] = None, **kwargs) -> None:
        ts = kwargs.get("timestamps")
        if ts is None:
            raise ValueError("AutoformerModel.fit requires 'timestamps' aligned with y.")

        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]  # Univariate -> [L, 1]
        L, C = y.shape
        if C != 1:
            raise ValueError(f"AutoformerModel expects univariate input (enc_in=1), got enc_in={C}")
        if len(ts) != L:
            raise ValueError(f"timestamps length {len(ts)} != window length {L}")

        # Cache encoder values and time features
        self._enc_len = L
        self._x_enc = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(self._device)  # [1, L, 1]
        self._x_mark_enc = self._build_x_mark(ts).to(self._device)                        # [1, L, Fm]

        # Assemble args to mirror the training configuration (DLinearModel style)
        class Args: ...
        args = Args()
        data_name = "power"
        args.task_name = "long_term_forecast"
        args.is_training = 0
        args.model = "Autoformer"
        args.freq = "t"                 # Keep consistent with the paired DLinearModel configuration
        args.checkpoints = "./checkpoints/"
        args.seq_len = L
        args.label_len = 48
        args.pred_len = 24              # The effective horizon is set again inside predict(h)
        args.seasonal_patterns = "Monthly"
        args.inverse = False
        args.individual = 0
        args.expand = 2
        args.d_conv = 4
        args.top_k = 5
        args.num_kernels = 6

        # Univariate configuration
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1
        args.features = "S"
        args.target = "real_power"

        # Model core & training hyperparameters (aligned with the provided DLinearModel setup)
        args.d_model = 512
        args.n_heads = 8
        args.e_layers = 2
        args.d_layers = 1
        args.d_ff = 2048
        args.moving_avg = 25
        args.factor = 3
        args.distil = True
        args.dropout = 0.1
        args.embed = "timeF"
        args.activation = "gelu"
        args.output_attention = True
        args.channel_independence = 1
        args.decomp_method = "moving_avg"
        args.use_norm = 0
        args.down_sampling_layers = 0
        args.down_sampling_window = 1
        args.down_sampling_method = None
        args.seg_len = 48
        args.num_workers = 10
        args.itr = 1
        args.train_epochs = 100
        args.batch_size = 32
        args.patience = 10
        args.learning_rate = 0.0001
        args.des = "Exp"
        args.loss = "MSE"
        args.lradj = "type1"
        args.use_amp = False
        args.use_gpu = self._device.startswith("cuda")
        args.gpu = 0
        args.use_multi_gpu = False
        args.devices = "0,1,2,3"
        args.p_hidden_dims = [128, 128]
        args.p_hidden_layers = 2
        args.use_dtw = False
        args.augmentation_ratio = 0
        args.seed = 2
        args.extra_tag = ""
        args.use_patch = False
        args.patch_len = 16
        args.model_id = f"{data_name}_96_96"
        args.data = "custom"
        args.data_path = f"{data_name}.csv"

        args.pred_len = _resolve_pred_len(args.pred_len)
        self._args = args

        runtime_path = _resolve_checkpoint(self.alias)
        if runtime_path:
            self.model_path = runtime_path
        if not self.model_path:
            raise FileNotFoundError(
                "Autoformer checkpoint path is not configured for the active dataset."
            )
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"checkpoint not found: {self.model_path}")

        # Instantiate and load weights
        self._model = Autoformer(args).to(self._device)
        try:
            state = torch.load(self.model_path, map_location=self._device, weights_only=True)
        except TypeError:
            state = torch.load(self.model_path, map_location=self._device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self._model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[Autoformer] missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
        self._model.eval()
        
    def predict(self, h: int, **kwargs) -> np.ndarray:
        if self._model is None or self._args is None or self._x_enc is None or self._x_mark_enc is None:
            raise RuntimeError("AutoformerModel not fitted.")

        # Write pred_len because some implementations rely on it
        self._args.pred_len = int(h)
        if hasattr(self._model, "pred_len"):
            setattr(self._model, "pred_len", int(h))
        if hasattr(self._model, "args"):
            setattr(self._model, "args", self._args)

        # === Build decoder-side inputs carefully to avoid shape errors ===
        B = self._x_enc.size(0)              # Typically 1
        C = self._x_enc.size(-1)             # Univariate = 1
        Fm = self._x_mark_enc.size(-1)       # Time-feature dimension
        L_lab = min(self.label_len, self._enc_len)

        # x_dec: take the tail label_len slice plus h zero-padded future steps
        x_hist = self._x_enc[:, -L_lab:, :]                                  # [B, label_len, C]
        x_zeros = torch.zeros(B, h, C, device=self._x_enc.device, dtype=self._x_enc.dtype)
        x_dec = torch.cat([x_hist, x_zeros], dim=1)                           # [B, label_len+h, C]

        # x_mark_dec: build in sync; fill future timestamps with zeros as placeholders
        mark_hist = self._x_mark_enc[:, -L_lab:, :]                           # [B, label_len, Fm]
        mark_future = torch.zeros(B, h, Fm, device=self._x_mark_enc.device, dtype=self._x_mark_enc.dtype)
        x_mark_dec = torch.cat([mark_hist, mark_future], dim=1)               # [B, label_len+h, Fm]

        with torch.no_grad():
            out = self._model(self._x_enc, self._x_mark_enc, x_dec, x_mark_dec)
            if isinstance(out, tuple):
                out = out[0]
            if out.ndim == 3:
                out = out[:, -h:, :1]  # Keep only the last h steps for the univariate channel
            y_hat = out.squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32)

        if y_hat.shape[0] != h:
            raise RuntimeError(f"AutoformerModel returned {y_hat.shape[0]} steps, expected {h}.")
        return y_hat


@dataclass
class DLinearModel(ForecastModel):
    alias: str = "DLinear"

    # === Required: checkpoint path consistent with training ===
    model_path: str = ""

    # Runtime device and decoder history length
    label_len: int = 48           
    timefeat_freq: str = "min"      # Training used 'h'; change to '15min' for 15-minute data if needed

    # Runtime cache
    _model: Optional[torch.nn.Module] = None
    _args: Optional[object] = None
    _x_enc: Optional[torch.Tensor] = None
    _x_mark_enc: Optional[torch.Tensor] = None
    _enc_len: int = 0
    _device: str = "cpu"

    def _build_x_mark(self, ts_list) -> torch.Tensor:
        """Generate encoder time features from historical timestamps; DLinear does not use them but we keep the interface uniform."""
        df = pd.DataFrame({"date": pd.to_datetime(list(ts_list))})
        stamp = time_features(pd.to_datetime(df["date"].values), freq=self.timefeat_freq)  # [F, L]
        stamp = stamp.transpose(1, 0)                                                      # [L, F]
        return torch.from_numpy(np.asarray(stamp)).float().unsqueeze(0)                    # [1, L, F]

    def fit(self, y: np.ndarray, season_length: Optional[int] = None, **kwargs) -> None:
        ts = kwargs.get("timestamps")
        if ts is None:
            raise ValueError("DLinearModel.fit requires 'timestamps' aligned with y.")


        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]   # Univariate -> [L, 1]
        L, C = y.shape
        if C != 1:
            raise ValueError(f"DLinearModel expects univariate input (enc_in=1), got enc_in={C}")
        if len(ts) != L:
            raise ValueError(f"timestamps length {len(ts)} != window length {L}")

        # Cache encoder values and time features (consistent interface)
        self._enc_len = L
        self._x_enc = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(self._device)  # [1, L, 1]
        self._x_mark_enc = self._build_x_mark(ts).to(self._device)                        # [1, L, Fm]

        # Assemble args identical to training with the provided parameters
        class Args: ...
        args = Args()
        data_name = "power"
        args.task_name = "long_term_forecast"
        args.is_training = 0
        args.model = "DLinear"
        args.freq = "t"
        args.checkpoints = "./checkpoints/"
        args.seq_len = L
        args.label_len = 48  # Slightly more stable
        args.pred_len = 24                        # The forecast horizon is set inside predict(h)
        args.seasonal_patterns = "Monthly"
        args.inverse = False
        args.individual = 0
        args.expand = 2
        args.d_conv = 4
        args.top_k = 5
        args.num_kernels = 6

        # Univariate configuration (predict only the last column → S)
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1
        args.features = "S"
        args.target = "real_power"

        # Keep the remaining hyperparameters identical to training
        args.d_model = 512
        args.n_heads = 8
        args.e_layers = 2
        args.d_layers = 1
        args.d_ff = 2048
        args.moving_avg = 25
        args.factor = 3
        args.distil = True
        args.dropout = 0.1
        args.embed = "timeF"
        args.activation = "gelu"
        args.output_attention = True
        args.channel_independence = 1
        args.decomp_method = "moving_avg"
        args.use_norm = 0
        args.down_sampling_layers = 0
        args.down_sampling_window = 1
        args.down_sampling_method = None
        args.seg_len = 48
        args.num_workers = 10
        args.itr = 1
        args.train_epochs = 100
        args.batch_size = 32
        args.patience = 10
        args.learning_rate = 0.0001
        args.des = "Exp"
        args.loss = "MSE"
        args.lradj = "type1"
        args.use_amp = False
        args.use_gpu = self._device.startswith("cuda")
        args.gpu = 0
        args.use_multi_gpu = False
        args.devices = "0,1,2,3"
        args.p_hidden_dims = [128, 128]
        args.p_hidden_layers = 2
        args.use_dtw = False
        args.augmentation_ratio = 0
        args.seed = 2
        args.extra_tag = ""
        args.use_patch = False
        args.patch_len = 16
        args.model_id = f"{data_name}_96_96"
        args.data = "custom"
        args.data_path = f"{data_name}.csv"

        args.pred_len = _resolve_pred_len(args.pred_len)
        self._args = args

        runtime_path = _resolve_checkpoint(self.alias)
        if runtime_path:
            self.model_path = runtime_path
        if not self.model_path:
            raise FileNotFoundError(
                "DLinear checkpoint path is not configured for the active dataset."
            )
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"checkpoint not found: {self.model_path}")

        # Instantiate and load weights
        self._model = DLinear(args).to(self._device)
        try:
            state = torch.load(self.model_path, map_location=self._device, weights_only=True)
        except TypeError:
            state = torch.load(self.model_path, map_location=self._device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self._model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[DLinear] missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
        self._model.eval()

    def predict(self, h: int, **kwargs) -> np.ndarray:
        if self._model is None or self._args is None or self._x_enc is None:
            raise RuntimeError("DLinearModel not fitted.")

        # Write pred_len into args because some implementations depend on it
        self._args.pred_len = int(h)
        if hasattr(self._model, "args"):
            setattr(self._model, "args", self._args)

        with torch.no_grad():
            # Prefer the signature with only x_enc; fall back to the legacy signature otherwise
            try:
                # Accept only the 4-argument forward signature: forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
                out = self._model(self._x_enc, self._x_mark_enc, 1, 1)
            except TypeError as e:
                raise TypeError(
                    f"{self.alias} requires a 4-arg forward(x_enc, x_mark_enc, x_dec, x_mark_dec), "
                    f"but the loaded model does not support it. Please align the model/ckpt."
                ) from e
            if isinstance(out, tuple):
                out = out[0]
            if out.ndim == 3:
                out = out[:, -h:, :1]  # Take the last h steps of the univariate output
            y_hat = out.squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32)

        if y_hat.shape[0] != h:
            raise RuntimeError(f"DLinearModel returned {y_hat.shape[0]} steps, expected {h}.")
        return y_hat

@dataclass
class PatchTSTModel(ForecastModel):
    alias: str = "PatchTST"

    # === Required: checkpoint path exactly matching training (prefer univariate enc_in=1, features='S') ===
    model_path: str = ""
    # Runtime device and decoder history length
    label_len: int = 48              # Training previously used 84; adjust as needed (≤ seq_len)
    # Encoder time-feature frequency (match training; use 't' for minutes or '15min' if supported)
    timefeat_freq: str = "min"


    # Runtime cache
    _model: Optional[torch.nn.Module] = None
    _args: Optional[object] = None
    _x_enc: Optional[torch.Tensor] = None
    _x_mark_enc: Optional[torch.Tensor] = None
    _enc_len: int = 0

    def _build_x_mark(self, ts_list) -> torch.Tensor:
        """Generate encoder time features from history only; do not create future timestamps."""
        df = pd.DataFrame({"date": pd.to_datetime(list(ts_list))})
        stamp = time_features(pd.to_datetime(df["date"].values), freq=self.timefeat_freq)  # [F, L]
        stamp = stamp.transpose(1, 0)                                                      # [L, F]
        return torch.from_numpy(np.asarray(stamp)).float().unsqueeze(0)                    # [1, L, F]


    def fit(self, y: np.ndarray, season_length: Optional[int] = None, **kwargs) -> None:
        ts = kwargs.get("timestamps")
        if ts is None:
            raise ValueError("PatchTSTModel.fit requires 'timestamps' aligned with y.")

        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]                           # Univariate -> [L, 1]
        L, C = y.shape
        if C != 1:
            raise ValueError(f"PatchTSTModel expects univariate input (enc_in=1), got enc_in={C}")

        # Cache encoder values and time features
        self._enc_len = L
        self._x_enc = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to('cpu')  # [1, L, 1]
        self._x_mark_enc = self._build_x_mark(ts).to('cpu')                        # [1, L, Fm]

        class Args: pass
        args = Args()
        data_name = 'power'   
        args.task_name = 'long_term_forecast'
        args.is_training = 0
        args.model = 'PatchTST'
        
        args.freq = 't'
        args.checkpoints = './checkpoints/'

        # Lengths: seq_len=L, label_len≤L, pred_len is set during predict(h)
        args.seq_len = L
        args.label_len = 48
        args.pred_len = 24

        # --- Architecture & training hyperparameters (identical to training) ---
        args.seasonal_patterns = 'Monthly'
        args.inverse = False
        args.individual = 0
        args.expand = 2
        args.d_conv = 4
        args.top_k = 5
        args.num_kernels = 6

        # Univariate configuration
        args.enc_in = 1
        args.dec_in = 1
        args.c_out  = 1
        args.features = 'S'
        args.target = 'real_power'

        # Transformer backbone
        args.d_model = 512
        args.n_heads = 2
        args.e_layers = 1
        args.d_layers = 1
        args.d_ff = 2048
        args.moving_avg = 25
        args.factor = 3
        args.distil = True
        args.dropout = 0.1
        args.embed = 'timeF'
        args.activation = 'gelu'
        args.output_attention =True
        args.channel_independence = 1
        args.decomp_method = 'moving_avg'
        args.use_norm = 0
        args.down_sampling_layers = 0
        args.down_sampling_window = 1
        args.down_sampling_method = None
        args.seg_len = 48
        args.num_workers = 10
        args.itr = 1
        args.train_epochs = 100
        args.batch_size = 32
        args.patience = 10
        args.learning_rate = 0.0001
        args.des = 'Exp'
        args.loss = 'MSE'
        args.lradj = 'type1'
        args.use_amp = False
        args.use_gpu = True
        args.gpu = 0
        args.use_multi_gpu = False
        args.devices = '0,1,2,3'
        args.p_hidden_dims = [128, 128]
        args.p_hidden_layers = 2
        args.use_dtw = False
        args.augmentation_ratio = 0
        args.seed = 2
        args.extra_tag = ""
        args.use_patch = False
        args.patch_len = 16
        args.model_id = f'{data_name}_96_96'
        args.data = 'custom'
        args.data_path = f'{data_name}.csv'

        args.pred_len = _resolve_pred_len(args.pred_len)
        self._args = args

        runtime_path = _resolve_checkpoint(self.alias)
        if runtime_path:
            self.model_path = runtime_path
        if not self.model_path:
            raise FileNotFoundError(
                "PatchTST checkpoint path is not configured for the active dataset."
            )
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"checkpoint not found: {self.model_path}")
        self._model = PatchTST(args).to('cpu')
        try:
            state = torch.load(self.model_path, map_location='cpu', weights_only=True)
        except TypeError:
            state = torch.load(self.model_path, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        self._model.load_state_dict(state, strict=False)
        self._model.eval()

    def predict(self, h: int,**kwargs) -> np.ndarray:
        if self._model is None or self._args is None or self._x_enc is None or self._x_mark_enc is None:
            raise RuntimeError("PatchTSTModel not fitted.")

        # Write pred_len into args because some implementations depend on it
        self._args.pred_len = int(h)
        if hasattr(self._model, 'args'):
            setattr(self._model, 'args', self._args)

        with torch.no_grad():
            # Support the existing forward signature: model(x, data_stamp, 1, 1)
            out = self._model(self._x_enc, self._x_mark_enc, 1, 1)
            if isinstance(out, tuple):
                out = out[0]
            # Extract the last h steps of the univariate output
            if out.ndim == 3:
                out = out[:, -h:, :1]
            y_hat = out.squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32)

        if y_hat.shape[0] != h:
            raise RuntimeError(f"PatchTSTModel returned {y_hat.shape[0]} steps, expected {h}.")
        return y_hat

@dataclass
class TimesNetModel(ForecastModel):
    alias: str = "TimesNet"

    # === Required: checkpoint path consistent with training (univariate enc_in=1, features='S') ===
    model_path: str = ""

    # Runtime configuration
    label_len: int = 48         # Training used 84
    timefeat_freq: str = "min"     # Match training; use "15min" for quarter-hour data if needed

    # Runtime cache
    _model: Optional[torch.nn.Module] = None
    _args: Optional[object] = None
    _x_enc: Optional[torch.Tensor] = None
    _x_mark_enc: Optional[torch.Tensor] = None
    _enc_len: int = 0
    _device: str = "cpu"

    def _build_x_mark(self, ts_list) -> torch.Tensor:
        """Generate encoder time features from history only; do not create future timestamps."""
        df = pd.DataFrame({"date": pd.to_datetime(list(ts_list))})
        stamp = time_features(pd.to_datetime(df["date"].values), freq=self.timefeat_freq)  # [F, L]
        stamp = stamp.transpose(1, 0)                                                      # [L, F]
        return torch.from_numpy(np.asarray(stamp)).float().unsqueeze(0)                    # [1, L, F]

    def fit(self, y: np.ndarray, season_length: Optional[int] = None, **kwargs) -> None:
        ts = kwargs.get("timestamps")
        if ts is None:
            raise ValueError("TimesNetModel.fit requires 'timestamps' aligned with y.")

        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]  # Univariate -> [L, 1]
        L, C = y.shape
        if C != 1:
            raise ValueError(f"TimesNetModel (univariate) expects enc_in=1, got enc_in={C}")
        if len(ts) != L:
            raise ValueError(f"timestamps length {len(ts)} != window length {L}")

        # Cache inputs and time features
        self._enc_len = L
        self._x_enc = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(self._device)  # [1, L, 1]
        self._x_mark_enc = self._build_x_mark(ts).to(self._device)                        # [1, L, Fm]

        # Assemble args identical to training (univariate S; keep provided hyperparameters)
        class Args: ...
        args = Args()
        data_name = "power"
        args.task_name = "long_term_forecast"
        args.is_training = 0
        args.model = "TimesNet"
        args.freq = 't'
        args.checkpoints = "./checkpoints/"

        args.seq_len = L
        args.label_len = 48
        args.pred_len = 24  # The forecast horizon is set inside predict(h)

        # --- Architecture & hyperparameters (as provided) ---
        args.seasonal_patterns = "Monthly"
        args.inverse = False
        args.individual = 0
        args.expand = 2
        args.d_conv = 4
        args.top_k = 5
        args.num_kernels = 6

        # Univariate configuration（S）
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1
        args.features = "S"
        args.target = "real_power"  # Target name must match training

        args.d_model = 16
        args.n_heads = 8
        args.e_layers = 2
        args.d_layers = 1
        args.d_ff = 32
        args.moving_avg = 25
        args.factor = 3
        args.distil = True
        args.dropout = 0.1
        args.embed = "timeF"
        args.activation = "gelu"
        args.output_attention = True
        args.channel_independence = 1
        args.decomp_method = "moving_avg"
        args.use_norm = 0
        args.down_sampling_layers = 0
        args.down_sampling_window = 1
        args.down_sampling_method = None
        args.seg_len = 48
        args.num_workers = 10
        args.itr = 1
        args.train_epochs = 100
        args.batch_size = 32
        args.patience = 10
        args.learning_rate = 0.005
        args.des = "Exp"
        args.loss = "MSE"
        args.lradj = "type1"
        args.use_amp = False
        args.use_gpu = self._device.startswith("cuda")
        args.gpu = 0
        args.use_multi_gpu = False
        args.devices = "0,1,2,3"
        args.p_hidden_dims = [128, 128]
        args.p_hidden_layers = 2
        args.use_dtw = False
        args.augmentation_ratio = 0
        args.seed = 2
        args.extra_tag = ""
        args.use_patch = False
        args.patch_len = 16
        args.data = "custom"
        args.model_id = f"{data_name}_96_96"
        args.data_path = f"{data_name}.csv"

        args.pred_len = _resolve_pred_len(args.pred_len)
        self._args = args

        runtime_path = _resolve_checkpoint(self.alias)
        if runtime_path:
            self.model_path = runtime_path
        if not self.model_path:
            raise FileNotFoundError(
                "TimesNet checkpoint path is not configured for the active dataset."
            )
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"checkpoint not found: {self.model_path}")

        # Instantiate and load weights (requires an ftS univariate checkpoint)
        self._model = TimesNet(args).to(self._device)
        try:
            state = torch.load(self.model_path, map_location=self._device, weights_only=True)
        except TypeError:
            state = torch.load(self.model_path, map_location=self._device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self._model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[TimesNet] missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
        self._model.eval()

    def predict(self, h: int, **kwargs) -> np.ndarray:
        if self._model is None or self._args is None or self._x_enc is None or self._x_mark_enc is None:
            raise RuntimeError("TimesNetModel not fitted.")

        # Enforce the 4-argument forward signature; raise if unsupported
        self._args.pred_len = int(h)
        if hasattr(self._model, "args"):
            setattr(self._model, "args", self._args)

        with torch.no_grad():
            try:
                out = self._model(self._x_enc, self._x_mark_enc, 1, 1)
            except TypeError as e:
                raise TypeError(
                    f"{self.alias} requires a 4-arg forward(x_enc, x_mark_enc, x_dec, x_mark_dec), "
                    f"but the loaded model does not support it. Please align the model/ckpt."
                ) from e

            if isinstance(out, tuple):
                out = out[0]
            if out.ndim == 3:
                out = out[:, -h:, :1]  # Extract the last h steps of the univariate output
            y_hat = out.squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32)

        if y_hat.shape[0] != h:
            raise RuntimeError(f"TimesNetModel returned {y_hat.shape[0]} steps, expected {h}.")
        return y_hat

@dataclass
class iTransformerModel(ForecastModel):
    alias: str = "iTransformer"

    # === Required: checkpoint path consistent with training (univariate enc_in=1, features='S') ===
    model_path: str = ""

    # Runtime configuration
    label_len: int = 48          # Training used 84; automatically align with seq_len
    timefeat_freq: str = "min"     # Hourly base; adjust to "15min" for quarter-hour data

    # Runtime cache
    _model: Optional[torch.nn.Module] = None
    _args: Optional[object] = None
    _x_enc: Optional[torch.Tensor] = None
    _x_mark_enc: Optional[torch.Tensor] = None
    _enc_len: int = 0
    _device: str = "cpu"

    def _build_x_mark(self, ts_list) -> torch.Tensor:
        df = pd.DataFrame({"date": pd.to_datetime(list(ts_list))})
        stamp = time_features(pd.to_datetime(df["date"].values), freq=self.timefeat_freq)  # [F, L]
        stamp = stamp.transpose(1, 0)                                                      # [L, F]
        return torch.from_numpy(np.asarray(stamp)).float().unsqueeze(0)                    # [1, L, F]

    def fit(self, y: np.ndarray, season_length: Optional[int] = None, **kwargs) -> None:
        ts = kwargs.get("timestamps")
        if ts is None:
            raise ValueError("ITransformerModel.fit requires 'timestamps' aligned with y.")

        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]  # Univariate -> [L, 1]
        L, C = y.shape
        if C != 1:
            raise ValueError(f"ITransformerModel (univariate) expects enc_in=1, got enc_in={C}")
        if len(ts) != L:
            raise ValueError(f"timestamps length {len(ts)} != window length {L}")

        # Cache inputs and time features
        self._enc_len = L
        self._x_enc = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(self._device)  # [1, L, 1]
        self._x_mark_enc = self._build_x_mark(ts).to(self._device)                        # [1, L, Fm]

        # Assemble args identical to training (univariate S)
        class Args: ...
        args = Args()
        data_name = "power"
        args.task_name = "long_term_forecast"
        args.is_training = 0
        args.model = "iTransformer"
        args.freq = 't'
        args.checkpoints = "./checkpoints/"

        args.seq_len = L
        args.label_len = 48
        args.pred_len = 24  # The forecast horizon is set inside predict(h)

        # --- Hyperparameters from training (keep identical) ---
        args.seasonal_patterns = "Monthly"
        args.inverse = False
        args.individual = 0
        args.expand = 2
        args.d_conv = 4
        args.top_k = 5
        args.num_kernels = 6

        # Univariate configuration（S）
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1
        args.features = "S"
        args.target = "real_power"  # Or your univariate column name (must match training)

        args.d_model = 128
        args.n_heads = 8
        args.e_layers = 2
        args.d_layers = 1
        args.d_ff = 128
        args.moving_avg = 25
        args.factor = 3
        args.distil = True
        args.dropout = 0.1
        args.embed = "timeF"
        args.activation = "gelu"
        args.output_attention = True
        args.channel_independence = 1
        args.decomp_method = "moving_avg"
        args.use_norm = 0
        args.down_sampling_layers = 0
        args.down_sampling_window = 1
        args.down_sampling_method = None
        args.seg_len = 48
        args.num_workers = 10
        args.itr = 1
        args.train_epochs = 100
        args.batch_size = 32
        args.patience = 10
        args.learning_rate = 0.005
        args.des = "Exp"
        args.loss = "MSE"
        args.lradj = "type1"
        args.use_amp = False
        args.use_gpu = self._device.startswith("cuda")
        args.gpu = 0
        args.use_multi_gpu = False
        args.devices = "0,1,2,3"
        args.p_hidden_dims = [128, 128]
        args.p_hidden_layers = 2
        args.use_dtw = False
        args.augmentation_ratio = 0
        args.seed = 2
        args.extra_tag = ""
        args.use_patch = False
        args.patch_len = 16
        args.data = "custom"
        args.model_id = f"{data_name}_96_96"
        args.data_path = f"{data_name}.csv"

        args.pred_len = _resolve_pred_len(args.pred_len)
        self._args = args

        runtime_path = _resolve_checkpoint(self.alias)
        if runtime_path:
            self.model_path = runtime_path
        if not self.model_path:
            raise FileNotFoundError(
                "iTransformer checkpoint path is not configured for the active dataset."
            )
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"checkpoint not found: {self.model_path}")

        # Instantiate and load weights (requires an ftS univariate checkpoint)
        self._model = iTransformer(args).to(self._device)
        try:
            state = torch.load(self.model_path, map_location=self._device, weights_only=True)
        except TypeError:
            state = torch.load(self.model_path, map_location=self._device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self._model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[iTransformer] missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
        self._model.eval()

    def predict(self, h: int, **kwargs) -> np.ndarray:
        if self._model is None or self._args is None or self._x_enc is None or self._x_mark_enc is None:
            raise RuntimeError("ITransformerModel not fitted.")

        self._args.pred_len = int(h)
        if hasattr(self._model, "args"):
            setattr(self._model, "args", self._args)

        with torch.no_grad():
            try:
                # Accept only the 4-argument forward signature: forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
                out = self._model(self._x_enc, self._x_mark_enc, 1, 1)
            except TypeError as e:
                raise TypeError(
                    f"{self.alias} requires a 4-arg forward(x_enc, x_mark_enc, x_dec, x_mark_dec), "
                    f"but the loaded model does not support it. Please align the model/ckpt."
                ) from e

            if isinstance(out, tuple):
                out = out[0]
            # Extract the last h steps of the univariate output
            if out.ndim == 3:
                out = out[:, -h:, :1]
            y_hat = out.squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32)

        if y_hat.shape[0] != h:
            raise RuntimeError(f"ITransformerModel returned {y_hat.shape[0]} steps, expected {h}.")
        return y_hat
    
@dataclass
class ProphetModel(ForecastModel):
    alias: str = "Prophet"
    yearly_seasonality: str = "auto"  # 'auto', True, False
    weekly_seasonality: str = "auto"
    daily_seasonality: str = "auto"
    
    _fitted: Optional[any] = None
    _ds_col: str = "ds"
    _y_col: str = "y"

    def fit(self, y: np.ndarray, season_length: Optional[int] = None, **kwargs) -> None:
        """
        Fit Prophet model to time series data.
        
        Args:
            y: Time series data as numpy array
            t: Optional time array for time series data
            season_length: Optional season length for seasonality modeling
        """
        logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
        
        # Convert time array to datetime if provided
        t = kwargs.get("timestamps", None)
        if t is not None:
            t = pd.to_datetime(t)
        else :
            raise ValueError("Prophet model requires time array (timestamps) for fitting")
        
        # Set seasonality based on season_length
        yearly_seasonality = self.yearly_seasonality
        weekly_seasonality = self.weekly_seasonality
        daily_seasonality = self.daily_seasonality
        
        if season_length and season_length > 1:
            # Adjust seasonality settings based on season_length
            yearly_seasonality = True
            weekly_seasonality = True
            daily_seasonality = True
        
        # Create Prophet model with seasonality parameters
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality
        )
        
        # Create dataframe for Prophet (requires 'ds' and 'y' columns)
        df = pd.DataFrame({
            self._ds_col: t,
            self._y_col: y
        })
        
        # Fit the model
        self._fitted = model.fit(df)

    def predict(self, h: int, **kwargs) -> np.ndarray:
        """
        Generate forecasts for h periods into the future.
        
        Args:
            h: Number of periods to forecast
            t: Optional time array for time series data
            
            
        Returns:
            Numpy array with forecast values
        """
        assert self._fitted is not None, "Model must be fitted before prediction"
        
        # Convert time array to datetime if provided
        t = kwargs.get("future_timestamps", None)
        if t is not None:
            t = pd.to_datetime(t)
        else :
            raise ValueError("Prophet model requires time array (future_timestamps) for fitting")
        
        # Create future dataframe
        future = pd.DataFrame({
            'ds': t
        })
        
        # Generate forecast
        fcst = self._fitted.predict(future)
        
        # Return only the forecast period (last h rows)
        return np.asarray(fcst.tail(h)['yhat'], dtype=float)    
    
@dataclass
class HoltWintersModel(ForecastModel):
    alias: str = "HoltWinters"
    trend: Optional[str] = "add"  # 'add', 'mul', or None
    seasonal: Optional[str] = "add"  # 'add', 'mul', or None
    damped_trend: bool = False
    
    _fitted: Optional[any] = None

    def fit(self, y: np.ndarray, season_length: Optional[int] = None, **kwargs) -> None:
        # Use season_length to determine seasonal periods
        seasonal_periods = None
        if season_length and season_length > 1:
            seasonal_periods = int(season_length)
            # Enable seasonal component if season_length is provided
            if self.seasonal is None:
                seasonal = "add"
            else:
                seasonal = self.seasonal
        else:
            # Disable seasonal component if no clear seasonality
            seasonal = None
            seasonal_periods = None
        
        self._fitted = ExponentialSmoothing(
            y,
            trend=self.trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            damped_trend=self.damped_trend
        ).fit()

    def predict(self, h: int, **kwargs) -> np.ndarray:
        assert self._fitted is not None
        fcst = self._fitted.forecast(steps=h)
        return np.asarray(fcst, dtype=float)

@dataclass
class ThetaModel(ForecastModel):
    alias: str = "Theta"
    theta: float = 2.0  # Theta parameter, typically between 0 and 3
    use_test: bool = True  # Whether to use statistical tests for model selection
    method: str = "auto"  # Method for theta estimation: 'auto', 'mm', 'additive'
    
    _fitted: Optional[any] = None

    def fit(self, y: np.ndarray, season_length: Optional[int] = None, **kwargs) -> None:
        # Theta model works best with at least 2 seasons of data
        if len(y) < 2:
            raise ValueError("Theta model requires at least 2 data points")
        
        # Determine seasonal period
        period = int(season_length) if season_length and season_length > 1 else 1
        
        self._fitted = StatsmodelsTheta(
            y, 
            period=period,
            use_test=self.use_test,
            method=self.method
        ).fit()

    def predict(self, h: int, **kwargs) -> np.ndarray:
        assert self._fitted is not None
        fcst = self._fitted.forecast(steps=h)
        return np.asarray(fcst, dtype=float)  
    
@dataclass
class CesModel(ForecastModel):
    alias: str = "AutoCES"
    season_length: Optional[int] = None
    model: Optional[str] = 'Z'  # CES model type, if None will be auto-selected
    
    _fitted: Optional[any] = None

    def fit(self, y: np.ndarray, season_length: Optional[int] = None, **kwargs) -> None:
        # Simple heuristic AutoCES: use season_length if provided
        season_len = season_length or self.season_length or 1
        
        t = kwargs.get("timestamps", None)
        if t is not None:
            t = pd.to_datetime(t)
            
            # Create dataframe for statsforecast
            df = pd.DataFrame({
                'unique_id': self.alias,
                'ds': t, 
                'y': y
                })
        else :
            raise ValueError(f"{self.alias} model requires time series data (timestamps) for model fitting")
            
        # Initialize AutoCES model
        if season_len > 1:
            self._fitted = StatsForecast(
                models=[SFAutoCES(season_length=season_len, model=self.model, alias=self.alias)],
                freq=pd.infer_freq(t),
                )
        else:
            self._fitted = StatsForecast(
                models=[SFAutoCES(season_length=1, model=self.model, alias=self.alias)],
                freq=pd.infer_freq(t),
                )
        
        # Store data for statsforecast usage
        self._fitted.fit(df)

    def predict(self, h: int, **kwargs) -> np.ndarray:
        assert self._fitted is not None
        
        # Use StatsForecast for prediction
        fcst_df = self._fitted.predict(h=h)
        
        return np.asarray(fcst_df[self.alias], dtype=float)
    
@dataclass
class CrostonModel(ForecastModel):
    alias: str = "CrostonClassic"
    alpha: Optional[float] = None  # Smoothing parameter for level, if None will be auto-selected
    
    _fitted: Optional[any] = None

    def fit(self, y: np.ndarray, season_length: Optional[int] = None, **kwargs) -> None:
        # Croston method is typically used for intermittent demand data
        t = kwargs.get("timestamps", None)
        if t is not None:
            t = pd.to_datetime(t)
            
            # Create dataframe for statsforecast
            df = pd.DataFrame({
                'unique_id': self.alias,
                'ds': t, 
                'y': y
                })
        else :
            raise ValueError(f"{self.alias} model requires time series data (timestamps) for model fitting")
            
        # Initialize CrostonClassic model
        if self.alpha is not None:
            self._fitted = StatsForecast(
                models=[SFCrostonClassic(alpha=self.alpha, alias=self.alias)],
                freq=pd.infer_freq(t),
                )
        else:
            self._fitted = StatsForecast(
                models=[SFCrostonClassic(alias=self.alias)],
                freq=pd.infer_freq(t),
                )
        
        # Store data for statsforecast usage
        self._fitted.fit(df)

    def predict(self, h: int, **kwargs) -> np.ndarray:
        assert self._fitted is not None
        
        # Use StatsForecast for prediction
        fcst_df = self._fitted.predict(h=h)
        
        return np.asarray(fcst_df[self.alias], dtype=float)
    
@dataclass
class DynamicOptimizedThetaModel(ForecastModel):
    alias: str = "DynamicOptimizedTheta"
    season_length: Optional[int] = None
    decomposition_type: str = "multiplicative"  # Decomposition type for theta method
    
    _fitted: Optional[any] = None

    def fit(self, y: np.ndarray, season_length: Optional[int] = None, **kwargs) -> None:
        # Simple heuristic DynamicOptimizedTheta: use season_length if provided
        season_len = season_length or self.season_length or 1
        
        t = kwargs.get("timestamps", None)   
        if t is not None:
            t = pd.to_datetime(t)
            
            # Create dataframe for statsforecast
            df = pd.DataFrame({
                'unique_id': self.alias,
                'ds': t, 
                'y': y
                })
        else :
            raise ValueError(f"{self.alias} model requires time series data (t) for model fitting")
            
        # Initialize DynamicOptimizedTheta model
        if season_len > 1:
            self._fitted = StatsForecast(
                models=[SFDynamicOptimizedTheta(season_length=season_len, decomposition_type=self.decomposition_type, alias=self.alias)],
                freq=pd.infer_freq(t),
                )
        else:
            self._fitted = StatsForecast(
                models=[SFDynamicOptimizedTheta(season_length=1, decomposition_type=self.decomposition_type, alias=self.alias)],
                freq=pd.infer_freq(t),
                )
        
        # Store data for statsforecast usage
        self._fitted.fit(df)

    def predict(self, h: int, **kwargs) -> np.ndarray:
        assert self._fitted is not None
        
        # Use StatsForecast for prediction
        fcst_df = self._fitted.predict(h=h)
        
        return np.asarray(fcst_df[self.alias], dtype=float)
    
@dataclass
class ZeroModel(ForecastModel):
    alias: str = "ZeroModel"
    
    _fitted: Optional[any] = None

    def fit(self, y: np.ndarray, season_length: Optional[int] = None, **kwargs) -> None:
        # ZeroModel always predicts zero, no fitting required but we follow the interface
        t = kwargs.get("timestamps", None)
        if t is not None:
            t = pd.to_datetime(t)
            
            # Create dataframe for statsforecast
            df = pd.DataFrame({
                'unique_id': self.alias,
                'ds': t, 
                'y': y
                })
        else :
            raise ValueError(f"{self.alias} model requires time series data (timestamps) for model fitting")
            
        # Initialize ZeroModel
        self._fitted = StatsForecast(
            models=[SFZeroModel(alias=self.alias)],
            freq=pd.infer_freq(t),
            )
        
        # Store data for statsforecast usage
        self._fitted.fit(df)

    def predict(self, h: int, **kwargs) -> np.ndarray:
        assert self._fitted is not None
        
        # Use StatsForecast for prediction
        fcst_df = self._fitted.predict(h=h)
        
        return np.asarray(fcst_df[self.alias], dtype=float)

#####foundation_model###################
from transformers import AutoModelForCausalLM, AutoTokenizer
# @dataclass
# class TimesFMModel:
    
#     alias: str = "TimesFM"
#     # Fixed local model directory (must exist in the runtime environment)
#     local_dir: str = "./castmind/foundation_models/timesfm"
#     # Fixed HuggingFace repo ID for offline parsing
#     hf_repo_id: str = "google/timesfm-1.0-200m-pytorch"
#     context_len: int = 512
#     backend: Optional[str] = None

#     _y: Optional[np.ndarray] = None
#     _model: Optional[any] = None

#     def fit(self, y: np.ndarray, season_length: Optional[int] = None, **kwargs) -> None:
#         """Store the training sequence as a NumPy array."""
#         self._y = np.asarray(y, dtype=float)

#     def _ensure_model(self, h: int):
#         """
#         Initialize the TimesFM model with the fixed local_dir and hf_repo_id.
#         Return `_model` immediately if it has already been initialized.
#         """
#         import os
#         # Force offline mode to avoid network access
#         os.environ.setdefault("HF_HUB_OFFLINE", "1")
#         os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

#         try:
#             import timesfm
#             import torch
#         except Exception as e:
#             raise ImportError(
#                 "timesfm is not installed; install timesfm[torch] and ensure a local checkpoint."
#             ) from e

#         data_len = len(self._y) if self._y is not None else 0
#         ctx_len = max(self.context_len, min(data_len, 2048))
#         # Choose GPU or CPU based on CUDA availability
#         backend = self.backend or ("gpu" if torch.cuda.is_available() else "cpu")

#         # Load the checkpoint using the fixed local_dir and hf_repo_id
#         checkpoint = timesfm.TimesFmCheckpoint(
#             huggingface_repo_id=self.hf_repo_id,
#             local_dir=self.local_dir,
#         )
#         self._model = timesfm.TimesFm(
#             hparams=timesfm.TimesFmHparams(
#                 backend=backend,
#                 horizon_len=h,
#                 context_len=ctx_len,
#             ),
#             checkpoint=checkpoint,
#         )
#         return self._model

#     def predict(self, h: int, **kwargs) -> np.ndarray:
#         assert self._y is not None and len(self._y) > 0
#         tfm = self._model or self._ensure_model(h)
#         ctx_len = getattr(tfm.hparams, "context_len", len(self._y))
#         # Select the last ctx_len points as input
#         ctx_arr = np.asarray(self._y[-min(len(self._y), ctx_len):], dtype=float)
#         # timesfm's forecast API no longer accepts the contexts keyword; only input and freq
#         point_fcst, _ = tfm.forecast([ctx_arr], freq=[0])  # Provide a list and frequency
#         # print("TimesFMModel succeeded")
#         print(f"point_fcst: {point_fcst.shape}")
#         return np.asarray(point_fcst[0], dtype=float)

# @dataclass
# class TimesFMModel:
    
#     alias: str = "TimesFM"
#     # Fixed local model directory (must exist in the runtime environment)
#     local_dir: str = "/data/Forever_Pan/AGI_sources/data/xhzhang/cast/castmind/castmind/foundation_models/timesfm"
#     # Fixed HuggingFace repo ID for offline parsing
#     hf_repo_id: str = "google/timesfm-1.0-200m-pytorch"
#     context_len: int = 512
#     backend: Optional[str] = None

#     _y: Optional[np.ndarray] = None
#     _model: Optional[any] = None

#     def fit(self, y: np.ndarray, season_length: Optional[int] = None, **kwargs) -> None:
#         """Store the training sequence as a NumPy array."""
#         self._y = np.asarray(y, dtype=float)

#     def _ensure_model(self, h: int):
#         """
#         Initialize the TimesFM model with the fixed local_dir and hf_repo_id.
#         Return `_model` immediately if it has already been initialized.
#         """
#         import os
#         # Force offline mode to avoid network access
#         os.environ.setdefault("HF_HUB_OFFLINE", "1")
#         os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

#         try:
#             import timesfm
#             import torch
#         except Exception as e:
#             raise ImportError(
#                 "timesfm is not installed; install timesfm[torch] and ensure a local checkpoint."
#             ) from e

#         data_len = len(self._y) if self._y is not None else 0
#         ctx_len = max(self.context_len, min(data_len, 2048))
#         # Choose GPU or CPU based on CUDA availability
#         backend = self.backend or ("gpu" if torch.cuda.is_available() else "cpu")

#         # Load the checkpoint using the fixed local_dir and hf_repo_id
#         checkpoint = timesfm.TimesFmCheckpoint(
#             huggingface_repo_id=self.hf_repo_id,
#             local_dir=self.local_dir,
#         )
#         self._model = timesfm.TimesFm(
#             hparams=timesfm.TimesFmHparams(
#                 backend=backend,
#                 horizon_len=h,
#                 context_len=ctx_len,
#             ),
#             checkpoint=checkpoint,
#         )
#         return self._model

#     def predict(self, h: int, **kwargs) -> np.ndarray:
#         assert self._y is not None and len(self._y) > 0
#         tfm = self._model or self._ensure_model(h)
#         ctx_len = getattr(tfm.hparams, "context_len", len(self._y))
#         # Select the last ctx_len points as input
#         ctx_arr = np.asarray(self._y[-min(len(self._y), ctx_len):], dtype=float)
#         # timesfm's forecast API no longer accepts the contexts keyword; only input and freq
#         point_fcst, _ = tfm.forecast([ctx_arr], freq=[0])  # Provide a list and frequency
#         # print("TimesFMModel succeeded")
#         return np.asarray(point_fcst[0], dtype=float)

@dataclass
class TimesFMModel:
    alias: str = "TimesFM"
    # Local model directory (must include config.json / pytorch_model.bin / hparams.json / etc.)
    local_dir: str = "./castmind/foundation_models/timesfm"
    # Used solely for offline parsing; HF_HUB_OFFLINE=1 prevents network calls
    hf_repo_id: str = "google/timesfm-1.0-200m-pytorch"
    # TimesFM 1.0 has a context limit of 512 (per the official documentation)
    context_len: int = 512
    # "gpu" | "cpu"; auto-detect by default
    backend: Optional[str] = None

    _y: Optional[np.ndarray] = None
    _model: Optional[any] = None

    def fit(self, y: np.ndarray, season_length: Optional[int] = None, **kwargs) -> None:
        """Cache the historical sequence."""
        self._y = np.asarray(y, dtype=float)

    def _ensure_model(self, h: int):
        """
        Initialize TimesFM using the local directory only (no network access).
        Return immediately if the model has already been initialized.
        """
        if self._model is not None:
            return self._model

        # Enforce offline mode and disable telemetry
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

        # Basic validation
        if not os.path.isdir(self.local_dir):
            raise FileNotFoundError(f"[TimesFM] Local directory does not exist: {self.local_dir}")
        if h is None or int(h) <= 0:
            raise ValueError("[TimesFM] Forecast horizon h must be a positive integer.")

        try:
            import torch
            import timesfm
        except Exception as e:
            raise ImportError("Please install timesfm[torch] and torch before running TimesFM.") from e

        # Device backend
        backend = self.backend or ("gpu" if torch.cuda.is_available() else "cpu")

        # Choose context_len based on data length (no more than 512)
        data_len = len(self._y) if self._y is not None else 0
        # Select a sensible value between data length and self.context_len, capped at 512
        ctx_len = max(1, min(int(self.context_len), 512, max(1, data_len)))

        # Load strictly from the local directory (no network access)
        checkpoint = timesfm.TimesFmCheckpoint(
            huggingface_repo_id=self.hf_repo_id,  # Used only for offline metadata parsing
            local_dir=self.local_dir,
        )
        self._model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=backend,
                horizon_len=int(h),
                context_len=int(ctx_len),
            ),
            checkpoint=checkpoint,
        )
        return self._model

    def predict(self, h: int, freq: int = 0, **kwargs) -> np.ndarray:
        """
        Forecast the next h steps.
        freq: TimesFM frequency class {0,1,2} (0: daily or faster; 1: weekly/monthly; 2: seasonal/annual)
        """
        if self._y is None or len(self._y) == 0:
            raise ValueError("Call fit(y) with historical data before predict().")

        tfm = self._model or self._ensure_model(int(h))
        # Slice the trailing context according to the model's context_len
        ctx_len = getattr(getattr(tfm, "hparams", None), "context_len", self.context_len)
        ctx_len = max(1, min(int(ctx_len), len(self._y)))
        ctx_arr = np.asarray(self._y[-ctx_len:], dtype=float)

        # TimesFM's forecast API expects list[np.ndarray] inputs and a frequency list
        point_fcst, _ = tfm.forecast([ctx_arr], freq=[int(freq)])
        return np.asarray(point_fcst[0], dtype=float)
    

@dataclass
class ChronosModel:
    alias: str = "Chronos"
    local_dir: Optional[str] = \
        "./castmind/foundation_models/chronos-bolt-base"
    hf_repo_id: Optional[str] = None
    dtype: str = "bfloat16"  # Optional parameter retained for logging or future use

    _y: Optional[np.ndarray] = None
    _pipeline: Optional[any] = None

    def fit(self, y: np.ndarray, season_length: Optional[int] = None, **kwargs) -> None:
        self._y = np.asarray(y, dtype=float)

    def _ensure_pipeline(self):
        try:
            from chronos import BaseChronosPipeline
            import torch
        except Exception as e:
            raise ImportError("chronos is not installed.") from e
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = self.local_dir or (self.hf_repo_id or "amazon/chronos-bolt-small")
        self._pipeline = BaseChronosPipeline.from_pretrained(
            model_id, device_map=device  # dtype parameter removed
        )
        return self._pipeline

    def predict(self, h: int, **kwargs) -> np.ndarray:
        assert self._y is not None and len(self._y) > 0
        import torch
        pipe = self._pipeline or self._ensure_pipeline()
        
        # Ensure context tensors are placed on the correct device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        context = torch.tensor(self._y.reshape(1, -1), dtype=torch.float32).to(device)

        # Forecast
        _, pred = pipe.predict_quantiles(
            context=context, prediction_length=h, quantile_levels=[0.5]
        )
        pred = pred.squeeze()
        
        # Validate output dimensions and reshape to 1D
        if pred.ndim == 3:
            pred = pred[0, :, 0]
        elif pred.ndim == 2:
            pred = pred[0]
        # print("ChronosModel succeeded")
        return np.asarray(pred, dtype=float)

# @dataclass
# class SundialModel:
#     alias: str = "Sundial"
#     local_dir: Optional[str] = "./castmind/foundation_models/sundial-base-128m"
#     hf_repo_id: Optional[str] = "thuml/sundial-base-128m"
#     _y: Optional[np.ndarray] = None
#     _model: Optional[any] = None
#     _tokenizer: Optional[any] = None

#     def fit(self, y: np.ndarray, season_length: Optional[int] = None,**kwargs) -> None:
#         """
#         Fit the model to the provided time series data.
#         """
#         self._y = np.asarray(y, dtype=float)

#     def _ensure_model(self):
#         """
#         Ensures the model is loaded. If the model is not loaded, it loads it.
#         """
#         import os
#         try:
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#             # Define model path or Hugging Face repository
#             model_id = self.local_dir if self.local_dir and os.path.isdir(self.local_dir) else self.hf_repo_id
#             print(f"Loading model from {model_id}")
            
#             # Load model and tokenizer
#             self._model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
#             # self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

#         except Exception as e:
#             raise ImportError("transformers is required for Sundial.") from e

#         return self._model
#     def predict(self, h: int, **kwargs) -> np.ndarray:
#         """
#         Predict the next `h` values based on the fitted data.
#         """
#         assert self._y is not None and len(self._y) > 0, "Model not fitted with data yet."
        
#         import torch
#         model = self._model or self._ensure_model()

#         # Prepare input: reshape to (1, context_length) for model input
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         x_enc = torch.tensor(self._y.reshape(1, -1), dtype=torch.float32).to(device)  # Ensure it is on the correct device

#         # Ensure that the input data is properly formatted
#         # print(f'h: {h}')
#         # print(f"Input shape for prediction: {x_enc.shape}")
# # Input shape for prediction: torch.Size([1, 96])
#         try:
#             # Generate predictions using model's generate method
#             output = model.generate(
#                 x_enc,
#                 max_new_tokens=h,      # Number of tokens to predict
#                 num_samples=1,         # Number of samples
#                 temperature=1.0        # Randomness factor
#             )

#             # Print the output shape for debugging
#             # print(f"Output shape: {output.shape}")

#             # Ensure the output shape is correct for extracting predictions
#             if output.dim() == 2:  # (B, pred_len) shape
#                 pred_tokens = output[:, -h:]  # Extract last h tokens
#             else:  # Handle case where output dimensions might vary
#                 pred_tokens = output.mean(dim=-1)[:, -h:]  # Extract mean prediction
#             print("type:", type(pred_tokens))
#             print("dtype:", pred_tokens.dtype)
#             print("shape:", pred_tokens.shape)
#             # print("SundialModel succeeded")
#             result=pred_tokens.squeeze(0).detach().cpu().numpy()
#             print("type:", type(result))
#             print("dtype:", result.dtype)
#             print("shape:", result.shape)
#             return pred_tokens.squeeze(0).detach().cpu().numpy()

#         except Exception as e:
#             print(f"Error during prediction: {e}")
#             return np.zeros(h)
@dataclass
class SundialModel:
    alias: str = "Sundial"
    local_dir: Optional[str] = "./castmind/foundation_models/sundial-base-128m"
    hf_repo_id: Optional[str] = "thuml/sundial-base-128m"
    _y: Optional[np.ndarray] = None
    _model: Optional[any] = None
    _tokenizer: Optional[any] = None

    def fit(self, y: np.ndarray, season_length: Optional[int] = None,**kwargs) -> None:
        """
        Fit the model to the provided time series data.
        """
        self._y = np.asarray(y, dtype=float)

    def _ensure_model(self):
        """
        Ensures the model is loaded. If the model is not loaded, it loads it.
        """
        import os
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Define model path or Hugging Face repository
            model_id = self.local_dir if self.local_dir and os.path.isdir(self.local_dir) else self.hf_repo_id
            print(f"Loading model from {model_id}")
            
            # Load model and tokenizer
            self._model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
            # self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        except Exception as e:
            raise ImportError("transformers is required for Sundial.") from e

        return self._model
    def predict(self, h: int,**kwargs) -> np.ndarray:
        """
        Predict the next `h` values based on the fitted data.
        """
        assert self._y is not None and len(self._y) > 0, "Model not fitted with data yet."
        
        import torch
        model = self._model or self._ensure_model()

        # Prepare input: reshape to (1, context_length) for model input
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_enc = torch.tensor(self._y.reshape(1, -1), dtype=torch.float32).to(device)  # Ensure it is on the correct device

        # Ensure that the input data is properly formatted
        # print(f'h: {h}')
        # print(f"Input shape for prediction: {x_enc.shape}")
# Input shape for prediction: torch.Size([1, 96])
        # try:
        # Generate predictions using model's generate method
        output = model.generate(
            x_enc,
            max_new_tokens=h,      # Number of tokens to predict
            num_samples=1,         # Number of samples
            temperature=1.0        # Randomness factor
        )

        # Print the output shape for debugging
        # print(f"Output shape: {output.shape}")
        
        # Take the last 96 values along the final dimension
        if output.dim() == 3:
            pred_tokens = output[0, 0, :]   # (96,)
        elif output.dim() == 2:
            pred_tokens = output[0, -h:]    # (h,)
        else:
            raise ValueError(f"Unexpected output shape: {output.shape}")

        return pred_tokens.detach().cpu().numpy()
    
def get_default_models() -> List[ForecastModel]:
    # Maintain three basic models: SeasonalNaive, HistoricAverage, and AutoARIMA
    return [
        SeasonalNaiveModel(),
        HistoricAverageModel(),
        ArimaModel(),
        #TimesFMModel(),
        #ChronosModel(),
        SundialModel(),
        AutoformerModel(),
        DLinearModel(),
        PatchTSTModel(),
        TimesNetModel(),
        iTransformerModel(),
        ProphetModel(),
        HoltWintersModel(),
        ThetaModel(),
        CesModel(),
        CrostonModel(),
        DynamicOptimizedThetaModel(),
        ZeroModel(),
    ]
