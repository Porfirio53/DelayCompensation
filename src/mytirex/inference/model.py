"""TiRex-only extraction of baseline_adapters.py and tirex_tuning.py.

All scaling, patching, trend detection, differencing and restoration remain in
the copied vendor code. The boundary returns the requested quantile trajectory.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from ..data.arrival import ReceiverView, causal_filled_context
from ..common import (assert_local_imports, current_source, project_path, read_json,
                     require_current_source, source_hash, verify_files)
from .gpu import cache_snapshot, configure_runtime
from ..timing.clock import horizon_steps, integer_us


def read_tuning(root: Path, path: str | Path, source: str = "current") -> dict:
    require_current_source(source)
    settings = read_json(project_path(root, path))
    expected = {"source_directory", "context_length", "pad_context", "tta_diff", "trend_window_points",
                "trend_threshold", "trend_min_r2"}
    if set(settings) != expected:
        raise ValueError(f"TiRex configuration fields must be exactly {sorted(expected)}")
    expected_source = "TiRex"
    if settings["source_directory"] != expected_source:
        raise ValueError(f"{source} must select source_directory={expected_source!r}")
    context = settings["context_length"]
    if isinstance(context, bool) or not isinstance(context, int) or not 2 <= context <= 2048:
        raise ValueError("context_length must be an integer from 2 to 2048")
    for key in ("pad_context", "tta_diff"):
        if not isinstance(settings[key], bool):
            raise ValueError(f"{key} must be a JSON boolean")
    window = settings["trend_window_points"]
    if window is not None and (isinstance(window, bool) or not isinstance(window, int) or window < 4):
        raise ValueError("trend_window_points must be null or an integer >= 4")
    threshold, r2 = settings["trend_threshold"], settings["trend_min_r2"]
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not np.isfinite(threshold) or threshold <= 0):
        raise ValueError("trend_threshold must be finite and > 0")
    if isinstance(r2, bool) or not isinstance(r2, (int, float)) or not np.isfinite(r2) or not 0 < r2 <= 1:
        raise ValueError("trend_min_r2 must be finite and in (0, 1]")
    return settings


@dataclass(frozen=True)
class ForecastTrajectory:
    source_time_us: int
    sample_period_us: int
    quantiles: np.ndarray
    values_mps: np.ndarray  # [quantile, requested future step], already on CPU.
    status: str = "ok"
    failure_reason: str = ""
    fallback: bool = False
    batch_elapsed_s: float = 0.0
    batch_size: int = 1

    @property
    def future_time_us(self) -> np.ndarray:
        return self.source_time_us + np.arange(1, self.values_mps.shape[1] + 1, dtype=np.int64) * self.sample_period_us

    def value_at(self, target_time_us: int) -> float:
        if isinstance(target_time_us, bool) or not isinstance(target_time_us, (int, np.integer)):
            raise ValueError("Target time must use integer microseconds")
        step, remainder = divmod(int(target_time_us) - self.source_time_us, self.sample_period_us)
        if remainder or not 1 <= step <= self.values_mps.shape[1]:
            raise ValueError("Target must be on the future grid within this trajectory")
        median = np.flatnonzero(np.isclose(self.quantiles, 0.5))
        if median.size != 1:
            raise ValueError("Trajectory has no unique median quantile")
        return float(self.values_mps[int(median[0]), step - 1])

    def value_at_offset(self, source_offset_us: float, last_observation_mps: float) -> float:
        """Linear median interpolation on [observed anchor, future trajectory].

        Offset is in the source clock, after receiver-time conversion. The
        strict legacy value_at() contract stays intact; no extrapolation or
        target-truth interpolation is involved here.
        """
        if (isinstance(source_offset_us, bool) or not np.isfinite(source_offset_us)
                or source_offset_us < 0 or source_offset_us > self.values_mps.shape[1] * self.sample_period_us + 1e-6
                or not np.isfinite(last_observation_mps)):
            raise ValueError("Offset must lie within this trajectory and the observed anchor must be finite")
        median = np.flatnonzero(np.isclose(self.quantiles, 0.5))
        if median.size != 1:
            raise ValueError("Trajectory has no unique median quantile")
        values = np.r_[last_observation_mps, self.values_mps[int(median[0])]]
        grid = np.arange(len(values), dtype=np.float64) * self.sample_period_us
        return float(np.interp(source_offset_us, grid, values))


def observation_horizon(view: ReceiverView, *, conservative: bool = True,
                        extra_delivery_us: float = 0.0, margin_steps: int = 0) -> int:
    """Cover measured observation age plus receiver-side queue/compute time."""
    if not np.isfinite(extra_delivery_us) or extra_delivery_us < 0:
        raise ValueError("Additional delivery time must be finite and nonnegative")
    integer_us(margin_steps, "margin_steps")
    if not view.ready:
        raise ValueError(f"Receiver observation is unavailable: {view.reason}")
    timing = view.timing
    if timing is None:
        offset = view.query_time_us - int(view.source_time_us[-1]) + extra_delivery_us
    else:
        if not timing.valid:
            raise ValueError(f"Observation timing is unavailable: {timing.reason}")
        if conservative:
            clock = timing.clock
            minimum_rate = clock.rate if clock.exact else 1.0 - clock.max_relative_drift_ppm * 1e-6
            offset = (max(0.0, timing.age_upper_us) + extra_delivery_us) / minimum_rate
        else:
            offset = timing.source_offset_us + extra_delivery_us / timing.clock.rate
    return max(1, horizon_steps(offset, view.sample_period_us) + margin_steps)


class TiRexModel:
    def __init__(self, root: Path, settings: dict, runtime: dict, source: str = "current"):
        require_current_source(source)
        if settings["source_directory"] != "TiRex":
            raise ValueError("The current TiRex model requires source_directory='TiRex'")
        self.root, self.settings, self.runtime, self.source = Path(root).resolve(), dict(settings), dict(runtime), source
        self.context_length = settings["context_length"]
        self.batch_size = int(runtime["batch_size"])
        if runtime["device"] != "cuda":
            raise ValueError("This GPU baseline requires device=cuda; CPU is not an implicit fallback")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        _, self.source_directory = current_source(self.root)
        self.source_sha256 = source_hash(self.source_directory)
        self.model_lock = read_json(self.root / "configs/model_manifest.json")["TiRex-2"]
        self.model_path = project_path(self.root, self.model_lock["local_directory"])
        verify_files(self.model_path, self.model_lock["sha256"])
        self.model = None
        self.batch_timings = []
        self.differencing_counts = {"target_rows_seen": 0, "differenced_rows": 0}
        self.first_forecast = None
        self.current_unit_seed = None
        self.seed_policy = "sha256 JSON [base_seed,sequence,delay_ms], first 8 bytes modulo 2**31; reset per replay unit"

    def load(self) -> None:
        if self.model is not None:
            return
        self.cache_before_load = cache_snapshot(self.root)
        started = perf_counter()
        configure_runtime(self.root)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        source = self.source_directory / "src"
        imported = sys.modules.get("tirex2")
        if imported is not None and Path(imported.__file__).resolve().parent != source / "tirex2":
            raise RuntimeError("A different tirex2 source is already imported; switch sources in a fresh process")
        sys.path.insert(0, str(source))
        import torch
        self.torch = torch
        torch.set_num_threads(int(self.runtime["num_threads"]))
        torch.manual_seed(int(self.runtime["seed"]))
        if not torch.cuda.is_available():
            raise RuntimeError("TiRex-2 GPU inference requires an available CUDA GPU")
        package = importlib.import_module("tirex2")
        if Path(package.__file__).resolve().parent != source / "tirex2":
            raise RuntimeError(f"Unexpected tirex2 import: {package.__file__}")
        self.timeseries_type = package.TimeseriesType
        self.model = package.load_model(self.model_path, device="cuda", compile=False,
                                        use_flex_attention=False, hf_kwargs={"local_files_only": True})
        self.model.eval()
        if self.context_length > int(self.model.context_len):
            raise ValueError("context_length exceeds this checkpoint's supported length")
        devices = {str(p.device) for p in self.model.parameters()}
        if not devices or any(not device.startswith("cuda:") for device in devices):
            raise RuntimeError(f"Model parameters are not all on the GPU: {devices}")
        self.parameter_devices = sorted(devices)
        processor = self.model.model.postprocessor
        if self.source == "current":
            processor.cfg = replace(processor.cfg, trend_window_points=self.settings["trend_window_points"],
                                    trend_threshold=self.settings["trend_threshold"],
                                    trend_min_r2=self.settings["trend_min_r2"])
        self.postprocessor_settings = asdict(processor.cfg)
        # Retain the old adapter's actual decision audit, without a second pass.
        transform_input = processor.transform_input
        def audited(*args, **kwargs):
            result = transform_input(*args, **kwargs)
            for mask in result[2]["diff_masks"]:
                self.differencing_counts["target_rows_seen"] += len(mask)
                self.differencing_counts["differenced_rows"] += sum(int(selected) for selected in mask)
            return result
        processor.transform_input = audited
        self.quantiles = self.model.quantiles.detach().cpu().numpy().copy()
        if np.flatnonzero(np.isclose(self.quantiles, 0.5)).size != 1:
            raise ValueError("TiRex checkpoint has no unique median quantile")
        torch.cuda.synchronize()
        self.load_time_s = perf_counter() - started
        self.cache_after_load = cache_snapshot(self.root)
        self.package_files = assert_local_imports(self.root)

    def synchronize(self) -> None:
        self.torch.cuda.synchronize()

    def reset_for_unit(self, sequence: str, delay_ms: int) -> None:
        payload = json.dumps([int(self.runtime["seed"]), str(sequence), int(delay_ms)],
                             separators=(",", ":")).encode()
        self.current_unit_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)
        self.torch.manual_seed(self.current_unit_seed)

    def reset_for_trace(self, series_id: str, profile: str, scenario: str, replicate: int) -> None:
        # Equal arrival histories use equal model seeds across clock scenarios.
        payload = json.dumps([int(self.runtime["seed"]), "aoi", series_id, profile, replicate],
                             separators=(",", ":")).encode()
        self.current_unit_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)
        self.torch.manual_seed(self.current_unit_seed)
        self.seed_policy = "sha256 JSON [base_seed,aoi,series_id,profile,replicate]; first 8 bytes modulo 2**31; shared across clock scenarios"

    def prepare_context(self, view: ReceiverView) -> np.ndarray:
        times, values, mask = view.source_time_us, view.values_mps, view.received_mask
        if (times.ndim != 1 or values.shape != times.shape or mask.shape != times.shape or times.size < 2
                or view.received_at_us.shape != times.shape or view.native_source_time_us.shape != times.shape):
            raise ValueError("Forecast needs matching one-dimensional receiver history of length >= 2")
        if not view.ready or not mask[-1] or np.any(np.diff(times) != view.sample_period_us) or view.sample_period_us != 25000:
            raise ValueError("Receiver history is not ready on the 25,000 us grid")
        if (not np.isfinite(values[mask]).all() or np.any(view.received_at_us[mask] > view.query_time_us)
                or np.any(view.received_at_us[mask] < 0)):
            raise ValueError("History contains invalid or not-yet-arrived observations")
        if view.timing is None:
            if int(times[-1]) > view.query_time_us or np.any(view.received_at_us[mask] < times[mask]):
                raise ValueError("History contains invalid or not-yet-arrived observations")
        elif view.timing.clock is not None and view.timing.clock.at_receiver_us > view.query_time_us:
            raise ValueError("History contains a future source-to-receiver clock mapping")
        # A source-relative trajectory can be computed from received history
        # without knowing clock offset. Query-time selection is gated separately
        # by forecast_observation(), which requires valid observation timing.
        if np.any(np.isfinite(values[~mask])) or np.any(view.received_at_us[~mask] != -1):
            raise ValueError("Missing positions expose unreceived information")
        context = values if mask.all() else causal_filled_context(view)
        return np.ascontiguousarray(context[-self.context_length:], dtype=np.float32)

    def forecast_options(self) -> dict:
        return {"pad_context": self.settings["pad_context"], "tta_diff": self.settings["tta_diff"],
                "tta_sign_flip": False}

    def forecast_batch(self, views: list[ReceiverView], prediction_length: int) -> list[ForecastTrajectory]:
        if self.model is None:
            raise RuntimeError("Load the local model before forecasting")
        if (isinstance(prediction_length, bool) or not isinstance(prediction_length, int) or prediction_length < 1
                or not views or len(views) > self.batch_size):
            raise ValueError("Use a positive trajectory length and 1..batch_size receiver views")
        if prediction_length > int(self.model.future_len):
            raise ValueError(f"Prediction horizon exceeds this checkpoint's {self.model.future_len} steps")
        self.synchronize()
        started = perf_counter()
        cache_before = cache_snapshot(self.root) if self.first_forecast is None else None
        results = [None] * len(views)
        contexts, positions = [], []
        for index, view in enumerate(views):
            try:
                contexts.append(self.prepare_context(view))
                positions.append(index)
            except (TypeError, ValueError) as exc:
                results[index] = self._fallback(view, prediction_length, str(exc))
        reason = ""
        if contexts:
            try:
                with self.torch.inference_mode():
                    series = [self.timeseries_type(target=self.torch.from_numpy(context).unsqueeze(0),
                                                  past_covariates=None, future_covariates=None) for context in contexts]
                    forecasts = self.model.forecast(series, prediction_length=prediction_length,
                                                   output_type="numpy", batch_size=len(contexts), **self.forecast_options())
                if len(forecasts) != len(contexts):
                    raise RuntimeError("TiRex returned an unexpected forecast count")
                for index, forecast in zip(positions, forecasts, strict=True):
                    values = np.asarray(forecast)[0].copy()
                    if values.shape != (len(self.quantiles), prediction_length) or not np.isfinite(values).all():
                        results[index] = self._fallback(views[index], prediction_length, "nonfinite/invalid model trajectory")
                    else:
                        values.flags.writeable = False
                        results[index] = ForecastTrajectory(int(views[index].source_time_us[-1]),
                            views[index].sample_period_us, self.quantiles.copy(), values)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                for index in positions:
                    results[index] = self._fallback(views[index], prediction_length, reason)
        self.synchronize()
        elapsed = perf_counter() - started
        results = [replace(result, batch_elapsed_s=elapsed, batch_size=len(views)) for result in results]
        self.batch_timings.append({"batch_size": len(views), "horizon_steps": prediction_length,
                                   "elapsed_s": elapsed, "failures": sum(r.status != "ok" for r in results),
                                   "failure_reason": reason})
        if self.first_forecast is None:
            self.first_forecast = {"elapsed_s": elapsed, "batch_size": len(views), "horizon_steps": prediction_length,
                                   "cache_before": cache_before, "cache_after": cache_snapshot(self.root),
                                   "note": "First request may include FlashRNN/Triton compilation; not steady-state latency."}
        return results

    def _fallback(self, view: ReceiverView, horizon: int, reason: str) -> ForecastTrajectory:
        observed = view.values_mps[view.received_mask]
        if not len(observed) or not np.isfinite(observed[-1]) or not len(view.source_time_us):
            raise RuntimeError(f"No finite received observation for fallback: {reason}")
        values = np.full((len(self.quantiles), horizon), float(observed[-1]), dtype=np.float64)
        values.flags.writeable = False
        latest_received = int(np.flatnonzero(view.received_mask)[-1])
        return ForecastTrajectory(int(view.source_time_us[latest_received]), view.sample_period_us, self.quantiles.copy(),
                                  values, "failed", reason, True)

    def forecast(self, view: ReceiverView, prediction_length: int) -> ForecastTrajectory:
        return self.forecast_batch([view], prediction_length)[0]

    def forecast_observation(self, view: ReceiverView, *, conservative: bool = True,
                             extra_delivery_us: float = 0.0, margin_steps: int = 0) -> ForecastTrajectory:
        """Cover AoI and, when supplied, estimated queue/compute delay."""
        length = observation_horizon(view, conservative=conservative,
                                     extra_delivery_us=extra_delivery_us, margin_steps=margin_steps)
        return self.forecast(view, length)

    def metadata(self) -> dict:
        return {"source": self.source, "source_directory": str(self.source_directory),
                "source_sha256": self.source_sha256, "model_directory": str(self.model_path),
                "model_lock": self.model_lock, "settings": self.settings, "runtime": self.runtime,
                "package_files": assert_local_imports(self.root), "parameter_devices": self.parameter_devices,
                "load_time_s": self.load_time_s, "first_forecast": self.first_forecast,
                "cache_before_load": self.cache_before_load, "cache_after_load": self.cache_after_load,
                "gpu_name": self.torch.cuda.get_device_name(), "gpu_capability": list(self.torch.cuda.get_device_capability()),
                "torch_version": self.torch.__version__, "torch_cuda_version": self.torch.version.cuda,
                "forecast_options": self.forecast_options(), "load_options": {"compile": False,
                    "use_flex_attention": False, "hf_kwargs": {"local_files_only": True}},
                "postprocessor_settings": self.postprocessor_settings, "differencing_counts": self.differencing_counts,
                "differencing_audit_definition": "Actual transform_input attempts per target, including adaptive OOM retries; not independent query count",
                "seed_policy": self.seed_policy,
                "last_unit_seed": self.current_unit_seed, "batch_count": len(self.batch_timings),
                "timing_note": "Synchronized batch wall time includes context preparation, forecast and CPU return; per-row amortization is not single-request latency.",
                "trajectory_clock": "source_time_us + step * sample_period_us; step starts at 1",
                "training": "none; shared original pretrained checkpoint"}
