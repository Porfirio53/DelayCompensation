"""Small batch=1 wrapper and paired delivery-time scoring; no inference queue."""
from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np
import pandas as pd

from ..reporting.aoi_metrics import AoiAccumulator, error_statistics
from ..data.arrival import ReceiverView
from ..evaluation.benchmark import latency_summary
from .model import ForecastTrajectory, observation_horizon
from ..timing.clock import horizon_steps, integer_us
from ..timing.transport import query_row

DELIVERY_POLICIES = ("aoi_only", "compute_compensated")


@dataclass
class ComputeLatencyEstimator:
    # Rounded-up maximum stable P99 from the two supplied current-model runs.
    estimate_us: float = 17_000.0
    samples: int = 0

    def update(self, measured_us: float) -> None:
        self.estimate_us = 0.8 * self.estimate_us + 0.2 * measured_us
        self.samples += 1


@dataclass(frozen=True)
class DeliveryForecast:
    trajectory: ForecastTrajectory
    prediction_start: float
    prediction_delivery: float
    compute_estimate_us: float
    compute_us: int
    queue_us: int
    delivery_time_us: int
    source_offset_start_us: float
    source_offset_delivery_us: float
    effective_horizon_us: float
    delivery_covered: bool


def forecast_delivery(model, view: ReceiverView, estimator: ComputeLatencyEstimator, *,
                      queue_us: int = 0) -> DeliveryForecast:
    """Generate once, then select the median at actual CPU-trajectory readiness.

    queue_us is elapsed receiver time since the detached view was captured.
    Replay uses zero; a future online caller can pass its measured queue time.
    The predictor receives only the view, never the replay clock or labels.
    """
    if model.batch_size != 1:
        raise ValueError("Delivery compensation requires batch_size=1")
    integer_us(queue_us, "queue_us")
    estimate = estimator.estimate_us
    model.synchronize()
    started = perf_counter_ns()
    trajectory = model.forecast_observation(view, extra_delivery_us=queue_us + estimate, margin_steps=1)
    model.synchronize()
    compute_us = (perf_counter_ns() - started + 999) // 1000
    timing = view.timing
    rate = timing.clock.rate if timing is not None else 1.0
    offset = timing.source_offset_us if timing is not None else view.query_time_us - int(view.source_time_us[-1])
    age = timing.age_us if timing is not None else offset
    delivery_offset = offset + (queue_us + compute_us) / rate
    covered = delivery_offset <= trajectory.values_mps.shape[1] * view.sample_period_us + 1e-6
    last = float(view.values_mps[-1])
    at_start, at_delivery = last, last
    if trajectory.status == "ok":
        at_start = trajectory.value_at_offset(offset, last)
        if covered:
            at_delivery = trajectory.value_at_offset(delivery_offset, last)
        estimator.update(compute_us)
    return DeliveryForecast(trajectory, at_start, at_delivery, estimate, compute_us, queue_us,
        view.query_time_us + queue_us + compute_us, offset, delivery_offset,
        age + queue_us + compute_us, covered)


def warmup_delivery(model, view: ReceiverView, estimator: ComputeLatencyEstimator, requests: int,
                    reuse_us: float = 0.0) -> dict:
    """Prewarm the actual short-horizon path; omit the first request from EWMA."""
    steps = observation_horizon(view, extra_delivery_us=estimator.estimate_us + reuse_us, margin_steps=1)
    seconds = []
    for index in range(requests):
        model.synchronize()
        started = perf_counter_ns()
        trajectory = model.forecast(view, steps)
        model.synchronize()
        elapsed = (perf_counter_ns() - started) / 1e9
        if trajectory.status != "ok":
            raise RuntimeError(f"Delivery warmup failed: {trajectory.failure_reason}")
        seconds.append(elapsed)
        if index:
            estimator.update(elapsed * 1e6)
    return {"requests": requests, "horizon_steps": steps, "reuse_us": reuse_us, "first_request_s": seconds[0],
            "subsequent_requests": latency_summary(seconds[1:]),
            "included_in_scoring": False, "first_request_in_estimator": False}


def evaluate_delivery_case(case, *, model, estimator: ComputeLatencyEstimator, target_for,
                           maximum_steps: int, batch_id: str, source_rate: float) -> pd.DataFrame:
    """Pair AoI-only and compensated selections from the same short trajectory.

    target_for(index, receiver_elapsed_us) belongs to the scorer and uses causal
    ZOH reference values. Both methods share the actual delivery timestamp.
    """
    view, result = case.view, None
    timing = view.timing
    reason = view.reason if not view.ready else ""
    planned_steps = 0
    if view.ready:
        planned_steps = observation_horizon(view, extra_delivery_us=estimator.estimate_us, margin_steps=1)
        if planned_steps > maximum_steps:
            reason = "horizon_limit"
        else:
            result = forecast_delivery(model, view, estimator)
    trajectory = result.trajectory if result is not None else None
    if trajectory is not None and trajectory.status != "ok":
        reason = trajectory.failure_reason
    last = float(view.values_mps[-1]) if len(view.values_mps) else None
    elapsed_us = result.queue_us + result.compute_us if result is not None else 0
    # No supervision is accessed before the forecast and CPU return complete.
    reference_start = float(target_for(case.query_index, 0))
    reference_delivery = float(target_for(case.query_index, elapsed_us))
    base = query_row(view, timing.clock.source_clock_id if timing and timing.clock else "")
    common = {**base, "query_index": case.query_index,
        "context_ready": bool(view.features.get("context_ready", 0)),
        "timing_valid": bool(timing and timing.valid), "timing_reason": timing.reason if timing else view.reason,
        "aoi_us": timing.age_us if timing else None,
        "aoi_lower_us": timing.age_lower_us if timing else None, "aoi_upper_us": timing.age_upper_us if timing else None,
        "native_aoi_us": timing.native_age_us if timing else None,
        "native_aoi_lower_us": timing.native_age_lower_us if timing else None,
        "native_aoi_upper_us": timing.native_age_upper_us if timing else None,
        "source_offset_us": timing.source_offset_us if timing else None,
        "horizon_steps": timing.horizon_steps if timing else None,
        "upper_horizon_steps": timing.upper_horizon_steps if timing else None,
        "true_aoi_us": case.true_aoi_us, "true_native_aoi_us": case.true_native_aoi_us,
        "true_source_offset_us": case.true_source_offset_us, "true_horizon_steps": case.true_horizon_steps,
        "true_delivery_aoi_us": case.true_aoi_us + elapsed_us if case.true_aoi_us is not None else None,
        "true_delivery_horizon_steps": horizon_steps(case.true_source_offset_us + elapsed_us / source_rate,
            view.sample_period_us) if case.true_source_offset_us is not None else None,
        "delivery_time_us": result.delivery_time_us if result else view.query_time_us,
        "queue_us": result.queue_us if result else 0,
        "compute_estimate_us": result.compute_estimate_us if result else estimator.estimate_us,
        "compute_us": result.compute_us if result else None,
        "effective_horizon_us": result.effective_horizon_us if result else None,
        "delivery_covered": result.delivery_covered if result else False,
        "planned_trajectory_steps": planned_steps,
        "prediction_start": result.prediction_start if result else last,
        "reference_start": reference_start, "reference": reference_delivery,
        "reference_available": bool(np.isfinite(reference_delivery)), "model_requested": True,
        "inference_status": trajectory.status if trajectory else "not_run",
        "trajectory_steps": trajectory.values_mps.shape[1] if trajectory else 0,
        "batch_id": batch_id, "batch_size": trajectory.batch_size if trajectory else 0,
        "batch_elapsed_s": trajectory.batch_elapsed_s if trajectory else 0.0,
        "elapsed_s": result.compute_us / 1e6 if result else 0.0}
    rows = []
    for policy in DELIVERY_POLICIES:
        compensated = policy == "compute_compensated"
        failure = reason
        if not failure and compensated and result is not None and not result.delivery_covered:
            failure = "delivery_horizon_exhausted"
        prediction = (result.prediction_delivery if compensated else result.prediction_start) if result else last
        selected = (result.source_offset_delivery_us if compensated else result.source_offset_start_us) if result else None
        rows.append({**common, "policy": policy, "prediction": prediction,
            "selected_source_offset_us": selected,
            "selected_horizon_steps": horizon_steps(selected, view.sample_period_us) if selected is not None else None,
            "status": "failed" if trajectory and trajectory.status != "ok" else "fallback" if failure else "ok",
            "fallback": bool(failure), "failure_reason": failure,
            "horizon_exceeded": failure in {"horizon_limit", "delivery_horizon_exhausted"}})
    return pd.DataFrame(rows)


class DeliveryAccumulator(AoiAccumulator):
    """Keep existing AoI metrics, adding start errors and actual request latency."""

    def update(self, frame: pd.DataFrame) -> None:
        super().update(frame)
        for key, group in frame.groupby(list(self.groups), sort=False, dropna=False):
            state = self._states[tuple(key)]
            chunks = state["chunks"]
            start_error = (group.prediction_start - group.reference_start).to_numpy(dtype=float)
            compute = group.compute_us.to_numpy(dtype=float)
            estimate_error = group.compute_estimate_us.to_numpy(dtype=float) - compute
            for name, values in (("start_speed", start_error), ("compute", compute), ("compute_estimate", estimate_error)):
                chunks.setdefault(name, []).append(values[np.isfinite(values)])
            measured = np.isfinite(compute)
            state["counts"]["delivery_measured_count"] += int(measured.sum())
            state["counts"]["delivery_covered_count"] += int((measured & group.delivery_covered.to_numpy(dtype=bool)).sum())

    def tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        table, reasons = super().tables()
        for index, row in table.iterrows():
            state = self._states[tuple(row[name] for name in self.groups)]
            chunks, counts = state["chunks"], state["counts"]
            start_error = np.concatenate(chunks["start_speed"])
            compute = np.concatenate(chunks["compute"])
            estimate_error = np.concatenate(chunks["compute_estimate"])
            extra = {**error_statistics(start_error, "start_speed_mps_"),
                "compute_count": len(compute),
                "compute_estimate_MAE_ms": float(np.abs(estimate_error).mean() / 1000) if len(estimate_error) else None,
                "delivery_covered_count": counts["delivery_covered_count"],
                "delivery_coverage": counts["delivery_covered_count"] / counts["delivery_measured_count"]
                    if counts["delivery_measured_count"] else None}
            extra.update({f"compute_ms_P{p}": float(np.percentile(compute, p) / 1000) if len(compute) else None
                          for p in (50, 95, 99)})
            for name, value in extra.items():
                table.at[index, name] = value
        return table, reasons
