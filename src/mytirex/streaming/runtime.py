"""Single-owner reception, one asynchronous model request, and timed publication."""
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import isfinite

import numpy as np

from ..data.arrival import ReceiverView
from ..inference.delivery import ComputeLatencyEstimator, warmup_delivery
from ..inference.model import ForecastTrajectory, observation_horizon
from ..signals import SignalForecast, SignalView
from ..timing.clock import integer_us, observation_timing
from ..timing.transport import CaptureSession, monotonic_us

INFERENCE_FIELDS = ("request_id", "source_clock_id", "receiver_clock_id", "snapshot_time_us", "started_at_us", "completed_at_us",
    "delivery_time_us", "queue_us", "compute_us", "dispatch_us", "compute_estimate_us", "reuse_us",
    "last_source_time_us", "aoi_us", "planned_steps", "effective_horizon_us", "selected_source_offset_us",
    "delivery_covered", "delivery_prediction_mps", "revision", "revision_at_delivery",
    "packets_during_inference", "installed", "completed_in_window", "status", "failure_reason",
    "delivery_prediction", "target_names", "sample_period_us", "future_covariate_observed_steps",
    "future_covariate_latest", "future_covariate_cutoff_time_us")

HORIZON_LIMITS = {"horizon_limit", "future_covariates_horizon_limit"}


async def sleep_until(deadline_us: int) -> None:
    while (remaining := deadline_us - monotonic_us()) > 0:
        await asyncio.sleep(remaining / 1e6)


@dataclass(frozen=True)
class StreamConfig:
    publication_period_us: int = 25_000
    inference_period_us: int = 25_000
    maximum_prediction_steps: int = 320
    compute_estimate_us: float = 17_000.0
    margin_steps: int = 1
    warmup_requests: int = 0

    def __post_init__(self):
        for name in ("publication_period_us", "inference_period_us", "maximum_prediction_steps"):
            integer_us(getattr(self, name), name, 1)
        integer_us(self.margin_steps, "margin_steps")
        integer_us(self.warmup_requests, "warmup_requests")
        if self.maximum_prediction_steps > 320:
            raise ValueError("Stream trajectory cap cannot exceed 320 steps")
        if not isfinite(self.compute_estimate_us) or self.compute_estimate_us <= 0:
            raise ValueError("Compute estimate must be finite and positive")


@dataclass(frozen=True)
class ForecastWork:
    request_id: int
    source_clock_id: str
    revision: int
    view: ReceiverView | SignalView
    started_at_us: int
    completed_at_us: int
    compute_estimate_us: float
    reuse_us: float
    planned_steps: int
    trajectory: ForecastTrajectory | SignalForecast | None
    failure_reason: str

    @property
    def queue_us(self) -> int:
        return self.started_at_us - self.view.query_time_us

    @property
    def compute_us(self) -> int:
        return self.completed_at_us - self.started_at_us


@dataclass(frozen=True)
class CachedForecast:
    work: ForecastWork
    installed_at_us: int
    delivery_values: dict[str, float] | None

    @property
    def delivery_prediction_mps(self) -> float | None:
        return self.delivery_values.get("speed") if self.delivery_values else None


def run_forecast(model, view: ReceiverView | SignalView, request_id: int, source_id: str, revision: int,
                 estimate_us: float, config: StreamConfig) -> ForecastWork:
    """Worker sees detached received targets and observed covariates, never scoring truth."""
    started = monotonic_us()
    # Cover this completion and the expected interval until the next completion.
    reuse_us = max(float(config.inference_period_us), estimate_us)
    steps, trajectory, reason = 0, None, ""
    try:
        steps = observation_horizon(view, extra_delivery_us=started - view.query_time_us
                                    + estimate_us + reuse_us, margin_steps=config.margin_steps)
        if steps > config.maximum_prediction_steps:
            reason = "horizon_limit"
        elif (isinstance(view, SignalView) and view.schema.future_covariates
              and (view.known_future is None or view.known_future.shape[1] < steps)):
            reason = "future_covariates_horizon_limit"
        else:
            model.synchronize()
            trajectory = model.forecast(view, steps)
            model.synchronize()
            if trajectory.status != "ok":
                reason = trajectory.failure_reason or "model_failure"
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
    completed = monotonic_us()
    return ForecastWork(request_id, source_id, revision, view, started, completed,
                        estimate_us, reuse_us, steps, trajectory, reason)


class StreamingRuntime:
    """The event loop owns all receiver/clock/cache mutations.

    Only one coroutine submits inference, and it awaits completion before taking
    the next snapshot. Packets update the bounded receiver during that await;
    they never create queued requests. Callbacks must return promptly.
    """

    def __init__(self, session: CaptureSession, config: StreamConfig, *, model=None,
                 estimator: ComputeLatencyEstimator | None = None, executor=None,
                 on_publish=None, on_inference=None, on_snapshot=None):
        if model is not None and model.batch_size != 1:
            raise ValueError("Streaming inference requires batch_size=1")
        self.session, self.config, self.model = session, config, model
        self.estimator = estimator or ComputeLatencyEstimator(config.compute_estimate_us)
        self.on_publish = on_publish or (lambda row: None)
        self.on_inference = on_inference or (lambda row: None)
        # Attach locally observed covariates to a detached view on the owner
        # thread, before submitting the native model request.
        self.on_snapshot = on_snapshot
        self._owns_executor = model is not None and executor is None
        self.executor = executor
        if self._owns_executor:
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tirex-stream")
        self._wake = asyncio.Event()
        self._running = self._in_flight = self._warming = False
        self.cache = None
        self.counts = Counter()
        self.warmup = None

    def process(self, event: dict) -> None:
        previous = self.session.source_clock_id
        self.session.process(event)
        if self.session.source_clock_id != previous:
            self.cache = None
        self._wake.set()

    def _cache_timing(self, timestamp: int, clock=None):
        if self.cache is None:
            return None
        if clock is None and self.session.clock is not None:
            clock = self.session.clock.snapshot(timestamp)
        work = self.cache.work
        return observation_timing(work.trajectory.source_time_us, None, timestamp,
            work.trajectory.sample_period_us, clock,
            source_error_us=self.session.timing_config.source_error_us,
            query_error_us=self.session.timing_config.query_error_us,
            maximum_uncertainty_us=self.session.timing_config.maximum_uncertainty_us)

    def _remaining_us(self, timestamp: int) -> float | None:
        timing = self._cache_timing(timestamp)
        if timing is None or not timing.valid:
            return None
        trajectory = self.cache.work.trajectory
        return (trajectory.prediction_steps * trajectory.sample_period_us
                - timing.source_offset_us) * timing.clock.rate

    async def _wait_changed(self, timeout_us: float) -> None:
        self._wake.clear()
        timeout = min(timeout_us, self.finish_us - monotonic_us()) / 1e6
        if not self._running or timeout <= 0:
            return
        try:
            await asyncio.wait_for(self._wake.wait(), timeout)
        except TimeoutError:
            pass

    async def _infer_loop(self) -> None:
        loop = asyncio.get_running_loop()
        next_allowed, last_key = self.start_us, None
        warmed = self.config.warmup_requests == 0
        while self._running and monotonic_us() < self.finish_us:
            await sleep_until(min(next_allowed, self.finish_us))
            now = monotonic_us()
            if not self._running or now >= self.finish_us:
                break
            view = self.session.snapshot(now)
            if view is not None and self.on_snapshot is not None:
                view = self.on_snapshot(view)
            receiver = self.session.receiver
            key = (self.session.source_clock_id, receiver.revision if receiver else 0,
                   view.features.get("covariate_revision", 0) if view is not None else 0)
            if key == last_key:
                remaining = self._remaining_us(now)
                threshold = self.estimator.estimate_us + self.config.publication_period_us
                if self.cache is None or remaining is None:
                    await self._wait_changed(self.finish_us - now)
                    continue
                if remaining > threshold:
                    await self._wait_changed(remaining - threshold)
                    continue
            if view is None or not view.ready:
                await self._wait_changed(self.config.publication_period_us)
                continue
            if not warmed:
                if isinstance(view, SignalView) and view.schema.future_covariates:
                    warmup_steps = observation_horizon(view, extra_delivery_us=self.estimator.estimate_us
                        + max(self.config.inference_period_us, self.estimator.estimate_us), margin_steps=1)
                    if view.known_future is None or view.known_future.shape[1] < warmup_steps:
                        self.counts["horizon_limited"] += 1
                        await self._wait_changed(self.config.publication_period_us)
                        continue
                self._in_flight = self._warming = True
                self.counts["max_in_flight"] = 1
                try:
                    self.warmup = await loop.run_in_executor(self.executor, warmup_delivery,
                        self.model, view, self.estimator, self.config.warmup_requests,
                        max(self.config.inference_period_us, self.estimator.estimate_us))
                finally:
                    self._in_flight = self._warming = False
                warmed = True
                # Warmup may be slow; always capture fresh history afterwards.
                continue
            request_id = self.counts["requests_started"]
            self.counts["requests_started"] += 1
            self.counts["max_in_flight"] = 1
            if last_key is not None and key[0] == last_key[0]:
                self.counts["coalesced_packet_updates"] += max(0, key[1] - last_key[1] - 1)
            last_key = key
            next_allowed = now + self.config.inference_period_us
            self._in_flight = True
            try:
                work = await loop.run_in_executor(self.executor, run_forecast, self.model, view,
                    request_id, key[0], key[1], self.estimator.estimate_us, self.config)
            finally:
                self._in_flight = False
            installed = monotonic_us()
            same_session = self.session.source_clock_id == work.source_clock_id
            successful = work.trajectory is not None and work.trajectory.status == "ok"
            accepted = successful and same_session and self._running and installed < self.finish_us
            prediction, timing = None, None
            if successful:
                self.estimator.update(work.compute_us)
            if accepted:
                self.cache = CachedForecast(work, installed, None)
                timing = self._cache_timing(installed)
                if timing.valid and timing.source_offset_us <= work.planned_steps * view.sample_period_us + 1e-6:
                    prediction = work.trajectory.values_at_offset(timing.source_offset_us, view.last_values)
                self.cache = CachedForecast(work, installed, prediction)
                self.counts["forecasts_installed"] += 1
            elif not same_session:
                self.counts["retired_forecasts_discarded"] += 1
            if work.failure_reason:
                self.counts["horizon_limited" if work.failure_reason in HORIZON_LIMITS else "model_failures"] += 1
            current_revision = self.session.receiver.revision if same_session else None
            self.on_inference({
                "request_id": request_id, "source_clock_id": work.source_clock_id,
                "receiver_clock_id": self.session.receiver_clock_id,
                "snapshot_time_us": view.query_time_us, "started_at_us": work.started_at_us,
                "completed_at_us": work.completed_at_us, "delivery_time_us": installed,
                "queue_us": work.queue_us, "compute_us": work.compute_us,
                "dispatch_us": installed - work.completed_at_us,
                "compute_estimate_us": work.compute_estimate_us, "reuse_us": work.reuse_us,
                "last_source_time_us": int(view.source_time_us[-1]),
                "aoi_us": view.timing.age_us, "planned_steps": work.planned_steps,
                "effective_horizon_us": view.timing.age_us + installed - view.query_time_us,
                "selected_source_offset_us": timing.source_offset_us if timing else None,
                "delivery_covered": prediction is not None, "delivery_prediction_mps": prediction.get("speed") if prediction else None,
                "delivery_prediction": dict(prediction) if prediction is not None else None,
                "target_names": list(view.last_values), "sample_period_us": view.sample_period_us,
                "revision": work.revision, "revision_at_delivery": current_revision,
                "packets_during_inference": current_revision - work.revision if same_session else None,
                "installed": accepted, "completed_in_window": installed < self.finish_us,
                "status": "ok" if successful else "not_run" if work.trajectory is None else "failed",
                "failure_reason": work.failure_reason,
                "future_covariate_observed_steps": (int(np.isfinite(view.known_future).all(axis=0).sum())
                    if isinstance(view, SignalView) and view.known_future is not None else 0),
                "future_covariate_latest": ({name: view.features["future_covariate_latest." + name]
                    for name in view.schema.future_covariates if "future_covariate_latest." + name in view.features}
                    if isinstance(view, SignalView) else {}),
                "future_covariate_cutoff_time_us": view.features.get("future_covariate_cutoff_time_us"),
            })
            if work.failure_reason and work.failure_reason not in HORIZON_LIMITS and not self.counts["forecasts_installed"]:
                raise RuntimeError(f"Initial streaming forecast failed: {work.failure_reason}")

    def _publication(self, scheduled: int, index: int, skipped: int) -> dict:
        view = self.session.snapshot(monotonic_us())
        published = monotonic_us()
        timing = view.timing if view is not None else None
        last = view.last_values if view is not None else {}
        elapsed = published - view.query_time_us if view is not None else 0
        age = timing.age_us + elapsed if timing is not None and timing.age_us is not None else None
        reason = "model_disabled" if self.model is None else "awaiting_forecast"
        mode, prediction, delivery_hold = "observation_hold", last, last
        offset = remaining = None
        cache = self.cache
        if not last:
            mode, reason = "no_observation", "no_received_observation"
        elif cache is not None:
            delivery_hold = cache.delivery_values if cache.delivery_values is not None else last
            selection = self._cache_timing(published, timing.clock if timing else None)
            if not selection.valid:
                reason = selection.reason
            else:
                trajectory = cache.work.trajectory
                offset = selection.source_offset_us
                remaining = (trajectory.prediction_steps * trajectory.sample_period_us - offset) * selection.clock.rate
                if remaining >= -1e-6:
                    prediction = trajectory.values_at_offset(offset, cache.work.view.last_values)
                    mode, reason = "trajectory", "ok"
                else:
                    reason = "trajectory_exhausted"
        elif timing is not None and not timing.valid:
            reason = timing.reason
        schema = self.session.signal_schema
        return {
            "publication_index": index, "scheduled_time_us": scheduled,
            "snapshot_time_us": view.query_time_us if view is not None else published,
            "publish_time_us": published, "deadline_lateness_us": published - scheduled,
            "missed_ticks": skipped, "source_clock_id": self.session.source_clock_id or "",
            "receiver_clock_id": self.session.receiver_clock_id,
            "last_source_time_us": int(view.source_time_us[-1]) if last else None,
            "last_observation": dict(last), "last_observation_mps": last.get("speed"), "aoi_us": age,
            "aoi_lower_us": timing.age_lower_us + elapsed if age is not None else None,
            "aoi_upper_us": timing.age_upper_us + elapsed if age is not None else None,
            "native_aoi_us": timing.native_age_us + elapsed if timing and timing.native_age_us is not None else None,
            "timing_valid": bool(timing and timing.valid),
            "context_ready": bool(view and view.features.get("context_ready", 0)),
            "prediction": dict(prediction), "delivery_hold": dict(delivery_hold),
            "prediction_mps": prediction.get("speed"), "delivery_hold_mps": delivery_hold.get("speed"),
            "target_names": list(schema.targets) if schema else ["speed"],
            "past_covariate_names": list(schema.past_covariates) if schema else [],
            "future_covariate_names": list(schema.future_covariates) if schema else [],
            "units": dict(schema.units) if schema else {"speed": "m/s"},
            "sample_period_us": self.session.sample_period_us,
            "mode": mode, "reason": reason, "fallback": self.model is not None and mode != "trajectory",
            "forecast_id": cache.work.request_id if cache else None,
            "forecast_source_time_us": cache.work.trajectory.source_time_us if cache else None,
            "forecast_delivery_time_us": cache.installed_at_us if cache else None,
            "forecast_age_us": published - cache.installed_at_us if cache else None,
            "selected_source_offset_us": offset, "coverage_remaining_us": remaining,
            "inference_running": self._in_flight, "warmup_running": self._warming,
            "pending_requests": 0,
        }

    async def _publish_loop(self) -> None:
        period, index, pending_skipped = self.config.publication_period_us, 0, 0
        try:
            while self._running:
                scheduled = self.start_us + index * period
                if scheduled >= self.finish_us:
                    await sleep_until(self.finish_us)
                    break
                await sleep_until(scheduled)
                now = monotonic_us()
                if not self._running or now >= self.finish_us:
                    break
                skipped = (now - scheduled) // period
                index += skipped
                scheduled += skipped * period
                row = self._publication(scheduled, index, pending_skipped + int(skipped))
                self.on_publish(row)
                self.counts["publications"] += 1
                self.counts[row["mode"] + "_publications"] += 1
                index += 1
                pending_skipped = 0
                # Keep absolute phase and skip passed deadlines instead of catching up.
                now = monotonic_us()
                next_tick = self.start_us + index * period
                if next_tick < now:
                    pending_skipped = int((now - next_tick) // period + 1)
                    index += pending_skipped
        finally:
            planned = (self.finish_us - self.start_us + period - 1) // period
            self.counts["missed_publication_ticks"] = planned - self.counts["publications"]
            self._running = False
            self._wake.set()

    async def run(self, duration_us: int, *, start_us: int | None = None) -> dict:
        integer_us(duration_us, "duration_us", 1)
        self.start_us = monotonic_us() if start_us is None else integer_us(start_us, "start_us")
        self.finish_us = self.start_us + duration_us
        self._running = True
        tasks = [asyncio.create_task(self._publish_loop())]
        if self.model is not None:
            tasks.append(asyncio.create_task(self._infer_loop()))
        try:
            await asyncio.gather(*tasks)
        finally:
            self._running = False
            self._wake.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self._owns_executor:
                self.executor.shutdown(wait=True)
        if self.model is not None and not self.counts["forecasts_installed"]:
            raise RuntimeError("No current-model forecast became available during the stream window")
        return {"start_us": self.start_us, "finish_us": self.finish_us,
            "duration_s": duration_us / 1e6, "counts": dict(self.counts),
            "pending_request_max": 0, "final_compute_estimate_us": self.estimator.estimate_us,
            "warmup": self.warmup}
