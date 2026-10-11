"""Essential asynchronous flow checks; synthetic CPU trajectories, no GPU loading."""
import asyncio
from types import SimpleNamespace
from time import sleep

import numpy as np
import pytest

from mytirex.model import ForecastTrajectory, TiRexModel
from mytirex.receiver import TimingConfig
from mytirex.streaming import StreamConfig, StreamingRuntime, sleep_until
from mytirex.timing import ClockConfig, ClockMapping
from mytirex.transport import CaptureSession, monotonic_us

PERIOD = 25000
DELAY = 175000


def packet(source_id, origin, sequence, receipt, speed_origin=10.0):
    return {"kind": "packet", "source_clock_id": source_id, "sequence": sequence,
        "source_time_us": origin + sequence * PERIOD, "native_source_time_us": origin + sequence * PERIOD - 10000,
        "value_mps": speed_origin + sequence * .05, "received_at_us": receipt, "event_time_us": receipt}


def calibrated(session, source_id, origin):
    mapping = ClockMapping.known(source_id, session.receiver_clock_id, origin, origin)
    session.clock = SimpleNamespace(snapshot=lambda query: mapping)


def initial_session():
    now = monotonic_us()
    origin = now - DELAY - 7 * PERIOD
    session = CaptureSession("receiver", ClockConfig(), TimingConfig(source_error_us=0, query_error_us=0),
        context_length=8, minimum_observations=2)
    for sequence in range(8):
        session.process(packet("source", origin, sequence, origin + sequence * PERIOD + DELAY))
    calibrated(session, "source", origin)
    return session, origin


class SlowTrajectory:
    batch_size = 1
    context_length = 8

    def __init__(self, compute_s=.07):
        self.compute_s = compute_s
        self.active = self.max_active = 0
        self.views = []

    def synchronize(self):
        pass

    def forecast(self, view, steps):
        assert not hasattr(view, "target_mps") and not hasattr(view, "reference_mps")
        assert not view.values_mps.flags.writeable and view.values_mps.base is None
        assert (view.received_at_us[view.received_mask] <= view.query_time_us).all()
        TiRexModel.prepare_context(self, view)
        before = view.values_mps.copy()
        self.views.append(view)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        sleep(self.compute_s)
        np.testing.assert_equal(view.values_mps, before)
        self.active -= 1
        values = np.tile(float(view.values_mps[-1]) + np.arange(1, steps + 1) * .05, (9, 1))
        return ForecastTrajectory(int(view.source_time_us[-1]), PERIOD, np.arange(1, 10) / 10, values)


def test_slow_worker_receives_new_inputs_and_reuses_time_aligned_trajectory_without_queue():
    async def run():
        session, origin = initial_session()
        model, states, requests = SlowTrajectory(), [], []
        runtime = StreamingRuntime(session, StreamConfig(compute_estimate_us=70000), model=model,
            on_publish=states.append, on_inference=requests.append)
        start = monotonic_us() + 2000
        duration = 350000
        async def feed():
            sequence = 8
            while (due := origin + sequence * PERIOD + DELAY) < start + duration:
                await sleep_until(due)
                runtime.process(packet("source", origin, sequence, monotonic_us()))
                sequence += 1
        await asyncio.gather(runtime.run(duration, start_us=start), feed())
        return runtime, model, states, requests, origin
    runtime, model, states, requests, origin = asyncio.run(run())
    assert model.max_active == runtime.counts["max_in_flight"] == 1
    assert all(row["pending_requests"] == 0 for row in states)
    assert len(states) >= 10 and len(requests) >= 3
    assert any(row["packets_during_inference"] >= 2 for row in requests)
    assert runtime.counts["coalesced_packet_updates"] > 0
    assert all(b["started_at_us"] >= a["completed_at_us"] for a, b in zip(requests, requests[1:]))
    assert model.views[1].source_time_us[-1] >= model.views[0].source_time_us[-1] + 2 * PERIOD
    trajectory = [row for row in states if row["mode"] == "trajectory"]
    assert len(trajectory) >= 5
    for row in trajectory:
        expected = 10.0 + (row["publish_time_us"] - origin) / 1e6 * 2
        assert row["prediction_mps"] == pytest.approx(expected)
    repeated = [(a, b) for a, b in zip(trajectory, trajectory[1:]) if a["forecast_id"] == b["forecast_id"]]
    assert repeated
    for a, b in repeated:
        assert b["prediction_mps"] > a["prediction_mps"]
        assert b["delivery_hold_mps"] == a["delivery_hold_mps"]


def test_unchanged_history_is_refreshed_before_cached_coverage_runs_out():
    session, _ = initial_session()
    states = []
    runtime = StreamingRuntime(session, StreamConfig(inference_period_us=100000, compute_estimate_us=20000),
        model=SlowTrajectory(.02), on_publish=states.append)
    asyncio.run(runtime.run(240000))
    assert runtime.counts["forecasts_installed"] >= 2
    first = next(row["publish_time_us"] for row in states if row["mode"] == "trajectory")
    assert all(row["mode"] == "trajectory" for row in states if row["publish_time_us"] >= first)


def test_source_restart_discards_inflight_result_and_uses_new_history():
    async def run():
        session, _ = initial_session()
        states, requests = [], []
        runtime = StreamingRuntime(session, StreamConfig(compute_estimate_us=80000), model=SlowTrajectory(.08),
            on_publish=states.append, on_inference=requests.append)
        async def restart():
            await asyncio.sleep(.03)
            now = monotonic_us()
            origin = now - DELAY - 7 * PERIOD
            for sequence in range(8):
                runtime.process(packet("new-source", origin, sequence, monotonic_us(), speed_origin=100))
            calibrated(session, "new-source", origin)
        await asyncio.gather(runtime.run(290000), restart())
        return runtime, states, requests
    runtime, states, requests = asyncio.run(run())
    assert runtime.counts["retired_forecasts_discarded"] == 1
    assert not requests[0]["installed"]
    new = [row for row in states if row["source_clock_id"] == "new-source"]
    assert new and any(row["mode"] == "trajectory" for row in new)
    assert all(row["prediction_mps"] >= 100 for row in new)


def test_compute_overrun_keeps_publishing_observation_when_trajectory_expires():
    session, _ = initial_session()
    states = []
    runtime = StreamingRuntime(session, StreamConfig(compute_estimate_us=1000), model=SlowTrajectory(.09),
        on_publish=states.append)
    asyncio.run(runtime.run(230000))
    expired = [row for row in states if row["reason"] == "trajectory_exhausted"]
    assert expired
    assert all(row["prediction_mps"] == row["last_observation_mps"] for row in expired)
    assert runtime.estimator.estimate_us > 1000


def test_absolute_publication_schedule_skips_stalled_slots_and_never_bursts():
    session, _ = initial_session()
    states = []
    def publish(row):
        states.append(row)
        if len(states) == 1:
            sleep(.07)
    runtime = StreamingRuntime(session, StreamConfig(), on_publish=publish)
    result = asyncio.run(runtime.run(160000))
    assert result["counts"]["missed_publication_ticks"] >= 2
    assert states[1]["publication_index"] >= 3
    assert states[1]["missed_ticks"] >= 2
    assert np.diff([row["publish_time_us"] for row in states]).min() > 15000
    assert monotonic_us() >= result["finish_us"]
