"""Generic data contracts, real current-model adapter shapes, and causal streams."""
import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mytirex.inference.model import TiRexModel
from mytirex.inference.delivery import ComputeLatencyEstimator, forecast_delivery
from mytirex.signals import SignalForecast, SignalSchema, SignalView, causal_signal_context
from mytirex.streaming.live import StreamDatagrams
from mytirex.streaming.runtime import StreamConfig, StreamingRuntime, run_forecast
from mytirex.streaming.source import SignalSource
from mytirex.timing.clock import ClockConfig, ClockMapping
from mytirex.timing.receiver import TimestampedPacket, TimingConfig
from mytirex.timing.transport import CaptureSession, monotonic_us

ROOT = Path(__file__).resolve().parents[1]


def calibrated(schema, period=25_000, context=4):
    session = CaptureSession("receiver", ClockConfig(), TimingConfig(source_error_us=0, query_error_us=0),
        context_length=context, minimum_observations=2, signal_schema=schema, sample_period_us=period)
    return session


def feed(session, sequence, receipt, values, origin=1_000_000, source="source", future=None):
    timestamp = origin + sequence * session.sample_period_us
    session.process({"kind": "packet", "source_clock_id": source, "sequence": sequence,
        "source_time_us": timestamp, "received_at_us": receipt, "event_time_us": receipt,
        "values": values, "future_values": future})
    mapping = ClockMapping.known(source, session.receiver_clock_id, origin, origin, at_receiver_us=receipt)
    session.clock = SimpleNamespace(snapshot=lambda query: mapping)


def test_named_history_is_causal_detached_and_late_packets_fill_each_row():
    schema = SignalSchema(("steer", "x"), ("throttle",))
    session = calibrated(schema, period=10_000)
    for sequence, receipt in ((0, 1_060_000), (2, 1_070_000), (3, 1_080_000)):
        feed(session, sequence, receipt, dict(steer=sequence, x=10 + sequence, throttle=.5))
    view = session.snapshot(1_080_000)
    assert view.ready and view.sample_period_us == 10_000
    assert view.last_values == {"steer": 3.0, "x": 13.0}
    assert np.isnan(view.values[:, 1]).all() and view.received_at_us[1] == -1
    np.testing.assert_equal(causal_signal_context(view)[:, 1], [0, 10, .5])
    assert view.values.base is None and not view.values.flags.writeable
    feed(session, 1, 1_090_000, dict(steer=1, x=11, throttle=.5))
    later = session.snapshot(1_090_000)
    assert later.received_mask.all() and later.last_values == view.last_values
    assert np.isnan(view.values[:, 1]).all()
    with pytest.raises(TypeError):
        schema.units["steer"] = "deg"


def test_malformed_schema_packet_cannot_retire_valid_source_history():
    session = calibrated(SignalSchema(("brake",)))
    feed(session, 0, 1_100_000, {"brake": .2})
    receiver = session.receiver
    with pytest.raises(ValueError, match="exactly"):
        feed(session, 0, 1_120_000, {"speed": 10}, source="invalid-new-source")
    assert session.receiver is receiver and session.source_clock_id == "source"
    with pytest.raises(ValueError, match="either"):
        TimestampedPacket("source", 0, 0, value_mps=1, values={"brake": .2})
    with pytest.raises(ValueError, match="finite"):
        TimestampedPacket("source", 0, 0, values={"brake": float("nan")})


def test_real_adapter_groups_targets_and_both_covariate_roles_in_one_series():
    import torch

    captured = []
    class SpyForecast:
        future_len = 320

        def forecast(self, series, **kwargs):
            captured.extend(series)
            return [np.broadcast_to(np.arange(len(item.target))[:, None, None],
                    (len(item.target), 9, kwargs["prediction_length"])).copy() for item in series]

    model = object.__new__(TiRexModel)
    model.model, model.torch, model.timeseries_type = SpyForecast(), torch, SimpleNamespace
    model.context_length, model.batch_size = 4, 1
    model.quantiles, model.batch_timings = np.arange(1, 10) / 10, []
    model.first_forecast = {"already_warm": True}
    model.synchronize = lambda: None
    model.forecast_options = lambda: {}
    result = model.predict_signals({"yaw": np.arange(6.), "x": np.arange(6.) + 10}, 3,
        past_covariates={"throttle": np.arange(6.) * .1},
        future_covariates={"planned_steer": np.arange(9.)}, sample_period_us=10_000)
    assert result.status == "ok" and result.values.shape == (2, 9, 3)
    assert result.target_names == ("yaw", "x")
    assert len(captured) == 1  # Two target rows in ONE joint series, not a batch of two scalar forecasts.
    item = captured[0]
    np.testing.assert_equal(item.target.numpy(), [[2, 3, 4, 5], [12, 13, 14, 15]])
    np.testing.assert_allclose(item.past_covariates.numpy(), [[.2, .3, .4, .5]])
    np.testing.assert_equal(item.future_covariates.numpy(), [[2, 3, 4, 5, 6, 7, 8]])
    assert result.values_at(result.source_time_us + 10_000) == {"yaw": 0., "x": 1.}
    with pytest.raises(ValueError, match="whole requested horizon"):
        model.predict_signals({"yaw": np.arange(6.)}, 4, future_covariates={"plan": np.arange(9.)})
    assert len(captured) == 1
    result = model.predict_signals({"speed": np.arange(6.)}, 3,
        future_covariates={"driver_throttle": [.8] * 6 + [0., 0., np.nan]})
    assert result.status == "ok"
    np.testing.assert_allclose(captured[-1].future_covariates.numpy(), [[.8, .8, .8, .8, 0., 0., np.nan]])


class JointTrajectory:
    batch_size = 1
    context_length = 4

    def synchronize(self):
        pass

    def forecast(self, view, steps):
        last = view.values[:len(view.schema.targets), -1]
        values = last[:, None, None] + np.arange(1, steps + 1)[None, None, :] * .01
        return SignalForecast(int(view.source_time_us[-1]), view.sample_period_us, np.array([.5]),
                              values, view.schema.targets)

    def forecast_observation(self, view, **kwargs):
        from mytirex.inference.model import observation_horizon
        return self.forecast(view, observation_horizon(view, **kwargs))


def test_synchronous_delivery_compensation_returns_named_values_for_every_target():
    view = SignalView.from_history({"brake": [.1, .2, .3, .4], "steer": [-.2] * 4})
    view = replace(view, query_time_us=view.query_time_us + 40_000)
    result = forecast_delivery(JointTrajectory(), view, ComputeLatencyEstimator(), queue_us=10_000)
    assert result.delivery_covered and set(result.prediction_delivery) == {"brake", "steer"}
    assert result.prediction_delivery["brake"] >= result.prediction_start["brake"]


def test_generic_stream_publishes_each_target_and_refreshes_cached_trajectory():
    schema = SignalSchema(("throttle", "steer"), ("speed",))
    session = calibrated(schema)
    now, period = monotonic_us(), session.sample_period_us
    origin = now - 175_000 - 3 * period
    for sequence in range(4):
        feed(session, sequence, now, dict(throttle=.3, steer=-.2, speed=10.), origin=origin)
    states = []
    runtime = StreamingRuntime(session, StreamConfig(inference_period_us=50_000),
                               model=JointTrajectory(), on_publish=states.append)
    result = asyncio.run(runtime.run(130_000))
    predicted = [row for row in states if row["mode"] == "trajectory"]
    assert predicted and result["counts"]["forecasts_installed"] >= 2
    assert all(set(row["prediction"]) == {"throttle", "steer"} for row in states)
    assert all(row["prediction_mps"] is None and row["pending_requests"] == 0 for row in states)
    assert all(row["past_covariate_names"] == ["speed"] for row in states)
    assert all(row["prediction"]["throttle"] > .3 for row in predicted)
    assert all(row["prediction"]["steer"] < 0 for row in predicted)


def test_future_known_covariates_are_copied_and_insufficient_plan_is_explicit():
    schema = SignalSchema(("x",), future_covariates=("steer_plan",))
    session = calibrated(schema)
    now = monotonic_us()
    origin = now - 3 * 25_000
    future = {"steer_plan": [.1]}
    for sequence in range(4):
        feed(session, sequence, now, dict(x=2., steer_plan=0.), origin=origin, future=future)
    view = session.snapshot(now)
    future["steer_plan"][0] = 999
    assert view.known_future[0, 0] == .1 and not view.known_future.flags.writeable
    work = run_forecast(JointTrajectory(), view, 0, "source", 4, 17_000, StreamConfig())
    assert work.trajectory is None and work.failure_reason == "future_covariates_horizon_limit"
    with pytest.raises(ValueError, match="match the schema"):
        feed(session, 4, now + 25_000, dict(x=2., steer_plan=0.), origin=origin, future={"x": [2.]})


def test_short_future_plan_keeps_publishing_until_a_warmup_plan_arrives():
    async def run():
        schema = SignalSchema(("x",), future_covariates=("steer_plan",))
        session = calibrated(schema)
        now = monotonic_us()
        origin = now - 125_000
        for sequence in range(4):
            feed(session, sequence, now, dict(x=2., steer_plan=0.), origin=origin, future={"steer_plan": [.1]})
        states = []
        runtime = StreamingRuntime(session, StreamConfig(warmup_requests=1),
                                   model=JointTrajectory(), on_publish=states.append)
        async def plan_arrives():
            await asyncio.sleep(.04)
            feed(session, 4, monotonic_us(), dict(x=2., steer_plan=0.), origin=origin, future={"steer_plan": [.1] * 32})
            runtime._wake.set()
        await asyncio.gather(runtime.run(150_000), plan_arrives())
        return runtime, states
    runtime, states = asyncio.run(run())
    assert runtime.counts["horizon_limited"] > 0 and runtime.counts["forecasts_installed"] > 0
    assert states[0]["mode"] == "observation_hold" and any(row["mode"] == "trajectory" for row in states)


def test_udp_adapter_receives_named_values_and_measures_age_on_custom_grid(tmp_path):
    async def run():
        schema, states = SignalSchema(("steer", "yaw")), []
        session = calibrated(schema, period=10_000)
        runtime = StreamingRuntime(session, StreamConfig(publication_period_us=10_000), on_publish=states.append)
        source = SignalSource(schema, sample_period_us=10_000)
        loop = asyncio.get_running_loop()
        sending, _ = await loop.create_datagram_endpoint(lambda: source, local_addr=("127.0.0.1", 0))
        with (tmp_path / "events.jsonl").open("w") as log:
            protocol = StreamDatagrams(runtime, log)
            receiving, _ = await loop.create_datagram_endpoint(lambda: protocol, remote_addr=sending.get_extra_info("sockname"))
            start = monotonic_us()
            sync = asyncio.create_task(protocol.synchronize(receiving, start, start + 220_000, 40_000))
            async def feed_wire():
                await asyncio.sleep(.03)
                origin = monotonic_us()
                for sequence in range(14):
                    timestamp = origin + sequence * 10_000
                    from mytirex.streaming.runtime import sleep_until
                    await sleep_until(timestamp)
                    source.publish({"steer": sequence * .01, "yaw": sequence},
                                   sequence=sequence, source_time_us=timestamp)
            try:
                await asyncio.gather(runtime.run(220_000, start_us=start), feed_wire(), sync)
            finally:
                sync.cancel()
                await asyncio.gather(sync, return_exceptions=True)
                receiving.close()
                sending.close()
        return session, states, protocol
    session, states, protocol = asyncio.run(run())
    assert session.counts["packets"] > 8 and session.counts["sync_accepted"] > 0
    assert not protocol.counts["malformed_datagrams"]
    available = [row for row in states if row["prediction"]]
    assert available and any(row["context_ready"] for row in available)
    assert any(row["timing_valid"] and row["aoi_us"] >= 0 for row in available)
    assert all(set(row["prediction"]) == {"steer", "yaw"} and row["sample_period_us"] == 10_000 for row in available)


@pytest.mark.parametrize("targets,past", [({"steer": [1, 2]}, {"speed": [1]}),
                                          ({"steer": [1, float("nan")]}, {})])
def test_direct_history_rejects_unaligned_or_nonfinite_input(targets, past):
    with pytest.raises(ValueError):
        SignalView.from_history(targets, past_covariates=past)


@pytest.mark.parametrize("plan", [[.8, np.nan, 0., 0.], [.8, .8, 0., np.inf]])
def test_partial_future_covariates_still_reject_missing_history_and_infinity(plan):
    with pytest.raises(ValueError, match="finite history"):
        SignalView.from_history({"speed": [1., 2.]}, future_covariates={"driver_throttle": plan})
