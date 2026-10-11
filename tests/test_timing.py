from dataclasses import replace

import numpy as np
import pytest

from mytirex.inference.model import ForecastTrajectory, TiRexModel
from mytirex.timing.receiver import TimestampedPacket, TimestampedReceiver, TimingConfig
from mytirex.timing.clock import ClockConfig, ClockMapping, ClockSynchronizer, SyncExchange, observation_timing


def exchange(source_origin, receiver_origin, source_tick, rate=1.0, up=5000, down=5000, process=2000):
    r1 = receiver_origin + round(rate * source_tick)
    s2 = source_origin + round((r1 + up - receiver_origin) / rate)
    s3 = s2 + process
    r4 = receiver_origin + round(rate * (s3 - source_origin)) + down
    return SyncExchange(r1, s2, s3, r4)


def synchronized(rate=1.0, up=5000, down=5000):
    source, receiver = 20_000_000_000_000, 40_000_000_000_000
    clock = ClockSynchronizer("source", "receiver", ClockConfig(timestamp_error_us=0))
    last = None
    for tick in range(0, 6_000_001, 250_000):
        last = exchange(source, receiver, tick, rate, up, down)
        assert clock.update(last)
    return clock, source, receiver, last.receiver_receive_us


def test_offset_conversion_preserves_small_age_with_large_different_boot_origins():
    clock, source, receiver, query = synchronized()
    sample = source + query - receiver - 175000
    mapping = clock.snapshot(query)
    age = mapping.age(sample, query)
    assert age.estimate_us == pytest.approx(175000, abs=1e-5)
    assert age.lower_us <= 175000 <= age.upper_us
    timing = observation_timing(sample, sample - 10000, query, 25000, mapping)
    assert timing.valid and timing.horizon_steps == 7
    assert timing.native_age_us - timing.age_us == pytest.approx(10000)


@pytest.mark.parametrize("ppm", [-120, 120])
def test_affine_fit_tracks_relative_drift_while_offset_only_accumulates_error(ppm):
    rate = 1 + ppm * 1e-6
    clock, source, receiver, last_sync = synchronized(rate)
    query = last_sync + 1_000_000
    sample = source + round((query - receiver) / rate) - 175000
    true_age = query - receiver - rate * (sample - source)
    mapping = clock.snapshot(query)
    assert mapping.drift_fitted
    assert mapping.relative_drift_ppm == pytest.approx(ppm, abs=0.4)
    assert mapping.age(sample, query).estimate_us == pytest.approx(true_age, abs=2.0)
    offset_only = clock.snapshot(query, estimate_drift=False)
    assert abs(offset_only.age(sample, query).estimate_us - true_age) > 100


def test_asymmetric_path_has_a_point_bias_but_retains_a_bound_containing_truth():
    clock, source, receiver, query = synchronized(up=1000, down=45000)
    sample = source + query - receiver - 300000
    age = clock.snapshot(query).age(sample, query)
    assert abs(age.estimate_us - 300000) > 20000
    assert age.lower_us <= 300000 <= age.upper_us
    assert age.upper_us - age.lower_us >= 40000


def test_no_sync_warmup_stale_and_future_sync_are_explicit():
    clock = ClockSynchronizer("source", "receiver", ClockConfig(timestamp_error_us=0))
    assert clock.snapshot(0) is None
    missing = observation_timing(0, 0, 0, 25000, None)
    assert not missing.valid and missing.reason == "clock_unavailable" and missing.age_us is None
    first = exchange(0, 1_000_000, 0)
    clock.update(first)
    assert clock.snapshot(first.receiver_receive_us).status == "warming_up"
    with pytest.raises(ValueError, match="future synchronization"):
        clock.snapshot(first.receiver_receive_us - 1)
    for tick in (250000, 500000):
        item = exchange(0, 1_000_000, tick)
        clock.update(item)
    now = item.receiver_receive_us
    before = clock.snapshot(now)
    clock.update(exchange(0, 1_000_000, 750000))
    assert len(before.exchanges) == 3  # A past immutable mapping never changes.
    later = item.receiver_receive_us + 4_000_000
    assert clock.snapshot(later).status == "stale"


def test_negative_age_and_inconsistent_clock_are_not_silently_clamped():
    mapping = ClockMapping.known("source", "receiver", 0, 0)
    timing = observation_timing(100000, 100000, 50000, 25000, mapping)
    assert timing.age_us == -50000
    assert not timing.valid and timing.reason == "negative_observation_age"
    clock = ClockSynchronizer("source", "receiver", ClockConfig(timestamp_error_us=0))
    clock.update(exchange(0, 1_000_000, 0))
    clock.update(exchange(0, 2_000_000, 250000))
    assert clock.snapshot(2_300_000).status == "inconsistent"


def test_round_trip_validation_and_window_are_bounded():
    clock = ClockSynchronizer("source", "receiver", ClockConfig(window_samples=4, minimum_samples=3,
                                                                 timestamp_error_us=0))
    assert not clock.update(SyncExchange(1000, 10000, 12000, 1000))
    assert clock.last_rejection == "negative_round_trip"
    for tick in range(0, 5_000_000, 250000):
        item = exchange(0, 1_000_000, tick)
        clock.update(item)
    assert len(clock.snapshot(item.receiver_receive_us).exchanges) == 4
    assert not clock.update(exchange(0, 1_000_000, 0))
    assert clock.last_rejection == "out_of_order_sync_completion"
    with pytest.raises(ValueError, match="integer microseconds"):
        SyncExchange(True, 1, 2, 3)


def receiver():
    return TimestampedReceiver("source", "receiver", capacity=8,
                               timing_config=TimingConfig(source_error_us=0, query_error_us=0))


def packet(index, value=None):
    return TimestampedPacket("source", index, 1_000_000 + index * 25000,
                             float(index if value is None else value), 990000 + index * 25000)


def test_live_receiver_accepts_different_clock_origins_and_model_context_uses_local_receipts():
    buffer = receiver()
    for index in range(10):
        buffer.ingest(packet(index), 275000 + index * 25000)
    mapping = ClockMapping.known("source", "receiver", 1_000_000, 100000)
    view = buffer.snapshot(500000, mapping, context_length=8, minimum_observations=1)
    assert view.ready and view.horizon_steps == 7
    assert view.source_time_us[-1] > view.query_time_us
    context = TiRexModel.prepare_context(type("Boundary", (), {"context_length": 8})(), view)
    np.testing.assert_array_equal(context, np.arange(2, 10, dtype=np.float32))
    assert view.features["native_observation_age_s"] == pytest.approx(.185)


def test_late_packets_fill_holes_without_reducing_age_and_idle_age_increases():
    buffer = receiver()
    mapping = ClockMapping.known("source", "receiver", 1_000_000, 0)
    buffer.ingest(packet(0), 0)
    buffer.ingest(packet(2), 60000)
    view = buffer.snapshot(75000, mapping, context_length=3, minimum_observations=1)
    assert view.timing.age_us == 25000
    assert view.received_mask.tolist() == [True, False, True]
    buffer.ingest(packet(1), 100000)
    later = buffer.snapshot(100000, mapping, context_length=3, minimum_observations=1)
    assert later.timing.age_us == 50000 and later.received_mask.all()
    assert later.features["observed_reordered_fraction"] == pytest.approx(1 / 3)
    assert np.isnan(view.values_mps[1]) and not view.values_mps.flags.writeable
    assert buffer.snapshot(200000, mapping, context_length=3, minimum_observations=1).timing.age_us == 150000


def test_stream_buffer_handles_duplicates_old_packets_and_session_mismatch():
    buffer = receiver()
    for index in range(100):
        buffer.ingest(packet(index), index * 25000)
    assert buffer.buffered_packets == 8
    assert not buffer.ingest(packet(99, 999), 2_500_000)
    assert not buffer.ingest(packet(0), 2_500_001)
    assert buffer.duplicates == buffer.too_old == 1
    with pytest.raises(ValueError, match="source clock changed"):
        buffer.ingest(replace(packet(100), source_clock_id="reboot"), 2_500_002)
    view = buffer.snapshot(2_500_002, None, context_length=8, minimum_observations=1)
    assert not view.ready and view.reason == "clock_unavailable"
    assert view.values_mps[-1] == 99 and view.horizon_steps is None


def test_non_grid_median_selection_interpolates_observed_anchor_without_extrapolation():
    trajectory = ForecastTrajectory(1_000_000, 25000, np.array([.1, .5, .9]),
                                     np.array([[0., 0.], [20., 30.], [100., 100.]]))
    assert trajectory.value_at_offset(0, 10) == 10
    assert trajectory.value_at_offset(12500, 10) == 15
    assert trajectory.value_at_offset(37500, 10) == 25
    assert trajectory.value_at_offset(50000, 10) == 30
    for offset in (-1, 50001, np.nan):
        with pytest.raises(ValueError, match="within this trajectory"):
            trajectory.value_at_offset(offset, 10)
    with pytest.raises(ValueError):
        trajectory.value_at(1_000_001)


def test_calibrated_clock_uses_its_known_rate_for_age_error_and_horizon_bounds():
    mapping = ClockMapping.known("source", "receiver", 0, 0, rate=3.0)
    timing = observation_timing(100000, 90000, 375000, 25000, mapping, source_error_us=1000)
    assert timing.valid and timing.source_offset_us == 25000
    assert timing.age_lower_us == 72000 and timing.age_upper_us == 78000
    assert timing.horizon_steps == timing.lower_horizon_steps == 1
    assert timing.upper_horizon_steps == 2


def test_dynamic_forecast_requests_uncertainty_coverage_and_gates_invalid_timing():
    buffer = receiver()
    for index in range(8):
        buffer.ingest(packet(index), index * 25000)
    mapping = ClockMapping.known("source", "receiver", 1_000_000, 0)
    view = buffer.snapshot(350000, mapping, context_length=8, minimum_observations=1)
    measured = replace(view.timing, age_upper_us=200000, upper_horizon_steps=8)
    calls = []
    class Boundary:
        def forecast(self, passed_view, length):
            calls.append((passed_view, length))
            return "trajectory"
    boundary = Boundary()
    view = replace(view, timing=measured)
    assert TiRexModel.forecast_observation(boundary, view) == "trajectory"
    assert calls[-1][1] == 8
    TiRexModel.forecast_observation(boundary, view, conservative=False)
    assert calls[-1][1] == 7
    with pytest.raises(ValueError, match="clock_stale"):
        TiRexModel.forecast_observation(boundary, replace(view, timing=replace(measured, valid=False, reason="clock_stale")))
    assert len(calls) == 2
    # Oversized requests fail before touching GPU synchronization or inference.
    frozen = type("CheckpointBoundary", (), {"model": type("Checkpoint", (), {"future_len": 320})(), "batch_size": 1})()
    with pytest.raises(ValueError, match="320 steps"):
        TiRexModel.forecast_batch(frozen, [view], 321)
