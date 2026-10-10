"""Retained causal preprocessing/receiver regression checks, without training dependencies."""
from dataclasses import fields
from pathlib import Path
import json
import numpy as np
import pytest
from mytirex.arrival import NEVER, PacketReceiver, ReceiverView, causal_filled_context, generate_arrivals, trace_seed
from mytirex.prepared import causal_blocks, discover_segments, read_speed, route_group, split_groups
ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = json.loads((ROOT / "data/prepared/comma2k19/v1/dataset_manifest.json").read_text())["identity"]["config"]

def write_array(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, np.asarray(values))


def raw_segment(raw, route, segment, times, values, extension=""):
    directory = raw / "Chunk_3" / route / str(segment) / "processed_log/CAN/speed"
    write_array(directory / ("t" + extension), times)
    write_array(directory / ("value" + extension), values)


def same_view(a, b):
    for name in ("source_time_us", "values_mps", "received_mask", "received_at_us", "native_source_time_us"):
        np.testing.assert_equal(getattr(a, name), getattr(b, name))
    assert dict(a.features) == dict(b.features)
    assert (a.query_time_us, a.horizon_steps, a.ready, a.reason) == (b.query_time_us, b.horizon_steps, b.ready, b.reason)


def test_split_is_day_grouped_and_legacy_quarantine_covers_other_routes_on_that_day():
    legacy = "car|2018-07-02--19-08-27"
    same_day = "car|2018-07-02--23-00-00"
    routes = [f"car|2018-07-{day:02d}--08-00-00" for day in range(1, 18)] + [legacy, same_day]
    excluded = {route_group(legacy)}
    assignment = split_groups({route_group(r) for r in routes}, excluded, DEFAULTS["split"]["ratios"], 42)
    assert assignment[route_group(same_day)] == "legacy_development"
    assert set(assignment.values()) == {"train", "validation", "test", "legacy_development"}
    assert assignment == split_groups({route_group(r) for r in reversed(routes)}, excluded,
                                      DEFAULTS["split"]["ratios"], 42)


def test_extending_corpus_keeps_frozen_split_assignments():
    groups = [f"car|2018-07-{day:02d}" for day in range(1, 11)]
    frozen = split_groups(groups, set(), DEFAULTS["split"]["ratios"], 42)
    extended = split_groups(groups + ["car|2018-08-01", "car|2018-08-02"], set(),
                            DEFAULTS["split"]["ratios"], 42, frozen)
    assert all(extended[group] == split for group, split in frozen.items())
    with pytest.raises(ValueError, match="at least 3"):
        split_groups(groups[:2], set(), DEFAULTS["split"]["ratios"], 42)


@pytest.mark.parametrize("extension", ["", ".npy"])
def test_official_native_format_dedup_causal_resampling_and_gap_cut(tmp_path, extension):
    raw_segment(tmp_path, "car|2018-07-01--08-00-00", 0,
                [0, .01, .04, .01, .3, .325, np.nan], [10, 20, 40, 21, 300, 325, 999], extension)
    segment, = discover_segments(tmp_path)
    times, speed, quality = read_speed(segment)
    blocks = list(causal_blocks(times, speed, 25000, 100000))
    assert quality["removed_invalid_records"] == quality["duplicate_timestamp_records"] == 1
    np.testing.assert_array_equal(blocks[0][0], [0, 25000])
    np.testing.assert_array_equal(blocks[0][1], [10, 21])  # Never borrow the value 40 at 40ms.
    np.testing.assert_array_equal(blocks[0][2], [0, 10000])
    np.testing.assert_array_equal(blocks[1][0], [300000, 325000])


def test_fixed_delay_boundary_matches_existing_protocol():
    times = np.arange(30, dtype=np.int64) * 25000
    arrivals = generate_arrivals(times, {"name": "fixed", "kind": "fixed", "delay_us": 175000}, 1)
    receiver = PacketReceiver(times, np.arange(len(times)), arrivals)
    assert receiver.snapshot(174999, 8, 1).reason == "no_received_observation"
    view = receiver.snapshot(175000, 8, 1)
    np.testing.assert_array_equal(view.source_time_us, [0])
    view = receiver.snapshot(500000, 8, 1)
    assert view.source_time_us[-1] == 325000 and view.horizon_steps == 7
    assert view.received_mask.all()
    assert view.features["received_latency_mean_s"] == pytest.approx(.175)


def test_reordering_keeps_internal_hole_and_forward_fill_uses_earlier_source_only():
    times = np.arange(4, dtype=np.int64) * 25000
    receiver = PacketReceiver(times, [1, 2, 3, 4], [0, 100000, 60000, 75000])
    view = receiver.snapshot(75000, 4, 1)
    np.testing.assert_array_equal(view.received_mask, [True, False, True, True])
    np.testing.assert_array_equal(causal_filled_context(view), [1, 1, 3, 4])
    assert np.isnan(view.values_mps[1]) and view.received_at_us[1] == -1
    assert view.features["window_unreceived_fraction"] == .25
    later = receiver.snapshot(100000, 4, 1)
    assert later.received_mask.all()
    assert later.features["observed_reordered_fraction"] == .25
    assert np.isnan(view.values_mps[1])  # A past immutable snapshot does not change.
    with pytest.raises(ValueError, match="nondecreasing"):
        receiver.snapshot(99999, 4, 1)


def test_not_yet_received_and_ultimately_lost_are_indistinguishable_until_arrival():
    times = np.arange(5, dtype=np.int64) * 25000
    delayed = PacketReceiver(times, [1, 2, 3, 4, 5], [0, 100000, 60000, 75000, 200000])
    lost = PacketReceiver(times, [1, 999, 3, 4, 99999], [0, NEVER, 60000, 75000, NEVER])
    same_view(delayed.snapshot(75000, 4, 1), lost.snapshot(75000, 4, 1))
    assert delayed.snapshot(100000, 4, 1).received_mask[1]
    assert not lost.snapshot(100000, 4, 1).received_mask[1]


def test_future_values_and_future_arrival_events_cannot_change_earlier_inputs():
    times = np.arange(6, dtype=np.int64) * 25000
    values = np.arange(6, dtype=float)
    schedule = times + 175000
    future_values = values.copy()
    future_values[4:] = 1e9
    future_schedule = schedule.copy()
    future_schedule[4:] += 999999
    a = PacketReceiver(times, values, schedule).snapshot(250000, 4, 1)
    b = PacketReceiver(times, future_values, future_schedule).snapshot(250000, 4, 1)
    same_view(a, b)
    assert set(f.name for f in fields(ReceiverView)).isdisjoint({"target_mps", "drop_mask", "schedule", "true_delay_us"})
    assert not any("drop" in name or "future" in name for name in a.features)
    with pytest.raises(ValueError):
        a.values_mps[0] = 20
    with pytest.raises(TypeError):
        a.features["observation_age_s"] = 20


def test_leading_unreceived_positions_remain_nan_and_no_truth_fallback():
    times = np.arange(3, dtype=np.int64) * 25000
    view = PacketReceiver(times, [999, 999, 3], [NEVER, NEVER, 50000]).snapshot(50000, 3, 1)
    np.testing.assert_equal(causal_filled_context(view), [np.nan, np.nan, 3])
    empty = PacketReceiver(times, [1, 2, 3], [NEVER, NEVER, NEVER]).snapshot(100000, 3, 1)
    assert not empty.ready and not len(empty.values_mps) and empty.horizon_steps is None


def test_random_traces_are_prefix_consistent_and_split_seeds_are_separate():
    profile = DEFAULTS["arrival"]["profiles"][-1]
    times = np.arange(500, dtype=np.int64) * 25000
    seed = trace_seed("corpus", "train", "series", profile["name"], 0, 42)
    arrivals = generate_arrivals(times, profile, seed)
    np.testing.assert_array_equal(generate_arrivals(times[:50], profile, seed), arrivals[:50])
    assert seed != trace_seed("corpus", "test", "series", profile["name"], 0, 42)
    finite = arrivals != NEVER
    assert (arrivals[finite] >= times[finite]).all()
    assert np.any(np.diff(arrivals[finite]) < 0)  # An actual reordered stream, not a prefix cut.

