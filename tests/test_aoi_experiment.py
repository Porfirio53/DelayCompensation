from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mytirex.evaluation.aoi import evaluate_chunk, eval_aoi
from mytirex.reporting.aoi_metrics import AoiAccumulator
from mytirex.evaluation.simulation import ClockScenario, timing_cases
from mytirex.common import read_json, source_hash
from mytirex.inference.model import ForecastTrajectory, TiRexModel
from mytirex.data.prepared import PreparedSeries
from mytirex.timing.receiver import TimingConfig
from mytirex.timing.clock import ClockConfig

ROOT = Path(__file__).resolve().parents[1]
FIXED = {"name": "fixed_175", "kind": "fixed", "delay_us": 175000}
JITTER = {"name": "jitter", "kind": "correlated", "base_delay_us": 175000, "rho": .9,
    "innovation_std_us": 35000, "min_delay_us": 0, "max_delay_us": 500000,
    "drop_probability": .02, "tail_probability": .01, "tail_min_us": 500000, "tail_max_us": 1500000}


def series(length=1400):
    times = 1_000_000 + np.arange(length, dtype=np.int64) * 25000
    return PreparedSeries({"series_id": "synthetic", "split": "validation", "route": "car|2018-05-01--00-00-00",
        "group": "car|2018-05-01"}, times, np.arange(length) * .02, times - 10000)


def cases(data=None, profile=FIXED, scenario=None, max_queries=None, query_stride=4):
    return list(timing_cases(data or series(), profile, scenario or ClockScenario("control", offset_us=0,
        uplink_us=0, downlink_us=0, network_jitter_us=0), namespace="synthetic", seed=42, replicate=0,
        clock_config=ClockConfig(timestamp_error_us=0), timing_config=TimingConfig(source_error_us=0),
        max_queries=max_queries, query_stride=query_stride))


def test_fixed_shared_clock_control_matches_old_seven_step_protocol():
    result = cases(max_queries=10)
    assert all(c.view.horizon_steps == c.true_horizon_steps == 7 for c in result)
    assert all(c.view.timing.age_us == pytest.approx(175000) for c in result)
    assert all(c.true_native_aoi_us == pytest.approx(185000) for c in result)


def test_future_truth_and_extended_schedule_do_not_change_the_visible_prefix():
    original = series()
    altered_values = original.speed_mps.copy()
    altered_values[400:] = 1e9
    altered = replace(original, speed_mps=altered_values)
    a = cases(original, JITTER, ClockScenario("noisy", relative_rate_ppm=120, measurement_error_us=5000), max_queries=8)
    b = cases(altered, JITTER, ClockScenario("noisy", relative_rate_ppm=120, measurement_error_us=5000), max_queries=12)
    for left, right in zip(a, b):
        np.testing.assert_equal(left.view.values_mps, right.view.values_mps)
        np.testing.assert_equal(left.view.received_at_us, right.view.received_at_us)
        np.testing.assert_equal(left.view.received_mask, right.view.received_mask)
        assert left.view.timing == right.view.timing
        assert dict(left.view.features) == dict(right.view.features)


@pytest.mark.parametrize("scenario", [
    ClockScenario("drift", relative_rate_ppm=120, network_jitter_us=0),
    ClockScenario("noise", relative_rate_ppm=-120, measurement_error_us=5000),
    ClockScenario("asymmetric", uplink_us=1000, downlink_us=45000, network_jitter_us=1000),
    ClockScenario("sensor_bias", relative_rate_ppm=120, source_stamp_bias_us=10000),
])
def test_declared_clock_timestamp_bounds_cover_true_grid_and_native_ages(scenario):
    result = cases(profile=JITTER, scenario=scenario, max_queries=128)
    for item in result:
        timing = item.view.timing
        assert timing.age_lower_us <= item.true_aoi_us <= timing.age_upper_us
        assert timing.native_age_lower_us <= item.true_native_aoi_us <= timing.native_age_upper_us
        assert timing.upper_horizon_steps >= item.true_horizon_steps


def test_sync_outage_stays_in_the_query_population_and_recovers_causally():
    result = cases(scenario=ClockScenario("outage", relative_rate_ppm=120,
        outage_start_s=20, outage_duration_s=10))
    assert any(c.view.reason == "clock_stale" for c in result)
    assert not next(c for c in result if c.query_index >= 24 * 40).view.ready
    assert next(c for c in result if c.query_index >= 32 * 40).view.ready
    assert len(result) == len(range(280, 1400, 4))


def test_misspecified_measurement_bound_can_reject_clock_instead_of_promising_coverage():
    matched = ClockScenario("matched", measurement_error_us=5000)
    assert matched.error_bounds(ClockConfig(), TimingConfig())[0] == 5001
    scenario = ClockScenario("underestimated", measurement_error_us=5000, timestamp_error_bound_us=0,
                             uplink_us=1000, downlink_us=1000, network_jitter_us=0)
    result = cases(scenario=scenario, max_queries=64)
    assert scenario.error_bounds(ClockConfig(), TimingConfig())[0] == 0
    assert any(case.view.reason == "clock_inconsistent" for case in result)
    assert all(case.view.horizon_steps is None for case in result if case.view.reason == "clock_inconsistent")


def test_clock_scenarios_preserve_received_values_and_model_random_seed():
    control = cases(profile=JITTER, max_queries=8)
    shifted = cases(profile=JITTER, scenario=ClockScenario("offset_and_drift", offset_us=500000,
        relative_rate_ppm=120), max_queries=8)
    for left, right in zip(control, shifted, strict=True):
        np.testing.assert_equal(left.view.values_mps, right.view.values_mps)
        np.testing.assert_equal(left.view.received_mask, right.view.received_mask)
    seeds = []
    boundary = SimpleNamespace(runtime={"seed": 42}, torch=SimpleNamespace(manual_seed=seeds.append))
    TiRexModel.reset_for_trace(boundary, "series", "profile", "control", 0)
    TiRexModel.reset_for_trace(boundary, "series", "profile", "shifted", 0)
    TiRexModel.reset_for_trace(boundary, "series", "profile", "shifted", 1)
    assert seeds[0] == seeds[1] and seeds[1] != seeds[2]


class RecordingModel:
    batch_size = 4
    context_length = 256

    def __init__(self, all_failed=False):
        self.called = False
        self.requested_lengths = []
        self.all_failed = all_failed
        self.source_directory = ROOT / "vendor/tirex"
        self.source_sha256 = source_hash(self.source_directory)

    def load(self):
        pass

    def reset_for_trace(self, *args):
        pass

    def metadata(self):
        return {"source": "test-double", "gpu_inference_validated": False}

    def forecast_batch(self, views, length):
        self.called = True
        self.requested_lengths.append(length)
        assert 1 <= length <= 320
        results = []
        for view in views:
            assert not hasattr(view, "target_mps") and not hasattr(view, "true_aoi_us")
            assert view.values_mps.base is None and not view.values_mps.flags.writeable
            assert (view.received_at_us[view.received_mask] <= view.query_time_us).all()
            TiRexModel.prepare_context(self, view)
            values = np.tile(view.values_mps[-1] + np.arange(1, length + 1) * .02, (9, 1))
            results.append(ForecastTrajectory(int(view.source_time_us[-1]), 25000, np.arange(1, 10) / 10,
                values, "failed" if self.all_failed else "ok", "test failure" if self.all_failed else "", self.all_failed))
        return results


def test_shared_trajectory_interpolates_clock_bias_and_reads_truth_after_forecast():
    model = RecordingModel()
    data = series()
    selected = cases(data, scenario=ClockScenario("sensor_bias", relative_rate_ppm=120,
                                                source_stamp_bias_us=10000, network_jitter_us=0), max_queries=4)

    def target(index):
        assert model.called
        return data.speed_mps[index]

    frame = evaluate_chunk(selected, model=model, target_for=target, maximum_steps=320, batch_id="test")
    oracle = frame.loc[frame.policy == "oracle"]
    np.testing.assert_allclose(oracle.prediction, oracle.reference, atol=1e-12)
    synced = frame.loc[frame.policy == "synchronized"]
    assert np.mean(np.abs(synced.prediction - synced.reference)) > .007
    assert len(frame) == 6 * len(selected) and not frame.fallback.any()


def test_generation_length_is_independent_of_oracle_age_and_time_selection():
    selected = cases(max_queries=2)
    perturbed = [replace(case, true_aoi_us=100.0, true_source_offset_us=100.0, true_horizon_steps=1) for case in selected]
    model = RecordingModel()
    original = evaluate_chunk(selected, model=model, target_for=lambda i: i * .02, maximum_steps=320, batch_id="original")
    altered = evaluate_chunk(perturbed, model=model, target_for=lambda i: i * .02, maximum_steps=320, batch_id="altered")
    assert model.requested_lengths == [320, 320]
    np.testing.assert_array_equal(original[original.policy != "oracle"].prediction, altered[altered.policy != "oracle"].prediction)


def test_no_received_observation_stays_in_metrics_as_missing_prediction():
    selected = cases(profile={"name": "unavailable", "kind": "fixed", "delay_us": 20_000_000}, max_queries=1)
    model = RecordingModel()
    frame = evaluate_chunk(selected, model=model, target_for=lambda i: i * .02, maximum_steps=320, batch_id="empty")
    assert not model.called and len(frame) == 6
    assert frame.prediction.isna().all() and frame.fallback.all()
    for key, value in {"source": "current", "arrival_profile": "unavailable", "clock_scenario": "control"}.items():
        frame[key] = value
    accumulator = AoiAccumulator()
    accumulator.update(frame)
    metrics, reasons = accumulator.tables()
    assert metrics.query_count.eq(1).all() and metrics.missing_prediction_count.eq(1).all()
    assert metrics.speed_mps_count.eq(0).all() and set(reasons.reason) == {"no_received_observation"}


def test_horizon_limit_and_sync_outage_use_finite_received_hold_values():
    selected = cases(scenario=ClockScenario("huge_offset", offset_us=10_000_000), max_queries=4)
    frame = evaluate_chunk(selected, model=RecordingModel(), target_for=lambda i: i * .02,
                           maximum_steps=320, batch_id="overflow")
    raw = frame[frame.policy == "raw_timestamps"]
    assert raw.horizon_exceeded.all() and raw.fallback.all()
    assert set(raw.failure_reason) == {"horizon_limit"}
    assert np.isfinite(raw.prediction).all()
    outage_cases = cases(scenario=ClockScenario("outage", outage_start_s=20, outage_duration_s=10))
    selected = [c for c in outage_cases if c.view.reason == "clock_stale"][:4]
    frame = evaluate_chunk(selected, model=RecordingModel(), target_for=lambda i: i * .02,
                           maximum_steps=320, batch_id="outage")
    stale = frame[frame.policy == "synchronized"]
    assert stale.fallback.all() and set(stale.failure_reason) == {"clock_stale"}
    assert not frame.loc[frame.policy == "oracle", "fallback"].any()


def arguments(**overrides):
    values = dict(aoi_config="configs/aoi.json", config="configs/experiment.json", source="current",
        tirex_config=None, corpus=None, split="validation", series_id=None, max_series=1, max_queries=4,
        profile=["jitter_loss_tail"], scenario=["clock_drift"], query_stride=None, replicates=None,
        with_model=False, batch_size=4)
    return SimpleNamespace(**(values | overrides))


def isolate_output(monkeypatch, tmp_path):
    output = tmp_path / "aoi"
    published = []
    def create(*args):
        output.mkdir()
        return output
    monkeypatch.setattr("mytirex.evaluation.aoi.new_output", create)
    monkeypatch.setattr("mytirex.evaluation.aoi.publish_success", lambda *args: published.append(args))
    return output, published


def test_cpu_cli_pipeline_writes_exact_metrics_artifact_and_report_without_gpu(monkeypatch, tmp_path):
    output, published = isolate_output(monkeypatch, tmp_path)
    eval_aoi(ROOT, arguments())
    metadata = read_json(output / "metadata.json")
    assert metadata["case_count"] == 4 and metadata["row_count"] == 24 and not metadata["model_inference"]
    assert metadata["dataset_id"] == read_json(ROOT / "configs/data_index.json")["prepared_comma2k19_v1"]["dataset_id"]
    frame = pd.read_csv(output / "aoi_queries.csv")
    summary = pd.read_csv(output / "metrics_summary.csv")
    assert len(frame) == 24 and len(summary) == 6 and summary.speed_mps_count.sum() == 0
    synced = frame[frame.policy == "synchronized"]
    measured = summary[summary.policy == "synchronized"].iloc[0]
    assert measured.aoi_ms_MAE == pytest.approx(np.abs(synced.aoi_us - synced.true_aoi_us).mean() / 1000)
    assert measured.interval_coverage == 1
    assert (output / "report.html").is_file() and (output / "plots/aoi_trace.png").is_file()
    assert len(published) == 1


def test_model_pipeline_failure_records_error_without_publishing_success(monkeypatch, tmp_path):
    output, published = isolate_output(monkeypatch, tmp_path)
    monkeypatch.setattr("mytirex.evaluation.aoi.TiRexModel", lambda *args: RecordingModel(all_failed=True))
    with pytest.raises(RuntimeError, match="entire initial AoI batch"):
        eval_aoi(ROOT, arguments(with_model=True))
    assert read_json(output / "metadata.json")["status"] == "failed" and not published
