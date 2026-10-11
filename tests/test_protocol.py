from dataclasses import fields
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import ast
import json
import sys

import numpy as np
import pandas as pd
import pytest

from mytirex.arrival import ReceiverView
from mytirex.common import read_json, source_hash, verify_migration
from mytirex.data import Sequence, load_dataset, query_indices
from mytirex.evaluation import load_config, replay
from mytirex.metrics import summarize
from mytirex.model import ForecastTrajectory, TiRexModel, read_tuning
from mytirex.prepared import PreparedCorpus
from mytirex.references import reference_identity, read_controls

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sequences():
    return load_dataset(ROOT)


def test_complete_migration_content_and_shared_model():
    result = verify_migration(ROOT)
    assert result["content_verified"] and result["files"] == 7462
    assert not result["source_needed_at_runtime"]
    index = read_json(ROOT / "configs/source_index.json")
    for entry in index.values():
        assert source_hash(ROOT / entry["directory"]) == entry["migrated_source_sha256"]
    assert not (ROOT / "data/raw/comma2k19/archives").exists()
    assert not list((ROOT / "data").rglob("*.zip"))
    assert not any(p.is_symlink() for p in (ROOT / "data").rglob("*"))


def test_original_five_sequence_query_population(sequences):
    assert [s.name for s in sequences] == [f"speed_seg{i:02d}" for i in range(1, 6)]
    assert all(len(s.values) == 7201 and len(query_indices(s)) == 7001 for s in sequences)
    assert all(query_indices(s)[0] == 200 and query_indices(s)[-1] == 7200 for s in sequences)
    assert sum(len(query_indices(s)) for s in sequences) * 2 == 70010
    assert len(query_indices(sequences[0], limit=32)) == 32


@pytest.mark.parametrize("delay_us,steps", [(175000, 7), (300000, 12)])
def test_inclusive_arrival_boundary_median_and_future_grid(sequences, delay_us, steps):
    view = sequences[0].request(200, delay_us)
    assert int(view.source_time_us[-1]) + delay_us == view.query_time_us == 5_000_000
    assert int(view.received_at_us[-1]) == view.query_time_us
    assert view.horizon_steps == steps
    assert len(view.values_mps) == 201 - steps
    assert np.all(view.source_time_us + delay_us <= view.query_time_us)
    # Distinct quantiles and steps expose either a wrong quantile or off-by-one target.
    values = np.arange(9)[:, None] * 100 + np.arange(1, steps + 1)[None, :]
    trajectory = ForecastTrajectory(int(view.source_time_us[-1]), 25000, np.arange(1, 10) / 10, values)
    assert trajectory.value_at(view.query_time_us) == 400 + steps
    assert trajectory.future_time_us[-1] == view.query_time_us
    with pytest.raises(ValueError, match="within this trajectory"):
        trajectory.value_at(view.query_time_us + 25000)
    with pytest.raises(ValueError, match="within this trajectory"):
        trajectory.value_at(view.query_time_us - 1)


def test_future_reference_and_numpy_base_are_isolated(sequences):
    original = sequences[0]
    view = original.request(200, 175000)
    changed = original.values.copy()
    changed[194:] = 1e9  # All invisible observations, including the query target.
    altered = Sequence(original.name, original.times_us, changed).request(200, 175000)
    np.testing.assert_array_equal(view.values_mps, altered.values_mps)
    assert view.values_mps.base is None and view.source_time_us.base is None
    assert not view.values_mps.flags.writeable
    assert not {"reference", "target_mps", "future_values", "arrival_schedule", "drop_mask"} & {f.name for f in fields(ReceiverView)}
    assert np.all(view.native_source_time_us == -1)  # Unknown in legacy interpolated CSV, not fabricated CAN times.


def test_context_cast_and_truncation_preserve_old_adapter_input(sequences):
    view = sequences[0].request(500, 300000)
    model_boundary = SimpleNamespace(context_length=256)
    context = TiRexModel.prepare_context(model_boundary, view)
    np.testing.assert_array_equal(context, sequences[0].values[233:489].astype(np.float32))
    assert context.dtype == np.float32 and context.flags.c_contiguous and len(context) == 256


def test_current_config_and_matching_history_are_fixed():
    config = load_config(ROOT)
    tuning = read_tuning(ROOT, "configs/tirex_tuning.json")
    assert tuning == {"source_directory": "TiRex", "context_length": 256, "pad_context": True,
                      "tta_diff": True, "trend_window_points": 32, "trend_threshold": 1.5, "trend_min_r2": 0.6}
    _, identity = reference_identity(ROOT, "current_speed", config, tuning, config["runtime"])
    assert identity["source_matches"] and identity["settings_match"] and identity["runtime_match"]
    assert identity["original_directory"] == "results/20261009_172124_750261_C"
    _, changed = reference_identity(ROOT, "current_speed", config, {**tuning, "trend_min_r2": 0.5}, config["runtime"])
    assert not changed["settings_match"]


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_prepared_views_remain_separate_and_keep_native_clock(split):
    corpus = PreparedCorpus(ROOT / "data/prepared/comma2k19/v1")
    sample = next(corpus.iter_examples(split, ["fixed_175"], max_examples=1))
    view = sample.view
    assert view.ready and view.horizon_steps == 7 and sample.split == split
    assert np.all(view.native_source_time_us[view.received_mask] <= view.source_time_us[view.received_mask])
    assert np.all(view.received_at_us[view.received_mask] <= view.query_time_us)
    assert "target_mps" not in {f.name for f in fields(view)}
    with pytest.raises(ValueError, match="different target definition"):
        load_dataset(ROOT, "prepared_comma2k19_v1")


class FakeReplayModel:
    batch_size = 2
    context_length = 256
    source = "current"

    def __init__(self, all_failed=False):
        self.all_failed = all_failed

    def reset_for_unit(self, *args):
        pass

    def forecast_batch(self, views, horizon):
        assert all(isinstance(v, ReceiverView) for v in views)
        output = []
        for index, view in enumerate(views):
            failed = self.all_failed or index == 1
            # The only predictor data here is the arrived history.
            values = np.full((9, horizon), view.values_mps[-1])
            output.append(ForecastTrajectory(int(view.source_time_us[-1]), 25000, np.arange(1, 10) / 10,
                                            values, "failed" if failed else "ok", "test" if failed else "", failed))
        return output


def test_finite_fallback_stays_in_scoring_and_full_initial_failure_is_fatal(sequences):
    protocol = load_config(ROOT)["protocol"]
    selection = {"delays_ms": [175], "max_queries": 2}
    frame, metadata = replay(FakeReplayModel(), sequences[:1], protocol, selection)
    metrics = summarize(frame).iloc[0]
    assert metrics.query_count == metrics.scored_count == 2
    assert metrics.failure_count == metrics.fallback_count == metadata["fallback_count"] == 1
    assert metrics.MAE == pytest.approx(np.abs(frame.prediction - frame.reference).mean())
    with pytest.raises(RuntimeError, match="entire initial batch"):
        replay(FakeReplayModel(all_failed=True), sequences[:1], protocol, selection)


def test_failure_trajectory_anchors_to_received_source_even_for_invalid_tail(sequences):
    view = sequences[0].request(200, 175000)
    mask, values = view.received_mask.copy(), view.values_mps.copy()
    mask[-1], values[-1] = False, np.nan
    invalid = replace(view, received_mask=mask, values_mps=values)
    boundary = SimpleNamespace(quantiles=np.arange(1, 10) / 10)
    result = TiRexModel._fallback(boundary, invalid, 7, "invalid history")
    assert result.source_time_us == view.source_time_us[-2]
    assert result.fallback and result.status == "failed"
    assert np.all(result.values_mps == view.values_mps[-2])


def test_references_validate_all_queries_and_select_subset_without_prior_run(sequences):
    config = load_config(ROOT)
    tuning = read_tuning(ROOT, "configs/tirex_tuning.json")
    selection = {"delays_ms": [175, 300], "max_queries": 2}
    frame, _ = replay(FakeReplayModel(), sequences[:1], config["protocol"], selection)
    controls, identities = read_controls(ROOT, frame, sequences, config, tuning, config["runtime"])
    assert len(controls) == 1 and all(len(frame) == len(control) == 4 for control in controls)
    assert set(identities) == {"current_speed"}
    assert all(identity["full_query_grid_validated"] for identity in identities.values())


def test_preloaded_foreign_package_requires_a_fresh_process(monkeypatch, tmp_path):
    config = load_config(ROOT)
    model = TiRexModel(ROOT, read_tuning(ROOT, "configs/tirex_tuning.json"), config["runtime"])
    wrong = tmp_path / "foreign/tirex2/__init__.py"
    monkeypatch.setitem(sys.modules, "tirex2", SimpleNamespace(__file__=str(wrong)))
    with pytest.raises(RuntimeError, match="fresh process"):
        model.load()


def test_no_legacy_integration_imports_in_project_code():
    forbidden = {"experiment", "adapter_training", "chronos", "multivariate_adapters", "ours_adapter"}
    for file in (ROOT / "src/mytirex").glob("*.py"):
        for node in ast.walk(ast.parse(file.read_text())):
            names = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                     [x.name for x in node.names] if isinstance(node, ast.Import) else [])
            assert not {name.split(".")[0] for name in names} & forbidden
