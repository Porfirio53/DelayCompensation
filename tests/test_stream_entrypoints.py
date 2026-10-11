"""Short checks of live/replay adapters and publication-time scoring; no GPU."""
import json
from pathlib import Path
import socket
from threading import Event, Thread
from time import sleep
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mytirex.cli import main
from mytirex.evaluation.simulation import ClockScenario
from mytirex.common import read_json, source_hash
from mytirex.inference.model import ForecastTrajectory, TiRexModel
from mytirex.data.prepared import PreparedSeries
from mytirex.timing.receiver import TimingConfig
from mytirex.streaming.replay import RealtimeReplay
from mytirex.timing.clock import ClockConfig
from mytirex.timing.transport import CaptureSession, sync_source

ROOT = Path(__file__).resolve().parents[1]


def isolated_output(monkeypatch, tmp_path, module):
    output = tmp_path / "run"
    def create(*args):
        output.mkdir()
        return output
    monkeypatch.setattr(module + ".new_output", create)
    monkeypatch.setattr(module + ".publish_success", lambda *args: None)
    return output


def test_model_free_real_time_replay_writes_live_states_and_valid_measurements(monkeypatch, tmp_path):
    output = isolated_output(monkeypatch, tmp_path, "mytirex.streaming.replay")
    def forbidden(*args):
        pytest.fail("Model-free streaming must not load a GPU model")
    monkeypatch.setattr(TiRexModel, "load", forbidden)
    assert main(["eval-stream", "--duration-s", ".3", "--profile", "jitter_loss_tail",
                 "--scenario", "timestamp_noise"]) == 0
    metadata = read_json(output / "metadata.json")
    frame = pd.read_csv(output / "publications.csv")
    states = [json.loads(line) for line in (output / "states.jsonl").read_text().splitlines()]
    assert metadata["status"] == "success" and not metadata["model_inference"]
    assert len(states) == len(frame) >= 8
    assert all("reference_mps" not in row for row in states)
    assert frame.publish_time_us.is_monotonic_increasing
    assert (frame.publish_time_us >= frame.scheduled_time_us).all()
    assert np.allclose(frame.prediction_mps, frame.last_observation_mps)
    assert pd.read_csv(output / "inferences.csv").empty
    assert metadata["summary"]["publication_hz"] >= 25
    assert (output / "report.html").is_file()


def test_reference_uses_actual_publication_time_instead_of_scheduled_tick_or_interpolated_truth():
    times = 1_000_000 + np.arange(400, dtype=np.int64) * 25000
    series = PreparedSeries({"series_id": "synthetic", "split": "validation"}, times,
                            np.arange(400, dtype=float) * .3, times - 10000)
    config = read_json(ROOT / "configs/aoi_frozen.json")
    replay = RealtimeReplay(series, {"name": "fixed_175", "kind": "fixed", "delay_us": 175000},
        ClockScenario("drift", relative_rate_ppm=120), config, "synthetic", 0, 250000)
    host_start = 1_000_000_000
    assert replay.reference_at(host_start, host_start) == pytest.approx(280 * .3)
    # A 41 ms source advance falls after tick 281 and before 282.
    published = host_start + round(41000 * replay.clock.rate)
    assert replay.reference_at(published, host_start) == pytest.approx(281 * .3)


class SyntheticBoundary:
    """Model boundary double, expressly not a TiRex accuracy/GPU result."""
    batch_size = 1
    context_length = 256

    def __init__(self, *args):
        self.source_directory = ROOT / "vendor/tirex"
        self.source_sha256 = source_hash(self.source_directory)

    def load(self):
        pass

    def synchronize(self):
        pass

    def reset_for_trace(self, *args):
        pass

    def forecast(self, view, steps):
        TiRexModel.prepare_context(self, view)
        assert not hasattr(view, "reference_mps")
        sleep(.003)
        return ForecastTrajectory(int(view.source_time_us[-1]), view.sample_period_us,
            np.arange(1, 10) / 10, np.tile(float(view.values_mps[-1]) + np.arange(1, steps + 1) * .01, (9, 1)))

    def metadata(self):
        return {"source": "current", "synthetic_test_double": True, "gpu_inference_validated": False}


def test_paired_scoring_shares_publication_times_and_separates_steady_latency_from_warmup(monkeypatch, tmp_path):
    output = isolated_output(monkeypatch, tmp_path, "mytirex.streaming.replay")
    monkeypatch.setattr("mytirex.streaming.replay.TiRexModel", SyntheticBoundary)
    assert main(["eval-stream", "--with-model", "--duration-s", ".25", "--warmup", "2",
                 "--inference-period-ms", "25", "--inference-period-ms", "100"]) == 0
    frame = pd.read_csv(output / "publications.csv")
    requests = pd.read_csv(output / "inferences.csv")
    metrics = pd.read_csv(output / "metrics_summary.csv")
    metadata = read_json(output / "metadata.json")
    assert metadata["model"]["synthetic_test_double"]
    assert metadata["warmup"]["included_in_scoring"] is False
    assert set(metrics.policy) == {"received_hold", "delivery_hold", "trajectory"}
    assert set(requests.inference_period_ms) == {25, 100}
    for period, group in frame.groupby("inference_period_ms"):
        scored = metrics[metrics.inference_period_ms == period]
        assert (scored.speed_mps_count == len(group)).all()
        for policy, column in (("received_hold", "last_observation_mps"), ("delivery_hold", "delivery_hold_mps"),
                               ("trajectory", "prediction_mps")):
            expected = np.abs(group[column] - group.reference_mps).mean()
            assert scored[scored.policy == policy].speed_mps_MAE.iloc[0] == pytest.approx(expected)
    assert (requests.delivery_time_us >= requests.completed_at_us).all()
    assert (requests.started_at_us >= requests.snapshot_time_us).all()
    assert (requests.planned_steps <= 320).all()
    assert metadata["summary"]["pending_request_max"] == 0
    assert (frame[frame.inference_period_ms == 100].forecast_id.dropna().diff() == 0).any()


def test_live_udp_stream_publishes_json_to_stdout_and_logs_replayable_inputs(monkeypatch, tmp_path, capsys):
    output = isolated_output(monkeypatch, tmp_path, "mytirex.streaming.live")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    ready, errors = Event(), []
    original_socket = socket.socket
    class ReadySocket(original_socket):
        def bind(self, address):
            super().bind(address)
            if address == ("127.0.0.1", port):
                ready.set()
    monkeypatch.setattr("mytirex.timing.transport.socket.socket", ReadySocket)
    def serve():
        try:
            sync_source(ROOT, SimpleNamespace(corpus=None, split="validation", series_id=None,
                bind="127.0.0.1", port=port, duration_s=1.3))
        except Exception as exc:
            errors.append(exc)
            ready.set()
    worker = Thread(target=serve)
    worker.start()
    try:
        assert ready.wait(3) and not errors
        capsys.readouterr()
        assert main(["stream-receive", "--host", "127.0.0.1", "--port", str(port), "--duration-s", ".8",
                     "--sync-interval-ms", "50", "--context-length", "8", "--minimum-observations", "2", "--stdout"]) == 0
        captured = capsys.readouterr()
        stdout_states = [json.loads(line) for line in captured.out.splitlines() if line.startswith("{")]
        states = [json.loads(line) for line in (output / "states.jsonl").read_text().splitlines()]
        assert stdout_states == states
        assert "Streaming UDP" in captured.err
        metadata = read_json(output / "metadata.json")
        assert metadata["status"] == "success" and not metadata["model_inference"]
        assert metadata["events"]["packets"] >= 8 and metadata["events"]["sync_accepted"] >= 3
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        assert events[0]["kind"] == "header"
        assert len([event for event in events if event["kind"] == "query"]) == len(states)
        assert any(row["timing_valid"] and row["context_ready"] for row in states)
        header = events[0]
        replayed = CaptureSession(header["receiver_clock_id"], ClockConfig(**header["clock_config"]),
            TimingConfig(**header["timing_config"]), context_length=header["context_length"],
            minimum_observations=header["minimum_observations"])
        queries = []
        for event in events[1:]:
            row = replayed.process(event)
            if row is not None:
                queries.append(row)
        for query, state in zip(queries, states, strict=True):
            assert query["query_time_us"] == state["snapshot_time_us"]
            assert query.get("last_source_time_us") == state["last_source_time_us"]
            if state["aoi_us"] is not None:
                assert query["aoi_us"] + state["publish_time_us"] - state["snapshot_time_us"] == pytest.approx(state["aoi_us"])
    finally:
        worker.join(3)
    assert not worker.is_alive() and not errors
