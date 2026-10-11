from pathlib import Path
from types import SimpleNamespace
import json
import socket
from threading import Event, Thread

import pandas as pd
import pytest

from mytirex.common import read_json
from mytirex.cli import main
from mytirex.receiver import TimingConfig
from mytirex.timing import ClockConfig
from mytirex.transport import CaptureSession, decode, encode, replay_capture, sync_capture, sync_source

ROOT = Path(__file__).resolve().parents[1]


def packet_event(clock_id, sequence, received, origin=1000000):
    return {"kind": "packet", "event_time_us": received, "received_at_us": received,
        "source_clock_id": clock_id, "sequence": sequence, "source_time_us": origin + sequence * 25000,
        "native_source_time_us": origin - 10000 + sequence * 25000, "value_mps": float(sequence)}


def test_source_restart_invalidates_sync_and_retired_packets_cannot_flip_session_back():
    session = CaptureSession("receiver", ClockConfig(), TimingConfig(), context_length=2, minimum_observations=1)
    session.process(packet_event("source-0", 0, 0))
    session.process(packet_event("source-0", 1, 25000))
    session.process(packet_event("source-1", 0, 50000, origin=2000000))
    session.process(packet_event("source-0", 2, 75000))
    row = session.process({"kind": "query", "event_time_us": 100000})
    assert row["source_clock_id"] == "source-1" and row["received_count"] == 1
    assert row["clock_status"] == "unavailable" and not row["ready"]
    assert session.counts["source_session_changes"] == session.counts["retired_session_events"] == 1


def test_trace_order_and_local_receipt_prevent_future_information():
    session = CaptureSession("receiver", ClockConfig(), TimingConfig(), context_length=2, minimum_observations=1)
    session.process({"kind": "query", "event_time_us": 100000})
    with pytest.raises(ValueError, match="future receiver information"):
        session.process(packet_event("source", 0, 99999))
    invalid = packet_event("source", 0, 200000)
    invalid["received_at_us"] = 200001
    with pytest.raises(ValueError, match="before its local receipt"):
        session.process(invalid)
    with pytest.raises(ValueError):
        decode(b'{"version":2,"kind":"sync_response"}')
    assert decode(encode({"kind": "query"}))["kind"] == "query"


def test_malformed_events_do_not_change_session_or_advance_visible_time():
    session = CaptureSession("receiver", ClockConfig(), TimingConfig(), context_length=2, minimum_observations=1)
    session.process(packet_event("source-0", 0, 0))
    original_receiver, original_clock = session.receiver, session.clock
    invalid = packet_event("source-1", 1000, 20000, origin=-10_000_000)
    with pytest.raises(ValueError, match="negative source-grid origin"):
        session.process(invalid)
    assert session.source_clock_id == "source-0"
    assert session.receiver is original_receiver and session.clock is original_clock
    with pytest.raises(ValueError, match="Unsupported captured event"):
        session.process({"kind": "unexpected", "event_time_us": 30000})
    session.process(packet_event("source-0", 1, 15000))
    row = session.process({"kind": "query", "event_time_us": 17500})
    assert row["received_count"] == 2 and row["last_observation_mps"] == 1
    assert session.counts["source_session_changes"] == 0


@pytest.mark.parametrize("version", [True, 1.0, 2, None])
def test_wire_version_is_an_integer_protocol_identity(version):
    with pytest.raises(ValueError, match="version"):
        decode(json.dumps({"version": version, "kind": "packet"}).encode())


def test_cli_capture_and_replay_have_independent_configuration_defaults(monkeypatch):
    captured = []
    monkeypatch.setattr("mytirex.transport.sync_capture", lambda root, args: captured.append(args.aoi_config))
    monkeypatch.setattr("mytirex.transport.replay_capture", lambda root, args: captured.append(args.aoi_config))
    assert main(["sync-capture", "--host", "127.0.0.1"]) == 0
    assert main(["replay-aoi", "--trace", "example.jsonl"]) == 0
    assert captured == ["configs/aoi.json", None]


def test_local_udp_four_timestamps_capture_and_replay_are_identical(monkeypatch, tmp_path):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    ready = Event()
    original_socket = socket.socket
    class ReadySocket(original_socket):
        def bind(self, address):
            super().bind(address)
            if address == ("127.0.0.1", port):
                ready.set()
    monkeypatch.setattr("mytirex.transport.socket.socket", ReadySocket)
    outputs = iter([tmp_path / "capture", tmp_path / "replay", tmp_path / "override"])
    def create(*args):
        output = next(outputs)
        output.mkdir()
        return output
    monkeypatch.setattr("mytirex.transport.new_output", create)
    monkeypatch.setattr("mytirex.transport.publish_success", lambda *args: None)
    configuration = read_json(ROOT / "configs/aoi.json")
    configuration["clock"]["timestamp_error_us"] = 17
    configuration["timing"]["source_error_us"] = 9
    capture_config = tmp_path / "capture_config.json"
    capture_config.write_text(json.dumps(configuration))
    errors = []
    def serve():
        try:
            sync_source(ROOT, SimpleNamespace(corpus=None, split="validation", series_id=None,
                bind="127.0.0.1", port=port, duration_s=2))
        except Exception as exc:
            errors.append(exc)
            ready.set()
    worker = Thread(target=serve)
    worker.start()
    try:
        assert ready.wait(3) and not errors
        capture = sync_capture(ROOT, SimpleNamespace(aoi_config=str(capture_config), host="127.0.0.1",
            port=port, duration_s=1.4, sync_interval_ms=100, query_period_ms=25,
            context_length=8, minimum_observations=2))
        trace = capture / "events.jsonl"
        events = [json.loads(line) for line in trace.read_text().splitlines()]
        assert any(e["kind"] == "sync" for e in events) and any(e["kind"] == "packet" for e in events)
        assert all(e["send_time_us"] >= e["source_time_us"] for e in events if e["kind"] == "packet")
        assert read_json(capture / "metadata.json")["summary"]["ready_queries"] > 0
        replay = replay_capture(ROOT, SimpleNamespace(aoi_config=None, trace=str(trace)))
        pd.testing.assert_frame_equal(pd.read_csv(capture / "aoi_queries.csv"), pd.read_csv(replay / "aoi_queries.csv"))
        assert read_json(replay / "metadata.json")["summary"] == read_json(capture / "metadata.json")["summary"]
        assert read_json(replay / "config.json")["configuration_source"] == "trace_header"
        override = replay_capture(ROOT, SimpleNamespace(aoi_config="configs/aoi.json", trace=str(trace)))
        assert read_json(override / "metadata.json")["configuration_source"] == "override"
        original_bounds = pd.read_csv(replay / "aoi_queries.csv").aoi_upper_us.dropna().to_numpy()
        override_bounds = pd.read_csv(override / "aoi_queries.csv").aoi_upper_us.dropna().to_numpy()
        assert (override_bounds > original_bounds).all()
    finally:
        worker.join(3)
    assert not worker.is_alive() and not errors
