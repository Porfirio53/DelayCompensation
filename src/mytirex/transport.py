"""Real UDP four-timestamp measurement and deterministic replay of receiver logs.

The source command replays an existing causal CAN block on its monotonic clock;
the capture command measures actual network arrivals on another monotonic clock.
These commands do not run a GPU model or claim live vehicle ground truth.
"""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import csv
import json
import select
import socket
from time import monotonic_ns
from uuid import uuid4

from .common import (atomic_json, integration_identity, new_output, project_path, publish_success,
                     read_json, sha256)
from .prepared import PreparedCorpus
from .receiver import TimestampedPacket, TimestampedReceiver, TimingConfig
from .timing import ClockConfig, ClockSynchronizer, SyncExchange, clock_identities, integer_us

WIRE_VERSION = 1
MAX_DATAGRAM_BYTES = 8192


def monotonic_us() -> int:
    return monotonic_ns() // 1000


def encode(message: dict) -> bytes:
    return json.dumps({"version": WIRE_VERSION, **message}, separators=(",", ":"), allow_nan=False).encode()


def decode(data: bytes) -> dict:
    if len(data) > MAX_DATAGRAM_BYTES:
        raise ValueError("Datagram exceeds the timestamp protocol size limit")
    message = json.loads(data)
    if (not isinstance(message, dict) or type(message.get("version")) is not int
            or message["version"] != WIRE_VERSION):
        raise ValueError("Unsupported timestamp datagram version")
    return message


def prepared_path(root: Path, override: str | None = None) -> Path:
    entry = read_json(root / "configs/data_index.json")["prepared_comma2k19_v1"]
    return project_path(root, override or entry["directory"])


def measurement_configs(root: Path, filename: str) -> tuple[dict, ClockConfig, TimingConfig]:
    config = read_json(project_path(root, filename))
    if config["schema_version"] != 1:
        raise ValueError("Unsupported AoI configuration schema")
    return config, ClockConfig(**config["clock"]), TimingConfig(**config["timing"])


def query_row(view, source_clock_id: str) -> dict:
    timing = view.timing
    clock = timing.clock if timing else None
    return {
        "source_clock_id": source_clock_id, "query_time_us": view.query_time_us,
        "last_source_time_us": int(view.source_time_us[-1]) if len(view.source_time_us) else None,
        "last_native_time_us": int(view.native_source_time_us[-1]) if len(view.source_time_us) else None,
        "last_received_at_us": int(view.received_at_us[-1]) if len(view.source_time_us) else None,
        "last_observation_mps": float(view.values_mps[-1]) if len(view.values_mps) else None,
        "received_count": int(view.received_mask.sum()), "ready": view.ready, "reason": view.reason,
        "aoi_us": timing.age_us if timing else None,
        "aoi_lower_us": timing.age_lower_us if timing else None,
        "aoi_upper_us": timing.age_upper_us if timing else None,
        "native_aoi_us": timing.native_age_us if timing else None,
        "native_aoi_lower_us": timing.native_age_lower_us if timing else None,
        "native_aoi_upper_us": timing.native_age_upper_us if timing else None,
        "timing_valid": timing.valid if timing else False,
        "timing_reason": timing.reason if timing else "no_received_observation",
        "source_offset_us": timing.source_offset_us if timing else None,
        "horizon_steps": view.horizon_steps,
        "upper_horizon_steps": timing.upper_horizon_steps if timing else None,
        "clock_status": clock.status if clock else "unavailable",
        "clock_rate": clock.rate if clock else None,
        "clock_relative_drift_ppm": clock.relative_drift_ppm if clock else None,
        "clock_drift_fitted": clock.drift_fitted if clock else False,
        "clock_drift_clipped": clock.drift_clipped if clock else False,
        "clock_sync_age_us": view.query_time_us - clock.last_sync_receiver_us if clock else None,
        "clock_fit_samples": clock.fit_samples if clock else 0,
        "clock_minimum_rtt_us": clock.minimum_round_trip_us if clock else None,
        "observed_reordered_fraction": view.features.get("observed_reordered_fraction", 0.0),
        "window_unreceived_fraction": view.features.get("window_unreceived_fraction"),
    }


class CaptureSession:
    """Same causal state machine for live capture and recorded event replay."""

    def __init__(self, receiver_clock_id: str, clock_config: ClockConfig, timing_config: TimingConfig,
                 *, sample_period_us: int = 25000, context_length: int = 256, minimum_observations: int = 32):
        if not isinstance(receiver_clock_id, str) or not receiver_clock_id:
            raise ValueError("Capture needs a nonempty receiver clock identity")
        self.receiver_clock_id = receiver_clock_id
        self.clock_config, self.timing_config = clock_config, timing_config
        self.sample_period_us = integer_us(sample_period_us, "sample_period_us", 1)
        self.context_length = integer_us(context_length, "context_length", 1)
        self.minimum_observations = integer_us(minimum_observations, "minimum_observations", 1)
        if self.minimum_observations > self.context_length:
            raise ValueError("minimum_observations exceeds capture context length")
        self.source_clock_id = None
        self.clock = self.receiver = None
        self._retired = set()
        self.counts = Counter()
        self._last_event_us = -1

    def _session(self, source_clock_id: str) -> bool:
        clock_identities(source_clock_id, self.receiver_clock_id)
        if source_clock_id in self._retired:
            self.counts["retired_session_events"] += 1
            return False
        if source_clock_id != self.source_clock_id:
            if self.source_clock_id is not None:
                self._retired.add(self.source_clock_id)
                self.counts["source_session_changes"] += 1
            self.source_clock_id = source_clock_id
            self.clock = ClockSynchronizer(source_clock_id, self.receiver_clock_id, self.clock_config)
            self.receiver = TimestampedReceiver(source_clock_id, self.receiver_clock_id,
                sample_period_us=self.sample_period_us, capacity=max(2048, self.context_length),
                timing_config=self.timing_config)
        return True

    def process(self, event: dict):
        if not isinstance(event, dict):
            raise ValueError("A captured event must be a JSON object")
        kind = event["kind"]
        timestamp = integer_us(event["event_time_us"], "event_time_us")
        if timestamp < self._last_event_us:
            raise ValueError("Recorded event order would expose future receiver information")
        if kind == "query":
            if self.receiver is None:
                row = {"source_clock_id": "", "query_time_us": timestamp, "ready": False,
                       "reason": "no_received_observation", "clock_status": "unavailable",
                       "timing_valid": False, "timing_reason": "no_received_observation"}
            else:
                view = self.receiver.snapshot(timestamp, self.clock.snapshot(timestamp),
                    context_length=self.context_length, minimum_observations=self.minimum_observations)
                row = query_row(view, self.source_clock_id)
            self._last_event_us = timestamp
            self.counts["queries"] += 1
            return row
        if kind not in {"sync", "packet"}:
            raise ValueError(f"Unsupported captured event: {kind}")
        clock_identities(event["source_clock_id"], self.receiver_clock_id)
        if kind == "sync":
            exchange = SyncExchange(**{name: event[name] for name in SyncExchange.__dataclass_fields__})
            if timestamp < exchange.receiver_receive_us:
                raise ValueError("Synchronization exchange is processed before receipt")
        else:
            receipt = integer_us(event["received_at_us"], "received_at_us")
            if receipt > timestamp:
                raise ValueError("Packet is processed before its local receipt timestamp")
            packet = TimestampedPacket(**{name: event.get(name) for name in TimestampedPacket.__dataclass_fields__})
            if packet.source_time_us - packet.sequence * self.sample_period_us < 0:
                raise ValueError("Packet sequence/source timestamps imply a negative source-grid origin")
            if event.get("send_time_us") is not None:
                send_time = integer_us(event["send_time_us"], "send_time_us")
                if send_time < packet.source_time_us:
                    raise ValueError("Packet send timestamp precedes its source-grid tick")
        # Validate the entire payload before changing session or event order.
        # A malformed datagram must not erase an otherwise valid clock/history.
        if not self._session(event["source_clock_id"]):
            self._last_event_us = timestamp
            return None
        if kind == "sync":
            accepted = self.clock.update(exchange, available_at_us=timestamp)
            self.counts["sync_accepted" if accepted else "sync_rejected"] += 1
        else:
            self.receiver.ingest(packet, receipt)
            self.counts["packets"] += 1
        self._last_event_us = timestamp
        return None


def capture_summary(rows: list[dict]) -> dict:
    import numpy as np
    ages = np.asarray([r["aoi_us"] for r in rows if r.get("aoi_us") is not None], dtype=float)
    return {"queries": len(rows), "ready_queries": sum(bool(r.get("ready")) for r in rows),
        "timing_usable_queries": sum(bool(r.get("timing_valid")) for r in rows),
        "reason_counts": dict(Counter(r["reason"] for r in rows)),
        "aoi_p50_ms": float(np.percentile(ages, 50) / 1000) if len(ages) else None,
        "aoi_p95_ms": float(np.percentile(ages, 95) / 1000) if len(ages) else None,
        "aoi_max_ms": float(ages.max() / 1000) if len(ages) else None,
        "ground_truth_age_available": False}


def write_capture_output(output: Path, rows: list[dict], metadata: dict) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "aoi_queries.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(output / "metadata.json", metadata)
    from .report import write_json_report
    write_json_report(output, "真实时间戳与动态观测年龄", metadata)


def sync_source(root: Path, args) -> None:
    corpus = PreparedCorpus(prepared_path(root, args.corpus))
    series = next(corpus.iter_series(args.split, [args.series_id] if args.series_id else None, max_series=1))
    period = corpus.config["resampling"]["sample_period_us"]
    source_id = "source-" + uuid4().hex
    clients = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((args.bind, args.port))
        sock.setblocking(False)
        start = monotonic_us()
        anchor = start + 250000
        finish = start + round(args.duration_s * 1e6)
        sent_index = -1
        skipped = 0
        print(f"Timestamp source {source_id}: UDP {sock.getsockname()}, prepared series={series.metadata['series_id']}", flush=True)
        try:
            while monotonic_us() < finish:
                now = monotonic_us()
                next_tick = anchor + (sent_index + 1) * period
                timeout = min(0.05, max(0.0, (min(next_tick, finish) - now) / 1e6))
                if select.select([sock], [], [], timeout)[0]:
                    data, address = sock.recvfrom(MAX_DATAGRAM_BYTES + 1)
                    t2 = monotonic_us()
                    try:
                        request = decode(data)
                        if request["kind"] != "sync_request":
                            raise ValueError("Invalid synchronization request")
                        clock_identities(source_id, request["receiver_clock_id"])
                        integer_us(request["request_id"], "request_id")
                        integer_us(request["receiver_send_us"], "receiver_send_us")
                        clients[address] = t2
                        t3 = monotonic_us()
                        sock.sendto(encode({"kind": "sync_response", "source_clock_id": source_id,
                            "receiver_clock_id": request["receiver_clock_id"], "request_id": request["request_id"],
                            "receiver_send_us": request["receiver_send_us"], "source_receive_us": t2,
                            "source_send_us": t3, "sample_period_us": period,
                            "series_id": series.metadata["series_id"],
                            "dataset_id": corpus.manifest["dataset_id"]}), address)
                    except (KeyError, TypeError, ValueError, OSError):
                        pass
                now = monotonic_us()
                due = (now - anchor) // period
                if due > sent_index and 0 <= due < len(series.source_time_us):
                    skipped += max(0, due - sent_index - 1)
                    sent_index = int(due)
                    packet = TimestampedPacket(source_id, sent_index, anchor + sent_index * period,
                        float(series.speed_mps[sent_index]),
                        anchor + int(series.native_time_us[sent_index] - series.source_time_us[0]))
                    payload = encode({"kind": "packet", **asdict(packet), "sample_period_us": period,
                                      "send_time_us": monotonic_us(), "series_id": series.metadata["series_id"]})
                    for address, last_seen in tuple(clients.items()):
                        if now - last_seen > 5_000_000:
                            del clients[address]
                        else:
                            sock.sendto(payload, address)
                if due >= len(series.source_time_us):
                    # Keep serving sync without a busy loop after the signal ends.
                    sent_index = max(sent_index, int(due))
        except KeyboardInterrupt:
            pass
        print(f"Source stopped: last_sequence={sent_index}, scheduler_skipped_ticks={skipped}", flush=True)


def sync_capture(root: Path, args) -> Path:
    config, clock_config, timing_config = measurement_configs(root, args.aoi_config)
    context = args.context_length or config["context_length"]
    minimum = args.minimum_observations or config["minimum_observations"]
    receiver_id = "receiver-" + uuid4().hex
    session = CaptureSession(receiver_id, clock_config, timing_config, context_length=context,
                             minimum_observations=minimum)
    output = new_output(root, "aoi_capture")
    resolved = {"config": config, "host": args.host, "port": args.port, "duration_s": args.duration_s,
        "receiver_clock_id": receiver_id, "context_length": context, "minimum_observations": minimum,
        "sync_interval_us": args.sync_interval_ms * 1000, "query_period_us": args.query_period_ms * 1000,
        "clock": "monotonic_ns // 1000; user-space capture; no OS clock adjustment"}
    atomic_json(output / "config.json", resolved)
    rows = []
    pending = {}
    request_id = 0
    malformed = 0
    start = monotonic_us()
    finish = start + round(args.duration_s * 1e6)
    next_sync = next_query = start
    print(f"Capturing real UDP timestamps from {args.host}:{args.port}; output={output}", flush=True)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, (output / "events.jsonl").open("w", encoding="utf-8") as log:
            sock.connect((args.host, args.port))
            sock.setblocking(False)
            header = {"kind": "header", "schema_version": 1, "receiver_clock_id": receiver_id,
                      "sample_period_us": 25000, "context_length": context, "minimum_observations": minimum,
                      "clock_config": asdict(clock_config), "timing_config": asdict(timing_config)}
            log.write(json.dumps(header) + "\n")

            def record(event):
                result = session.process(event)
                log.write(json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n")
                if result is not None:
                    rows.append(result)

            try:
                while monotonic_us() < finish:
                    now = monotonic_us()
                    if now >= next_sync:
                        t1 = monotonic_us()
                        sock.send(encode({"kind": "sync_request", "receiver_clock_id": receiver_id,
                            "request_id": request_id, "receiver_send_us": t1}))
                        pending[request_id] = t1
                        request_id += 1
                        next_sync = now + args.sync_interval_ms * 1000
                    timeout = min(0.05, max(0.0, (min(next_sync, next_query, finish) - monotonic_us()) / 1e6))
                    if select.select([sock], [], [], timeout)[0]:
                        try:
                            data = sock.recv(MAX_DATAGRAM_BYTES + 1)
                            received = monotonic_us()
                            message = decode(data)
                            if message.get("sample_period_us") != 25000:
                                raise ValueError("Capture requires the 25,000 us source grid")
                            if message["kind"] == "sync_response":
                                t1 = pending.get(message["request_id"])
                                if t1 is None or message["receiver_clock_id"] != receiver_id or message["receiver_send_us"] != t1:
                                    raise ValueError("Unmatched synchronization response")
                                event = {"kind": "sync", "source_clock_id": message["source_clock_id"],
                                    "receiver_send_us": t1, "source_receive_us": message["source_receive_us"],
                                    "source_send_us": message["source_send_us"], "receiver_receive_us": received,
                                    "event_time_us": monotonic_us(), "series_id": message.get("series_id"),
                                    "dataset_id": message.get("dataset_id")}
                            elif message["kind"] == "packet":
                                event = {"kind": "packet", "received_at_us": received, "event_time_us": monotonic_us(),
                                         "send_time_us": message.get("send_time_us"), "series_id": message.get("series_id"),
                                         **{name: message.get(name) for name in TimestampedPacket.__dataclass_fields__}}
                            else:
                                raise ValueError("Unknown timestamp datagram kind")
                            record(event)
                            if message["kind"] == "sync_response":
                                del pending[message["request_id"]]
                        except (KeyError, TypeError, ValueError, OSError):
                            malformed += 1
                    now = monotonic_us()
                    pending = {key: value for key, value in pending.items() if now - value <= 5_000_000}
                    if now >= next_query:
                        record({"kind": "query", "event_time_us": now})
                        next_query = now + args.query_period_ms * 1000
            except KeyboardInterrupt:
                pass
        if not session.counts["sync_accepted"] or not session.counts["packets"]:
            raise RuntimeError("No completed synchronization exchange or no source packets were captured")
        metadata = {"status": "success", "kind": "sync-capture", "summary": capture_summary(rows),
            "events": dict(session.counts), "malformed_datagrams": malformed,
            "trace_sha256": sha256(output / "events.jsonl"), "integration": integration_identity(root),
            "clock_measurement": "Real UDP with two monotonic clock domains and four user-space timestamps",
            "model_inference": False, "signal_source": "replayed prepared CAN; not a live vehicle sensor"}
        write_capture_output(output, rows, metadata)
        publish_success(root, output, "aoi_capture")
        print(f"Capture complete: {output / 'report.html'}", flush=True)
        return output
    except Exception as exc:
        atomic_json(output / "metadata.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


def replay_capture(root: Path, args) -> Path:
    trace = project_path(root, args.trace)
    trace_digest = sha256(trace)
    output = new_output(root, "aoi_trace")
    rows = []
    try:
        with trace.open(encoding="utf-8") as stream:
            header = json.loads(next(stream))
            if (not isinstance(header, dict) or header.get("kind") != "header"
                    or type(header.get("schema_version")) is not int or header["schema_version"] != 1):
                raise ValueError("AoI trace must begin with a versioned receiver-clock header")
            recorded_clock, recorded_timing = ClockConfig(**header["clock_config"]), TimingConfig(**header["timing_config"])
            if args.aoi_config is not None:
                _, clock_config, timing_config = measurement_configs(root, args.aoi_config)
                config_source = "override"
            else:
                clock_config, timing_config = recorded_clock, recorded_timing
                config_source = "trace_header"
            atomic_json(output / "config.json", {"trace": str(trace), "trace_sha256": trace_digest,
                "configuration_source": config_source, "override_file": args.aoi_config,
                "clock_config": asdict(clock_config), "timing_config": asdict(timing_config), "recorded_header": header})
            session = CaptureSession(header["receiver_clock_id"], clock_config, timing_config,
                sample_period_us=header["sample_period_us"], context_length=header["context_length"],
                minimum_observations=header["minimum_observations"])
            for line in stream:
                row = session.process(json.loads(line))
                if row is not None:
                    rows.append(row)
        if not rows or not session.counts["sync_accepted"] or not session.counts["packets"]:
            raise ValueError("Trace needs queries, source packets and completed synchronization measurements")
        if sha256(trace) != trace_digest:
            raise RuntimeError("Input trace changed during replay")
        metadata = {"status": "success", "kind": "replay-aoi", "summary": capture_summary(rows),
            "events": dict(session.counts), "trace": str(trace), "trace_sha256": trace_digest,
            "configuration_source": config_source,
            "integration": integration_identity(root), "model_inference": False}
        write_capture_output(output, rows, metadata)
        publish_success(root, output, "aoi_trace")
        print(f"Trace replay complete: {output / 'report.html'}", flush=True)
        return output
    except Exception as exc:
        atomic_json(output / "metadata.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
