"""Live UDP adapter for the shared streaming runtime and existing timestamp wire format."""
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import json
import sys
from uuid import uuid4

import pandas as pd

from .common import atomic_json, integration_identity, new_output, publish_success, require_current_source, source_hash
from .evaluation import resolved_config
from .model import TiRexModel
from .stream_experiment import UNIT_FIELDS, stream_metrics
from .streaming import INFERENCE_FIELDS, StreamConfig, StreamingRuntime, sleep_until
from .transport import (CaptureSession, decode, encode, measurement_configs,
                        monotonic_us, timestamp_event)


class StreamDatagrams(asyncio.DatagramProtocol):
    def __init__(self, runtime: StreamingRuntime, log):
        self.runtime, self.log = runtime, log
        self.pending = {}
        self.counts = Counter()
        self.remote_series_id = None

    def datagram_received(self, data, address):
        received = monotonic_us()
        if hasattr(self.runtime, "finish_us") and received >= self.runtime.finish_us:
            self.counts["after_window_datagrams"] += 1
            return
        try:
            message = decode(data)
            event = timestamp_event(message, received, self.runtime.session.receiver_clock_id, self.pending)
            self.runtime.process(event)
            self.log.write(json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n")
            self.remote_series_id = message.get("series_id", self.remote_series_id)
            if message["kind"] == "sync_response":
                del self.pending[message["request_id"]]
        except (KeyError, TypeError, ValueError, OSError):
            self.counts["malformed_datagrams"] += 1

    def error_received(self, exc):
        self.counts["socket_errors"] += 1

    async def synchronize(self, transport, start_us: int, finish_us: int, interval_us: int):
        request_id, due = 0, start_us
        while due < finish_us:
            await sleep_until(due)
            sent = monotonic_us()
            if sent >= finish_us:
                break
            self.pending = {key: timestamp for key, timestamp in self.pending.items() if sent - timestamp <= 5_000_000}
            self.pending[request_id] = sent
            transport.sendto(encode({"kind": "sync_request", "receiver_clock_id": self.runtime.session.receiver_clock_id,
                "request_id": request_id, "receiver_send_us": sent}))
            request_id += 1
            due += interval_us
            if due <= sent:
                due += (sent - due) // interval_us * interval_us + interval_us


def stream_receive(root: Path, args) -> Path:
    require_current_source(args.source)
    config, clock_config, timing_config = measurement_configs(root, args.aoi_config)
    resolved = resolved_config(root, args, batch_size=1)
    stream_config = StreamConfig(publication_period_us=args.publish_period_ms * 1000,
        inference_period_us=args.inference_period_ms * 1000,
        maximum_prediction_steps=config["maximum_prediction_steps"],
        compute_estimate_us=args.compute_estimate_ms * 1000,
        warmup_requests=args.warmup if args.with_model else 0)
    receiver_id = "stream-receiver-" + uuid4().hex
    session = CaptureSession(receiver_id, clock_config, timing_config,
        context_length=args.context_length or config["context_length"],
        minimum_observations=args.minimum_observations or config["minimum_observations"])
    output = new_output(root, "stream_live_current" if args.with_model else "stream_live_timing")
    integration = integration_identity(root)
    selection = {"host": args.host, "port": args.port, "duration_s": args.duration_s,
        "with_model": args.with_model, "publication_period_ms": args.publish_period_ms,
        "inference_period_ms": args.inference_period_ms, "stdout": args.stdout}
    atomic_json(output / "config.json", {"selection": selection, "stream": asdict(stream_config),
        "clock": asdict(clock_config), "timing": asdict(timing_config), "model_configuration": resolved,
        "receiver_clock_id": receiver_id, "context_length": session.context_length,
        "minimum_observations": session.minimum_observations})
    model = TiRexModel(root, resolved["settings"], resolved["runtime"]) if args.with_model else None
    publications, requests = [], []
    identity = {"unit_id": receiver_id, "source": "current" if args.with_model else "none",
        "series_id": "udp", "split": "live", "arrival_profile": "real_udp", "clock_scenario": "measured",
        "replicate": 0, "inference_period_ms": args.inference_period_ms}
    progress = sys.stderr if args.stdout else sys.stdout

    async def receive(executor):
        loop = asyncio.get_running_loop()
        if model is not None:
            print(f"Loading current TiRex; output={output}", file=progress, flush=True)
            await loop.run_in_executor(executor, model.load)
        with (output / "states.jsonl").open("w", encoding="utf-8", buffering=1) as states, \
                (output / "events.jsonl").open("w", encoding="utf-8", buffering=1) as events:
            events.write(json.dumps({"kind": "header", "schema_version": 1, "receiver_clock_id": receiver_id,
                "sample_period_us": 25000, "context_length": session.context_length,
                "minimum_observations": session.minimum_observations,
                "clock_config": asdict(clock_config), "timing_config": asdict(timing_config)}) + "\n")
            def publish(row):
                serialized = json.dumps(row, separators=(",", ":"), allow_nan=False)
                states.write(serialized + "\n")
                events.write(json.dumps({"kind": "query", "event_time_us": row["snapshot_time_us"]}) + "\n")
                publications.append({**identity, **row})
                if args.stdout:
                    print(serialized, flush=True)
            runtime = StreamingRuntime(session, stream_config, model=model, executor=executor,
                on_publish=publish, on_inference=lambda row: requests.append({**identity, **row}))
            protocol = StreamDatagrams(runtime, events)
            transport, _ = await loop.create_datagram_endpoint(lambda: protocol, remote_addr=(args.host, args.port))
            start = monotonic_us()
            duration = round(args.duration_s * 1e6)
            sync_task = asyncio.create_task(protocol.synchronize(transport, start, start + duration,
                                                                 args.sync_interval_ms * 1000))
            print(f"Streaming UDP from {args.host}:{args.port}; states={output / 'states.jsonl'}", file=progress, flush=True)
            try:
                run = await runtime.run(duration, start_us=start)
            finally:
                sync_task.cancel()
                await asyncio.gather(sync_task, return_exceptions=True)
                transport.close()
            run.update({**identity, "publication_period_us": stream_config.publication_period_us,
                "online_packets": session.counts["packets"],
                "reordered_packets": session.receiver.reordered if session.receiver else 0,
                "input_lateness_us": []})
            return run, protocol

    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="tirex-stream") as executor:
            run, protocol = asyncio.run(receive(executor))
        if not session.counts["packets"] or not session.counts["sync_accepted"]:
            raise RuntimeError("No source packets or completed synchronization were received")
        frame = pd.DataFrame(publications)
        measured = pd.DataFrame(requests, columns=[*UNIT_FIELDS, *INFERENCE_FIELDS])
        frame.to_csv(output / "publications.csv", index=False)
        measured.to_csv(output / "inferences.csv", index=False)
        performance, summary, _, overall = stream_metrics(frame, measured, [run], args.with_model)
        performance.to_csv(output / "runtime_metrics.csv", index=False)
        if integration_identity(root) != integration or model is not None and source_hash(model.source_directory) != model.source_sha256:
            raise RuntimeError("Runtime source changed during streaming")
        metadata = {"status": "success", "kind": "stream-receive", "selection": selection,
            "summary": overall, "runs": [run], "events": dict(session.counts),
            "datagrams": dict(protocol.counts), "remote_series_id": protocol.remote_series_id,
            "model_inference": model is not None, "model": model.metadata() if model is not None else None,
            "integration": integration, "ground_truth_age_available": False, "speed_ground_truth_available": False,
            "warmup": run["warmup"], "clock_measurement": "real UDP four timestamps and local monotonic publication time",
            "scheduler": "one in-flight request; no pending queue; fresh history after completion; absolute publication deadlines"}
        atomic_json(output / "metadata.json", metadata)
        from .report import write_stream_report
        write_stream_report(output, performance, summary, metadata, frame)
        publish_success(root, output, "stream_live" if args.with_model else "stream_live_timing")
        print(f"Live stream complete: {output / 'report.html'}", file=progress, flush=True)
        return output
    except Exception as exc:
        atomic_json(output / "metadata.json", {"status": "failed", "selection": selection,
            "error": f"{type(exc).__name__}: {exc}", "integration": integration})
        raise
