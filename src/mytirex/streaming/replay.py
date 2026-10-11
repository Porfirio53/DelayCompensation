"""Real-time causal CAN playback through the same asynchronous runtime as UDP."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from math import ceil
from pathlib import Path
import json

import numpy as np
import pandas as pd

from ..evaluation.aoi import load_aoi_config
from ..reporting.aoi_metrics import error_statistics
from ..evaluation.simulation import ClockScenario, SimulationClock, synchronization_trials
from ..data.arrival import NEVER, generate_arrivals, trace_seed
from ..common import (atomic_json, fingerprint, integration_identity, new_output, project_path,
                     publish_success, read_json, require_current_source, sha256, source_hash)
from ..inference.delivery import ComputeLatencyEstimator, warmup_delivery
from ..evaluation.speed import resolved_config
from ..inference.model import TiRexModel
from ..data.prepared import PreparedCorpus
from ..timing.receiver import TimingConfig
from .runtime import INFERENCE_FIELDS, StreamConfig, StreamingRuntime, sleep_until
from ..timing.clock import ClockConfig
from ..timing.transport import CaptureSession, monotonic_us, prepared_path

UNIT_FIELDS = ("unit_id", "source", "series_id", "split", "arrival_profile", "clock_scenario",
               "replicate", "inference_period_ms")


class RealtimeReplay:
    """Private source/schedule/scorer. Only completed packet events reach the runtime."""

    def __init__(self, series, profile: dict, scenario: ClockScenario, config: dict,
                 namespace: str, replicate: int, duration_us: int):
        self.series, self.scenario = series, scenario
        self.period = int(series.source_time_us[1] - series.source_time_us[0])
        origin = max(10_000_000, 10_000_000 - scenario.offset_us, 10_000_000 - scenario.source_stamp_bias_us)
        self.times = series.source_time_us - int(series.source_time_us[0]) + origin
        native = series.native_time_us - int(series.source_time_us[0]) + origin
        self.clock = SimulationClock(origin, origin + scenario.offset_us, 1 + scenario.relative_rate_ppm * 1e-6)
        self.sim_start_us = ceil(self.clock.receiver_time(origin + config["query_start_us"]))
        available_us = ceil(self.clock.receiver_time(int(self.times[-1]))) - self.sim_start_us
        self.duration_us = min(duration_us, available_us)
        if self.duration_us <= 0:
            raise ValueError(f"No stream window after query_start_us in {series.metadata['series_id']}")
        self.sim_finish_us = self.sim_start_us + self.duration_us
        clock_config, timing_config = ClockConfig(**config["clock"]), TimingConfig(**config["timing"])
        sync_error, source_error = scenario.error_bounds(clock_config, timing_config)
        self.clock_config = replace(clock_config, timestamp_error_us=sync_error)
        self.timing_config = replace(timing_config, source_error_us=source_error)
        self.context, self.minimum = config["context_length"], config["minimum_observations"]
        source_id = "stream-source-" + series.metadata["series_id"]
        self.receiver_id = "stream-receiver-" + series.metadata["series_id"]
        self.events = []
        metadata = series.metadata
        sync_seed = trace_seed(namespace + "/clock", metadata["split"], metadata["series_id"],
                               scenario.name, replicate, config["seed"])
        trials = synchronization_trials(self.clock, scenario,
            self.sim_finish_us - self.clock.receiver_origin_us, config["sync_interval_us"],
            config["sync_processing_us"], sync_seed)
        for trial in trials:
            if trial.available_at_us <= self.sim_finish_us:
                self.events.append({"kind": "sync", "event_time_us": trial.available_at_us,
                    "source_clock_id": source_id, **asdict(trial.exchange)})
        arrival_seed = trace_seed(namespace, metadata["split"], metadata["series_id"],
                                  profile["name"], replicate, config["seed"])
        arrivals = generate_arrivals(self.times, profile, arrival_seed)
        for index in np.argsort(arrivals, kind="stable"):
            if arrivals[index] == NEVER:
                break
            receipt = ceil(self.clock.receiver_time(int(arrivals[index])))
            if receipt > self.sim_finish_us:
                break
            self.events.append({"kind": "packet", "event_time_us": receipt, "received_at_us": receipt,
                "source_clock_id": source_id, "sequence": int(index),
                "source_time_us": int(self.times[index]) + scenario.source_stamp_bias_us,
                "native_source_time_us": int(native[index]) + scenario.source_stamp_bias_us,
                "value_mps": float(series.speed_mps[index])})
        self.events.sort(key=lambda event: event["event_time_us"])
        self.prefix_length = sum(event["event_time_us"] <= self.sim_start_us for event in self.events)

    @staticmethod
    def _shifted(event: dict, shift_us: int) -> dict:
        event = dict(event)
        for field in ("event_time_us", "received_at_us", "receiver_send_us", "receiver_receive_us"):
            if field in event:
                event[field] += shift_us
        return event

    def prefill(self, start_us: int | None = None) -> CaptureSession:
        shift = 0 if start_us is None else start_us - self.sim_start_us
        session = CaptureSession(self.receiver_id, self.clock_config, self.timing_config,
            sample_period_us=self.period, context_length=self.context, minimum_observations=self.minimum)
        for event in self.events[:self.prefix_length]:
            session.process(self._shifted(event, shift))
        return session

    async def feed(self, runtime: StreamingRuntime, start_us: int, lateness: list[int]) -> None:
        shift = start_us - self.sim_start_us
        finish = start_us + self.duration_us
        for original in self.events[self.prefix_length:]:
            event = self._shifted(original, shift)
            planned = event["event_time_us"]
            await sleep_until(planned)
            available = monotonic_us()
            if available >= finish:
                break
            event["event_time_us"] = available
            if event["kind"] == "packet":
                # Record when the application actually handles the arriving packet.
                event["received_at_us"] = available
                lateness.append(available - planned)
            runtime.process(event)

    def reference_at(self, receiver_time_us: int, start_us: int) -> float:
        source = self.clock.source_time(receiver_time_us - (start_us - self.sim_start_us))
        if source < self.times[0] or source > self.times[-1] + 1e-6:
            return float("nan")
        index = int(np.searchsorted(self.times, source + 1e-6, side="right")) - 1
        return float(self.series.speed_mps[index])


def percentiles(values, prefix: str, scale: float = 1000) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)] / scale
    return {f"{prefix}_P{p}": float(np.percentile(values, p)) if len(values) else None for p in (50, 95, 99)}


def stream_metrics(publications: pd.DataFrame, requests: pd.DataFrame, runs: list[dict],
                   with_model: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    performance, intervals = [], []
    for run in runs:
        identity = {name: run[name] for name in UNIT_FIELDS}
        frame = publications[publications.unit_id == run["unit_id"]]
        measured = requests[requests.unit_id == run["unit_id"]] if not requests.empty else requests
        gaps = np.diff(frame.publish_time_us.to_numpy(dtype=np.int64))
        intervals.extend(gaps)
        installed = int(measured.installed.sum()) if not measured.empty else 0
        trajectory = frame[frame["mode"] == "trajectory"]
        reused = trajectory.forecast_id.eq(trajectory.forecast_id.shift()).sum()
        row = {**identity, "duration_s": run["duration_s"], "publication_count": len(frame),
            "publication_hz": len(frame) / run["duration_s"], "inference_update_hz": installed / run["duration_s"],
            "missed_publication_ticks": run["counts"]["missed_publication_ticks"],
            "gap_over_1_5_period_count": int((gaps > run["publication_period_us"] * 1.5).sum()),
            "max_gap_ms": float(gaps.max() / 1000) if len(gaps) else None,
            "trajectory_fraction": len(trajectory) / len(frame),
            "reused_trajectory_publications": int(reused), "fallback_count": int(frame.fallback.sum()),
            "max_in_flight": run["counts"].get("max_in_flight", 0), "pending_request_max": run["pending_request_max"],
            "coalesced_packet_updates": run["counts"].get("coalesced_packet_updates", 0),
            "input_packets": run["online_packets"], "reordered_packets": run["reordered_packets"],
            **percentiles(gaps, "publication_interval_ms"),
            **percentiles(frame.deadline_lateness_us, "publication_lateness_ms"),
            **percentiles(run["input_lateness_us"], "input_lateness_ms")}
        successful = measured[measured.status == "ok"] if not measured.empty else measured
        for field, prefix in (("compute_us", "compute_ms"), ("queue_us", "queue_ms"), ("dispatch_us", "dispatch_ms")):
            row.update(percentiles(successful[field] if not successful.empty else [], prefix))
        row["request_count"] = len(measured)
        row["requests_with_packets_during_inference"] = int((measured.packets_during_inference > 0).sum()) if not measured.empty else 0
        performance.append(row)

    controls = [("received_hold", "last_observation_mps")]
    if with_model:
        controls += [("delivery_hold", "delivery_hold_mps"), ("trajectory", "prediction_mps")]
    if "reference_mps" not in publications:
        controls = []
    summary, by_series = [], []
    groups = ["source", "arrival_profile", "clock_scenario", "inference_period_ms"]
    for fields, destination in ((groups, summary), (list(UNIT_FIELDS), by_series)):
        for key, frame in publications.groupby(fields, sort=False):
            identity = dict(zip(fields, key))
            for policy, column in controls:
                error = (frame[column] - frame.reference_mps).to_numpy(dtype=float)
                destination.append({**identity, "policy": policy, "publication_count": len(frame),
                    **error_statistics(error[np.isfinite(error)], "speed_mps_")})
    table = pd.DataFrame(performance)
    measured = requests[requests.status == "ok"] if not requests.empty else requests
    total_duration = sum(run["duration_s"] for run in runs)
    overall = {"publication_count": len(publications), "duration_s": total_duration,
        "publication_hz": len(publications) / total_duration,
        "inference_update_hz": float(table.inference_update_hz @ table.duration_s / total_duration),
        "missed_publication_ticks": int(table.missed_publication_ticks.sum()),
        "max_in_flight": int(table.max_in_flight.max()), "pending_request_max": int(table.pending_request_max.max()),
        "trajectory_fraction": float((publications["mode"] == "trajectory").mean()),
        "fallback_count": int(publications.fallback.sum()),
        **percentiles(intervals, "publication_interval_ms"),
        **percentiles(publications.deadline_lateness_us, "publication_lateness_ms"),
        **percentiles(measured.compute_us if not measured.empty else [], "compute_ms")}
    return table, pd.DataFrame(summary), pd.DataFrame(by_series), overall


def eval_stream(root: Path, args) -> Path:
    require_current_source(args.source)
    config = load_aoi_config(root, args.aoi_config)
    resolved = resolved_config(root, args, batch_size=1)
    corpus_path = prepared_path(root, args.corpus)
    corpus = PreparedCorpus(corpus_path)
    index = read_json(root / "configs/data_index.json")[config["dataset"]]
    if args.corpus is None and corpus.manifest["dataset_id"] != index["dataset_id"]:
        raise ValueError("Prepared corpus differs from the frozen dataset index")
    split = args.split or config["split"]
    profiles = {entry["name"]: entry for entry in corpus.config["arrival"]["profiles"]}
    scenarios = {entry["name"]: ClockScenario(**entry) for entry in config["scenarios"]}
    selected_profiles = args.profile or ["fixed_175"]
    selected_scenarios = args.scenario or ["shared_clock_control"]
    periods = args.inference_period_ms or [25]
    for selected, available in ((selected_profiles, profiles), (selected_scenarios, scenarios), (periods, periods)):
        if len(selected) != len(set(selected)) or set(selected) - set(available):
            raise ValueError("Choose distinct configured stream profiles, scenarios and inference periods")
    # Read selection before creating outputs or loading the model.
    series = list(corpus.iter_series(split, args.series_id, args.max_series))
    if not series:
        raise ValueError("No prepared CAN series selected")
    output = new_output(root, "stream_current" if args.with_model else "stream_timing")
    selection = {"split": split, "series_ids": [item.metadata["series_id"] for item in series],
        "profiles": selected_profiles, "scenarios": selected_scenarios, "inference_period_ms": periods,
        "duration_s": args.duration_s, "publication_period_ms": args.publish_period_ms,
        "replicates": args.replicates, "with_model": args.with_model, "warmup": args.warmup}
    atomic_json(output / "config.json", {"aoi": config, "selection": selection, "model_configuration": resolved,
        "dataset_id": corpus.manifest["dataset_id"], "dataset_directory": str(corpus_path),
        "compute_estimate_ms": args.compute_estimate_ms, "coverage_margin_steps": 1,
        "aoi_config_sha256": sha256(project_path(root, args.aoi_config))})
    integration = integration_identity(root)
    publications, requests, runs, warmup = [], [], [], None
    warmed_estimate_us = args.compute_estimate_ms * 1000
    model = TiRexModel(root, resolved["settings"], resolved["runtime"]) if args.with_model else None

    async def experiment(executor):
        nonlocal warmup, warmed_estimate_us
        loop = asyncio.get_running_loop()
        if model is not None:
            print(f"Loading current TiRex for real-time streaming; output={output}", flush=True)
            await loop.run_in_executor(executor, model.load)
        with (output / "states.jsonl").open("w", encoding="utf-8", buffering=1) as state_log:
            for item in series:
                for profile_name in selected_profiles:
                    profile = profiles[profile_name]
                    for scenario_name in selected_scenarios:
                        scenario = scenarios[scenario_name]
                        for replicate in range(1 if profile["kind"] == "fixed" else args.replicates):
                            replay = RealtimeReplay(item, profile, scenario, config,
                                corpus.config["dataset_namespace"], replicate, round(args.duration_s * 1e6))
                            for period_ms in periods:
                                estimator = ComputeLatencyEstimator(warmed_estimate_us)
                                identity = {"source": "current" if model is not None else "none",
                                    "series_id": item.metadata["series_id"], "split": split,
                                    "arrival_profile": profile_name, "clock_scenario": scenario_name,
                                    "replicate": replicate, "inference_period_ms": period_ms}
                                identity["unit_id"] = fingerprint(identity)[:20]
                                if model is not None:
                                    await loop.run_in_executor(executor, model.reset_for_trace,
                                        identity["series_id"], profile_name, scenario_name, replicate)
                                    if warmup is None:
                                        initial = replay.prefill().snapshot(replay.sim_start_us)
                                        if initial is None or not initial.ready:
                                            raise ValueError("The selected first stream window has no ready warmup history")
                                        print(f"Prewarming {args.warmup} requests before timed replay (first may compile)", flush=True)
                                        warmup = await loop.run_in_executor(executor, warmup_delivery, model,
                                            initial, estimator, args.warmup, max(period_ms * 1000, estimator.estimate_us))
                                        warmed_estimate_us = estimator.estimate_us
                                        await loop.run_in_executor(executor, model.reset_for_trace,
                                            identity["series_id"], profile_name, scenario_name, replicate)
                                start = monotonic_us() + 50_000
                                session = replay.prefill(start)
                                prefix_packets = session.counts["packets"]
                                lateness = []
                                def publish(row):
                                    # The public stream contains no scoring reference.
                                    state_log.write(json.dumps({**identity, **row}, separators=(",", ":"), allow_nan=False) + "\n")
                                    reference = replay.reference_at(row["publish_time_us"], start)
                                    publications.append({**identity, **row, "reference_mps": reference})
                                runtime = StreamingRuntime(session, StreamConfig(
                                    publication_period_us=args.publish_period_ms * 1000,
                                    inference_period_us=period_ms * 1000,
                                    maximum_prediction_steps=config["maximum_prediction_steps"],
                                    compute_estimate_us=args.compute_estimate_ms * 1000),
                                    model=model, estimator=estimator, executor=executor,
                                    on_publish=publish, on_inference=lambda row: requests.append({**identity, **row}))
                                print(f"Stream {len(runs) + 1}: {item.metadata['series_id']} / {profile_name} / "
                                      f"{scenario_name}, inference period={period_ms} ms, window={replay.duration_us / 1e6:g} s", flush=True)
                                runtime_task = asyncio.create_task(runtime.run(replay.duration_us, start_us=start))
                                feeder = asyncio.create_task(replay.feed(runtime, start, lateness))
                                try:
                                    await asyncio.gather(runtime_task, feeder)
                                finally:
                                    if not feeder.done():
                                        feeder.cancel()
                                    await asyncio.gather(feeder, return_exceptions=True)
                                run = runtime_task.result()
                                run.update({**identity, "publication_period_us": args.publish_period_ms * 1000,
                                    "online_packets": session.counts["packets"] - prefix_packets,
                                    "reordered_packets": session.receiver.reordered if session.receiver else 0,
                                    "input_lateness_us": lateness})
                                runs.append(run)
                                print(f"Published {run['counts']['publications']} states; installed "
                                      f"{run['counts'].get('forecasts_installed', 0)} forecasts; "
                                      f"missed ticks={run['counts']['missed_publication_ticks']}", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="tirex-stream") as executor:
            asyncio.run(experiment(executor))
        frame = pd.DataFrame(publications)
        measured = pd.DataFrame(requests, columns=[*UNIT_FIELDS, *INFERENCE_FIELDS])
        if frame.empty:
            raise RuntimeError("No states were published")
        frame.to_csv(output / "publications.csv", index=False)
        measured.to_csv(output / "inferences.csv", index=False)
        performance, summary, by_series, overall = stream_metrics(frame, measured, runs, args.with_model)
        performance.to_csv(output / "runtime_metrics.csv", index=False)
        summary.to_csv(output / "metrics_summary.csv", index=False)
        by_series.to_csv(output / "metrics_by_series.csv", index=False)
        if integration_identity(root) != integration or model is not None and source_hash(model.source_directory) != model.source_sha256:
            raise RuntimeError("Runtime source changed during streaming")
        metadata = {"status": "success", "kind": "eval-stream", "selection": selection, "summary": overall,
            "model_inference": model is not None, "model": model.metadata() if model is not None else None,
            "dataset_id": corpus.manifest["dataset_id"], "integration": integration, "warmup": warmup,
            "initial_stream_compute_estimate_us": warmed_estimate_us,
            "clock_measurement": "controlled arrival/clock schedules paced by real monotonic time; actual packet handling, GPU readiness and publication timestamps",
            "scoring_clock": "actual application publication time; causal 40 Hz ZOH reference; no future-reference interpolation",
            "trajectory_policy": "measured AoI upper bound + actual worker dispatch queue + EWMA compute estimate + max(inference period, compute estimate) + one source-grid step",
            "scheduler": "one in-flight request; no pending request queue; fresh snapshot at every start; passed publication deadlines skipped",
            "controls": "received_hold and delivery_hold share the trajectory method's publication timestamps; no additional model calls",
            "historical_fixed_references_used": False,
            "runs": [{key: value for key, value in run.items() if key != "input_lateness_us"} for run in runs]}
        atomic_json(output / "metadata.json", metadata)
        from ..reporting.html import write_stream_report
        write_stream_report(output, performance, summary, metadata, frame)
        publish_success(root, output, "stream" if args.with_model else "stream_timing")
        print(f"Streaming complete: {output / 'report.html'}", flush=True)
        return output
    except Exception as exc:
        atomic_json(output / "metadata.json", {"status": "failed", "selection": selection,
            "error": f"{type(exc).__name__}: {exc}", "integration": integration})
        raise
