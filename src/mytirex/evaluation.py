"""Independent extraction of the old causal replay, without cross-method caches."""
from collections import Counter
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from .common import (atomic_json, dependency_versions, integration_identity, new_output, project_path,
                     publish_success, read_json, sha256, source_hash, verify_files)
from .data import dataset_identity, load_dataset, query_indices
from .metrics import write_metrics
from .model import TiRexModel, read_tuning
from .references import comparison_table, read_controls


def load_config(root: Path, path: str | Path = "configs/experiment.json") -> dict:
    config = read_json(project_path(root, path))
    expected_protocol = {"score_start_s": 5.0, "delays_ms": [175, 300],
                         "clock": "integer_microseconds", "target": "query_time"}
    if config["schema_version"] != 1 or config["dataset"] != "regression_speed_v1" or config["protocol"] != expected_protocol:
        raise ValueError("eval-speed must preserve the original five-sequence 40 Hz, 175/300 ms, score-from-5s protocol")
    runtime = config["runtime"]
    if runtime["device"] != "cuda":
        raise ValueError("This project baseline requires CUDA")
    for key in ("batch_size", "num_threads", "seed"):
        value = runtime[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == "seed" else 1):
            raise ValueError(f"runtime.{key} must be a valid integer")
    return config


def resolved_config(root: Path, args, batch_size: int | None = None) -> dict:
    config = load_config(root, args.config)
    runtime = dict(config["runtime"])
    if batch_size is not None:
        runtime["batch_size"] = batch_size
    elif getattr(args, "batch_size", None) is not None:
        runtime["batch_size"] = args.batch_size
    source = args.source
    tuning_path = args.tirex_config or config["tuning_config" if source == "current" else "official_config"]
    tuning = read_tuning(root, tuning_path, source)
    return {"experiment": config, "source": source, "settings": tuning, "runtime": runtime,
            "config_file": str(project_path(root, args.config)), "config_sha256": sha256(project_path(root, args.config)),
            "tuning_file": str(project_path(root, tuning_path)), "tuning_sha256": sha256(project_path(root, tuning_path))}


def replay(model: TiRexModel, sequences: list, protocol: dict, selection: dict) -> tuple[pd.DataFrame, dict]:
    chunks = []
    started = perf_counter()
    for sequence in sequences:
        for delay_ms in selection["delays_ms"]:
            indices = query_indices(sequence, protocol["score_start_s"], selection["max_queries"])
            rows = []
            model.reset_for_unit(sequence.name, delay_ms)
            unit_start = perf_counter()
            horizon = delay_ms * 1000 // 25000
            for offset in range(0, len(indices), model.batch_size):
                selected = indices[offset:offset + model.batch_size]
                views = [sequence.request(int(index), delay_ms * 1000) for index in selected]
                trajectories = model.forecast_batch(views, horizon)
                if len(trajectories) != len(views):
                    raise RuntimeError("Model returned an invalid batch length")
                if offset == 0 and all(p.status != "ok" for p in trajectories):
                    raise RuntimeError(f"TiRex failed the entire initial batch: {trajectories[0].failure_reason}")
                for index, view, trajectory in zip(selected, views, trajectories, strict=True):
                    if view.query_time_us - trajectory.source_time_us != horizon * 25000:
                        raise RuntimeError("Prediction trajectory is not aligned to the fixed-delay target")
                    point = trajectory.value_at(view.query_time_us)
                    if not np.isfinite(point):
                        raise RuntimeError("No finite published model/fallback prediction")
                    row = {
                        "method": "TiRex-2" if model.source == "current" else "TiRex-2 (official run)",
                        "sequence": sequence.name, "delay_ms": delay_ms, "query_index": int(index),
                        "query_time_us": view.query_time_us, "target_time_us": view.query_time_us,
                        "last_source_time_us": trajectory.source_time_us, "last_received_at_us": int(view.received_at_us[-1]),
                        "query_time_s": view.query_time_us / 1e6, "target_time_s": view.query_time_us / 1e6,
                        "last_source_time_s": trajectory.source_time_us / 1e6, "last_observation": float(view.values_mps[-1]),
                        "visible_count": len(view.values_mps), "context_count": min(len(view.values_mps), model.context_length),
                        "horizon_steps": horizon, "prediction": point, "status": trajectory.status,
                        "failure_reason": trajectory.failure_reason, "fallback": trajectory.fallback,
                        "batch_elapsed_s": trajectory.batch_elapsed_s, "elapsed_s": trajectory.batch_elapsed_s / trajectory.batch_size,
                        "batch_size": trajectory.batch_size, "batch_id": f"{sequence.name}/{delay_ms}/{offset // model.batch_size}",
                    }
                    # Truth enters scoring only after GPU completion and CPU return.
                    row["reference"] = float(sequence.values[index])
                    rows.append(row)
                if (offset // model.batch_size) % max(1, 1024 // model.batch_size) == 0 or offset + model.batch_size >= len(indices):
                    print(f"{sequence.name} {delay_ms}ms: {min(offset + model.batch_size, len(indices))}/{len(indices)}"
                          f" ({perf_counter() - unit_start:.2f}s)", flush=True)
            chunks.append(pd.DataFrame(rows))
    frame = pd.concat(chunks, ignore_index=True)
    if frame.duplicated(["sequence", "delay_ms", "query_index"]).any():
        raise RuntimeError("Duplicate query results")
    expected = sum(len(query_indices(s, protocol["score_start_s"], selection["max_queries"]))
                   for s in sequences) * len(selection["delays_ms"])
    if len(frame) != expected:
        raise RuntimeError(f"Incomplete replay: {len(frame)} != {expected}")
    return frame, {"replay_wall_s": perf_counter() - started, "status_counts": dict(Counter(frame.status)),
                   "failure_count": int((frame.status != "ok").sum()), "fallback_count": int(frame.fallback.sum())}


def eval_speed(root: Path, args) -> Path:
    started = perf_counter()
    resolved = resolved_config(root, args)
    config, settings, runtime = resolved["experiment"], resolved["settings"], resolved["runtime"]
    all_sequences = load_dataset(root)
    names = [s.name for s in all_sequences]
    if args.sequence and set(args.sequence) - set(names):
        raise ValueError(f"Unknown sequence; choose from {names}")
    if args.delay_ms and set(args.delay_ms) - {175, 300}:
        raise ValueError("Choose fixed protocol delay 175 or 300 ms")
    sequences = [s for s in all_sequences if not args.sequence or s.name in args.sequence]
    selection = {"sequences": [s.name for s in sequences], "delays_ms": sorted(set(args.delay_ms or [175, 300])),
                 "max_queries": args.max_queries}
    output = new_output(root, f"speed_{args.source}")
    resolved["selection"] = selection
    atomic_json(output / "config.json", resolved)
    integration = integration_identity(root)
    try:
        print(f"Loading {args.source} TiRex locally on CUDA; output={output}", flush=True)
        model = TiRexModel(root, settings, runtime, args.source)
        model.load()
        model.torch.cuda.reset_peak_memory_stats()
        frame, replay_metadata = replay(model, sequences, config["protocol"], selection)
        frame.to_csv(output / "predictions.csv", index=False)
        metrics = write_metrics(frame, output)
        controls, identities = (read_controls(root, frame, all_sequences, config, settings, runtime)
                                if args.reference == "auto" else ([], {"policy": "disabled"}))
        comparison = comparison_table(frame, controls) if controls else None
        if comparison is not None:
            comparison.to_csv(output / "reference_comparison.csv", index=False)
        if source_hash(model.source_directory) != model.source_sha256 or integration_identity(root) != integration:
            raise RuntimeError("Runtime source changed during evaluation; result was not published")
        verify_files(model.model_path, model.model_lock["sha256"])
        load_dataset(root)  # Recheck the small regression files before publishing.
        metadata = {
            "status": "success", "kind": "eval-speed", "selection": selection, "query_count": len(frame),
            "full_protocol_query_count": 70010, "is_full_evaluation": len(frame) == 70010,
            "dataset": dataset_identity(root), "model": model.metadata(), "integration": integration,
            "dependencies": dependency_versions(), "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "references": identities, **replay_metadata,
            "gpu_peak_allocated_bytes": model.torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved_bytes": model.torch.cuda.max_memory_reserved(),
            "scoring_clock": "offline source/query microseconds; target is query_time, not inference delivery time",
            "causal_rule": "source_time_us + delay_us <= query_time_us",
            "truth_attached_after_inference": True, "wall_time_s_before_report": perf_counter() - started,
        }
        from .report import make_plots, write_report
        plots = make_plots(pd.concat([frame, *controls], ignore_index=True), output / "plots")
        atomic_json(output / "metadata.json", metadata)
        write_report(output, metrics, metadata, comparison, plots)
        publish_success(root, output, "speed")
        print(metrics["summary"][["method", "delay_ms", "query_count", "MAE", "RMSE", "failure_count", "fallback_count"]].to_string(index=False), flush=True)
        print(f"Complete: {output / 'report.html'}", flush=True)
        return output
    except Exception as exc:
        atomic_json(output / "metadata.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                               "selection": selection, "integration": integration})
        raise
