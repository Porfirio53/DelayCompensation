"""Real batch=1 end-to-end latency, with separate deployment/warmup phases."""
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from .common import (atomic_json, dependency_versions, integration_identity, new_output, publish_success,
                     source_hash)
from .data import dataset_identity, load_dataset
from .evaluation import resolved_config
from .model import TiRexModel


def latency_summary(seconds: list[float]) -> dict:
    if not seconds:
        return {"samples": 0}
    values = np.asarray(seconds) * 1000
    return {"samples": len(seconds), "P50_ms": float(np.percentile(values, 50)),
            "P95_ms": float(np.percentile(values, 95)), "P99_ms": float(np.percentile(values, 99)),
            "mean_ms": float(values.mean()), "min_ms": float(values.min()), "max_ms": float(values.max())}


def peak_memory(model: TiRexModel) -> dict:
    return {"peak_allocated_bytes": model.torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": model.torch.cuda.max_memory_reserved()}


def timed_request(model: TiRexModel, sequence, query_index: int, delay_ms: int) -> tuple[float, dict]:
    # Timer begins before detached receiver history creation, not after preprocessing.
    model.synchronize()
    started = perf_counter()
    view = sequence.request(query_index, delay_ms * 1000)
    forecast = model.forecast(view, delay_ms * 1000 // view.sample_period_us)
    point = forecast.value_at(view.query_time_us)
    model.synchronize()
    elapsed = perf_counter() - started
    if forecast.status != "ok" or not np.isfinite(point):
        raise RuntimeError(f"Single-request benchmark forecast failed: {forecast.failure_reason}")
    return elapsed, {"sequence": sequence.name, "delay_ms": delay_ms, "query_index": query_index,
                     "source_time_us": forecast.source_time_us, "query_time_us": view.query_time_us,
                     "horizon_steps": forecast.values_mps.shape[1], "context_count": min(len(view.values_mps), model.context_length),
                     "batch_size": 1, "point_mps": point, "end_to_end_s": elapsed}


def benchmark(root: Path, args) -> Path:
    if args.batch_size != 1:
        raise ValueError("Single-request latency requires --batch-size 1; batch amortization is not supported here")
    resolved = resolved_config(root, args, batch_size=1)
    sequence = next((s for s in load_dataset(root) if s.name == args.sequence), None)
    if sequence is None:
        raise ValueError("Unknown benchmark sequence")
    if not 200 <= args.query_index < len(sequence.values):
        raise ValueError("Benchmark query-index must be in the scored grid 200..7200")
    output = new_output(root, f"benchmark_{args.source}")
    resolved["benchmark"] = {"sequence": args.sequence, "delay_ms": args.delay_ms, "samples": args.samples,
                             "warmup": args.warmup, "query_index": args.query_index, "batch_size": 1,
                             "query_policy": "consecutive source-grid queries, wrapping at end of selected sequence"}
    atomic_json(output / "config.json", resolved)
    integration = integration_identity(root)
    try:
        model = TiRexModel(root, resolved["settings"], resolved["runtime"], args.source)
        print(f"Loading {args.source} TiRex; benchmark output={output}", flush=True)
        model.load()
        load_peak = peak_memory(model)
        rows = []
        def run_one(phase: str, index: int):
            query = args.query_index + index % (len(sequence.values) - args.query_index)
            seconds, row = timed_request(model, sequence, query, args.delay_ms)
            rows.append({"phase": phase, "phase_index": index, **row})
            return seconds
        model.torch.cuda.reset_peak_memory_stats()
        first = run_one("first_request", 0)
        first_peak = peak_memory(model)
        print(f"First request (possible kernel compilation): {first * 1000:.3f} ms", flush=True)
        warmup = [run_one("warmup", index) for index in range(args.warmup)]
        model.torch.cuda.reset_peak_memory_stats()
        stable = []
        for index in range(args.samples):
            stable.append(run_one("stable", index))
            if (index + 1) % 50 == 0:
                print(f"Steady-state samples: {index + 1}/{args.samples}", flush=True)
        stable_peak = peak_memory(model)
        if source_hash(model.source_directory) != model.source_sha256 or integration_identity(root) != integration:
            raise RuntimeError("Source changed during benchmark; result was not published")
        phases = {"load": {"elapsed_s": model.load_time_s, **load_peak},
                  "first_request": {"elapsed_s": first, "contains_possible_kernel_compilation": True,
                                    "kernel_cache": model.first_forecast, **first_peak},
                  "warmup": {**latency_summary(warmup), "total_s": sum(warmup)},
                  "stable": {**latency_summary(stable), **stable_peak}}
        metadata = {"status": "success", "kind": "benchmark", "batch_size": 1, "phases": phases,
                    "model": model.metadata(), "dataset": dataset_identity(root), "integration": integration,
                    "dependencies": dependency_versions(), "python": sys.version, "executable": sys.executable,
                    "platform": platform.platform(),
                    "timer": "perf_counter with CUDA synchronization before and after each individual request",
                    "includes": ["receiver history copy", "context validation/truncation/cast", "TiRex forecast",
                                 "GPU completion", "CPU trajectory return", "target median selection"],
                    "small_sample_note": "Tail percentiles describe the recorded sample only; use >=1000 stable requests for formal tail measurements."}
        pd.DataFrame(rows).to_csv(output / "benchmark.csv", index=False)
        atomic_json(output / "metadata.json", metadata)
        from .report import write_json_report
        write_json_report(output, "TiRex-2 batch=1 单请求性能", {"batch_size": 1, "phases": phases,
                          "gpu": metadata["model"]["gpu_name"], "configuration": resolved["settings"]})
        publish_success(root, output, "benchmark")
        print(f"Stable latency: {phases['stable']}", flush=True)
        print(f"Complete: {output / 'report.html'}", flush=True)
        return output
    except Exception as exc:
        atomic_json(output / "metadata.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
