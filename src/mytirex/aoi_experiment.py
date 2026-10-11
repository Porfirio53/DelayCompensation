"""AoI experiments with optional query-time or batch=1 delivery-time GPU scoring."""
from collections import Counter
from dataclasses import dataclass, replace
from itertools import islice
from pathlib import Path
import csv
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from .aoi_metrics import AoiAccumulator
from .aoi_simulation import ClockScenario, TimingCase, timing_cases
from .common import (atomic_json, dependency_versions, fingerprint, integration_identity, new_output,
                     project_path, publish_success, read_json, require_current_source, sha256, source_hash)
from .evaluation import resolved_config
from .delivery import (DELIVERY_POLICIES, ComputeLatencyEstimator, DeliveryAccumulator,
                       evaluate_delivery_case, warmup_delivery)
from .model import TiRexModel, observation_horizon
from .prepared import PreparedCorpus, verify_prepared
from .receiver import TimingConfig
from .timing import ClockConfig, horizon_steps, integer_us
from .transport import prepared_path, query_row

POLICIES = ("fixed_175", "fixed_300", "raw_timestamps", "offset_only", "synchronized", "oracle")


def load_aoi_config(root: Path, path: str) -> dict:
    config = read_json(project_path(root, path))
    expected = {"schema_version", "dataset", "split", "context_length", "minimum_observations", "query_start_us",
        "query_stride", "seed", "replicates", "arrival_profiles", "sync_interval_us", "sync_processing_us",
        "maximum_prediction_steps", "clock", "timing", "scenarios"}
    if set(config) != expected or config["schema_version"] != 1 or config["dataset"] != "prepared_comma2k19_v1":
        raise ValueError("Invalid AoI experiment schema or dataset")
    for name in ("context_length", "minimum_observations", "query_stride", "replicates", "sync_interval_us",
                 "sync_processing_us", "maximum_prediction_steps"):
        integer_us(config[name], name, 1)
    for name in ("seed", "query_start_us"):
        integer_us(config[name], name)
    if not 2 <= config["context_length"] <= 2048 or config["minimum_observations"] > config["context_length"]:
        raise ValueError("Use a context from 2 to 2048 points and a valid minimum observation count")
    if config["maximum_prediction_steps"] > 320:
        raise ValueError("AoI horizon cap cannot exceed the frozen checkpoint's 320 steps")
    ClockConfig(**config["clock"])
    TimingConfig(**config["timing"])
    if not config["scenarios"] or len({s["name"] for s in config["scenarios"]}) != len(config["scenarios"]):
        raise ValueError("Clock scenario names must be nonempty and distinct")
    for entry in config["scenarios"]:
        scenario = ClockScenario(**entry)
        if abs(scenario.relative_rate_ppm) > config["clock"]["max_relative_drift_ppm"]:
            raise ValueError("Scenario drift exceeds the estimator's declared frequency bound")
        if config["sync_processing_us"] < 2 * scenario.measurement_error_us:
            raise ValueError("Sync processing interval must accommodate the timestamp noise experiment")
    return config


@dataclass(frozen=True)
class PolicyEstimate:
    name: str
    aoi_us: float | None
    native_aoi_us: float | None
    source_offset_us: float | None
    lower_us: float | None = None
    upper_us: float | None = None
    upper_steps: int | None = None
    valid: bool = True
    reason: str = "ok"
    native_lower_us: float | None = None
    native_upper_us: float | None = None

    def steps(self, period: int) -> int | None:
        if self.source_offset_us is None or self.source_offset_us < 0:
            return None
        return horizon_steps(self.source_offset_us, period)


def policy_estimates(case: TimingCase) -> list[PolicyEstimate]:
    view = case.view
    if not len(view.source_time_us):
        return [PolicyEstimate(name, None, None, None, valid=False, reason="no_received_observation") for name in POLICIES]
    source, native = int(view.source_time_us[-1]), int(view.native_source_time_us[-1])
    hold = source - native
    raw_age = view.query_time_us - source
    estimates = [PolicyEstimate("fixed_175", 175000.0, 175000.0 + hold, 175000.0),
                 PolicyEstimate("fixed_300", 300000.0, 300000.0 + hold, 300000.0),
                 PolicyEstimate("raw_timestamps", float(raw_age), float(view.query_time_us - native), float(raw_age),
                                valid=raw_age >= 0, reason="ok" if raw_age >= 0 else "negative_observation_age")]
    for name, timing in (("offset_only", case.offset_only), ("synchronized", view.timing)):
        if timing is None:
            estimates.append(PolicyEstimate(name, None, None, None, valid=False, reason="clock_unavailable"))
        else:
            estimates.append(PolicyEstimate(name, timing.age_us, timing.native_age_us, timing.source_offset_us,
                timing.age_lower_us, timing.age_upper_us, timing.upper_horizon_steps, timing.valid, timing.reason,
                timing.native_age_lower_us, timing.native_age_upper_us))
    estimates.append(PolicyEstimate("oracle", case.true_aoi_us, case.true_native_aoi_us, case.true_source_offset_us,
        case.true_aoi_us, case.true_aoi_us, case.true_horizon_steps,
        native_lower_us=case.true_native_aoi_us, native_upper_us=case.true_native_aoi_us))
    return estimates


def evaluate_chunk(cases: list[TimingCase], *, model, target_for, maximum_steps: int, batch_id: str) -> pd.DataFrame:
    """One source-relative trajectory per history, shared by the time-selection controls.

    The configured coverage cap alone sets trajectory length; neither oracle
    age nor policy availability can affect generation. This isolates time
    selection errors from model outputs. Production forecast_observation()
    separately requests a dynamic horizon from measured timing.
    """
    plans = [policy_estimates(case) for case in cases]
    positions, views = [], []
    for index, case in enumerate(cases):
        context_ready = bool(case.view.features.get("context_ready", 0))
        if model is not None and context_ready:
            positions.append(index)
            # This detached view still carries its actual, possibly unavailable
            # timing record. Only history readiness is needed for source forecasts.
            views.append(replace(case.view, ready=True, reason="received_history"))
    trajectories = {}
    if views:
        results = model.forecast_batch(views, maximum_steps)
        if len(results) != len(views):
            raise RuntimeError("Model returned an incomplete AoI trajectory batch")
        trajectories = dict(zip(positions, results, strict=True))
    rows = []
    for index, (case, estimates) in enumerate(zip(cases, plans, strict=True)):
        view = case.view
        trajectory = trajectories.get(index)
        context_ready = bool(view.features.get("context_ready", 0))
        base = query_row(view, view.timing.clock.source_clock_id if view.timing and view.timing.clock else "")
        last = float(view.values_mps[-1]) if len(view.values_mps) else None
        # Supervision is read only after every model call and CPU trajectory return.
        reference = float(target_for(case.query_index))
        for estimate in estimates:
            steps = estimate.steps(view.sample_period_us)
            required = estimate.upper_steps if estimate.upper_steps is not None else steps
            prediction = None
            fallback = False
            status, reason = "not_run", ""
            if model is not None:
                if not context_ready:
                    reason = "insufficient_context" if last is not None else "no_received_observation"
                elif not estimate.valid:
                    reason = estimate.reason
                elif required is None or required > maximum_steps:
                    reason = "horizon_limit"
                elif trajectory is None:
                    raise RuntimeError("A usable AoI query has no generated trajectory")
                elif trajectory.status != "ok":
                    reason, status = trajectory.failure_reason, "failed"
                else:
                    prediction = trajectory.value_at_offset(estimate.source_offset_us, last)
                    status = "ok"
                if reason:
                    prediction, fallback = last, True
                    if status != "failed":
                        status = "fallback"
            rows.append({**base, "query_index": case.query_index, "policy": estimate.name,
                "context_ready": context_ready, "timing_valid": estimate.valid, "timing_reason": estimate.reason,
                "aoi_us": estimate.aoi_us, "aoi_lower_us": estimate.lower_us, "aoi_upper_us": estimate.upper_us,
                "native_aoi_us": estimate.native_aoi_us, "source_offset_us": estimate.source_offset_us,
                "native_aoi_lower_us": estimate.native_lower_us, "native_aoi_upper_us": estimate.native_upper_us,
                "horizon_steps": steps, "upper_horizon_steps": estimate.upper_steps,
                "true_aoi_us": case.true_aoi_us, "true_native_aoi_us": case.true_native_aoi_us,
                "true_source_offset_us": case.true_source_offset_us,
                "true_horizon_steps": case.true_horizon_steps, "horizon_exceeded": required is not None and required > maximum_steps,
                "prediction": prediction, "reference": reference, "status": status,
                "fallback": fallback, "failure_reason": reason,
                "model_requested": model is not None,
                "inference_status": trajectory.status if trajectory is not None else "not_run",
                "trajectory_steps": trajectory.values_mps.shape[1] if trajectory is not None else 0,
                "batch_id": batch_id, "batch_size": trajectory.batch_size if trajectory else 0,
                "batch_elapsed_s": trajectory.batch_elapsed_s if trajectory else 0.0,
                "elapsed_s": trajectory.batch_elapsed_s / trajectory.batch_size if trajectory else 0.0,
            })
    return pd.DataFrame(rows)


def eval_aoi(root: Path, args) -> Path:
    require_current_source(args.source)
    delivery = getattr(args, "delivery", False)
    if delivery and not args.with_model:
        raise ValueError("--delivery requires --with-model")
    if delivery and args.batch_size not in (None, 1):
        raise ValueError("--delivery measures individual requests; use --batch-size 1")
    started = perf_counter()
    config = load_aoi_config(root, args.aoi_config)
    split = args.split or config["split"]
    corpus_path = prepared_path(root, args.corpus)
    corpus = PreparedCorpus(corpus_path)
    index = read_json(root / "configs/data_index.json")[config["dataset"]]
    if not args.corpus and corpus.manifest["dataset_id"] != index["dataset_id"]:
        raise ValueError("Prepared corpus identity differs from the frozen dataset index")
    available_profiles = {p["name"]: p for p in corpus.config["arrival"]["profiles"]}
    selected_profiles = args.profile or config["arrival_profiles"]
    available_scenarios = {s["name"]: ClockScenario(**s) for s in config["scenarios"]}
    selected_scenarios = args.scenario or list(available_scenarios)
    for selected, available, name in ((selected_profiles, available_profiles, "arrival profiles"),
                                      (selected_scenarios, available_scenarios, "clock scenarios")):
        if not selected or len(set(selected)) != len(selected) or set(selected) - set(available):
            raise ValueError(f"Choose distinct configured {name}: {list(available)}")
    # Validate split/series selection before creating an output or loading CUDA.
    first_series = next(corpus.iter_series(split, args.series_id, args.max_series), None)
    if first_series is None:
        raise ValueError("No prepared series in the selected split")
    query_stride = args.query_stride or config["query_stride"]
    replicates = args.replicates or config["replicates"]
    resolved = resolved_config(root, args, batch_size=1 if delivery else None)
    policies = DELIVERY_POLICIES if delivery else POLICIES
    selection = {"split": split, "series_ids": args.series_id, "max_series": args.max_series,
        "max_queries_per_unit": args.max_queries, "profiles": selected_profiles, "scenarios": selected_scenarios,
        "query_stride": query_stride, "replicates": replicates, "with_model": args.with_model}
    selection["delivery"] = delivery
    if delivery:
        selection.update({"compute_estimate_ms": args.compute_estimate_ms, "warmup": args.warmup,
                          "queue_us": 0, "coverage_margin_steps": 1, "compute_ewma_alpha": 0.2})
    settings = {"aoi": config, "selection": selection, "model_configuration": resolved,
        "aoi_config_sha256": sha256(project_path(root, args.aoi_config)),
        "dataset_id": corpus.manifest["dataset_id"], "dataset_directory": str(corpus_path), "policies": list(policies)}
    output = new_output(root, ("delivery_" if delivery else "aoi_") + (args.source if args.with_model else "timing"))
    atomic_json(output / "config.json", settings)
    integration = integration_identity(root)
    accumulator = DeliveryAccumulator if delivery else AoiAccumulator
    summary = accumulator()
    by_series = accumulator(("source", "series_id", "route", "group", "replicate", "arrival_profile", "clock_scenario", "policy"))
    estimator = ComputeLatencyEstimator(args.compute_estimate_ms * 1000) if delivery else None
    warmup_metadata = None
    rows_written = cases_written = units = 0
    outcomes = Counter()
    example = []
    example_unit = None
    example_preferred = False
    model = None
    initial_batch_checked = False
    try:
        if args.with_model:
            model = TiRexModel(root, resolved["settings"], resolved["runtime"], args.source)
            scoring = "delivery-time compensation" if delivery else "query-time AoI controls"
            print(f"Loading {args.source} TiRex for {scoring}; output={output}", flush=True)
            model.load()
        batch_size = model.batch_size if model is not None else (args.batch_size or 256)
        with (output / "aoi_queries.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = None
            for series in corpus.iter_series(split, args.series_id, args.max_series):
                for profile_name in selected_profiles:
                    profile = available_profiles[profile_name]
                    count_replicates = 1 if profile["kind"] == "fixed" else replicates
                    for scenario_name in selected_scenarios:
                        scenario = available_scenarios[scenario_name]
                        for replicate in range(count_replicates):
                            identity = {"source": args.source if model is not None else "none",
                                "series_id": series.metadata["series_id"], "route": series.metadata["route"],
                                "group": series.metadata["group"], "split": split, "replicate": replicate,
                                "arrival_profile": profile_name, "clock_scenario": scenario_name}
                            unit_id = fingerprint(identity)[:20]
                            if model is not None:
                                model.reset_for_trace(identity["series_id"], profile_name, scenario_name, replicate)
                            iterator = timing_cases(series, profile, scenario, namespace=corpus.config["dataset_namespace"],
                                seed=config["seed"], replicate=replicate, clock_config=ClockConfig(**config["clock"]),
                                timing_config=TimingConfig(**config["timing"]), sync_interval_us=config["sync_interval_us"],
                                sync_processing_us=config["sync_processing_us"], context_length=config["context_length"],
                                minimum_observations=config["minimum_observations"], query_start_us=config["query_start_us"],
                                query_stride=query_stride, max_queries=args.max_queries)
                            unit_queries = 0
                            batch_index = 0
                            source_rate = 1.0 + scenario.relative_rate_ppm * 1e-6
                            def delivery_reference(index: int, elapsed_us: int) -> float:
                                target = int(series.source_time_us[index]) + elapsed_us / source_rate
                                if target > int(series.source_time_us[-1]) + 1e-6:
                                    return float("nan")
                                position = int(np.searchsorted(series.source_time_us, target + 1e-6, side="right")) - 1
                                return float(series.speed_mps[position])
                            while chunk := list(islice(iterator, batch_size)):
                                if delivery:
                                    case = chunk[0]
                                    if (warmup_metadata is None and case.view.ready and
                                            observation_horizon(case.view, extra_delivery_us=estimator.estimate_us,
                                                margin_steps=1) <= config["maximum_prediction_steps"]):
                                        print(f"Prewarming {args.warmup} unscored delivery requests (first may compile kernels)", flush=True)
                                        warmup_metadata = warmup_delivery(model, case.view, estimator, args.warmup)
                                        model.reset_for_trace(identity["series_id"], profile_name, scenario_name, replicate)
                                    frame = evaluate_delivery_case(case, model=model, estimator=estimator,
                                        target_for=delivery_reference, maximum_steps=config["maximum_prediction_steps"],
                                        batch_id=f"{unit_id}/{batch_index}", source_rate=source_rate)
                                else:
                                    frame = evaluate_chunk(chunk, model=model, target_for=lambda i: series.speed_mps[i],
                                        maximum_steps=config["maximum_prediction_steps"], batch_id=f"{unit_id}/{batch_index}")
                                for name, value in identity.items():
                                    frame[name] = value
                                if model is not None and not initial_batch_checked and (frame.batch_size > 0).any():
                                    if not frame.inference_status.eq("ok").any():
                                        raise RuntimeError("TiRex failed the entire initial AoI batch")
                                    initial_batch_checked = True
                                records = frame.to_dict("records")
                                if writer is None:
                                    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                                    writer.writeheader()
                                writer.writerows(records)
                                summary.update(frame)
                                by_series.update(frame)
                                outcomes.update(frame.status)
                                rows_written += len(frame)
                                cases_written += len(chunk)
                                unit_queries += len(chunk)
                                batch_index += 1
                                preferred = profile_name == "jitter_loss_tail" and scenario_name == "timestamp_noise"
                                if example_unit is None or preferred and not example_preferred:
                                    example = []
                                    example_unit, example_preferred = unit_id, preferred
                                if example_unit == unit_id and len(example) < 6000:
                                    example.extend(records[:6000 - len(example)])
                            if unit_queries == 0:
                                raise ValueError(f"No query after the configured start in series {identity['series_id']}")
                            units += 1
                            if units == 1 or units % 10 == 0:
                                print(f"AoI units={units}, queries={cases_written}, rows={rows_written}, wall_s={perf_counter() - started:.1f}", flush=True)
        if not rows_written or model is not None and not initial_batch_checked:
            raise RuntimeError("No usable query/model trajectory was produced")
        tables, failures = summary.tables()
        series_table, _ = by_series.tables()
        tables.to_csv(output / "metrics_summary.csv", index=False)
        series_table.to_csv(output / "metrics_by_series.csv", index=False)
        failures.to_csv(output / "metrics_timing_reasons.csv", index=False)
        if integration_identity(root) != integration:
            raise RuntimeError("Integration source changed during the AoI run")
        if model is not None and source_hash(model.source_directory) != model.source_sha256:
            raise RuntimeError("TiRex source changed during the AoI run")
        verify_prepared(corpus_path)
        metadata = {"status": "success", "kind": "eval-aoi-delivery" if delivery else "eval-aoi", "selection": selection,
            "case_count": cases_written, "row_count": rows_written, "unit_count": units,
            "status_counts": dict(outcomes), "dataset_id": corpus.manifest["dataset_id"],
            "dataset_manifest_sha256": sha256(corpus_path / "dataset_manifest.json"),
            "dataset_reference": "causal ZOH CAN speed; frozen vehicle/recording-day splits; legacy quarantine preserved",
            "scoring_clock": ("replay query time + measured batch=1 CPU trajectory readiness latency; queue=0; paired methods share delivery time"
                if delivery else "query time; queue, GPU and delivery delay are not added to the target"),
            "clock_measurement": "controlled simulated clocks and four timestamps; not a real-network evaluation",
            "clock_bounds": "nonnegative paths, configured maximum relative frequency and timestamp errors; asymmetric path point bias remains observable",
            "effective_scenario_error_bounds": {name: dict(zip(("sync_timestamp_error_us", "source_timestamp_error_us"),
                available_scenarios[name].error_bounds(ClockConfig(**config["clock"]), TimingConfig(**config["timing"]))))
                for name in selected_scenarios},
            "model_inference": bool(model), "model": model.metadata() if model is not None else None,
            "receiver_context_points": config["context_length"], "policies": list(policies),
            "effective_model_context_points": min(config["context_length"], resolved["settings"]["context_length"]) if model is not None else None,
            "trajectory_policy": ("one shared short trajectory per history; measured AoI upper bound + EWMA compute estimate + one grid-step margin; actual completion determines compensated median selection"
                if delivery else "one trajectory per received history; configured coverage cap sets length without oracle timing; controls select the median with observed-anchor interpolation; no extrapolation"),
            "fallback_policy": "hold last finite received observation and retain scoring; no observation yields missing prediction",
            "truth_attached_after_inference": True, "historical_fixed_references_used": False,
            "integration": integration, "dependencies": dependency_versions(), "executable": sys.executable,
            "wall_time_s": perf_counter() - started}
        if delivery:
            metadata["delivery"] = {"warmup": warmup_metadata,
                "initial_compute_estimate_ms": args.compute_estimate_ms,
                "final_compute_estimate_ms": estimator.estimate_us / 1000,
                "compute_ewma_alpha": 0.2, "coverage_margin_steps": 1, "queue_us": 0,
                "timer": "perf_counter_ns; horizon planning, input preparation, synchronized forecast and CPU trajectory return; receiver/event replay and value selection excluded",
                "reference": "previous-sample ZOH on the frozen 40 Hz corpus; no interpolation of future truth; targets past the series end are unscored",
                "comparison": "paired time selection on one current-model trajectory, with identical history and delivery timestamp"}
        atomic_json(output / "metadata.json", metadata)
        if delivery:
            from .report import write_delivery_report
            write_delivery_report(output, tables, metadata, pd.DataFrame(example))
        else:
            from .report import write_aoi_report
            write_aoi_report(output, tables, metadata, pd.DataFrame(example))
        publish_success(root, output, "delivery" if delivery else "aoi")
        print(f"AoI complete: {output / 'report.html'}", flush=True)
        return output
    except Exception as exc:
        atomic_json(output / "metadata.json", {"status": "failed", "selection": selection,
                                               "error": f"{type(exc).__name__}: {exc}", "integration": integration})
        raise
