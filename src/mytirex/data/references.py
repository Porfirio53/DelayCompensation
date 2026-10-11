"""Matching current-source history is an optional development regression reference."""
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import current_source, project_path, read_json, sha256, source_hash, verify_files
from .regression import Sequence, dataset_identity, query_indices
from ..reporting.metrics import summarize

HISTORICAL = "TiRex-2 (current historical)"
QUERY_KEYS = ["sequence", "delay_ms", "query_index"]
ARRIVAL_FIELDS = ["reference", "query_time_s", "target_time_s", "last_source_time_s", "last_observation", "visible_count"]


def select_queries(frame, expected_keys):
    # Same one-to-one query selection as experiment/tirex_reference.py.
    keys = ["method", *QUERY_KEYS]
    if frame.duplicated(keys).any() or expected_keys.duplicated(keys).any():
        raise ValueError("Duplicate cached or requested query keys")
    selected = expected_keys[keys].merge(frame, on=keys, how="left", validate="one_to_one", indicator=True)
    if not selected["_merge"].eq("both").all():
        raise ValueError("Reference predictions do not cover the requested query grid")
    return selected.drop(columns="_merge")


def assert_same_observations(frame, reference):
    paired = frame[QUERY_KEYS + ARRIVAL_FIELDS].merge(
        reference[QUERY_KEYS + ARRIVAL_FIELDS], on=QUERY_KEYS, suffixes=("", "_reference"),
        how="outer", validate="one_to_one", indicator=True)
    if not paired["_merge"].eq("both").all():
        raise ValueError("Reference and current prediction query grids differ")
    if not np.allclose(paired[ARRIVAL_FIELDS].to_numpy(dtype=float),
                       paired[[field + "_reference" for field in ARRIVAL_FIELDS]].to_numpy(dtype=float),
                       rtol=1e-12, atol=1e-12):
        raise ValueError("Reference and current truth/arrival data disagree")


def _validate_full_grid(frame: pd.DataFrame, sequences: list[Sequence], protocol: dict) -> None:
    expected = []
    for sequence in sequences:
        for delay in protocol["delays_ms"]:
            indices = query_indices(sequence, protocol["score_start_s"])
            steps = delay * 1000 // 25000
            expected.append(pd.DataFrame({
                "sequence": sequence.name, "delay_ms": delay, "query_index": indices,
                "reference": sequence.values[indices], "query_time_s": sequence.times_us[indices] / 1e6,
                "target_time_s": sequence.times_us[indices] / 1e6,
                "last_source_time_s": sequence.times_us[indices - steps] / 1e6,
                "last_observation": sequence.values[indices - steps], "visible_count": indices + 1 - steps,
            }))
    expected = pd.concat(expected, ignore_index=True)
    if len(frame) != 70010 or set(frame.method) != {"TiRex-2"} or frame.duplicated(QUERY_KEYS).any():
        raise ValueError("Reference must have the original 70,010 unique TiRex speed queries")
    if not np.isfinite(pd.to_numeric(frame.prediction, errors="coerce")).all():
        raise ValueError("Reference contains nonfinite published predictions")
    assert_same_observations(frame, expected)


def reference_identity(root: Path, name: str, config: dict, settings: dict | None = None,
                       runtime: dict | None = None) -> tuple[dict, dict]:
    """Validate historical identities without resolving any historical paths."""
    if name != "current_speed":
        raise ValueError("Only current-source historical predictions are enabled")
    entry = read_json(root / "configs/references.json")[name]
    if entry["source_id"] != "TiRex" or entry["directory"] != "references/current_speed":
        raise ValueError("Historical predictions must belong to the current TiRex source")
    directory = project_path(root, entry["directory"])
    verify_files(directory, entry["sha256"])
    recorded_config = read_json(directory / "experiment_config.json")
    metadata = read_json(directory / "run_metadata.json")
    data = dataset_identity(root)
    if entry["dataset"] != config["dataset"] or recorded_config["protocol"] != config["protocol"]:
        raise ValueError("Fixed references apply only to the original speed dataset and fixed-delay protocol")
    recorded = metadata["current_baseline_fingerprint"]
    if data["manifest"] != recorded["dataset"] or data["delay_profiles"] != recorded["delay_profiles"]:
        raise ValueError("Reference data identity no longer matches the copied regression data")
    lock = read_json(root / "configs/model_manifest.json")["TiRex-2"]
    if lock != entry["model_lock"] or lock != recorded["model_lock"]["TiRex-2"]:
        raise ValueError("The reference uses a different TiRex checkpoint")
    _, directory_source = current_source(root)
    actual_source = source_hash(directory_source)
    result = {"directory": str(directory), "original_directory": entry["original_directory"],
              "source_sha256": entry["source_sha256"], "source_matches": actual_source == entry["source_sha256"],
              "weights_match": True, "dataset_match": True, "protocol_match": True,
              "sha256": entry["sha256"], "rows": entry["rows"],
              "historical_timings_are_current_single_request_measurements": False}
    snapshot = read_json(directory / "tirex_tuning.json")
    result["settings"] = snapshot["settings"]
    result["settings_match"] = settings == snapshot["settings"] if settings is not None else None
    if snapshot["source_sha256"] != entry["source_sha256"]:
        raise ValueError("Historical current-source snapshot is inconsistent")
    if runtime is not None:
        method = metadata["method_metadata"]["TiRex-2"]
        old_runtime = {k: method[k] for k in ("device", "batch_size", "seed")}
        old_runtime["num_threads"] = recorded_config["baselines"]["TiRex-2"]["num_threads"]
        result["historical_runtime"] = old_runtime
        result["runtime_match"] = old_runtime == runtime
    result["reference_manifest_sha256"] = sha256(root / "configs/references.json")
    return entry, result


def read_controls(root: Path, predictions: pd.DataFrame, sequences: list[Sequence], config: dict,
                  settings: dict, runtime: dict) -> tuple[list[pd.DataFrame], dict]:
    controls, identities = [], {}
    keys = predictions[QUERY_KEYS].copy()
    keys.insert(0, "method", "TiRex-2")
    for name, label in (("current_speed", HISTORICAL),):
        entry = read_json(root / "configs/references.json")[name]
        if not project_path(root, entry["directory"]).is_dir():
            identities[name] = {"used": False, "reason": "Optional imported reference directory is absent"}
            continue
        entry, identity = reference_identity(root, name, config, settings, runtime)
        if not (identity["source_matches"] and identity["settings_match"]):
            identities[name] = {**identity, "used": False,
                                "reason": "Current source/configuration differs; historical predictions are not a reproduction target"}
            continue
        frame = pd.read_csv(project_path(root, entry["directory"]) / "tirex2_predictions.csv",
                            keep_default_na=False, na_values=[""])
        _validate_full_grid(frame, sequences, config["protocol"])
        selected = select_queries(frame, keys)
        assert_same_observations(predictions, selected)
        identity["actual_batch_layout_matches"] = bool(np.array_equal(predictions.batch_size, selected.batch_size))
        identity["runtime_match_definition"] = "Configured device/batch size/seed/threads; actual batch layout is checked separately"
        delta = np.abs(predictions.prediction.to_numpy() - selected.prediction.to_numpy())
        identity["prediction_difference_mps"] = {"max_absolute": float(delta.max()),
                                                "mean_absolute": float(delta.mean())}
        identity["numerical_comparison_note"] = "Batch size can change floating-point rounding; runtime identity is recorded separately."
        selected["method"] = label
        controls.append(selected)
        identities[name] = {**identity, "used": True, "selected_rows": len(selected), "full_query_grid_validated": True}
    return controls, identities


def comparison_table(predictions: pd.DataFrame, controls: list[pd.DataFrame]) -> pd.DataFrame:
    # Reuse the established scorer, retaining every finite failure fallback.
    columns = ["method", "delay_ms", "query_count", "MAE", "RMSE", "Bias", "abs_error_p95", "abs_error_p99",
               "failure_count", "fallback_count"]
    table = summarize(pd.concat([predictions, *controls], ignore_index=True))[columns]
    current = table.loc[table.method == predictions.method.iloc[0], ["delay_ms", "MAE", "RMSE"]]
    table = table.merge(current, on="delay_ms", suffixes=("", "_new_run"), validate="many_to_one")
    table["MAE_delta_to_new_run"] = table.MAE - table.MAE_new_run
    table["RMSE_delta_to_new_run"] = table.RMSE - table.RMSE_new_run
    return table.drop(columns=["MAE_new_run", "RMSE_new_run"])
