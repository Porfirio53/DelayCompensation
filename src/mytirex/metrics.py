"""Existing MAE/RMSE/bias/tail metrics; finite failure fallbacks stay in scoring."""
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

def numeric_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    """Read a numeric field, preserving missing entries as NaN."""
    if name not in frame:
        return np.full(len(frame), np.nan, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)


def boolean_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    """Read booleans from in-memory results or round-tripped CSV files."""
    if name not in frame:
        return np.zeros(len(frame), dtype=bool)
    values = frame[name]
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).to_numpy(dtype=bool)
    return values.fillna("").astype(str).str.strip().str.lower().isin(
        {"true", "1", "1.0", "yes"}
    ).to_numpy(dtype=bool)


def interval_mask(frame: pd.DataFrame) -> np.ndarray:
    """Identify published, finite, ordered intervals."""
    lower = numeric_column(frame, "lower")
    upper = numeric_column(frame, "upper")
    finite = np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)
    return finite & boolean_column(frame, "valid_interval")


def _percentile(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, percentile)) if finite.size else float("nan")


def _failure_mask(frame: pd.DataFrame) -> np.ndarray:
    reference = numeric_column(frame, "reference")
    prediction = numeric_column(frame, "prediction")
    failures = ~np.isfinite(reference) | ~np.isfinite(prediction) | boolean_column(frame, "fallback")
    if "failure_reason" in frame:
        reason = frame["failure_reason"].fillna("").astype(str).str.strip()
        failures |= (~reason.str.lower().isin({"", "none", "nan", "null"})).to_numpy()
    if "status" in frame:
        status = frame["status"].fillna("").astype(str).str.lower()
        failures |= status.isin({
            "error", "failed", "failure", "missing", "timeout",
            "numerical_rejection", "infeasible", "fallback",
        }).to_numpy()
    return failures


def _batch_timings(frame: pd.DataFrame) -> np.ndarray:
    if "batch_elapsed_s" not in frame:
        return np.array([], dtype=float)
    timed = frame
    if "batch_id" in frame:
        keys = [key for key in ("method", "sequence", "delay_ms", "batch_id") if key in frame]
        timed = frame.drop_duplicates(keys)
    elapsed = numeric_column(timed, "batch_elapsed_s")
    return elapsed[np.isfinite(elapsed) & (elapsed >= 0)]


def _score_group(frame: pd.DataFrame) -> dict[str, float | int]:
    count = len(frame)
    reference = numeric_column(frame, "reference")
    prediction = numeric_column(frame, "prediction")
    scored = np.isfinite(reference) & np.isfinite(prediction)
    errors = prediction[scored] - reference[scored]
    absolute = np.abs(errors)
    fallback = boolean_column(frame, "fallback")
    failures = _failure_mask(frame)
    elapsed = numeric_column(frame, "elapsed_s")
    elapsed = elapsed[np.isfinite(elapsed) & (elapsed >= 0)]
    batch_elapsed = _batch_timings(frame)
    row: dict[str, float | int] = {
        "query_count": count,
        "scored_count": int(np.count_nonzero(scored)),
        "MAE": float(np.mean(absolute)) if absolute.size else float("nan"),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))) if errors.size else float("nan"),
        "Bias": float(np.mean(errors)) if errors.size else float("nan"),
        "abs_error_p95": _percentile(absolute, 95),
        "abs_error_p99": _percentile(absolute, 99),
        "inference_p50_s": _percentile(elapsed, 50),
        "inference_p95_s": _percentile(elapsed, 95),
        "inference_p99_s": _percentile(elapsed, 99),
        "batch_elapsed_p50_s": _percentile(batch_elapsed, 50),
        "batch_elapsed_p95_s": _percentile(batch_elapsed, 95),
        "batch_elapsed_p99_s": _percentile(batch_elapsed, 99),
        "failure_count": int(np.count_nonzero(failures)),
        "failure_rate": float(np.mean(failures)) if count else float("nan"),
        "fallback_count": int(np.count_nonzero(fallback)),
    }
    return row


def summarize(
    frame: pd.DataFrame, by: Iterable[str] = ("method", "delay_ms")
) -> pd.DataFrame:
    """Summarize all query rows without thinning or removing finite fallbacks.

    Error units are m/s and inference-time units are seconds. The supplied
    elapsed_s field may be an amortized per-query value for batch inference;
    the runner records that timing convention separately.
    """
    keys = list(by)
    missing = [key for key in keys if key not in frame]
    if missing:
        raise ValueError(f"Missing grouping columns: {missing}")
    if frame.empty:
        return pd.DataFrame(columns=keys)
    result = []
    grouper = keys[0] if len(keys) == 1 else keys
    for group_key, group in frame.groupby(grouper, sort=True, dropna=False):
        values = (group_key,) if len(keys) == 1 else tuple(group_key)
        result.append(dict(zip(keys, values)) | _score_group(group))
    return pd.DataFrame(result)


def write_metrics(frame: pd.DataFrame, output_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Write aggregate and per-sequence metrics CSVs and return both tables."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    failures = frame.loc[_failure_mask(frame)].copy()
    for key in ("status", "failure_reason"):
        if key not in failures:
            failures[key] = ""
        failures[key] = failures[key].fillna("").astype(str)
    failures["fallback"] = boolean_column(failures, "fallback").astype(int)
    breakdown_keys = ["method", "delay_ms", "status", "failure_reason"]
    breakdown = failures.groupby(breakdown_keys, sort=True, dropna=False).agg(
        query_count=("method", "size"), fallback_count=("fallback", "sum")
    ).reset_index()
    tables = {
        "summary": summarize(frame),
        "by_sequence": summarize(frame, ("method", "sequence", "delay_ms")),
        "failure_reasons": breakdown,
    }
    tables["summary"].to_csv(destination / "metrics_summary.csv", index=False)
    tables["by_sequence"].to_csv(destination / "metrics_by_sequence.csv", index=False)
    tables["failure_reasons"].to_csv(destination / "metrics_failure_reasons.csv", index=False)
    return tables

