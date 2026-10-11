"""Streaming query counts with exact percentiles for timing and speed errors."""
from collections import Counter

import numpy as np
import pandas as pd

from .metrics import boolean_column, numeric_column

GROUPS = ("source", "arrival_profile", "clock_scenario", "policy")


def error_statistics(errors: np.ndarray, prefix: str, scale: float = 1.0) -> dict:
    values = errors / scale
    absolute = np.abs(values)
    return {
        prefix + "count": len(values),
        prefix + "MAE": float(absolute.mean()) if len(values) else None,
        prefix + "RMSE": float(np.sqrt(np.mean(values ** 2))) if len(values) else None,
        prefix + "Bias": float(values.mean()) if len(values) else None,
        prefix + "abs_P95": float(np.percentile(absolute, 95)) if len(values) else None,
        prefix + "abs_P99": float(np.percentile(absolute, 99)) if len(values) else None,
    }


class AoiAccumulator:
    """Retain numeric error chunks, not full query rows or duplicate contexts."""

    def __init__(self, groups=GROUPS):
        self.groups = tuple(groups)
        self._states = {}

    def update(self, frame: pd.DataFrame) -> None:
        for key, group in frame.groupby(list(self.groups), sort=False, dropna=False):
            state = self._states.setdefault(tuple(key), {"counts": Counter(), "reasons": Counter(),
                "chunks": {name: [] for name in ("aoi", "usable_aoi", "native_aoi", "speed", "width")}})
            counts, chunks = state["counts"], state["chunks"]
            counts["query_count"] += len(group)
            valid = boolean_column(group, "timing_valid")
            counts["timing_usable_count"] += int(valid.sum())
            counts["context_ready_count"] += int(boolean_column(group, "context_ready").sum())
            counts["fallback_count"] += int(boolean_column(group, "fallback").sum())
            counts["model_failure_count"] += int(group["status"].eq("failed").sum())
            counts["horizon_exceeded_count"] += int(boolean_column(group, "horizon_exceeded").sum())
            counts["model_requested_count"] += int(boolean_column(group, "model_requested").sum())
            counts["model_success_count"] += int(group["status"].eq("ok").sum())
            state["reasons"].update(group.loc[~valid, "timing_reason"])
            error = numeric_column(group, "aoi_us") - numeric_column(group, "true_aoi_us")
            finite = np.isfinite(error)
            chunks["aoi"].append(error[finite])
            chunks["usable_aoi"].append(error[finite & valid])
            native_error = numeric_column(group, "native_aoi_us") - numeric_column(group, "true_native_aoi_us")
            chunks["native_aoi"].append(native_error[np.isfinite(native_error)])
            speed_error = numeric_column(group, "prediction") - numeric_column(group, "reference")
            chunks["speed"].append(speed_error[np.isfinite(speed_error)])
            counts["missing_prediction_count"] += int((boolean_column(group, "model_requested") & ~np.isfinite(speed_error)).sum())
            lower, upper, truth = (numeric_column(group, name) for name in ("aoi_lower_us", "aoi_upper_us", "true_aoi_us"))
            intervals = np.isfinite(lower) & np.isfinite(upper) & np.isfinite(truth) & (lower <= upper)
            counts["interval_count"] += int(intervals.sum())
            counts["interval_covered_count"] += int((intervals & (lower <= truth) & (truth <= upper)).sum())
            chunks["width"].append((upper - lower)[intervals])
            native_lower, native_upper, native_truth = (numeric_column(group, name)
                for name in ("native_aoi_lower_us", "native_aoi_upper_us", "true_native_aoi_us"))
            native_intervals = (np.isfinite(native_lower) & np.isfinite(native_upper) & np.isfinite(native_truth)
                                & (native_lower <= native_upper))
            counts["native_interval_count"] += int(native_intervals.sum())
            counts["native_interval_covered_count"] += int((native_intervals & (native_lower <= native_truth)
                                                          & (native_truth <= native_upper)).sum())
            estimated, conservative, actual = (numeric_column(group, name)
                for name in ("horizon_steps", "upper_horizon_steps", "true_horizon_steps"))
            comparable = np.isfinite(estimated) & np.isfinite(actual)
            counts["horizon_count"] += int(comparable.sum())
            counts["horizon_match_count"] += int((comparable & (estimated == actual)).sum())
            counts["horizon_underestimate_count"] += int((comparable & (estimated < actual)).sum())
            bounded = np.isfinite(conservative) & np.isfinite(actual)
            counts["upper_horizon_count"] += int(bounded.sum())
            counts["upper_horizon_underestimate_count"] += int((bounded & (conservative < actual)).sum())

    def tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows, reasons = [], []
        for key, state in sorted(self._states.items()):
            identity = dict(zip(self.groups, key))
            counts = state["counts"]
            row = {**identity, **{name: int(counts[name]) for name in (
                "query_count", "timing_usable_count", "context_ready_count", "fallback_count", "model_failure_count",
                "horizon_exceeded_count", "model_requested_count", "model_success_count", "missing_prediction_count",
                "interval_count", "interval_covered_count", "horizon_count", "horizon_match_count",
                "horizon_underestimate_count", "upper_horizon_count", "upper_horizon_underestimate_count",
                "native_interval_count", "native_interval_covered_count")}}
            arrays = {name: np.concatenate(chunks) if chunks else np.array([], dtype=float)
                      for name, chunks in state["chunks"].items()}
            row.update(error_statistics(arrays["aoi"], "aoi_ms_", 1000))
            row.update(error_statistics(arrays["usable_aoi"], "usable_aoi_ms_", 1000))
            row.update(error_statistics(arrays["native_aoi"], "native_aoi_ms_", 1000))
            row.update(error_statistics(arrays["speed"], "speed_mps_"))
            row["interval_coverage"] = (counts["interval_covered_count"] / counts["interval_count"]
                                         if counts["interval_count"] else None)
            row["interval_width_P95_ms"] = (float(np.percentile(arrays["width"], 95) / 1000)
                                             if len(arrays["width"]) else None)
            row["native_interval_coverage"] = (counts["native_interval_covered_count"] / counts["native_interval_count"]
                                                if counts["native_interval_count"] else None)
            row["horizon_match_rate"] = counts["horizon_match_count"] / counts["horizon_count"] if counts["horizon_count"] else None
            rows.append(row)
            reasons.extend({**identity, "reason": reason, "count": count} for reason, count in sorted(state["reasons"].items()))
        return pd.DataFrame(rows), pd.DataFrame(reasons, columns=[*self.groups, "reason", "count"])
