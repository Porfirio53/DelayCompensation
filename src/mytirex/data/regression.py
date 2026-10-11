"""The unchanged legacy sampling protocol, with detached receiver-safe views."""
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from .arrival import ReceiverView, _readonly
from ..common import project_path, read_json, verify_files


@dataclass(frozen=True)
class Sequence:
    name: str
    times_us: np.ndarray
    values: np.ndarray  # Scorer/replay only; never passed to the predictor.

    def request(self, query_index: int, delay_us: int) -> ReceiverView:
        if (isinstance(query_index, bool) or not isinstance(query_index, (int, np.integer))
                or not 0 <= query_index < len(self.times_us)):
            raise ValueError("query_index must identify a source-grid point")
        if isinstance(delay_us, bool) or not isinstance(delay_us, (int, np.integer)) or delay_us < 0:
            raise ValueError("delay_us must be nonnegative integer microseconds")
        query_us = int(self.times_us[query_index])
        # Retained from dataset_api.Sequence.request: arrival at q is usable.
        end = int(np.searchsorted(self.times_us, query_us - int(delay_us), side="right"))
        times = _readonly(self.times_us[:end])
        values = _readonly(self.values[:end])
        period = int(self.times_us[1] - self.times_us[0])
        age = query_us - int(times[-1]) if end else None
        features = {"received_count": float(end), "window_unreceived_fraction": 0.0}
        if end:
            features["observation_age_s"] = age / 1e6
        return ReceiverView(query_us, period, times, values, _readonly(np.ones(end, dtype=bool)),
                            _readonly(times + int(delay_us)),
                            # Native CAN timestamps cannot be recovered from interpolated CSV.
                            _readonly(np.full(end, -1, dtype=np.int64)), MappingProxyType(features),
                            (age + period - 1) // period if age is not None else None,
                            end >= 2, "ok" if end >= 2 else "insufficient_context")


def load_dataset(root: Path, dataset: str = "regression_speed_v1") -> list[Sequence]:
    if dataset != "regression_speed_v1":
        raise ValueError("eval-speed scores only legacy interpolated regression data; ZOH corpus has a different target definition")
    entry = read_json(root / "configs/data_index.json")[dataset]
    directory = project_path(root, entry["directory"])
    verify_files(directory, entry["sha256"])
    manifest = read_json(directory / "dataset_manifest.json")
    verify_files(directory, manifest["sha256"])
    frame = pd.read_csv(directory / "speed_segments.csv")
    times = frame.pop("time").to_numpy(dtype=np.float64)
    times_us = np.rint(times * 1_000_000).astype(np.int64)
    period_us = int(manifest["sample_period_us"])
    if not np.allclose(times * 1_000_000, times_us, atol=1e-5, rtol=0):
        raise ValueError("Source timestamps must lie on the integer microsecond clock")
    if (len(times_us) != manifest["samples_per_sequence"] or not np.all(np.diff(times_us) == period_us)
            or period_us != 25000 or len(times_us) != 7201 or times_us[0] != 0 or times_us[-1] != 180_000_000):
        raise ValueError("Unexpected legacy source sampling grid")
    if list(frame.columns) != manifest["sequences"] or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Invalid sequence columns or nonfinite reference values")
    if len(pd.read_csv(directory / "segments_manifest.csv")) != len(frame.columns) or len(frame.columns) != 5:
        raise ValueError("Segment provenance manifest does not match the five regression sequences")
    return [Sequence(name, _readonly(times_us), _readonly(frame[name].to_numpy(dtype=np.float64))) for name in frame]


def query_indices(sequence: Sequence, score_start_s: float = 5.0, limit: int | None = None) -> np.ndarray:
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("max_queries must be a positive integer")
    indices = np.flatnonzero(sequence.times_us >= round(score_start_s * 1_000_000))
    return indices if limit is None else indices[:limit]


def dataset_identity(root: Path) -> dict:
    entry = read_json(root / "configs/data_index.json")["regression_speed_v1"]
    directory = project_path(root, entry["directory"])
    return {"id": "regression_speed_v1", "directory": str(directory),
            "reference_definition": entry["reference_definition"], "sha256": entry["sha256"],
            "manifest": read_json(directory / "dataset_manifest.json"),
            "delay_profiles": read_json(directory / "delay_profiles.json")}
