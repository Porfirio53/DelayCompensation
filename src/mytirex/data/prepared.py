"""Existing causal CAN readers and prepared corpus; no training or feature caches."""
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import re
import numpy as np
from ..common import fingerprint, sha256
from .arrival import PacketReceiver, ReceiverView, _readonly, generate_arrivals, trace_seed

SPLITS = ("train", "validation", "test")
ROUTE = re.compile(r"^(?P<vehicle>[A-Za-z0-9_-]+)\|(?P<day>\d{4}-\d{2}-\d{2})--\d{2}-\d{2}-\d{2}$")

def route_group(route: str) -> str:
    match = ROUTE.fullmatch(route)
    if not match:
        raise ValueError(f"Unsupported Comma2k19 route identifier: {route}")
    return f"{match['vehicle']}|{match['day']}"


def legacy_routes(manifest: Path) -> set[str]:
    with manifest.open(newline="") as stream:
        return {row["route"] for row in csv.DictReader(stream)}


def split_groups(groups, excluded_groups, ratios: dict, seed: int, frozen: dict | None = None) -> dict:
    """Assign before windowing. A reused manifest never moves an existing group."""
    if set(ratios) != set(SPLITS) or any(not 0 < float(v) < 1 for v in ratios.values()):
        raise ValueError("Specify positive train/validation/test ratios")
    if not np.isclose(sum(ratios.values()), 1.0, atol=1e-9, rtol=0):
        raise ValueError("Split ratios must sum to one")
    eligible = set(groups) - set(excluded_groups)
    if len(eligible) < 3:
        raise ValueError("Need at least 3 new vehicle-recording-day groups after legacy quarantine")
    assigned = dict(frozen or {})
    if any(value not in (*SPLITS, "legacy_development") for value in assigned.values()):
        raise ValueError("Invalid frozen split assignment")
    for group in set(groups) & set(excluded_groups):
        if group in assigned and assigned[group] != "legacy_development":
            raise ValueError(f"Previously assigned group now overlaps legacy development: {group}")
        assigned[group] = "legacy_development"
    if any(assigned.get(g) == "legacy_development" for g in eligible):
        raise ValueError("Cannot release a quarantined group into training/evaluation")
    ordered = sorted(eligible - assigned.keys(), key=lambda g: fingerprint({"seed": seed, "group": g}))
    all_eligible = eligible | {g for g, s in assigned.items() if s in SPLITS}
    n = len(all_eligible)
    validation = max(1, round(n * ratios["validation"]))
    test = max(1, round(n * ratios["test"]))
    while validation + test >= n:
        if validation >= test and validation > 1:
            validation -= 1
        else:
            test -= 1
    targets = {"train": n - validation - test, "validation": validation, "test": test}
    counts = {s: sum(v == s for v in assigned.values()) for s in SPLITS}
    for group in ordered:
        selected = max(SPLITS, key=lambda s: targets[s] - counts[s])
        assigned[group] = selected
        counts[selected] += 1
    return dict(sorted(assigned.items()))


@dataclass(frozen=True)
class RawSegment:
    route: str
    group: str
    segment: int
    times_path: Path
    values_path: Path


def array_path(directory: Path, name: str) -> Path | None:
    return next((p for p in (directory / name, directory / (name + ".npy")) if p.is_file()), None)


def discover_segments(raw: Path) -> list[RawSegment]:
    segments = []
    seen = set()
    for processed in sorted(raw.rglob("processed_log")):
        if not processed.is_dir():
            continue
        route = next((p for p in reversed(processed.parts) if ROUTE.fullmatch(p)), None)
        if route is None or not processed.parent.name.isdigit():
            raise ValueError(f"Cannot identify route/segment from {processed}")
        for signal in ("speed", "car_speed"):
            directory = processed / "CAN" / signal
            times, values = array_path(directory, "t"), array_path(directory, "value")
            if times is not None and values is not None:
                key = (route, int(processed.parent.name))
                if key in seen:
                    raise ValueError(f"Duplicated raw route/segment: {key}")
                seen.add(key)
                segments.append(RawSegment(route, route_group(route), key[1], times, values))
                break
    if not segments:
        raise ValueError("No raw CAN speed t/value arrays found. Supply existing extracted Comma2k19 arrays")
    return segments


def read_speed(segment: RawSegment) -> tuple[np.ndarray, np.ndarray, dict]:
    times = np.asarray(np.load(segment.times_path, allow_pickle=False), dtype=np.float64)
    values = np.asarray(np.load(segment.values_path, allow_pickle=False), dtype=np.float64)
    if times.ndim == 2 and times.shape[1] == 1:
        times = times[:, 0]
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if times.ndim != 1 or values.ndim != 1 or len(times) != len(values):
        raise ValueError("CAN speed must have matching scalar t/value arrays")
    valid = np.isfinite(times) & np.isfinite(values) & (times >= 0)
    if not valid.any():
        raise ValueError("No finite, nonnegative native timestamps with finite speed")
    times, values = times[valid], values[valid]
    if np.max(times) >= np.iinfo(np.int64).max / 1e6:
        raise ValueError("Native timestamps cannot be represented in integer microseconds")
    times_us = np.rint(times * 1e6).astype(np.int64)
    order = np.argsort(times_us, kind="stable")
    times_us, values = times_us[order], values[order]
    keep = np.r_[times_us[1:] != times_us[:-1], True]  # Last record at a duplicate source timestamp.
    return times_us[keep], values[keep], {
        "native_records": len(valid), "removed_invalid_records": int((~valid).sum()),
        "duplicate_timestamp_records": int((~keep).sum()),
        "native_time_unit": "boot_seconds", "speed_unit": "m/s",
        "deduplication": "stable_sort_then_last_record_at_same_integer_microsecond"
    }


def causal_blocks(times_us: np.ndarray, values: np.ndarray, period_us: int, max_gap_us: int):
    """ZOH from earlier native records only; never cross a large native gap."""
    cuts = np.r_[0, np.flatnonzero(np.diff(times_us) > max_gap_us) + 1, len(times_us)]
    for start, stop in zip(cuts[:-1], cuts[1:]):
        times, speed = times_us[start:stop], values[start:stop]
        first = ((int(times[0]) + period_us - 1) // period_us) * period_us
        last = (int(times[-1]) // period_us) * period_us
        if last < first:
            continue
        grid = np.arange(first, last + 1, period_us, dtype=np.int64)
        previous = np.searchsorted(times, grid, side="right") - 1
        yield grid, speed[previous], times[previous]


def verify_prepared(directory: Path) -> dict:
    manifest = json.loads((directory / "dataset_manifest.json").read_text())
    identity = manifest["identity"]
    if fingerprint(identity) != manifest["dataset_id"]:
        raise ValueError("Prepared dataset identity is damaged")
    for name in ("split_manifest", "inventory"):
        if sha256(directory / (name + ".json")) != identity[name + "_sha256"]:
            raise ValueError(f"Prepared {name} checksum mismatch")
    split_manifest = json.loads((directory / "split_manifest.json").read_text())
    inventory = json.loads((directory / "inventory.json").read_text())
    routes, groups = {}, {}
    for series in inventory["series"]:
        file = (directory / series["file"]).resolve()
        if not file.is_relative_to(directory.resolve()) or sha256(file) != series["sha256"]:
            raise ValueError(f"Prepared series checksum/path mismatch: {series['series_id']}")
        for container, key in ((routes, series["route"]), (groups, series["group"])):
            if key in container and container[key] != series["split"]:
                raise ValueError("Route/day crosses dataset splits")
            container[key] = series["split"]
        if split_manifest["assignments"][series["group"]] != series["split"]:
            raise ValueError("Series does not match the frozen split")
        if series["group"] in split_manifest["legacy_groups"] or series["route"] in split_manifest["legacy_routes"]:
            raise ValueError("Legacy development data leaked into a held-out corpus split")
    return manifest


@dataclass(frozen=True)
class ScoringExample:
    view: ReceiverView
    target_mps: float  # Scorer only; the predictor receives example.view.
    series_id: str
    split: str
    profile_name: str  # Replay metadata, not a predictor feature.
    replicate: int
    query_index: int


@dataclass(frozen=True)
class PreparedSeries:
    metadata: dict
    source_time_us: np.ndarray
    speed_mps: np.ndarray  # Source/scorer only; predictors receive ReceiverView.
    native_time_us: np.ndarray


class PreparedCorpus:
    def __init__(self, directory: Path, verify: bool = True):
        self.directory = Path(directory)
        self.manifest = (verify_prepared(self.directory) if verify else
                         json.loads((self.directory / "dataset_manifest.json").read_text()))
        self.config = self.manifest["identity"]["config"]
        self.inventory = json.loads((self.directory / "inventory.json").read_text())["series"]

    def iter_series(self, split: str, series_ids: list[str] | None = None, max_series: int | None = None):
        """Load existing blocks for event replay; retain the frozen day-group split."""
        if split not in SPLITS:
            raise ValueError(f"Choose a prepared-corpus split from {SPLITS}")
        if max_series is not None and (isinstance(max_series, bool) or not isinstance(max_series, int) or max_series < 1):
            raise ValueError("max_series must be a positive integer")
        eligible = {s["series_id"] for s in self.inventory if s["split"] == split}
        if series_ids is not None and (not series_ids or len(set(series_ids)) != len(series_ids)
                                       or set(series_ids) - eligible):
            raise ValueError("Choose distinct series_ids belonging to the selected frozen split")
        selected = [s for s in sorted(self.inventory, key=lambda s: s["series_id"])
                    if s["split"] == split and (series_ids is None or s["series_id"] in series_ids)]
        period = self.config["resampling"]["sample_period_us"]
        for series in selected[:max_series]:
            with np.load(self.directory / series["file"], allow_pickle=False) as arrays:
                times, values, native = (arrays[k] for k in ("source_time_us", "speed_mps", "native_time_us"))
            if (times.ndim != 1 or len(times) < 2 or values.shape != times.shape or native.shape != times.shape
                    or times.dtype != np.int64 or native.dtype != np.int64 or np.any(times < 0)
                    or np.any(np.diff(times) != period) or not np.isfinite(values).all()
                    or np.any(native < 0) or np.any(native > times)):
                raise ValueError(f"Invalid prepared source grid/values/native timestamps: {series['series_id']}")
            yield PreparedSeries(dict(series), _readonly(times), _readonly(values), _readonly(native))

    def iter_examples(self, split: str, profiles: list[str] | None = None, max_examples: int | None = None):
        if split not in SPLITS:
            raise ValueError(f"Choose a prepared-corpus split from {SPLITS}")
        if max_examples is not None and max_examples < 1:
            raise ValueError("max_examples must be positive")
        protocol = self.config["arrival"]
        available = {p["name"]: p for p in protocol["profiles"]}
        selected = list(available) if profiles is None else profiles
        if not selected or len(set(selected)) != len(selected) or any(name not in available for name in selected):
            raise ValueError("Choose distinct configured arrival profile names")
        windows = self.config["windows"]
        period = self.config["resampling"]["sample_period_us"]
        stride = windows["train_query_stride"] if split == "train" else windows["evaluation_query_stride"]
        first_query = (round(windows["query_start_s"] * 1e6) + period - 1) // period
        published = 0
        for series in sorted(self.inventory, key=lambda s: s["series_id"]):
            if series["split"] != split:
                continue
            with np.load(self.directory / series["file"], allow_pickle=False) as arrays:
                times, values, native = (arrays[k] for k in ("source_time_us", "speed_mps", "native_time_us"))
            for profile_name in selected:
                profile = available[profile_name]
                # Fixed-delay replicas would duplicate the same inputs and labels exactly.
                replicates = 1 if profile["kind"] == "fixed" else protocol["replicates"]
                for replicate in range(replicates):
                    seed = trace_seed(self.config["dataset_namespace"], split, series["series_id"],
                                      profile_name, replicate, protocol["seed"])
                    arrivals = generate_arrivals(times, profile, seed)
                    receiver = PacketReceiver(times, values, arrivals, period, native,
                                              protocol["receiver_statistics_packets"])
                    for query_index in range(first_query, len(times), stride):
                        view = receiver.snapshot(int(times[query_index]), windows["context_length"],
                                                 windows["minimum_observations"])
                        if not view.ready:
                            continue
                        # Materialize the model-safe view before attaching the supervision label.
                        yield ScoringExample(view, float(values[query_index]), series["series_id"], split,
                                              profile_name, replicate, query_index)
                        published += 1
                        if max_examples is not None and published >= max_examples:
                            return


def check_examples(directory: Path) -> dict:
    """A few examples per split/profile, with no inference, training or target metrics."""
    corpus = PreparedCorpus(directory)
    report = {}
    for split in SPLITS:
        report[split] = {}
        for profile in corpus.config["arrival"]["profiles"]:
            examples = list(corpus.iter_examples(split, [profile["name"]], max_examples=3))
            if not examples:
                raise ValueError(f"No ready example for {split}/{profile['name']}")
            report[split][profile["name"]] = {
                "checked_examples": len(examples), "context_points": len(examples[0].view.values_mps),
                "first_horizon_steps": examples[0].view.horizon_steps,
                "first_observation_age_s": examples[0].view.features["observation_age_s"],
                "first_unreceived_fraction": examples[0].view.features["window_unreceived_fraction"],
                "predictor_receives_target_truth": False}
    return report


