"""Packet-event replay. Only ReceiverView is allowed to reach a predictor."""
from collections import deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .common import fingerprint

NEVER = np.iinfo(np.int64).max


def validate_profile(profile: dict) -> None:
    if not isinstance(profile.get("name"), str) or not profile["name"]:
        raise ValueError("Arrival profile needs a name")
    kind = profile.get("kind")
    if kind not in {"fixed", "uniform_trace", "correlated"}:
        raise ValueError(f"Unsupported arrival profile: {kind}")
    integer_fields = ["delay_us"] if kind == "fixed" else ["min_delay_us", "max_delay_us"]
    if kind == "correlated":
        integer_fields += ["base_delay_us", "tail_min_us", "tail_max_us"]
    for field in integer_fields:
        value = profile[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a nonnegative integer number of microseconds")
    if kind != "fixed" and profile["min_delay_us"] > profile["max_delay_us"]:
        raise ValueError("Minimum delay exceeds maximum delay")
    if kind == "correlated":
        if profile["tail_min_us"] > profile["tail_max_us"]:
            raise ValueError("Minimum tail delay exceeds maximum tail delay")
        if not np.isfinite(profile["rho"]) or not -1 < profile["rho"] < 1:
            raise ValueError("Correlated jitter requires -1 < rho < 1")
        if not np.isfinite(profile["innovation_std_us"]) or profile["innovation_std_us"] < 0:
            raise ValueError("Jitter standard deviation must be finite and nonnegative")
        for field in ("drop_probability", "tail_probability"):
            if not np.isfinite(profile[field]) or not 0 <= profile[field] <= 1:
                raise ValueError(f"{field} must be in [0, 1]")


def trace_seed(dataset_namespace: str, split: str, series_id: str, profile: str, replicate: int, seed: int) -> int:
    # A namespace/route identifier, not a hash of future speed labels, determines the trace.
    return int(fingerprint({"dataset": dataset_namespace, "split": split, "series": series_id,
                            "profile": profile, "replicate": replicate, "seed": seed})[:16], 16)


def generate_arrivals(source_time_us: np.ndarray, profile: dict, seed: int) -> np.ndarray:
    """Private simulator schedule; NEVER denotes ultimate loss, never a model feature."""
    validate_profile(profile)
    times = np.asarray(source_time_us, dtype=np.int64)
    generator = np.random.default_rng(seed)
    if profile["kind"] == "fixed":
        delay = np.full(len(times), profile["delay_us"], dtype=np.int64)
        lost = np.zeros(len(times), dtype=bool)
    elif profile["kind"] == "uniform_trace":
        chosen = generator.integers(profile["min_delay_us"], profile["max_delay_us"] + 1)
        delay = np.full(len(times), chosen, dtype=np.int64)
        lost = np.zeros(len(times), dtype=bool)
    else:
        delay = np.empty(len(times), dtype=np.int64)
        lost = np.zeros(len(times), dtype=bool)
        jitter = 0.0
        # Draw packet by packet so extending a trace cannot change an existing prefix.
        for index in range(len(times)):
            jitter = profile["rho"] * jitter + generator.normal(0, profile["innovation_std_us"])
            base = round(np.clip(profile["base_delay_us"] + jitter,
                                 profile["min_delay_us"], profile["max_delay_us"]))
            has_tail = generator.random() < profile["tail_probability"]
            tail = generator.integers(profile["tail_min_us"], profile["tail_max_us"] + 1)
            lost[index] = generator.random() < profile["drop_probability"]
            delay[index] = base + int(tail) if has_tail else base
    if len(times) and (np.any(times < 0) or int(times.max()) > NEVER - int(delay.max()) - 1):
        raise ValueError("Arrival times overflow the integer-microsecond clock")
    arrivals = times + delay
    arrivals[lost] = NEVER
    return arrivals


def _readonly(array: np.ndarray) -> np.ndarray:
    result = array.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class ReceiverView:
    """Predictor input: observed values, masks and observable clocks only."""
    query_time_us: int
    sample_period_us: int
    source_time_us: np.ndarray
    values_mps: np.ndarray
    received_mask: np.ndarray
    received_at_us: np.ndarray       # -1 means not received as of this query.
    native_source_time_us: np.ndarray  # -1 at the same unreceived positions.
    features: Mapping[str, float]
    horizon_steps: int | None
    ready: bool
    reason: str


class PacketReceiver:
    """Driver-side event state. Never pass this object or its schedule to a model."""

    def __init__(self, source_time_us: np.ndarray, values_mps: np.ndarray, arrival_time_us: np.ndarray,
                 sample_period_us: int = 25000, native_source_time_us: np.ndarray | None = None,
                 statistics_packets: int = 64):
        times = np.asarray(source_time_us, dtype=np.int64)
        values = np.asarray(values_mps, dtype=np.float64)
        arrivals = np.asarray(arrival_time_us, dtype=np.int64)
        native = times if native_source_time_us is None else np.asarray(native_source_time_us, dtype=np.int64)
        if (times.ndim != 1 or not len(times) or values.shape != times.shape or arrivals.shape != times.shape
                or native.shape != times.shape or not np.isfinite(values).all()):
            raise ValueError("Receiver needs nonempty, matching source/value/arrival/native arrays")
        if (sample_period_us <= 0 or statistics_packets <= 0 or np.any(np.diff(times) != sample_period_us)
                or np.any(times < 0) or np.any(native < 0) or np.any(native > times)
                or np.any(arrivals < times)):
            raise ValueError("Invalid source grid, native timestamps or arrival times")
        self._times, self._truth, self._arrivals, self._native = [a.copy() for a in (times, values, arrivals, native)]
        self._order = np.argsort(arrivals, kind="stable")
        self._seen = np.zeros(len(times), dtype=bool)
        self._seen_values = np.full(len(times), np.nan)
        self._seen_native = np.full(len(times), -1, dtype=np.int64)
        self._seen_arrivals = np.full(len(times), -1, dtype=np.int64)
        self._recent = deque(maxlen=statistics_packets)
        self._position, self._latest, self._query = 0, -1, -1
        self._period = sample_period_us

    def snapshot(self, query_time_us: int, context_length: int = 256, minimum_observations: int = 32) -> ReceiverView:
        if isinstance(query_time_us, bool) or not isinstance(query_time_us, (int, np.integer)):
            raise ValueError("Query time must be integer microseconds")
        query = int(query_time_us)
        if query < self._query or query >= NEVER or query < 0:
            raise ValueError("Receiver queries must be nondecreasing, finite and nonnegative")
        if context_length < 1 or not 1 <= minimum_observations <= context_length:
            raise ValueError("Invalid receiver context or minimum observation count")
        self._query = query
        while self._position < len(self._order):
            index = int(self._order[self._position])
            arrival = int(self._arrivals[index])
            if arrival > query:
                break
            reordered = index < self._latest
            self._seen[index] = True
            self._seen_values[index] = self._truth[index]
            self._seen_native[index] = self._native[index]
            self._seen_arrivals[index] = arrival
            self._recent.append((int(self._times[index]), arrival, reordered))
            self._latest = max(self._latest, index)
            self._position += 1
        if self._latest < 0:
            empty_int = _readonly(np.array([], dtype=np.int64))
            return ReceiverView(query, self._period, empty_int, _readonly(np.array([], dtype=float)),
                                _readonly(np.array([], dtype=bool)), empty_int, empty_int,
                                MappingProxyType({"received_count": 0.0}), None, False, "no_received_observation")
        end = self._latest + 1
        start = max(0, end - context_length)
        mask = self._seen[start:end]
        recent = np.asarray(self._recent, dtype=np.int64)
        latency = (recent[:, 1] - recent[:, 0]) / 1e6
        interarrival = np.diff(recent[:, 1]) / 1e6
        latest_source = int(self._times[self._latest])
        age = query - latest_source
        features = {
            "observation_age_s": age / 1e6,
            "native_observation_age_s": (query - int(self._seen_native[self._latest])) / 1e6,
            "latest_sample_hold_age_s": (latest_source - int(self._seen_native[self._latest])) / 1e6,
            "time_since_last_receipt_s": (query - int(recent[-1, 1])) / 1e6,
            "received_latency_mean_s": float(latency.mean()),
            "received_latency_std_s": float(latency.std()),
            "received_latency_mad_s": float(np.median(np.abs(latency - np.median(latency)))),
            "received_interarrival_mean_s": float(interarrival.mean()) if len(interarrival) else 0.0,
            "received_interarrival_std_s": float(interarrival.std()) if len(interarrival) else 0.0,
            "observed_reordered_fraction": float(recent[:, 2].mean()),
            "window_unreceived_fraction": float(1.0 - mask.mean()),
            "received_count": float(mask.sum()),
        }
        ready = len(mask) == context_length and int(mask.sum()) >= minimum_observations
        return ReceiverView(query, self._period, _readonly(self._times[start:end]),
                            _readonly(self._seen_values[start:end]), _readonly(mask),
                            _readonly(self._seen_arrivals[start:end]), _readonly(self._seen_native[start:end]),
                            MappingProxyType(features), (age + self._period - 1) // self._period,
                            ready, "ok" if ready else "insufficient_context")


def causal_filled_context(view: ReceiverView) -> np.ndarray:
    """Forward fill from received earlier-source positions inside this window only."""
    result = view.values_mps.copy()
    last = np.nan
    for index, received in enumerate(view.received_mask):
        if received:
            last = result[index]
        else:
            result[index] = last
    return _readonly(result)
