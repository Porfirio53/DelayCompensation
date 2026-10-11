"""Causal clock-domain conversion and observation age, in integer microseconds.

Four timestamps are receiver send, source receive, source send, receiver receive.
The receiver estimates its own monotonic clock as an affine function of the
source's monotonic clock. No operating-system clock is changed. RTT is a clock
filter input, never a substitute for the age of the newest source observation.
"""
from collections import deque
from dataclasses import dataclass
from math import ceil, isfinite

import numpy as np

MAX_TIME_US = np.iinfo(np.int64).max


def integer_us(value, name: str, minimum: int = 0) -> int:
    if (isinstance(value, bool) or not isinstance(value, (int, np.integer))
            or not minimum <= value < MAX_TIME_US):
        raise ValueError(f"{name} must be integer microseconds in [{minimum}, {MAX_TIME_US})")
    return int(value)


def clock_identities(source_clock_id: str, receiver_clock_id: str) -> None:
    if (not isinstance(source_clock_id, str) or not source_clock_id
            or not isinstance(receiver_clock_id, str) or not receiver_clock_id
            or source_clock_id == receiver_clock_id):
        raise ValueError("Provide distinct, nonempty source and receiver clock identities")


@dataclass(frozen=True)
class SyncExchange:
    receiver_send_us: int
    source_receive_us: int
    source_send_us: int
    receiver_receive_us: int

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            integer_us(getattr(self, name), name)
        if self.receiver_receive_us < self.receiver_send_us or self.source_send_us < self.source_receive_us:
            raise ValueError("Synchronization timestamps reverse a local monotonic clock")

    @property
    def round_trip_us(self) -> int:
        return ((self.receiver_receive_us - self.receiver_send_us)
                - (self.source_send_us - self.source_receive_us))


@dataclass(frozen=True)
class ClockConfig:
    window_samples: int = 64
    minimum_samples: int = 3
    minimum_drift_span_us: int = 2_000_000
    filter_bucket_us: int = 500_000
    max_relative_drift_ppm: float = 500.0
    max_round_trip_us: int = 1_000_000
    stale_after_us: int = 3_000_000
    timestamp_error_us: int = 1000

    def __post_init__(self):
        for name in ("window_samples", "minimum_samples", "minimum_drift_span_us",
                     "filter_bucket_us", "max_round_trip_us", "stale_after_us"):
            integer_us(getattr(self, name), name, 1)
        integer_us(self.timestamp_error_us, "timestamp_error_us")
        if self.minimum_samples > self.window_samples:
            raise ValueError("minimum_samples exceeds the clock filter window")
        if (isinstance(self.max_relative_drift_ppm, bool) or not isfinite(self.max_relative_drift_ppm)
                or not 0 <= self.max_relative_drift_ppm < 1_000_000):
            raise ValueError("max_relative_drift_ppm must be finite in [0, 1000000)")


@dataclass(frozen=True)
class AgeInterval:
    estimate_us: float
    lower_us: float
    upper_us: float


@dataclass(frozen=True)
class ClockMapping:
    """Immutable receiver-time snapshot; only completed past exchanges enter it.

    R(S) = receiver_anchor + offset + rate * (S - source_anchor).
    Anchored subtraction preserves microsecond differences with large boot-time
    origins. Bounds do not assume a symmetric path or use fit residuals as a
    bound: they follow from nonnegative one-way delays and the configured clock
    frequency / timestamp error bounds. Their validity depends on those bounds.
    """
    source_clock_id: str
    receiver_clock_id: str
    source_anchor_us: int
    receiver_anchor_us: int
    offset_us: float
    rate: float
    at_receiver_us: int
    last_sync_receiver_us: int
    exchanges: tuple[SyncExchange, ...]
    max_relative_drift_ppm: float
    timestamp_error_us: int
    status: str
    fit_samples: int = 0
    drift_fitted: bool = False
    drift_clipped: bool = False
    fit_residual_rms_us: float = 0.0
    minimum_round_trip_us: float = 0.0
    exact: bool = False

    @property
    def valid(self) -> bool:
        return self.status == "ok"

    @property
    def relative_drift_ppm(self) -> float:
        return (self.rate - 1.0) * 1e6

    def age(self, source_time_us: int, query_time_us: int, *, source_error_us: int = 0,
            query_error_us: int = 0) -> AgeInterval:
        source = integer_us(source_time_us, "source_time_us")
        query = integer_us(query_time_us, "query_time_us")
        integer_us(source_error_us, "source_error_us")
        integer_us(query_error_us, "query_error_us")
        if query < self.at_receiver_us:
            raise ValueError("A clock mapping cannot be used before its receiver snapshot")
        point = self.offset_us + self.rate * (source - self.source_anchor_us)
        if self.exact:
            lower = upper = point
        else:
            if not self.exchanges:
                raise ValueError("No synchronization measurements in this mapping")
            drift = self.max_relative_drift_ppm * 1e-6
            # One receiver timestamp plus one source timestamp, with source
            # error scaled by the worst permitted relative clock rate.
            margin = (2.0 + drift) * self.timestamp_error_us
            lower = max((e.receiver_send_us - self.receiver_anchor_us)
                        + (source - e.source_receive_us)
                        - drift * abs(source - e.source_receive_us) - margin for e in self.exchanges)
            upper = min((e.receiver_receive_us - self.receiver_anchor_us)
                        + (source - e.source_send_us)
                        + drift * abs(source - e.source_send_us) + margin for e in self.exchanges)
            if lower > upper:
                raise ValueError("Clock measurement intervals are inconsistent with the configured bounds")
            point = float(np.clip(point, lower, upper))
        maximum_rate = self.rate if self.exact else 1.0 + self.max_relative_drift_ppm * 1e-6
        error = source_error_us * maximum_rate + query_error_us
        elapsed = query - self.receiver_anchor_us
        return AgeInterval(elapsed - point, elapsed - upper - error, elapsed - lower + error)

    @classmethod
    def known(cls, source_clock_id: str, receiver_clock_id: str, source_anchor_us: int,
              receiver_anchor_us: int, rate: float = 1.0, at_receiver_us: int = 0):
        """Explicit calibrated/oracle mapping, never inferred from packet delays."""
        clock_identities(source_clock_id, receiver_clock_id)
        integer_us(source_anchor_us, "source_anchor_us")
        integer_us(receiver_anchor_us, "receiver_anchor_us")
        integer_us(at_receiver_us, "at_receiver_us")
        if isinstance(rate, bool) or not isfinite(rate) or rate <= 0:
            raise ValueError("A known clock rate must be finite and positive")
        return cls(source_clock_id, receiver_clock_id, source_anchor_us, receiver_anchor_us, 0.0,
                   rate, at_receiver_us, at_receiver_us, (), abs(rate - 1.0) * 1e6,
                   0, "ok", exact=True)


class ClockSynchronizer:
    """Bounded online filter; callers supply completed exchanges in receiver order."""

    def __init__(self, source_clock_id: str, receiver_clock_id: str, config: ClockConfig | None = None):
        clock_identities(source_clock_id, receiver_clock_id)
        self.source_clock_id, self.receiver_clock_id = source_clock_id, receiver_clock_id
        self.config = config or ClockConfig()
        self._exchanges = deque(maxlen=self.config.window_samples)
        self._last_update_us = -1
        self.accepted = self.rejected = 0
        self.last_rejection = ""

    def update(self, exchange: SyncExchange, *, available_at_us: int | None = None) -> bool:
        """available_at_us may include a software measurement/processing delay."""
        available = exchange.receiver_receive_us if available_at_us is None else integer_us(available_at_us, "available_at_us")
        if available < exchange.receiver_receive_us:
            raise ValueError("A synchronization exchange is unavailable before its recorded receipt")
        drift = self.config.max_relative_drift_ppm * 1e-6
        processing = exchange.source_send_us - exchange.source_receive_us
        elapsed = exchange.receiver_receive_us - exchange.receiver_send_us
        minimum_possible_rtt = elapsed - (1.0 - drift) * processing
        error = 2.0 * (2.0 + drift) * self.config.timestamp_error_us
        reason = ""
        if available < self._last_update_us:
            reason = "out_of_order_sync_completion"
        elif minimum_possible_rtt < -error:
            reason = "negative_round_trip"
        elif elapsed - (1.0 + drift) * processing > self.config.max_round_trip_us:
            reason = "round_trip_limit"
        if reason:
            self.rejected += 1
            self.last_rejection = reason
            return False
        self._exchanges.append((exchange, available))
        self._last_update_us = available
        self.accepted += 1
        return True

    def snapshot(self, query_time_us: int, *, estimate_drift: bool = True) -> ClockMapping | None:
        query = integer_us(query_time_us, "query_time_us")
        if query < self._last_update_us:
            raise ValueError("Clock snapshot would contain future synchronization measurements")
        if not self._exchanges:
            return None
        # One minimum-RTT sample per receiver-time bucket preserves temporal
        # coverage; a single globally best RTT cannot estimate clock drift.
        buckets = {}
        origin = self._exchanges[0][0].receiver_receive_us
        for exchange, _ in self._exchanges:
            bucket = (exchange.receiver_receive_us - origin) // self.config.filter_bucket_us
            if bucket not in buckets or exchange.round_trip_us < buckets[bucket].round_trip_us:
                buckets[bucket] = exchange
        selected = list(buckets.values())
        reference = min(selected, key=lambda e: e.round_trip_us)
        source_anchor = reference.source_receive_us
        receiver_anchor = reference.receiver_send_us
        x = np.asarray([(e.source_receive_us - source_anchor) + (e.source_send_us - e.source_receive_us) / 2
                        for e in selected], dtype=np.float64)
        y = np.asarray([(e.receiver_send_us - receiver_anchor) + (e.receiver_receive_us - e.receiver_send_us) / 2
                        for e in selected], dtype=np.float64)
        weights = np.asarray([1.0 / max(1.0, float(e.round_trip_us)) ** 2 for e in selected])
        weights /= weights.sum()
        center_x, center_y = float(weights @ x), float(weights @ y)
        variance = float(weights @ ((x - center_x) ** 2))
        fitted = (estimate_drift and len(selected) >= self.config.minimum_samples
                  and float(np.ptp(x)) >= self.config.minimum_drift_span_us and variance > 0)
        raw_rate = float(weights @ ((x - center_x) * (y - center_y)) / variance) if fitted else 1.0
        drift = self.config.max_relative_drift_ppm * 1e-6
        rate = float(np.clip(raw_rate, 1.0 - drift, 1.0 + drift))
        # Before drift is identifiable, use the lowest-RTT exchange for offset.
        offset = center_y - rate * center_x if fitted else (
            (reference.receiver_receive_us - reference.receiver_send_us)
            - (reference.source_send_us - reference.source_receive_us)) / 2.0
        residual = float(np.sqrt(weights @ ((y - offset - rate * x) ** 2)))
        status = "ok"
        if len(self._exchanges) < self.config.minimum_samples:
            status = "warming_up"
        elif query - self._last_update_us > self.config.stale_after_us:
            status = "stale"
        mapping = ClockMapping(self.source_clock_id, self.receiver_clock_id, source_anchor,
            receiver_anchor, offset, rate, query, self._last_update_us,
            tuple(e for e, _ in self._exchanges), self.config.max_relative_drift_ppm,
            self.config.timestamp_error_us, status, len(selected), bool(fitted), raw_rate != rate,
            residual, max(0.0, min(e.round_trip_us for e in selected)))
        try:
            mapping.age(source_anchor, query)
        except ValueError:
            from dataclasses import replace
            mapping = replace(mapping, status="inconsistent")
        return mapping


def horizon_steps(source_offset_us: float, sample_period_us: int) -> int:
    integer_us(sample_period_us, "sample_period_us", 1)
    if not isfinite(source_offset_us) or source_offset_us < 0:
        raise ValueError("Prediction offset must be finite and nonnegative")
    # The tolerance only absorbs floating-point roundoff at an exact boundary.
    return max(0, ceil((source_offset_us - 1e-6) / sample_period_us))


@dataclass(frozen=True)
class ObservationTiming:
    clock: ClockMapping | None
    age_us: float | None
    age_lower_us: float | None
    age_upper_us: float | None
    native_age_us: float | None
    native_age_lower_us: float | None
    native_age_upper_us: float | None
    source_offset_us: float | None
    horizon_steps: int | None
    lower_horizon_steps: int | None
    upper_horizon_steps: int | None
    valid: bool
    reason: str


def observation_timing(source_time_us: int, native_time_us: int | None, query_time_us: int,
                       sample_period_us: int, clock: ClockMapping | None, *,
                       source_error_us: int = 0, query_error_us: int = 0,
                       maximum_uncertainty_us: int = 100_000) -> ObservationTiming:
    integer_us(maximum_uncertainty_us, "maximum_uncertainty_us", 1)
    if clock is None:
        return ObservationTiming(None, *(None,) * 10, False, "clock_unavailable")
    try:
        age = clock.age(source_time_us, query_time_us, source_error_us=source_error_us,
                        query_error_us=query_error_us)
        native = (clock.age(native_time_us, query_time_us, source_error_us=source_error_us,
                            query_error_us=query_error_us) if native_time_us is not None else None)
    except ValueError:
        return ObservationTiming(clock, *(None,) * 10, False, "clock_inconsistent")
    reason = "ok" if clock.valid else "clock_" + clock.status
    if age.upper_us < 0 or age.estimate_us < 0:
        reason = "negative_observation_age"
    elif age.upper_us - age.lower_us > maximum_uncertainty_us:
        reason = "clock_uncertainty_limit"
    offset = age.estimate_us / clock.rate
    drift = clock.max_relative_drift_ppm * 1e-6
    lower_rate, upper_rate = (clock.rate, clock.rate) if clock.exact else (1.0 - drift, 1.0 + drift)
    lower = max(0.0, age.lower_us) / upper_rate
    upper = max(0.0, age.upper_us) / lower_rate
    return ObservationTiming(clock, age.estimate_us, age.lower_us, age.upper_us,
        native.estimate_us if native else None, native.lower_us if native else None,
        native.upper_us if native else None, offset,
        horizon_steps(offset, sample_period_us) if offset >= 0 else None,
        horizon_steps(lower, sample_period_us), horizon_steps(upper, sample_period_us),
        reason == "ok", reason)
