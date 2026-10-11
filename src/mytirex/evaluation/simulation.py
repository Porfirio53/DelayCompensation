"""Private experiment clocks and event schedules; never predictor features."""
from dataclasses import dataclass, replace
from math import ceil, isfinite

import numpy as np

from ..data.arrival import NEVER, ReceiverView, generate_arrivals, trace_seed
from ..data.prepared import PreparedSeries
from ..timing.receiver import TimestampedPacket, TimestampedReceiver, TimingConfig
from ..timing.clock import ClockConfig, ClockSynchronizer, ObservationTiming, SyncExchange, horizon_steps, integer_us, observation_timing


@dataclass(frozen=True)
class ClockScenario:
    name: str
    offset_us: int = 500000
    relative_rate_ppm: float = 0.0
    uplink_us: int = 5000
    downlink_us: int = 5000
    network_jitter_us: int = 2000
    measurement_error_us: int = 0
    source_stamp_bias_us: int = 0
    timestamp_error_bound_us: int | None = None
    source_error_bound_us: int | None = None
    sync_drop_probability: float = 0.0
    outage_start_s: float | None = None
    outage_duration_s: float = 0.0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Clock scenario needs a name")
        for name in ("offset_us", "source_stamp_bias_us", "uplink_us", "downlink_us",
                     "network_jitter_us", "measurement_error_us"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or (name not in {"offset_us", "source_stamp_bias_us"} and value < 0):
                raise ValueError(f"Invalid integer clock scenario field: {name}")
        if not isfinite(self.relative_rate_ppm) or abs(self.relative_rate_ppm) >= 1_000_000:
            raise ValueError("Relative clock rate must stay positive and below 2")
        if not isfinite(self.sync_drop_probability) or not 0 <= self.sync_drop_probability <= 1:
            raise ValueError("Synchronization loss probability must lie in [0, 1]")
        if (not isfinite(self.outage_duration_s) or self.outage_duration_s < 0
                or self.outage_start_s is not None and (not isfinite(self.outage_start_s) or self.outage_start_s < 0)):
            raise ValueError("Synchronization outage times must be finite and nonnegative")
        for name in ("timestamp_error_bound_us", "source_error_bound_us"):
            if getattr(self, name) is not None:
                integer_us(getattr(self, name), name)

    def error_bounds(self, clock_config: ClockConfig, timing_config: TimingConfig) -> tuple[int, int]:
        """Matched declared bounds by default; explicit overrides can test misspecification."""
        # Rounded simulated clock reads add at most one microsecond beyond
        # the injected integer noise; include it in matched declared bounds.
        sync_error = (max(clock_config.timestamp_error_us, self.measurement_error_us + 1)
                      if self.timestamp_error_bound_us is None else self.timestamp_error_bound_us)
        source_error = (max(timing_config.source_error_us, abs(self.source_stamp_bias_us))
                        if self.source_error_bound_us is None else self.source_error_bound_us)
        return sync_error, source_error


@dataclass(frozen=True)
class SimulationClock:
    source_origin_us: int
    receiver_origin_us: int
    rate: float

    def receiver_time(self, source_time_us: int | float) -> float:
        return self.receiver_origin_us + self.rate * (source_time_us - self.source_origin_us)

    def source_time(self, receiver_time_us: int | float) -> float:
        return self.source_origin_us + (receiver_time_us - self.receiver_origin_us) / self.rate


@dataclass(frozen=True)
class SyncTrial:
    available_at_us: int
    exchange: SyncExchange


def synchronization_trials(clock: SimulationClock, scenario: ClockScenario, duration_us: int,
                           interval_us: int, processing_us: int, seed: int) -> list[SyncTrial]:
    generator = np.random.default_rng(seed)
    trials = []
    for elapsed in range(0, duration_us + 1, interval_us):
        up_jitter = int(generator.integers(-scenario.network_jitter_us, scenario.network_jitter_us + 1))
        down_jitter = int(generator.integers(-scenario.network_jitter_us, scenario.network_jitter_us + 1))
        errors = [int(generator.integers(-scenario.measurement_error_us, scenario.measurement_error_us + 1)) for _ in range(4)]
        dropped = generator.random() < scenario.sync_drop_probability
        outage = (scenario.outage_start_s is not None and scenario.outage_start_s * 1e6 <= elapsed
                  < (scenario.outage_start_s + scenario.outage_duration_s) * 1e6)
        if dropped or outage:
            continue
        r1 = clock.receiver_origin_us + elapsed
        s2 = clock.source_time(r1 + max(0, scenario.uplink_us + up_jitter))
        s3 = s2 + processing_us
        r4 = clock.receiver_time(s3) + max(0, scenario.downlink_us + down_jitter)
        exchange = SyncExchange(round(r1) + errors[0], round(s2) + errors[1], round(s3) + errors[2], round(r4) + errors[3])
        # Negative measurement error must never make a future response visible.
        available = max(ceil(r4), exchange.receiver_receive_us)
        trials.append(SyncTrial(available, exchange))
    # Responses themselves can be reordered. Availability, not request number,
    # decides which synchronization information has actually been received.
    return sorted(trials, key=lambda item: item.available_at_us)


@dataclass(frozen=True)
class TimingCase:
    view: ReceiverView
    offset_only: ObservationTiming | None
    query_index: int
    true_aoi_us: float | None
    true_native_aoi_us: float | None
    true_source_offset_us: float | None
    true_horizon_steps: int | None


def timing_cases(series: PreparedSeries, profile: dict, scenario: ClockScenario, *, namespace: str,
                 seed: int, replicate: int, clock_config: ClockConfig, timing_config: TimingConfig,
                 sync_interval_us: int = 250000, sync_processing_us: int = 20000,
                 context_length: int = 256, minimum_observations: int = 32,
                 query_start_us: int = 7_000_000, query_stride: int = 1, max_queries: int | None = None):
    """Advance only completed sync and packet events, then detach a receiver view.

    True clock, future arrivals and speed labels stay on this driver side. The
    source clock is rebased to an arbitrary positive boot origin while preserving
    source/native differences and the immutable corpus split / target definition.
    """
    metadata = series.metadata
    period = int(series.source_time_us[1] - series.source_time_us[0])
    source_origin = max(10_000_000, 10_000_000 - scenario.offset_us,
                        10_000_000 - scenario.source_stamp_bias_us)
    times = series.source_time_us - series.source_time_us[0] + source_origin
    native = series.native_time_us - series.source_time_us[0] + source_origin
    clock = SimulationClock(source_origin, source_origin + scenario.offset_us,
                            1.0 + scenario.relative_rate_ppm * 1e-6)
    sync_error, source_error = scenario.error_bounds(clock_config, timing_config)
    effective_clock_config = replace(clock_config, timestamp_error_us=sync_error)
    effective_timing_config = replace(timing_config, source_error_us=source_error)
    source_id, receiver_id = "sim-source-" + metadata["series_id"], "sim-receiver-" + metadata["series_id"]
    synchronizer = ClockSynchronizer(source_id, receiver_id, effective_clock_config)
    receiver = TimestampedReceiver(source_id, receiver_id, sample_period_us=period,
        capacity=max(2048, context_length), timing_config=effective_timing_config)
    arrival_seed = trace_seed(namespace, metadata["split"], metadata["series_id"], profile["name"], replicate, seed)
    arrivals = generate_arrivals(times, profile, arrival_seed)
    arrival_order = np.argsort(arrivals, kind="stable")
    indices = np.flatnonzero(times - source_origin >= query_start_us)[::query_stride]
    if max_queries is not None:
        indices = indices[:max_queries]
    if not len(indices):
        return
    sync_seed = trace_seed(namespace + "/clock", metadata["split"], metadata["series_id"],
                           scenario.name, replicate, seed)
    duration = ceil(clock.rate * (int(times[int(indices[-1])]) - source_origin))
    trials = synchronization_trials(clock, scenario, duration, sync_interval_us, sync_processing_us, sync_seed)
    packet_position = sync_position = 0
    for query_index in indices:
        query = int(clock.receiver_time(int(times[query_index])))
        while sync_position < len(trials) and trials[sync_position].available_at_us <= query:
            trial = trials[sync_position]
            synchronizer.update(trial.exchange, available_at_us=trial.available_at_us)
            sync_position += 1
        while packet_position < len(arrival_order):
            index = int(arrival_order[packet_position])
            if arrivals[index] == NEVER:
                break
            if arrivals[index] > times[query_index]:
                break
            receipt = int(clock.receiver_time(int(arrivals[index])))
            packet = TimestampedPacket(source_id, index, int(times[index]) + scenario.source_stamp_bias_us,
                float(series.speed_mps[index]), int(native[index]) + scenario.source_stamp_bias_us)
            receiver.ingest(packet, receipt)
            packet_position += 1
        view = receiver.snapshot(query, synchronizer.snapshot(query), context_length=context_length,
                                 minimum_observations=minimum_observations)
        if not len(view.source_time_us):
            yield TimingCase(view, None, int(query_index), None, None, None, None)
            continue
        last_source = int(view.source_time_us[-1]) - scenario.source_stamp_bias_us
        last_native = int(view.native_source_time_us[-1]) - scenario.source_stamp_bias_us
        offset_only = observation_timing(int(view.source_time_us[-1]), int(view.native_source_time_us[-1]),
            query, period, synchronizer.snapshot(query, estimate_drift=False),
            source_error_us=effective_timing_config.source_error_us,
            query_error_us=effective_timing_config.query_error_us,
            maximum_uncertainty_us=effective_timing_config.maximum_uncertainty_us)
        true_age = clock.rate * (int(times[query_index]) - last_source)
        true_native_age = clock.rate * (int(times[query_index]) - last_native)
        offset = true_age / clock.rate
        yield TimingCase(view, offset_only, int(query_index), true_age, true_native_age, offset,
                         horizon_steps(offset, period))
