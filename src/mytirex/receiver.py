"""A bounded receiver that stores only packets actually received, with two clocks."""
from collections import deque
from dataclasses import asdict, dataclass
from math import isfinite
from types import MappingProxyType

import numpy as np

from .arrival import ReceiverView, _readonly
from .timing import ClockMapping, clock_identities, integer_us, observation_timing


@dataclass(frozen=True)
class TimingConfig:
    source_error_us: int = 1000
    query_error_us: int = 1
    maximum_uncertainty_us: int = 100_000

    def __post_init__(self):
        integer_us(self.source_error_us, "source_error_us")
        integer_us(self.query_error_us, "query_error_us")
        integer_us(self.maximum_uncertainty_us, "maximum_uncertainty_us", 1)


@dataclass(frozen=True)
class TimestampedPacket:
    source_clock_id: str
    sequence: int
    source_time_us: int
    value_mps: float
    native_source_time_us: int | None = None

    def __post_init__(self):
        if not isinstance(self.source_clock_id, str) or not self.source_clock_id:
            raise ValueError("Packet needs a source monotonic-clock identity")
        integer_us(self.sequence, "sequence")
        integer_us(self.source_time_us, "source_time_us")
        if isinstance(self.value_mps, bool) or not isfinite(self.value_mps):
            raise ValueError("Packet speed must be finite")
        if self.native_source_time_us is not None:
            integer_us(self.native_source_time_us, "native_source_time_us")
            if self.native_source_time_us > self.source_time_us:
                raise ValueError("Native observation follows the packet source tick")


class TimestampedReceiver:
    """Sparse source-grid buffer; loss and not-yet-arrived remain indistinguishable.

    Sequence numbers locate source positions, not receipt order. A late packet
    may fill a retained hole but can never replace the newest source observation.
    Caller owns session changes and feeds local receipt timestamps in order.
    """

    def __init__(self, source_clock_id: str, receiver_clock_id: str, *, sample_period_us: int = 25000,
                 capacity: int = 2048, statistics_packets: int = 64,
                 timing_config: TimingConfig | None = None):
        clock_identities(source_clock_id, receiver_clock_id)
        self.source_clock_id, self.receiver_clock_id = source_clock_id, receiver_clock_id
        self.sample_period_us = integer_us(sample_period_us, "sample_period_us", 1)
        self.capacity = integer_us(capacity, "capacity", 1)
        integer_us(statistics_packets, "statistics_packets", 1)
        self.timing_config = timing_config or TimingConfig()
        self._packets = {}
        self._recent = deque(maxlen=statistics_packets)
        self._origin = None
        self._latest = -1
        self._query = self._last_receipt = -1
        self.received = self.duplicates = self.too_old = self.reordered = 0

    @property
    def buffered_packets(self) -> int:
        return len(self._packets)

    @property
    def revision(self) -> int:
        """Accepted history changes, including late packets filling a hole."""
        return self.received - self.duplicates - self.too_old

    def ingest(self, packet: TimestampedPacket, received_at_us: int) -> bool:
        receipt = integer_us(received_at_us, "received_at_us")
        if packet.source_clock_id != self.source_clock_id:
            raise ValueError("Packet source clock changed; start a new receiver and synchronizer")
        if receipt < max(self._last_receipt, self._query):
            raise ValueError("Packets must be ingested in nondecreasing receiver time before snapshotting")
        origin = packet.source_time_us - packet.sequence * self.sample_period_us
        if origin < 0 or self._origin is not None and origin != self._origin:
            raise ValueError("Packet sequence/source timestamps do not describe one regular source grid")
        self._origin = origin
        self._last_receipt = receipt
        self.received += 1
        if packet.sequence in self._packets:
            self.duplicates += 1
            return False
        reordered = packet.sequence < self._latest
        self.reordered += int(reordered)
        self._recent.append((packet.source_time_us, receipt, reordered))
        if packet.sequence <= self._latest - self.capacity:
            self.too_old += 1
            return False
        self._latest = max(self._latest, packet.sequence)
        self._packets[packet.sequence] = (packet, receipt)
        cutoff = self._latest - self.capacity + 1
        for sequence in tuple(self._packets):
            if sequence < cutoff:
                del self._packets[sequence]
        return True

    def snapshot(self, query_time_us: int, clock: ClockMapping | None, *, context_length: int = 256,
                 minimum_observations: int = 32) -> ReceiverView:
        query = integer_us(query_time_us, "query_time_us")
        if query < max(self._query, self._last_receipt):
            raise ValueError("Receiver queries must be nondecreasing and include ingested receipt times")
        integer_us(context_length, "context_length", 1)
        integer_us(minimum_observations, "minimum_observations", 1)
        if not minimum_observations <= context_length <= self.capacity:
            raise ValueError("Invalid receiver context/minimum count or buffer capacity")
        if clock is not None and (clock.source_clock_id != self.source_clock_id
                                  or clock.receiver_clock_id != self.receiver_clock_id):
            raise ValueError("Clock mapping belongs to another source/receiver session")
        self._query = query
        if self._latest < 0:
            empty = _readonly(np.array([], dtype=np.int64))
            return ReceiverView(query, self.sample_period_us, empty, _readonly(np.array([], dtype=float)),
                _readonly(np.array([], dtype=bool)), empty, empty, MappingProxyType({"received_count": 0.0}),
                None, False, "no_received_observation")
        start = max(0, self._latest + 1 - context_length)
        indices = np.arange(start, self._latest + 1, dtype=np.int64)
        times = self._origin + indices * self.sample_period_us
        values = np.full(len(indices), np.nan)
        native = np.full(len(indices), -1, dtype=np.int64)
        receipts = np.full(len(indices), -1, dtype=np.int64)
        mask = np.zeros(len(indices), dtype=bool)
        for index, sequence in enumerate(indices):
            observed = self._packets.get(int(sequence))
            if observed is not None:
                packet, receipt = observed
                mask[index], values[index], receipts[index] = True, packet.value_mps, receipt
                if packet.native_source_time_us is not None:
                    native[index] = packet.native_source_time_us
        timing = observation_timing(int(times[-1]), int(native[-1]) if native[-1] >= 0 else None,
            query, self.sample_period_us, clock, **asdict(self.timing_config))
        recent = np.asarray(self._recent, dtype=np.int64)
        interarrival = np.diff(recent[:, 1]) / 1e6
        context_ready = len(indices) == context_length and int(mask.sum()) >= minimum_observations
        features = {
            "received_count": float(mask.sum()), "context_ready": float(context_ready),
            "window_unreceived_fraction": float(1.0 - mask.mean()),
            "time_since_last_receipt_s": (query - self._last_receipt) / 1e6,
            "observed_reordered_fraction": float(recent[:, 2].mean()),
            "received_interarrival_mean_s": float(interarrival.mean()) if len(interarrival) else 0.0,
            "received_interarrival_std_s": float(interarrival.std()) if len(interarrival) else 0.0,
            "clock_valid": float(timing.valid),
        }
        if timing.age_us is not None:
            features.update({"observation_age_s": timing.age_us / 1e6,
                             "observation_age_lower_s": timing.age_lower_us / 1e6,
                             "observation_age_upper_s": timing.age_upper_us / 1e6})
        if timing.native_age_us is not None:
            features.update({"native_observation_age_s": timing.native_age_us / 1e6,
                             "latest_sample_hold_age_s": (timing.native_age_us - timing.age_us) / 1e6})
        if clock is not None:
            features.update({"clock_sync_age_s": (query - clock.last_sync_receiver_us) / 1e6,
                             "clock_relative_drift_ppm": clock.relative_drift_ppm,
                             "clock_minimum_rtt_s": clock.minimum_round_trip_us / 1e6})
            try:
                latency = np.asarray([clock.age(int(s), query).estimate_us - (query - int(r))
                                      for s, r, _ in recent]) / 1e6
                features.update({"received_latency_mean_s": float(latency.mean()),
                    "received_latency_std_s": float(latency.std()),
                    "received_latency_mad_s": float(np.median(np.abs(latency - np.median(latency))))})
            except ValueError:
                pass  # An inconsistent clock is already explicit in timing.reason.
        ready = context_ready and timing.valid
        reason = timing.reason if context_ready else "insufficient_context"
        return ReceiverView(query, self.sample_period_us, _readonly(times), _readonly(values), _readonly(mask),
            _readonly(receipts), _readonly(native), MappingProxyType(features),
            timing.horizon_steps if timing.valid else None, ready, reason, timing)
