"""A bounded receiver that stores only packets actually received, with two clocks."""
from collections import deque
from dataclasses import asdict, dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..data.arrival import ReceiverView, _readonly
from ..signals import SignalSchema, SignalView
from .clock import ClockMapping, clock_identities, integer_us, observation_timing


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
    value_mps: float | None = None
    native_source_time_us: int | None = None
    values: Mapping[str, float] | None = None
    future_values: Mapping[str, tuple[float, ...]] | None = None

    def __post_init__(self):
        if not isinstance(self.source_clock_id, str) or not self.source_clock_id:
            raise ValueError("Packet needs a source monotonic-clock identity")
        integer_us(self.sequence, "sequence")
        integer_us(self.source_time_us, "source_time_us")
        if (self.value_mps is None) == (self.values is None):
            raise ValueError("Supply either legacy value_mps or named values")
        if self.value_mps is not None and (isinstance(self.value_mps, bool) or not isfinite(self.value_mps)):
            raise ValueError("Packet speed must be finite")
        if self.values is not None:
            if (not isinstance(self.values, Mapping) or not self.values
                    or any(not isinstance(k, str) or not k.strip() or isinstance(v, bool) or not isfinite(v)
                           for k, v in self.values.items())):
                raise ValueError("Packet values must map nonempty signal names to finite numbers")
            object.__setattr__(self, "values", MappingProxyType({k: float(v) for k, v in self.values.items()}))
        if self.future_values is not None:
            if not isinstance(self.future_values, Mapping) or self.values is None:
                raise ValueError("Future-known values require a named signal packet")
            future = {}
            for name, values in self.future_values.items():
                if (name not in self.values or isinstance(values, (str, bytes)) or not len(values)
                        or any(isinstance(v, bool) or not isfinite(v) for v in values)):
                    raise ValueError("Future-known values must be finite arrays of declared signals")
                future[name] = tuple(float(v) for v in values)
            if len({len(values) for values in future.values()}) > 1:
                raise ValueError("Future-known arrays must have matching lengths")
            object.__setattr__(self, "future_values", MappingProxyType(future))
        if self.native_source_time_us is not None:
            integer_us(self.native_source_time_us, "native_source_time_us")
            if self.native_source_time_us > self.source_time_us:
                raise ValueError("Native observation follows the packet source tick")

    def validate_schema(self, schema: SignalSchema | None) -> None:
        if schema is None:
            if self.values is not None:
                raise ValueError("Named packets require a signal_schema on the receiver")
        else:
            schema.validate_values(self.values)
            if self.future_values is not None and set(self.future_values) != set(schema.future_covariates):
                raise ValueError("Future-known packet values must match the schema's future covariates")


class TimestampedReceiver:
    """Sparse source-grid buffer; loss and not-yet-arrived remain indistinguishable.

    Sequence numbers locate source positions, not receipt order. A late packet
    may fill a retained hole but can never replace the newest source observation.
    Caller owns session changes and feeds local receipt timestamps in order.
    """

    def __init__(self, source_clock_id: str, receiver_clock_id: str, *, sample_period_us: int = 25000,
                 capacity: int = 2048, statistics_packets: int = 64,
                 timing_config: TimingConfig | None = None, signal_schema: SignalSchema | None = None):
        clock_identities(source_clock_id, receiver_clock_id)
        self.source_clock_id, self.receiver_clock_id = source_clock_id, receiver_clock_id
        self.sample_period_us = integer_us(sample_period_us, "sample_period_us", 1)
        self.capacity = integer_us(capacity, "capacity", 1)
        integer_us(statistics_packets, "statistics_packets", 1)
        self.timing_config = timing_config or TimingConfig()
        self.signal_schema = signal_schema
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
        self.validate_packet(packet)
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

    def validate_packet(self, packet: TimestampedPacket) -> None:
        packet.validate_schema(self.signal_schema)

    def snapshot(self, query_time_us: int, clock: ClockMapping | None, *, context_length: int = 256,
                 minimum_observations: int = 32) -> ReceiverView | SignalView:
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
            if self.signal_schema is not None:
                return SignalView(query, self.sample_period_us, empty, self.signal_schema,
                    _readonly(np.empty((len(self.signal_schema.names), 0))), _readonly(np.array([], dtype=bool)),
                    empty, empty, MappingProxyType({"received_count": 0.0}), None, False, "no_received_observation")
            return ReceiverView(query, self.sample_period_us, empty, _readonly(np.array([], dtype=float)),
                _readonly(np.array([], dtype=bool)), empty, empty, MappingProxyType({"received_count": 0.0}),
                None, False, "no_received_observation")
        start = max(0, self._latest + 1 - context_length)
        indices = np.arange(start, self._latest + 1, dtype=np.int64)
        times = self._origin + indices * self.sample_period_us
        schema = self.signal_schema
        values = np.full(len(indices) if schema is None else (len(schema.names), len(indices)), np.nan)
        native = np.full(len(indices), -1, dtype=np.int64)
        receipts = np.full(len(indices), -1, dtype=np.int64)
        mask = np.zeros(len(indices), dtype=bool)
        for index, sequence in enumerate(indices):
            observed = self._packets.get(int(sequence))
            if observed is not None:
                packet, receipt = observed
                mask[index], receipts[index] = True, receipt
                if schema is None:
                    values[index] = packet.value_mps
                else:
                    values[:, index] = [packet.values[name] for name in schema.names]
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
        if schema is not None:
            future = self._packets[self._latest][0].future_values
            if schema.future_covariates and not future:
                ready, reason = False, "future_covariates_unavailable"
            known = _readonly(np.asarray([future[name] for name in schema.future_covariates])) if future else None
            return SignalView(query, self.sample_period_us, _readonly(times), schema, _readonly(values),
                _readonly(mask), _readonly(receipts), _readonly(native), MappingProxyType(features),
                timing.horizon_steps if timing.valid else None, ready, reason, timing, known)
        return ReceiverView(query, self.sample_period_us, _readonly(times), _readonly(values), _readonly(mask),
            _readonly(receipts), _readonly(native), MappingProxyType(features),
            timing.horizon_steps if timing.valid else None, ready, reason, timing)
