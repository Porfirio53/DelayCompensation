"""UDP source adapter for arbitrary application signals and clock exchanges."""
import asyncio
from collections import Counter
from uuid import uuid4

from ..signals import SignalSchema
from ..timing.clock import clock_identities, integer_us
from ..timing.receiver import TimestampedPacket
from ..timing.transport import decode, encode, monotonic_us


class SignalSource(asyncio.DatagramProtocol):
    """Attach with loop.create_datagram_endpoint; call publish on that loop.

    A stream-receive client subscribes by sending the existing sync requests.
    This adapter never loads a predictor and works on either logical endpoint.
    Each instance represents one source grid / clock identity.
    """

    def __init__(self, schema: SignalSchema, *, sample_period_us: int = 25_000,
                 source_clock_id: str | None = None):
        if not isinstance(schema, SignalSchema):
            raise ValueError("SignalSource needs a SignalSchema")
        if source_clock_id is not None and (not isinstance(source_clock_id, str) or not source_clock_id):
            raise ValueError("SignalSource needs a nonempty source-clock identity")
        self.schema = schema
        self.sample_period_us = integer_us(sample_period_us, "sample_period_us", 1)
        self.source_clock_id = source_clock_id or "signal-source-" + uuid4().hex
        self.clients = {}
        self.counts = Counter()
        self.transport = None
        self._origin = None

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, address):
        received = monotonic_us()
        try:
            request = decode(data)
            if request["kind"] != "sync_request":
                raise ValueError("SignalSource accepts sync_request datagrams")
            receiver = request["receiver_clock_id"]
            clock_identities(self.source_clock_id, receiver)
            integer_us(request["request_id"], "request_id")
            integer_us(request["receiver_send_us"], "receiver_send_us")
            response = {"kind": "sync_response", "source_clock_id": self.source_clock_id,
                "receiver_clock_id": receiver, "request_id": request["request_id"],
                "receiver_send_us": request["receiver_send_us"], "source_receive_us": received,
                "source_send_us": monotonic_us(), "sample_period_us": self.sample_period_us}
            self.transport.sendto(encode(response), address)
            self.clients[address] = received
            self.counts["sync_responses"] += 1
        except (KeyError, TypeError, ValueError, OSError):
            self.counts["malformed_datagrams"] += 1

    def error_received(self, exc):
        self.counts["socket_errors"] += 1

    def publish(self, values: dict, *, sequence: int, source_time_us: int,
                native_source_time_us: int | None = None, future_values: dict | None = None) -> int:
        """Publish an acquired, aligned tick; future_values are known plans only.

        Signal values must contain every schema name. Future arrays start one
        source tick after this packet, with historical values in ``values``.
        The caller owns sampling/alignment and must skip, rather than fabricate,
        source ticks when acquisition cannot keep up.
        """
        packet = TimestampedPacket(self.source_clock_id, sequence, source_time_us,
            native_source_time_us=native_source_time_us, values=values, future_values=future_values)
        packet.validate_schema(self.schema)
        origin = packet.source_time_us - packet.sequence * self.sample_period_us
        if origin < 0 or self._origin is not None and origin != self._origin:
            raise ValueError("SignalSource needs one regular source grid")
        sent = monotonic_us()
        if packet.source_time_us > sent:
            raise ValueError("Acquire a source tick before publishing it")
        if self.transport is None:
            raise RuntimeError("Attach SignalSource to a datagram endpoint before publishing")
        payload = encode({"kind": "packet", "source_clock_id": packet.source_clock_id,
            "sequence": packet.sequence, "source_time_us": packet.source_time_us,
            "native_source_time_us": packet.native_source_time_us, "values": dict(packet.values),
            "future_values": {k: list(v) for k, v in packet.future_values.items()} if packet.future_values else None,
            "send_time_us": sent, "sample_period_us": self.sample_period_us})
        self._origin = origin
        subscribers = 0
        for address, last_seen in tuple(self.clients.items()):
            if sent - last_seen > 5_000_000:
                del self.clients[address]
            else:
                self.transport.sendto(payload, address)
                subscribers += 1
        self.counts["published_ticks"] += 1
        return subscribers
