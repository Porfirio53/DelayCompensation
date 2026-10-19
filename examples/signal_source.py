"""Manual UDP wiring example; replace sample_values with acquired application data."""
import argparse
import asyncio
import math
from pathlib import Path

from mytirex.streaming.runtime import sleep_until
from mytirex.streaming.source import SignalSource
from mytirex.timing.transport import monotonic_us, signal_configuration


def sample_values(sequence, schema):
    # Synthetic numeric variables exercise transport only, not driving accuracy.
    return {name: math.sin(sequence * .02 + index) for index, name in enumerate(schema.names)}


async def serve(args):
    schema, period = signal_configuration(args.root, args.signals_config)
    source = SignalSource(schema, sample_period_us=period)
    transport, _ = await asyncio.get_running_loop().create_datagram_endpoint(
        lambda: source, local_addr=(args.bind, args.port))
    origin = monotonic_us() + 100_000
    finish, sequence = origin + round(args.duration_s * 1e6), 0
    try:
        while origin + sequence * period < finish:
            await sleep_until(origin + sequence * period)
            sequence = max(sequence, (monotonic_us() - origin) // period)
            timestamp = origin + sequence * period
            if timestamp >= finish:
                break
            future = {name: [sample_values(sequence + step, schema)[name] for step in range(1, 65)]
                      for name in schema.future_covariates} or None
            source.publish(sample_values(sequence, schema), sequence=sequence, source_time_us=timestamp,
                           native_source_time_us=timestamp, future_values=future)
            sequence += 1
    finally:
        transport.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--signals-config", default="configs/signals_example.json")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--duration-s", type=float, default=30)
    asyncio.run(serve(parser.parse_args()))
