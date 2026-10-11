"""Project entry point; model operations use the current TiRex source only."""
import argparse
from math import isfinite
from pathlib import Path
import sys

from .common import project_root


def positive(value: str) -> int:
    integer = int(value)
    if integer < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return integer


def positive_seconds(value: str) -> float:
    seconds = float(value)
    if not isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be finite positive seconds")
    return seconds


def port_number(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("must be a port from 1 to 65535")
    return number


def main(argv=None) -> int:
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description="TiRex-2 causal replay, clock-aware compensation and asynchronous streaming")
    roots = argparse.ArgumentParser(add_help=False)
    roots.add_argument("--root", type=Path, default=project_root(), help="Project root; independent of current working directory")
    common = argparse.ArgumentParser(add_help=False, parents=[roots])
    common.add_argument("--config", default="configs/experiment.json")
    common.add_argument("--tirex-config", help="Override the current TiRex model configuration")
    common.add_argument("--source", choices=("current",), default="current",
                        help="Only the current TiRex source is enabled")
    commands = parser.add_subparsers(dest="command", required=True)
    diagnostic = commands.add_parser("doctor", parents=[common], help="Verify environment/assets; --smoke runs real GPU predictions")
    diagnostic.add_argument("--smoke", action="store_true")
    diagnostic.add_argument("--verify-assets", action="store_true", help="Rehash every migrated file, including raw arrays")
    speed = commands.add_parser("eval-speed", parents=[common], help="Independent original five-trajectory speed replay")
    speed.add_argument("--sequence", action="append", help="Sequence column, repeatable")
    speed.add_argument("--delay-ms", type=int, choices=(175, 300), action="append")
    speed.add_argument("--max-queries", type=positive, help="First N scored queries per sequence/delay; no full-run prerequisite")
    speed.add_argument("--batch-size", type=positive, help="Replay throughput setting; not a single-request benchmark")
    speed.add_argument("--reference", choices=("auto", "none"), default="auto", help="Optionally compare matching current-source historical predictions")
    bench = commands.add_parser("benchmark", parents=[common], help="Synchronized batch=1 end-to-end latency")
    bench.add_argument("--batch-size", type=positive, default=1)
    bench.add_argument("--samples", type=positive, default=200)
    bench.add_argument("--warmup", type=positive, default=10)
    bench.add_argument("--sequence", default="speed_seg01")
    bench.add_argument("--delay-ms", type=int, choices=(175, 300), default=175)
    bench.add_argument("--query-index", type=int, default=1200, help="Initial query on 40 Hz grid; default 30s, then consecutive queries")
    aoi = commands.add_parser("eval-aoi", parents=[common], help="Controlled clock/AoI experiments on causal prepared CAN data")
    aoi.add_argument("--aoi-config", default="configs/aoi.json")
    aoi.add_argument("--corpus", help="Prepared corpus directory; defaults to the frozen data index")
    aoi.add_argument("--split", choices=("train", "validation", "test"))
    aoi.add_argument("--series-id", action="append")
    aoi.add_argument("--profile", action="append", help="Arrival profile name, repeatable")
    aoi.add_argument("--scenario", action="append", help="Clock scenario name, repeatable")
    aoi.add_argument("--max-series", type=positive)
    aoi.add_argument("--max-queries", type=positive, help="Query limit per series/profile/scenario/replicate, including unavailable queries")
    aoi.add_argument("--query-stride", type=positive)
    aoi.add_argument("--replicates", type=positive)
    aoi.add_argument("--batch-size", type=positive)
    aoi.add_argument("--with-model", action="store_true", help="Also run local TiRex and score speed; otherwise measure timing only")
    aoi.add_argument("--delivery", action="store_true",
                     help="With --with-model: batch=1 short trajectories and paired delivery-time scoring")
    aoi.add_argument("--compute-estimate-ms", type=positive_seconds, default=17.0,
                     help="Initial compute estimate for --delivery, in ms; updated by EWMA (default: 17)")
    aoi.add_argument("--warmup", type=positive, default=5,
                     help="Unscored startup requests for --delivery (default: 5; first excluded from EWMA)")
    measurement = argparse.ArgumentParser(add_help=False, parents=[roots])
    source = commands.add_parser("sync-source", parents=[roots], help="Serve real four-timestamp UDP sync and replay an existing CAN block")
    source.add_argument("--bind", default="127.0.0.1")
    source.add_argument("--port", type=port_number, default=5005)
    source.add_argument("--duration-s", type=positive_seconds, default=60)
    source.add_argument("--corpus")
    source.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    source.add_argument("--series-id")
    capture = commands.add_parser("sync-capture", parents=[measurement], help="Capture real UDP receipts and measure dynamic AoI using local monotonic time")
    capture.add_argument("--aoi-config", default="configs/aoi.json", help="Clock bounds and timing configuration")
    capture.add_argument("--host", required=True)
    capture.add_argument("--port", type=port_number, default=5005)
    capture.add_argument("--duration-s", type=positive_seconds, default=60)
    capture.add_argument("--sync-interval-ms", type=positive, default=250)
    capture.add_argument("--query-period-ms", type=positive, default=25)
    capture.add_argument("--context-length", type=positive)
    capture.add_argument("--minimum-observations", type=positive)
    trace = commands.add_parser("replay-aoi", parents=[measurement], help="Replay recorded clock/packet/query events in receiver order")
    trace.add_argument("--aoi-config", help="Override recorded clock/timing configuration; default restores the trace header")
    trace.add_argument("--trace", required=True)
    streaming = argparse.ArgumentParser(add_help=False, parents=[common])
    streaming.add_argument("--aoi-config", default="configs/aoi_frozen.json")
    streaming.add_argument("--with-model", action="store_true", help="Enable current TiRex batch=1; default publishes received values without loading a model")
    streaming.add_argument("--publish-period-ms", type=positive, default=25, help="Absolute publication period; default 25 ms / 40 Hz")
    streaming.add_argument("--compute-estimate-ms", type=positive_seconds, default=17.0)
    streaming.add_argument("--warmup", type=positive, default=5, help="Unscored startup requests; first omitted from compute EWMA")
    stream = commands.add_parser("eval-stream", parents=[streaming], help="Real-time CAN arrival replay, asynchronous inference and publication-time scoring")
    stream.add_argument("--corpus")
    stream.add_argument("--split", choices=("train", "validation", "test"))
    stream.add_argument("--series-id", action="append")
    stream.add_argument("--profile", action="append")
    stream.add_argument("--scenario", action="append")
    stream.add_argument("--max-series", type=positive, default=1)
    stream.add_argument("--duration-s", type=positive_seconds, default=10, help="Timed stream window per unit, after historical prefill and warmup")
    stream.add_argument("--replicates", type=positive, default=1)
    stream.add_argument("--inference-period-ms", type=positive, action="append", help="Minimum request start interval; repeat to compare rates (default: 25)")
    live = commands.add_parser("stream-receive", parents=[streaming], help="Receive real UDP timestamps while independently publishing continuous estimates")
    live.add_argument("--host", required=True)
    live.add_argument("--port", type=port_number, default=5005)
    live.add_argument("--duration-s", type=positive_seconds, default=60)
    live.add_argument("--sync-interval-ms", type=positive, default=250)
    live.add_argument("--inference-period-ms", type=positive, default=25)
    live.add_argument("--context-length", type=positive)
    live.add_argument("--minimum-observations", type=positive)
    live.add_argument("--stdout", action="store_true", help="Publish JSON lines to stdout; progress messages go to stderr")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    try:
        if args.command == "doctor":
            from .evaluation.doctor import doctor
            doctor(root, args)
        elif args.command == "eval-speed":
            from .evaluation.speed import eval_speed
            eval_speed(root, args)
        elif args.command == "benchmark":
            from .evaluation.benchmark import benchmark
            benchmark(root, args)
        elif args.command == "eval-aoi":
            from .evaluation.aoi import eval_aoi
            eval_aoi(root, args)
        elif args.command == "sync-source":
            from .timing.transport import sync_source
            sync_source(root, args)
        elif args.command == "sync-capture":
            from .timing.transport import sync_capture
            sync_capture(root, args)
        elif args.command == "replay-aoi":
            from .timing.transport import replay_capture
            replay_capture(root, args)
        elif args.command == "eval-stream":
            from .streaming.replay import eval_stream
            eval_stream(root, args)
        elif args.command == "stream-receive":
            from .streaming.live import stream_receive
            stream_receive(root, args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0
