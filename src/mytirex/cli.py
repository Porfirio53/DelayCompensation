"""One entry point; each invocation selects exactly one tirex2 source."""
import argparse
from pathlib import Path
import sys

from .common import project_root


def positive(value: str) -> int:
    integer = int(value)
    if integer < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return integer


def main(argv=None) -> int:
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description="Offline TiRex-2 speed baseline and real GPU single-request measurement")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, default=project_root(), help="Project root; independent of current working directory")
    common.add_argument("--config", default="configs/experiment.json")
    common.add_argument("--tirex-config", help="Override the selected source's model configuration")
    common.add_argument("--source", choices=("current", "official"), default="current",
                        help="Run one source in this process; switch sources by starting a new command")
    commands = parser.add_subparsers(dest="command", required=True)
    diagnostic = commands.add_parser("doctor", parents=[common], help="Verify environment/assets; --smoke runs real GPU predictions")
    diagnostic.add_argument("--smoke", action="store_true")
    diagnostic.add_argument("--verify-assets", action="store_true", help="Rehash every migrated file, including raw arrays")
    speed = commands.add_parser("eval-speed", parents=[common], help="Independent original five-trajectory speed replay")
    speed.add_argument("--sequence", action="append", help="Sequence column, repeatable")
    speed.add_argument("--delay-ms", type=int, choices=(175, 300), action="append")
    speed.add_argument("--max-queries", type=positive, help="First N scored queries per sequence/delay; no full-run prerequisite")
    speed.add_argument("--batch-size", type=positive, help="Replay throughput setting; not a single-request benchmark")
    speed.add_argument("--reference", choices=("auto", "none"), default="auto", help="Optionally compare copied fixed controls after inference")
    bench = commands.add_parser("benchmark", parents=[common], help="Synchronized batch=1 end-to-end latency")
    bench.add_argument("--batch-size", type=positive, default=1)
    bench.add_argument("--samples", type=positive, default=200)
    bench.add_argument("--warmup", type=positive, default=10)
    bench.add_argument("--sequence", default="speed_seg01")
    bench.add_argument("--delay-ms", type=int, choices=(175, 300), default=175)
    bench.add_argument("--query-index", type=int, default=1200, help="Initial query on 40 Hz grid; default 30s, then consecutive queries")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    try:
        if args.command == "doctor":
            from .doctor import doctor
            doctor(root, args)
        elif args.command == "eval-speed":
            from .evaluation import eval_speed
            eval_speed(root, args)
        else:
            from .benchmark import benchmark
            benchmark(root, args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0
