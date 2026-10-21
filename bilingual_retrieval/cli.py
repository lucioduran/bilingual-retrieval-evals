"""Command-line entry points with explicit download and offline run modes."""

import argparse
import json
import sys
from pathlib import Path

from .contracts import load_dataset
from .experiment import run, verify
from .models import snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download")
    download.add_argument("--cache", type=Path, default=Path(".cache/models"))
    validate = sub.add_parser("validate")
    validate.add_argument("--dataset", type=Path, default=Path("data/dataset.json"))
    execute = sub.add_parser("run")
    execute.add_argument("--dataset", type=Path, default=Path("data/dataset.json"))
    execute.add_argument("--cache", type=Path, default=Path(".cache/models"))
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--split", choices=("dev", "eval"), required=True)
    execute.add_argument("--k", type=int, default=3)
    execute.add_argument("--threads", type=int, default=2)
    replay = sub.add_parser("verify")
    replay.add_argument("--dataset", type=Path, default=Path("data/dataset.json"))
    replay.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            snapshot(args.cache, offline=False)
        elif args.command == "validate":
            dataset = load_dataset(args.dataset)
            print(
                json.dumps({"documents": len(dataset.documents), "queries": len(dataset.queries)})
            )
        elif args.command == "run":
            run(args.dataset, args.output, args.cache, args.split, args.k, args.threads)
        else:
            verify(args.dataset, args.report)
        print(json.dumps({"status": "ok", "command": args.command}))
        return 0
    except (ValueError, OSError, ImportError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({"status": "error", "message": str(error)}), file=sys.stderr)
        return 2
