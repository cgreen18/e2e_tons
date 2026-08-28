#!/usr/bin/env python3
"""Manifest-driven TONS experiment entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tons_sim import analyze, prepare, run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "analyze"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.stage == "prepare":
        result = prepare(args.manifest)
    elif args.stage == "run":
        result = run(args.manifest, dry_run=args.dry_run, limit=args.limit)
    else:
        result = analyze(args.manifest)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
