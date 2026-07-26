from __future__ import annotations

import argparse
from collections.abc import Sequence

from stc_research import __version__


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stc")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version")
    args = parser.parse_args(argv)
    if args.command == "version":
        print(f"stc-research {__version__}")
        return 0
    raise AssertionError(args.command)
