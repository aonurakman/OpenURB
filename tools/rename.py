#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import string
import sys
from pathlib import Path


ID_CHARS = string.ascii_lowercase + string.digits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename an experiment directory in results/ and update references.",
    )
    parser.add_argument("old_id", help="Existing experiment id (directory name).")
    parser.add_argument(
        "new_id",
        nargs="?",
        help="New experiment id (defaults to a random 5-char alphanumeric).",
    )
    return parser.parse_args()


def validate_id(value: str) -> None:
    if not value or Path(value).name != value:
        raise ValueError("id must be a simple directory name (no path separators)")


def generate_id(results_dir: Path, length: int = 5) -> str:
    for _ in range(10_000):
        candidate = "".join(random.choice(ID_CHARS) for _ in range(length))
        if not (results_dir / candidate).exists():
            return candidate
    raise RuntimeError("could not generate a unique id")


def replace_in_text_files(root: Path, old_id: str, new_id: str) -> int:
    old_bytes = old_id.encode("utf-8")
    updated_count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if old_bytes not in data:
            continue
        if b"\0" in data:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        updated = text.replace(old_id, new_id)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            updated_count += 1
    return updated_count


def main() -> int:
    args = parse_args()
    try:
        validate_id(args.old_id)
        if args.new_id is not None:
            validate_id(args.new_id)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    results_dir = Path(__file__).resolve().parents[1] / "results"
    if not results_dir.is_dir():
        print(f"Error: results directory not found at {results_dir}", file=sys.stderr)
        return 2

    old_dir = results_dir / args.old_id
    if not old_dir.exists():
        print(f"Error: experiment {args.old_id} not found in {results_dir}", file=sys.stderr)
        return 2
    if not old_dir.is_dir():
        print(f"Error: {old_dir} is not a directory", file=sys.stderr)
        return 2

    new_id = args.new_id or generate_id(results_dir)
    if new_id == args.old_id:
        print("Error: new id matches old id", file=sys.stderr)
        return 2

    new_dir = results_dir / new_id
    if new_dir.exists():
        print(f"Error: experiment {new_id} already exists in {results_dir}", file=sys.stderr)
        return 2

    old_dir.rename(new_dir)
    updated_files = replace_in_text_files(new_dir, args.old_id, new_id)

    print(f"Renamed {args.old_id} -> {new_id}")
    print(f"Updated {updated_files} file(s) in {new_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
