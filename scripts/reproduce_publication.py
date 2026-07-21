"""CLI for generating or verifying publication artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.publication_workflow import (
    PublicationWorkflowError,
    generate_publication_bundle,
    verify_publication_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or verify deterministic offline publication artifacts."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path, help="Repository-local YAML config.")
    mode.add_argument("--verify", type=Path, help="Manifest to verify.")
    parser.add_argument("--output-dir", type=Path, help="Generation destination.")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a dirty-tree preview; do not use its artifacts for publication.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only known generated files in a non-empty destination.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.verify is not None:
            result = verify_publication_manifest(args.verify)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.output_dir is None:
            raise PublicationWorkflowError("--output-dir is required with --config.")
        manifest = generate_publication_bundle(
            args.config,
            args.output_dir,
            allow_dirty=args.allow_dirty,
            overwrite=args.overwrite,
        )
        print(f"Publication bundle generated: {manifest}")
        return 0
    except PublicationWorkflowError as exc:
        print(f"Publication workflow failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
