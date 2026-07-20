"""Print or intentionally replace the reviewed numerical golden baseline."""

from __future__ import annotations

import argparse
import json

from tests.numerical_baseline import GOLDEN_PATH, compute_numerical_baseline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the deterministic numerical regression baseline."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Replace the committed golden JSON after an explicit model review.",
    )
    args = parser.parse_args()

    rendered = json.dumps(
        compute_numerical_baseline(),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    if args.write:
        GOLDEN_PATH.write_text(rendered, encoding="utf-8")
        print(f"Wrote reviewed candidate baseline to {GOLDEN_PATH}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
