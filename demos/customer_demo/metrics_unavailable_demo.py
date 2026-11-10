"""Customer demo: repeatable inference with unavailable cache metrics.

The HTTP server supplies simulated completion token IDs.
No usable /metrics endpoint is provided.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from demos.customer_demo.healthy_demo import run_demo


def main() -> int:
    parser = argparse.ArgumentParser(description="Demonstrate unavailable prefix-cache metrics")

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination for the JSON verification report",
    )

    args = parser.parse_args()

    return run_demo(
        args.output,
        observe_prefix_cache=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
