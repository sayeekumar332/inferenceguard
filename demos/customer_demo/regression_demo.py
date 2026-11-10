"""CPU-only inference regression customer demonstration.

The second simulated response changes one token ID.
InferenceGuard's existing CLI must detect the mismatch.
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

from demos.customer_demo.healthy_demo import (
    SIMULATED_TOKEN_IDS,
    DemoHandler,
    run_demo,
)


def run_regression_demo(output: Path) -> int:
    """Execute three requests with a deliberate second-response mismatch."""

    request_count = 0
    request_lock = threading.Lock()

    class RegressionHandler(DemoHandler):
        def response_token_ids(self) -> list[int]:
            nonlocal request_count

            with request_lock:
                request_count += 1
                current_request = request_count

            if current_request == 2:
                return [101, 205, 999, 412]

            return SIMULATED_TOKEN_IDS

    return run_demo(output, handler_class=RegressionHandler)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="InferenceGuard inference regression demonstration"
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination for the JSON verification report",
    )

    args = parser.parse_args()

    return run_regression_demo(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
