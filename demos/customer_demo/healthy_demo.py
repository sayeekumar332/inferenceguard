"""CPU-only healthy inference customer demonstration.

Uses simulated token IDs, not actual model-generated tokens.
The existing InferenceGuard CLI performs verification.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from inferenceguard.cli import main as inferenceguard_main

SIMULATED_TOKEN_IDS = [101, 205, 309, 412]


class DemoHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible completion endpoint."""

    def response_token_ids(self) -> list[int]:
        """Return simulated token IDs for this response."""
        return SIMULATED_TOKEN_IDS

    def do_POST(self):
        """Serve simulated completion responses."""
        if self.path != "/v1/completions":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))

            if request.get("model") != "customer-demo-model":
                self.send_error(400, "Unexpected model")
                return

            response = {
                "id": "demo-completion",
                "object": "text_completion",
                "choices": [
                    {
                        "index": 0,
                        "text": "Inference is repeatable.",
                        "finish_reason": "stop",
                        "token_ids": self.response_token_ids(),
                    }
                ],
            }

            payload = json.dumps(response).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        except (ValueError, UnicodeDecodeError):
            self.send_error(400, "Invalid request")

    def log_message(self, format, *args):
        """Suppress HTTP server console logging."""


def run_demo(
    output: Path,
    *,
    handler_class: type[BaseHTTPRequestHandler] = DemoHandler,
    observe_prefix_cache: bool = False,
) -> int:
    """Run the existing InferenceGuard CLI against a local demo server."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_class)

    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )

    try:
        thread.start()

        endpoint = f"http://127.0.0.1:{server.server_port}/v1/completions"

        cli_args = [
            "repeat",
            "--endpoint",
            endpoint,
            "--model",
            "customer-demo-model",
            "--prompt",
            "Demonstrate deterministic inference.",
            "--repeats",
            "3",
            "--comparison-mode",
            "tokens",
            "--max-tokens",
            "24",
            "--output",
            str(output),
        ]

        if observe_prefix_cache:
            cli_args.append("--observe-prefix-cache")

        return inferenceguard_main(cli_args)

    finally:
        server.shutdown()
        server.server_close()

        if thread.is_alive():
            thread.join(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description="InferenceGuard healthy inference demonstration")

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination for the JSON verification report",
    )

    args = parser.parse_args()

    return run_demo(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
