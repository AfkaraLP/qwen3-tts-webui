#!/usr/bin/env python3
"""
Voice Cloner Server Launcher
Start the FastAPI server for voice cloning functionality
"""

import os
import sys
import argparse
import threading
import time
import itertools
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import uvicorn  # noqa: E402
import urllib.request  # noqa: E402
import urllib.error  # noqa: E402


class LoadingSpinner:
    """A terminal loading spinner that shows while the model is loading"""

    def __init__(self, port: int):
        self.port = port
        self.spinning = True
        self.spinner_chars = itertools.cycle(
            ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        )
        self.thread = None
        self.model_loaded = False

    def _check_health(self) -> bool:
        """Check if the server is healthy and model is loaded"""
        try:
            url = f"http://localhost:{self.port}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as response:
                import json

                data = json.loads(response.read().decode())
                return data.get("voice_cloner_loaded", False)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, Exception):
            return False

    def _spin(self):
        """Run the spinner animation"""
        start_time = time.time()
        while self.spinning:
            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            time_str = f"{mins:02d}:{secs:02d}"

            char = next(self.spinner_chars)
            status = f"\r{char} Loading model... ({time_str}) "
            sys.stdout.write(status)
            sys.stdout.flush()

            # Check health every iteration
            if self._check_health():
                self.model_loaded = True
                self.spinning = False
                break

            time.sleep(0.1)

        # Clear the spinner line and print success message
        if self.model_loaded:
            sys.stdout.write("\r" + " " * 80 + "\r")  # Clear line
            sys.stdout.write("✓ Model loaded successfully! Server is ready.\n")
            sys.stdout.flush()

    def start(self):
        """Start the spinner in a background thread"""
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the spinner"""
        self.spinning = False
        if self.thread:
            self.thread.join(timeout=1)


def main():
    parser = argparse.ArgumentParser(description="Start the Voice Cloner HTTP server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("VOICE_CLONER_PORT", "8000")),
        help="Port to bind to",
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes"
    )

    args = parser.parse_args()

    # Change to the project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)

    print("Starting Qwen TTS WebUI Server")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Auto-reload: {args.reload}")
    print(f"Workers: {args.workers}")
    print(f"Working directory: {project_dir}")
    print()
    print(f"Web UI will be available at: http://localhost:{args.port}")
    print(f"API Documentation: http://localhost:{args.port}/docs")
    print()

    # Start the loading spinner in a background thread
    spinner = LoadingSpinner(args.port)
    spinner.start()

    # Start the server
    try:
        uvicorn.run(
            "voice_cloner.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=1 if args.reload else args.workers,
            log_level="info",
        )
    finally:
        spinner.stop()


if __name__ == "__main__":
    main()
