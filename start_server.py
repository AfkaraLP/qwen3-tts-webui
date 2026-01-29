#!/usr/bin/env python3
"""
Voice Cloner Server Launcher
Start the FastAPI server for voice cloning functionality
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import uvicorn

def main():
    parser = argparse.ArgumentParser(description="Start the Voice Cloner HTTP server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=int(os.environ.get("VOICE_CLONER_PORT", "8000")), help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Change to the project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print(f"Starting Qwen TTS WebUI Server")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Auto-reload: {args.reload}")
    print(f"Workers: {args.workers}")
    print(f"Working directory: {project_dir}")
    print()
    print(f"Web UI will be available at: http://localhost:{args.port}")
    print(f"API Documentation: http://localhost:{args.port}/docs")
    print()
    
    # Start the server
    uvicorn.run(
        "voice_cloner.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1 if args.reload else args.workers,
        log_level="info"
    )

if __name__ == "__main__":
    main()