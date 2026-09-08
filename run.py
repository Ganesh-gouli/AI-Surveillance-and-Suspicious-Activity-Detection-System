"""
SentinelVision AI - Unified Full-Stack Runner
Launches both FastAPI AI Backend and React Vite Frontend concurrently.
"""

import os
import sys
import subprocess
import signal
import time

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(root_dir, "backend")
    frontend_dir = os.path.join(root_dir, "frontend")

    print("=" * 65)
    print("  SENTINELVISION AI - UNIFIED FULL-STACK LAUNCHER")
    print("=" * 65)
    print("[1/2] Starting FastAPI & AI Inference Engine (Port 8000)...")
    print("[2/2] Starting React + Vite SOC Web Dashboard (Port 5173)...")
    print("-" * 65)
    print("Dashboard UI:   http://localhost:5173")
    print("API Swagger:    http://localhost:8000/docs")
    print("Press Ctrl+C at any time to shut down both servers.")
    print("=" * 65)

    backend_cmd = [
        sys.executable, "-m", "uvicorn", "app.main:app",
        "--host", "0.0.0.0", "--port", "8000", "--reload",
        "--reload-dir", "app",
        "--reload-exclude", "*.db",
        "--reload-exclude", "uploads/*"
    ]
    backend_process = subprocess.Popen(
        backend_cmd,
        cwd=backend_dir
    )

    # Launch Frontend
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    frontend_process = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=frontend_dir
    )

    def signal_handler(sig, frame):
        print("\nShutting down SentinelVision AI servers...")
        try:
            frontend_process.terminate()
            backend_process.terminate()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while True:
            time.sleep(1)
            # If any process terminated prematurely
            if backend_process.poll() is not None:
                print("Backend exited. Shutting down frontend...")
                frontend_process.terminate()
                break
            if frontend_process.poll() is not None:
                print("Frontend exited. Shutting down backend...")
                backend_process.terminate()
                break
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main()
