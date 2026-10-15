"""
VRP Visualizer — Entry Point

Usage:
    python main.py <problem.vrp> <solution.sol>

Opens an interactive Dash app in the browser with real-time file watching.
"""

import sys
import os
import time
import threading
import webbrowser

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from app import AppState, create_app


class VRPFileHandler(FileSystemEventHandler):
    """Watchdog handler that reloads data when .vrp or .sol files change."""

    def __init__(self, state: AppState, vrp_path: str, sol_path: str):
        super().__init__()
        self.state = state
        self.vrp_name = os.path.basename(vrp_path)
        self.sol_name = os.path.basename(sol_path)
        self._debounce_time = 0

    def on_modified(self, event):
        if event.is_directory:
            return
        basename = os.path.basename(event.src_path)
        if basename not in (self.vrp_name, self.sol_name):
            return
        # Simple debounce — ignore events within 0.5s of each other
        now = time.time()
        if now - self._debounce_time < 0.5:
            return
        self._debounce_time = now
        print(f"[watchdog] Detected change in {basename}, reloading...")
        try:
            self.state.reload()
        except Exception as e:
            print(f"[watchdog] Error reloading: {e}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <problem.vrp> <solution.sol>")
        print("Example: python main.py A-n32-k5.vrp A-n32-k5.sol")
        sys.exit(1)

    vrp_path = os.path.abspath(sys.argv[1])
    sol_path = os.path.abspath(sys.argv[2])

    if not os.path.isfile(vrp_path):
        print(f"Error: VRP file not found: {vrp_path}")
        sys.exit(1)

    # SOL file is optional (will show nodes without routes)
    if not os.path.isfile(sol_path):
        print(f"Warning: SOL file not found: {sol_path}")
        print("Will display nodes without routes. Create the file to add routes.")

    print(f"VRP file: {vrp_path}")
    print(f"SOL file: {sol_path}")

    # Initialize shared state
    state = AppState(vrp_path, sol_path)
    vrp = state.vrp_data

    print(f"Loaded: {vrp['name']} -- {len(vrp['coords'])} nodes, "
          f"depot={vrp['depot']}, capacity={vrp['capacity']}")
    if state.sol_data["routes"]:
        print(f"Solution: {len(state.sol_data['routes'])} routes, "
              f"cost={state.sol_data['cost']}")

    # Start watchdog observer
    watch_dir = os.path.dirname(vrp_path) or "."
    handler = VRPFileHandler(state, vrp_path, sol_path)
    observer = Observer()
    observer.schedule(handler, watch_dir, recursive=False)
    observer.daemon = True
    observer.start()
    print(f"[watchdog] Monitoring {watch_dir} for file changes...")

    # Create and run Dash app
    app = create_app(state)

    port = 8051
    print(f"\n>> Starting VRP Visualizer at http://127.0.0.1:{port}")
    print("   Press Ctrl+C to stop.\n")

    # Open browser after a short delay
    def open_browser():
        time.sleep(1.5)
        webbrowser.open(f"http://127.0.0.1:{port}")

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        app.run(debug=False, port=port, host="127.0.0.1")
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
