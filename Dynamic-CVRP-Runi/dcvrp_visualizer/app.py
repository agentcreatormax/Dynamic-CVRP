#!/usr/bin/env python3
"""
Local Dynamic-CVRP Runi visualizer backend.

No third-party Python packages are required. It serves the frontend and runs
the project's dcvrp.out for ADD / DELETE / refine actions.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

VIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", str(VIS_DIR.parent))).resolve()
DCVRP_BIN = Path(os.environ.get("DCVRP_BIN", str(PROJECT_DIR / "dcvrp.out"))).resolve()

STATIC_DIR = VIS_DIR / "static"
SAMPLE_DIR = VIS_DIR / "sample_data"
RUNTIME_DIR = VIS_DIR / "runtime"
CURRENT_VRP = RUNTIME_DIR / "current.vrp"
CURRENT_SOL = RUNTIME_DIR / "current.sol"
LAST_JSON = RUNTIME_DIR / "last_run.json"
HISTORY_DIR = RUNTIME_DIR / "history"

DEFAULT_START_VRP = Path(
    os.environ.get("START_VRP", str(SAMPLE_DIR / "toy_start.vrp"))
).resolve()
DEFAULT_START_SOL = Path(
    os.environ.get("START_SOL", str(SAMPLE_DIR / "toy_start.sol"))
).resolve()

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
RUN_TIMEOUT = int(os.environ.get("RUN_TIMEOUT", "600"))

DEFAULT_HMAX = int(os.environ.get("HMAX", "5"))
DEFAULT_REFINE_K = int(os.environ.get("REFINE_K", "5"))
DEFAULT_QUEUE_CAP = int(os.environ.get("QUEUE_CAP", "5"))
DEFAULT_PASSES = int(os.environ.get("PASSES", "1"))

STATE_LOCK = threading.Lock()


def ensure_runtime(reset: bool = False) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    if reset or not CURRENT_VRP.exists() or not CURRENT_SOL.exists():
        if not DEFAULT_START_VRP.exists():
            raise FileNotFoundError(f"Start VRP not found: {DEFAULT_START_VRP}")
        if not DEFAULT_START_SOL.exists():
            raise FileNotFoundError(f"Start SOL not found: {DEFAULT_START_SOL}")

        shutil.copy2(DEFAULT_START_VRP, CURRENT_VRP)
        shutil.copy2(DEFAULT_START_SOL, CURRENT_SOL)
        if LAST_JSON.exists():
            LAST_JSON.unlink()


def parse_vrp(path: Path) -> dict:
    coords = {}
    demands = {}
    capacity = None
    name = path.stem
    mode = None

    with path.open() as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue

            upper = s.upper()
            if upper.startswith("NAME"):
                if ":" in s:
                    name = s.split(":", 1)[1].strip()
            elif upper.startswith("CAPACITY"):
                capacity = float(s.replace(":", " ").split()[-1])
            elif upper == "NODE_COORD_SECTION":
                mode = "coords"
            elif upper == "DEMAND_SECTION":
                mode = "demands"
            elif upper == "DEPOT_SECTION":
                mode = "depot"
            elif upper == "EOF":
                break
            elif mode == "coords":
                p = s.split()
                if len(p) >= 3:
                    coords[int(p[0])] = (float(p[1]), float(p[2]))
            elif mode == "demands":
                p = s.split()
                if len(p) >= 2:
                    demands[int(p[0])] = float(p[1])

    if 1 not in coords:
        raise ValueError(f"{path} does not contain depot node 1")

    if capacity is None:
        raise ValueError(f"{path} does not contain CAPACITY")

    depot_xy = coords[1]
    customers = {}
    for node, (x, y) in coords.items():
        if node == 1:
            continue
        # Dynamic-CVRP internal customer ID convention:
        # TSPLIB node 2 -> customer 1, etc.
        cid = node - 1
        customers[cid] = {
            "id": cid,
            "node": node,
            "x": x,
            "y": y,
            "demand": float(demands.get(node, 0.0)),
        }

    return {
        "name": name,
        "capacity": capacity,
        "depot": {"x": depot_xy[0], "y": depot_xy[1]},
        "customers": customers,
    }


def parse_sol(path: Path) -> dict:
    routes = []
    cost = None

    with path.open() as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue

            m = re.match(r"Route\s*#?\s*\d+\s*:\s*(.*)$", s, re.I)
            if m:
                body = m.group(1).strip()
                ids = [int(x) for x in re.findall(r"-?\d+", body)]
                routes.append(ids)
                continue

            m = re.match(r"Cost\s*:?\s*([-+0-9.eE]+)", s, re.I)
            if m:
                cost = float(m.group(1))

    return {"routes": routes, "cost": cost}


def read_profile(path: Path) -> dict:
    if not path.exists():
        return {}

    out = {}
    with path.open(newline="") as f:
        rows = list(csv.reader(f))

    for row in rows[1:]:
        if len(row) < 2:
            continue
        key = row[0].strip()
        raw = row[1].strip()
        try:
            val = float(raw)
            if val.is_integer() and (
                key.endswith("_warps") or key.startswith("max_")
            ):
                val = int(val)
            out[key] = val
        except ValueError:
            out[key] = raw

    # Convenient derived value used by the UI.
    try:
        out["algorithm_time_ms"] = (
            float(out.get("delete_total_ms", 0.0))
            + float(out.get("add_total_ms", 0.0))
            + float(out.get("refine_total_ms", 0.0))
        )
    except Exception:
        pass

    return out


def parse_refine_passes(log_text: str) -> tuple[int | None, int | None]:
    matches = re.findall(r"\[gpu-refine\]\s+passes=(\d+)/(\d+)", log_text)
    if not matches:
        return None, None
    run, requested = matches[-1]
    return int(run), int(requested)


def capture_current_state() -> dict:
    """Capture the current solution before an interactive operation.

    This snapshot is returned with the next result so the browser can overlay
    the previous routes and show route transitions.
    """
    inst = parse_vrp(CURRENT_VRP)
    sol = parse_sol(CURRENT_SOL)

    customers = inst["customers"]
    routes_payload = []
    customer_route = {}

    for r_idx, route in enumerate(sol["routes"], start=1):
        clean = [cid for cid in route if cid in customers]
        load = sum(customers[cid]["demand"] for cid in clean)
        routes_payload.append({
            "index": r_idx,
            "customers": clean,
            "load": load,
        })
        for pos, cid in enumerate(clean, start=1):
            customer_route[cid] = {"route": r_idx, "position": pos}

    customer_payload = []
    for cid in sorted(customers):
        c = dict(customers[cid])
        c.update(customer_route.get(cid, {"route": None, "position": None}))
        customer_payload.append(c)

    return {
        "instance": inst["name"],
        "capacity": inst["capacity"],
        "depot": inst["depot"],
        "customers": customer_payload,
        "routes": routes_payload,
        "cost": sol["cost"],
    }


def build_state(last: dict | None = None) -> dict:
    ensure_runtime()
    inst = parse_vrp(CURRENT_VRP)
    sol = parse_sol(CURRENT_SOL)

    customers = inst["customers"]
    routes_payload = []
    customer_route = {}

    for r_idx, route in enumerate(sol["routes"], start=1):
        clean = [cid for cid in route if cid in customers]
        load = sum(customers[cid]["demand"] for cid in clean)
        routes_payload.append(
            {
                "index": r_idx,
                "customers": clean,
                "load": load,
            }
        )
        for pos, cid in enumerate(clean, start=1):
            customer_route[cid] = {"route": r_idx, "position": pos}

    customer_payload = []
    for cid in sorted(customers):
        c = dict(customers[cid])
        rp = customer_route.get(cid, {"route": None, "position": None})
        c.update(rp)
        customer_payload.append(c)

    if last is None and LAST_JSON.exists():
        try:
            last = json.loads(LAST_JSON.read_text())
        except Exception:
            last = None

    return {
        "ok": True,
        "instance": inst["name"],
        "capacity": inst["capacity"],
        "depot": inst["depot"],
        "customers": customer_payload,
        "routes": routes_payload,
        "cost": sol["cost"],
        "last": last,
        "defaults": {
            "passes": DEFAULT_PASSES,
            "hmax": DEFAULT_HMAX,
            "refine_k": DEFAULT_REFINE_K,
            "queue_cap": DEFAULT_QUEUE_CAP,
        },
    }


def finite_number(value, field: str) -> float:
    try:
        x = float(value)
    except Exception:
        raise ValueError(f"{field} must be a number")
    if not math.isfinite(x):
        raise ValueError(f"{field} must be finite")
    return x


def safe_int(value, field: str, lo: int, hi: int) -> int:
    try:
        x = int(value)
    except Exception:
        raise ValueError(f"{field} must be an integer")
    if not (lo <= x <= hi):
        raise ValueError(f"{field} must be between {lo} and {hi}")
    return x


def run_action(action: str, payload: dict) -> dict:
    if not DCVRP_BIN.exists():
        raise RuntimeError(
            f"dcvrp.out not found at {DCVRP_BIN}. "
            "Run ./start_visualizer.sh so the project is built first."
        )

    ensure_runtime()

    passes = safe_int(payload.get("passes", DEFAULT_PASSES), "passes", 0, 100)
    hmax = safe_int(payload.get("hmax", DEFAULT_HMAX), "hmax", 1, 64)
    refine_k = safe_int(payload.get("refine_k", DEFAULT_REFINE_K), "refine_k", 1, 64)
    queue_cap = safe_int(payload.get("queue_cap", DEFAULT_QUEUE_CAP), "queue_cap", 1, 16)

    inst_before = parse_vrp(CURRENT_VRP)
    sol_before = parse_sol(CURRENT_SOL)
    before_cost = sol_before["cost"]
    previous_state = capture_current_state()

    event_line = ""
    action_desc = action

    if action == "add":
        x = finite_number(payload.get("x"), "x")
        y = finite_number(payload.get("y"), "y")
        demand = finite_number(payload.get("demand"), "demand")
        if demand <= 0:
            raise ValueError("demand must be > 0")
        if demand > inst_before["capacity"]:
            raise ValueError(
                f"demand {demand:g} exceeds vehicle capacity "
                f"{inst_before['capacity']:g}"
            )
        event_line = f"A {x:.12g} {y:.12g} {demand:.12g}\n"
        action_desc = f"ADD ({x:g}, {y:g}) demand={demand:g}"

    elif action == "delete":
        cid = safe_int(payload.get("id"), "customer id", 1, 10**9)
        if cid not in inst_before["customers"]:
            raise ValueError(f"customer {cid} does not exist in the current state")
        event_line = f"D {cid}\n"
        action_desc = f"DELETE customer {cid}"

    elif action == "update_demand":
        cid = safe_int(payload.get("id"), "customer id", 1, 10**9)
        if cid not in inst_before["customers"]:
            raise ValueError(f"customer {cid} does not exist in the current state")
        demand = finite_number(payload.get("demand"), "demand")
        if demand <= 0:
            raise ValueError("demand must be > 0")
        if demand > inst_before["capacity"]:
            raise ValueError(
                f"demand {demand:g} exceeds vehicle capacity "
                f"{inst_before['capacity']:g}"
            )
        c = inst_before["customers"][cid]
        old_demand = c["demand"]
        # The controller has no separate UPDATE event.  Model a quantity change
        # as DELETE(old request) + ADD(same location, new demand).  The current
        # controller already applies all deletes before adds and refines once.
        event_line = (
            f"D {cid}\n"
            f"A {c['x']:.12g} {c['y']:.12g} {demand:.12g}\n"
        )
        action_desc = (
            f"UPDATE quantity customer {cid}: "
            f"{old_demand:g} -> {demand:g}"
        )

    elif action == "refine":
        # An empty event stream still causes the controller to run refinement.
        event_line = ""
        action_desc = "REFINE only"

    else:
        raise ValueError(f"unknown action: {action}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = HISTORY_DIR / f"{stamp}_{action}"
    run_dir.mkdir(parents=True, exist_ok=False)

    event_file = run_dir / "event.txt"
    event_file.write_text(event_line)

    cmd = [
        str(DCVRP_BIN),
        "--vrp", str(CURRENT_VRP),
        "--sol", str(CURRENT_SOL),
        "--events", str(event_file),
        "--out-dir", str(run_dir),
        "--engine", "gpu",
        "--passes", str(passes),
        "--hmax", str(hmax),
        "--refine-k", str(refine_k),
        "--queue-cap", str(queue_cap),
        "--snap-vrp",
        "--verbose",
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=RUN_TIMEOUT,
        check=False,
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0

    log_text = proc.stdout
    (run_dir / "run.log").write_text(log_text)

    if proc.returncode != 0:
        tail = "\n".join(log_text.splitlines()[-35:])
        raise RuntimeError(
            f"dcvrp.out failed with exit code {proc.returncode}\n\n{tail}"
        )

    final_vrp = run_dir / "final.vrp"
    final_sol = run_dir / "final.sol"
    if not final_vrp.exists() or not final_sol.exists():
        raise RuntimeError(
            "dcvrp.out finished but final.vrp/final.sol were not produced"
        )

    profile = read_profile(run_dir / "summary.csv")
    passes_run, passes_requested_from_log = parse_refine_passes(log_text)

    # Commit this successful result as the next interactive state.
    shutil.copy2(final_vrp, CURRENT_VRP)
    shutil.copy2(final_sol, CURRENT_SOL)

    sol_after = parse_sol(CURRENT_SOL)
    after_cost = sol_after["cost"]

    last = {
        "action": action,
        "description": action_desc,
        "event": event_line.strip(),
        "before_cost": before_cost,
        "after_cost": after_cost,
        "cost_change": (
            None if before_cost is None or after_cost is None
            else after_cost - before_cost
        ),
        "wall_time_ms": wall_ms,
        "passes_requested": (
            passes_requested_from_log
            if passes_requested_from_log is not None
            else passes
        ),
        "passes_run": passes_run,
        "profile": profile,
        "previous_state": previous_state,
        "run_dir": str(run_dir),
        "timestamp": stamp,
    }

    LAST_JSON.write_text(json.dumps(last, indent=2))
    return build_state(last)


def reset_state() -> dict:
    ensure_runtime(reset=True)
    return build_state(
        {
            "action": "reset",
            "description": "Reset to toy start state",
            "profile": {},
            "passes_requested": None,
            "passes_run": None,
        }
    )


class Handler(BaseHTTPRequestHandler):
    server_version = "DCVRPVisualizer/1.0"

    def log_message(self, fmt, *args):
        print("[web]", fmt % args)

    def send_json(self, obj: dict, status=HTTPStatus.OK):
        raw = json.dumps(obj, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def send_file(self, path: Path, content_type: str):
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        raw = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(raw)

    def read_json_body(self) -> dict:
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0:
            return {}
        if n > 1024 * 1024:
            raise ValueError("request is too large")
        data = self.rfile.read(n)
        return json.loads(data.decode("utf-8"))

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/":
            self.send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return
        if path == "/static/app.js":
            self.send_file(STATIC_DIR / "app.js", "text/javascript; charset=utf-8")
            return
        if path == "/static/style.css":
            self.send_file(STATIC_DIR / "style.css", "text/css; charset=utf-8")
            return
        if path == "/api/state":
            try:
                with STATE_LOCK:
                    self.send_json(build_state())
            except Exception as exc:
                self.send_json(
                    {"ok": False, "error": str(exc)},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            payload = self.read_json_body()

            with STATE_LOCK:
                if path == "/api/add":
                    result = run_action("add", payload)
                elif path == "/api/delete":
                    result = run_action("delete", payload)
                elif path == "/api/update-demand":
                    result = run_action("update_demand", payload)
                elif path == "/api/refine":
                    result = run_action("refine", payload)
                elif path == "/api/reset":
                    result = reset_state()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return

            self.send_json(result)

        except ValueError as exc:
            self.send_json(
                {"ok": False, "error": str(exc)},
                HTTPStatus.BAD_REQUEST,
            )
        except subprocess.TimeoutExpired:
            self.send_json(
                {
                    "ok": False,
                    "error": f"dcvrp.out exceeded the {RUN_TIMEOUT}s timeout",
                },
                HTTPStatus.GATEWAY_TIMEOUT,
            )
        except Exception as exc:
            self.send_json(
                {"ok": False, "error": str(exc)},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def main():
    ensure_runtime()
    print()
    print("Dynamic-CVRP Runi Visualizer")
    print("-----------------------------")
    print(f"Project : {PROJECT_DIR}")
    print(f"Binary  : {DCVRP_BIN}")
    print(f"Start   : {DEFAULT_START_VRP}")
    print(f"URL     : http://{HOST}:{PORT}")
    print()
    print("Use Ctrl+C to stop.")
    print()

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping visualizer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
