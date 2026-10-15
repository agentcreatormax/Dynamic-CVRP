#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit(
        "ERROR: matplotlib is required for plotting. "
        "Use a Python environment with matplotlib installed."
    )

events = []
times = []

with csv_path.open(newline="") as f:
    for row in csv.DictReader(f):
        if row["status"] != "OK":
            continue
        try:
            x = int(row["events"])
            y = float(row["algorithm_time_sec"])
        except (TypeError, ValueError):
            continue
        events.append(x)
        times.append(y)

if not events:
    raise SystemExit("ERROR: no successful load-test runs are available to plot.")

# Natural view for doubling workloads.
fig, ax = plt.subplots(figsize=(10, 5.8))
ax.plot(events, times, marker="o", linewidth=2)
ax.set_xscale("log", base=2)
ax.set_xticks(events)
ax.set_xticklabels([f"{x:,}" for x in events], rotation=42, ha="right")
ax.set_xlabel("Total dynamic events (50% DELETE + 50% ADD)")
ax.set_ylabel("Algorithm time (seconds)")
ax.set_title("Large5 Runi load test: event count vs algorithm time")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(out_dir / "events_vs_algorithm_time.png", dpi=180)
plt.close(fig)

# Also save an ordinary linear-x version.
fig, ax = plt.subplots(figsize=(10, 5.8))
ax.plot(events, times, marker="o", linewidth=2)
ax.set_xlabel("Total dynamic events (50% DELETE + 50% ADD)")
ax.set_ylabel("Algorithm time (seconds)")
ax.set_title("Large5 Runi load test: event count vs algorithm time")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(out_dir / "events_vs_algorithm_time_linear.png", dpi=180)
plt.close(fig)

print(f"Plotted {len(events)} successful runs.")
