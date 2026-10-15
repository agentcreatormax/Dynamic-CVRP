#!/usr/bin/env python3
from pathlib import Path
import argparse
import random
import re

SIZES = [50,100,200,400,800,1600,3200,6400,12800,25600,51200,102400,204800,409600]

def clean_lines(path):
    return [
        s.strip() for s in Path(path).read_text().splitlines()
        if s.strip() and not s.lstrip().startswith("#")
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", default=str(Path.home() / "large5_benchmark"))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "events"))
    args = ap.parse_args()

    bench = Path(args.bench_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    adds0 = clean_lines(bench / "events" / "A_200_add.txt")
    dels0 = clean_lines(bench / "events" / "B_200_delete.txt")

    max_half = max(SIZES) // 2

    adds = list(adds0)
    r = random.Random(500042)
    while len(adds) < max_half:
        adds.append(f"A {r.randint(0,1000)} {r.randint(0,1000)} {r.randint(1,100)}")

    seen = set()
    dids = []
    for line in dels0:
        m = re.fullmatch(r"D\s+(\d+)", line)
        if not m:
            raise SystemExit(f"Unexpected DELETE line: {line}")
        v = int(m.group(1))
        if v not in seen:
            seen.add(v)
            dids.append(v)

    r = random.Random(500043)
    while len(dids) < max_half:
        v = r.randint(1, 500000)
        if v not in seen:
            seen.add(v)
            dids.append(v)

    dels = [f"D {v}" for v in dids]

    for total in SIZES:
        half = total // 2
        path = out / f"large5_mixed_{total}.txt"
        with path.open("w") as f:
            f.write(f"# Large5 mixed load test: total={total}, deletes={half}, adds={half}\n")
            f.write("# DELETE -> ADD -> one refinement\n")
            for line in dels[:half]:
                f.write(line + "\n")
            for line in adds[:half]:
                f.write(line + "\n")
        print(path)

if __name__ == "__main__":
    main()
