"""
Stitch together several test_ppe_models.py --only <combo> --out <file>.json
outputs (each a single-element JSON list) into one combined test_results.json,
in CONFIGS order. Exists because a full 4-combo run takes ~15 minutes, which
this environment's background-command runtime turned out not to tolerate
reliably -- running each combo as its own short-lived process and merging
afterward sidesteps that without changing what gets measured.

Usage:
    python merge_results.py out_tiny_base.json out_base_base.json \
        out_tiny_large.json out_base_large.json --out test_results.json
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", default="test_results.json")
    args = parser.parse_args()

    merged = []
    for path in args.inputs:
        with open(path) as f:
            data = json.load(f)
        merged.extend(data if isinstance(data, list) else [data])

    with open(args.out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Wrote {args.out} ({len(merged)} configs)")


if __name__ == "__main__":
    main()
