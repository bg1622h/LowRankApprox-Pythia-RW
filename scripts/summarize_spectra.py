"""Summarize singular-value diagnostics JSONL files."""

import argparse
import csv
import json
from pathlib import Path


def iter_records(directory):
    for path in sorted(Path(directory).glob("singular_values_*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def summarize(records):
    rows = []
    for record in records:
        angles = record.get("principal_angles_degrees", [])
        rows.append(
            {
                "projector": record.get("projector"),
                "param_group": record.get("param_group"),
                "param_name": record.get("param_name"),
                "step": record.get("step"),
                "rank": record.get("rank"),
                "rank_energy": record.get("rank_energy"),
                "rank_90": record.get("rank_90"),
                "rank_95": record.get("rank_95"),
                "rank_99": record.get("rank_99"),
                "effective_rank": record.get("effective_rank"),
                "mean_angle": sum(angles) / len(angles) if angles else "",
                "max_angle": max(angles) if angles else "",
                "adaptive_rank": record.get("adaptive_rank", ""),
                "temperature": record.get("temperature", ""),
                "candidate_count": record.get("candidate_count", ""),
            }
        )
    return rows


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spectrum_log_dir")
    parser.add_argument("--out", default="runs/spectrum_summary.csv")
    args = parser.parse_args()

    rows = summarize(iter_records(args.spectrum_log_dir))
    write_csv(args.out, rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
