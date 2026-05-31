"""Summarize CSV training logs into article-ready tables."""

import argparse
import csv
import math
import statistics as st
from collections import defaultdict
from pathlib import Path


def _float(row, key):
    value = row.get(key, "")
    if value in ("", "nan", "None"):
        return float("nan")
    return float(value)


def _mean(values):
    values = [value for value in values if not math.isnan(value)]
    return st.mean(values) if values else float("nan")


def _median(values):
    values = [value for value in values if not math.isnan(value)]
    return st.median(values) if values else float("nan")


def _fmt(value, digits=4):
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def load_rows(paths):
    rows = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["_source"] = str(path)
                rows.append(row)
    return rows


def method_family(name):
    if "adaptive_stochastic" in name:
        return "new_stochastic"
    if "stochastic_old" in name:
        return "old_stochastic"
    if "stochastic" in name:
        return "new_stochastic"
    if "adamw" in name or "adam8bit" in name:
        return "baseline_optimizer"
    return "baseline_projector"


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["run_name"]].append(row)

    summary = []
    memory = []
    for name, run_rows in sorted(groups.items()):
        run_rows = sorted(run_rows, key=lambda row: int(row["step"]))
        losses = [_float(row, "train_loss") for row in run_rows]
        vals = [_float(row, "val_perplexity") for row in run_rows]
        valid_vals = [value for value in vals if not math.isnan(value)]
        peak_row = max(run_rows, key=lambda row: _float(row, "vram_total_peak_gb"))

        summary.append(
            {
                "method": name,
                "method_family": method_family(name),
                "max_step": run_rows[-1]["step"],
                "rows": len(run_rows),
                "final_loss": losses[-1],
                "median_loss": _median(losses),
                "min_loss": min(losses),
                "max_loss": max(losses),
                "last_val_ppl": valid_vals[-1] if valid_vals else float("nan"),
                "min_val_ppl": min(valid_vals) if valid_vals else float("nan"),
                "peak_vram_gb": _float(peak_row, "vram_total_peak_gb"),
                "avg_tokens_per_sec": _mean(
                    [_float(row, "tokens_per_sec") for row in run_rows]
                ),
                "has_nan": any(math.isnan(value) for value in losses + valid_vals),
            }
        )
        memory.append(
            {
                "method": name,
                "method_family": method_family(name),
                "model_gb": _float(peak_row, "vram_model_gb"),
                "optimizer_gb": _float(peak_row, "vram_optimizer_gb"),
                "gradients_gb": _float(peak_row, "vram_gradients_gb"),
                "activations_gb": _float(peak_row, "vram_activations_gb"),
                "peak_vram_gb": _float(peak_row, "vram_total_peak_gb"),
            }
        )
    return summary, memory


def stochastic_comparison(summary):
    return [
        row
        for row in summary
        if row["method_family"] in {"old_stochastic", "new_stochastic"}
    ]


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_table(title, rows, fields):
    print(f"\n{title}")
    print(",".join(fields))
    for row in rows:
        print(
            ",".join(
                _fmt(row[field], 4) if isinstance(row[field], float) else str(row[field])
                for field in fields
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_logs", nargs="+")
    parser.add_argument("--out-prefix", default="runs/summary")
    args = parser.parse_args()

    rows = load_rows(args.csv_logs)
    summary, memory = summarize(rows)
    stochastic_rows = stochastic_comparison(summary)
    write_csv(f"{args.out_prefix}_methods.csv", summary)
    write_csv(f"{args.out_prefix}_memory.csv", memory)
    write_csv(f"{args.out_prefix}_stochastic_ablation.csv", stochastic_rows)

    print_table(
        "METHOD SUMMARY",
        summary,
        [
            "method",
            "method_family",
            "max_step",
            "final_loss",
            "median_loss",
            "last_val_ppl",
            "peak_vram_gb",
            "avg_tokens_per_sec",
            "has_nan",
        ],
    )
    print_table(
        "MEMORY DECOMPOSITION",
        memory,
        [
            "method",
            "method_family",
            "model_gb",
            "optimizer_gb",
            "gradients_gb",
            "activations_gb",
            "peak_vram_gb",
        ],
    )
    if stochastic_rows:
        print_table(
            "STOCHASTIC OLD VS NEW",
            stochastic_rows,
            [
                "method",
                "method_family",
                "max_step",
                "final_loss",
                "last_val_ppl",
                "peak_vram_gb",
                "avg_tokens_per_sec",
            ],
        )


if __name__ == "__main__":
    main()
