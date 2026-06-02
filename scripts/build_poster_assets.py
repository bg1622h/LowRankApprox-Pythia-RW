"""Build poster bundle: graphs + raw data co-located in each section."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POSTER = ROOT / "runs" / "poster"
REPO = "https://github.com/bg1622h/LowRankApprox-Pythia-RW"
BRANCH = "main"
BASE = f"{REPO}/blob/{BRANCH}/runs/poster"
TREE = f"{REPO}/tree/{BRANCH}/runs/poster"

BASELINE_CSV = [
    ("refinedweb_full_galore2_1500_seq2048.csv", ROOT / "runs/refinedweb_full_galore2_1500_seq2048.csv"),
    ("refinedweb_full_lotus_1500_seq2048.csv", ROOT / "runs/refinedweb_full_lotus_1500_seq2048.csv"),
]

COMPARISON_CSV = [
    (
        "refinedweb_full_adaptive_stochastic_1500_seq2048.csv",
        ROOT / "runs/refinedweb_full_adaptive_stochastic_1500_seq2048.csv",
    ),
    (
        "refinedweb_full_fisher_both_1500_seq2048_cr128.csv",
        ROOT / "runs/refinedweb_full_fisher_both_1500_seq2048_cr128.csv",
    ),
    (
        "refinedweb_full_fisher_topk_softmax_1500_seq2048_cr64.csv",
        ROOT / "runs/refinedweb_full_fisher_topk_softmax_1500_seq2048_cr64.csv",
    ),
    (
        "summary_methods.csv",
        ROOT / "runs/refinedweb_full_summary_1500_seq2048_methods.csv",
    ),
    (
        "summary_memory.csv",
        ROOT / "runs/refinedweb_full_summary_1500_seq2048_memory.csv",
    ),
]

FISHER_SECTIONS = [
    {
        "tag": "cr128",
        "jsonl": ROOT / "runs/fisher_cr128_jsonl",
        "training_csv": (
            "refinedweb_full_fisher_both_1500_seq2048_cr128.csv",
            ROOT / "runs/refinedweb_full_fisher_both_1500_seq2048_cr128.csv",
        ),
    },
    {
        "tag": "cr64",
        "jsonl": ROOT / "runs/fisher_both_cr64_jsonl",
        "training_csv": (
            "refinedweb_full_fisher_both_1500_seq2048_cr64.csv",
            ROOT / "runs/refinedweb_full_fisher_both_1500_seq2048_cr64.csv",
        ),
    },
    {
        "tag": "topk_softmax_cr64",
        "jsonl": ROOT / "runs/fisher_topk_softmax_cr64_jsonl",
        "training_csv": (
            "refinedweb_full_fisher_topk_softmax_1500_seq2048_cr64.csv",
            ROOT / "runs/refinedweb_full_fisher_topk_softmax_1500_seq2048_cr64.csv",
        ),
    },
]

SPECTRUM_CSV = [
    ("spectrum_all_layers_summary.csv", ROOT / "runs/spectrum_all_layers_summary.csv"),
    ("spectrum_all_layers.csv", ROOT / "runs/spectrum_all_layers.csv"),
]

SPECTRUM_PLOT_SOURCES = [
    ROOT / "runs/spectrum_all_layers_plots",
    ROOT / "runs/spectrum_plots",
]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def _copy_file(src: Path, dest: Path) -> bool:
    if not src.is_file():
        print(f"  skip missing {src}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def _copy_tree(src: Path, dest: Path) -> int:
    if not src.is_dir():
        print(f"  skip missing dir {src}")
        return 0
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    return sum(1 for _ in dest.rglob("*") if _.is_file())


def _clean_section(section: Path, keep_names: set[str]) -> None:
    if not section.exists():
        return
    for item in section.iterdir():
        if item.name not in keep_names and item.name != "data":
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


def stage_baselines(poster_dir: Path) -> Path:
    section = poster_dir / "baselines"
    section.mkdir(parents=True, exist_ok=True)
    data = section / "data"
    data.mkdir(exist_ok=True)
    for name, src in BASELINE_CSV:
        _copy_file(src, data / name)
    return section


def stage_comparison(poster_dir: Path) -> Path:
    section = poster_dir / "comparison"
    section.mkdir(parents=True, exist_ok=True)
    data = section / "data"
    data.mkdir(exist_ok=True)
    for name, src in COMPARISON_CSV:
        _copy_file(src, data / name)
    return section


def stage_fisher_section(section_dir: Path, jsonl_src: Path, training_name: str, training_src: Path) -> None:
    data = section_dir / "data"
    data.mkdir(parents=True, exist_ok=True)
    _copy_file(training_src, data / training_name)
    jsonl_dest = data / "jsonl"
    count = _copy_tree(jsonl_src, jsonl_dest)
    print(f"  {section_dir.name}: copied training csv + {count} jsonl files")


def build_fisher_boxplots(poster_dir: Path) -> None:
    fisher_root = poster_dir / "fisher_boxplots"
    fisher_root.mkdir(parents=True, exist_ok=True)

    jsonl_dirs: list[Path] = []
    for spec in FISHER_SECTIONS:
        tag = spec["tag"]
        section_dir = fisher_root / tag
        section_dir.mkdir(parents=True, exist_ok=True)
        stage_fisher_section(
            section_dir,
            spec["jsonl"],
            spec["training_csv"][0],
            spec["training_csv"][1],
        )
        jsonl_src = spec["jsonl"]
        if jsonl_src.is_dir():
            jsonl_dirs.append(jsonl_src)
            _run(
                [
                    sys.executable,
                    str(ROOT / "scripts/plot_fisher.py"),
                    str(jsonl_src),
                    "--out-dir",
                    str(section_dir),
                    "--per-projector",
                ]
            )

    combined = fisher_root / "all_datasets"
    combined.mkdir(parents=True, exist_ok=True)
    note = combined / "data" / "README.txt"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "Combined Fisher boxplots only.\n"
        "Raw JSONL + training CSV live in sibling folders:\n"
        "  cr128/, cr64/, topk_softmax_cr64/\n",
        encoding="utf-8",
    )
    if jsonl_dirs:
        _run(
            [
                sys.executable,
                str(ROOT / "scripts/plot_fisher.py"),
                *[str(p) for p in jsonl_dirs],
                "--out-dir",
                str(combined),
                "--per-projector",
            ]
        )


def build_spectrum_boxplots(poster_dir: Path) -> None:
    section = poster_dir / "spectrum_boxplots"
    section.mkdir(parents=True, exist_ok=True)
    data = section / "data"
    data.mkdir(exist_ok=True)
    for name, src in SPECTRUM_CSV:
        _copy_file(src, data / name)

    copied = 0
    for src_dir in SPECTRUM_PLOT_SOURCES:
        if not src_dir.is_dir():
            continue
        for png in src_dir.glob("*.png"):
            shutil.copy2(png, section / png.name)
            copied += 1
    print(f"  spectrum_boxplots: {copied} png + csv in data/")


def cleanup_legacy(poster_dir: Path) -> None:
    for name in ("poster_baselines_slide.png", "poster_final_val_bars.png", "poster_val_ppl_curves.png"):
        path = poster_dir / name
        if path.exists():
            path.unlink()

    fisher_root = poster_dir / "fisher_boxplots"
    if not fisher_root.is_dir():
        return
    keep = {"cr128", "cr64", "topk_softmax_cr64", "all_datasets"}
    for item in fisher_root.iterdir():
        if item.name in keep:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        elif item.suffix == ".png":
            item.unlink()


def write_readme(poster_dir: Path) -> None:
    b = f"{BASE}/baselines"
    c = f"{BASE}/comparison"
    f = f"{TREE}/fisher_boxplots"
    s = f"{BASE}/spectrum_boxplots"

    text = f"""# Poster bundle (graphs + raw data)

Regenerate: `python scripts/build_poster_assets.py`

---

## Baselines — GaLore2 + Lotus

Folder: {TREE}/baselines

### Graphs
- {b}/baseline_val_curves.png
- {b}/baseline_final_bars.png

### Data
- {b}/data/refinedweb_full_galore2_1500_seq2048.csv
- {b}/data/refinedweb_full_lotus_1500_seq2048.csv

---

## Comparison — all methods

Folder: {TREE}/comparison

### Graphs
- {c}/all_methods_val_curves.png
- {c}/all_methods_final_bars.png
- {c}/comparison_baselines_vs_fisher.png

### Data
- {c}/data/refinedweb_full_adaptive_stochastic_1500_seq2048.csv
- {c}/data/refinedweb_full_fisher_both_1500_seq2048_cr128.csv
- {c}/data/refinedweb_full_fisher_topk_softmax_1500_seq2048_cr64.csv
- {c}/data/summary_methods.csv
- {c}/data/summary_memory.csv

---

## Fisher boxplots

Folder: {f}

Each subfolder has `data/training_*.csv`, `data/jsonl/`, and PNG boxplots.

### cr128
- {BASE}/fisher_boxplots/cr128/fisher_group_overlap_boxplots.png
- {BASE}/fisher_boxplots/cr128/fisher_group_mass_boxplots.png
- {BASE}/fisher_boxplots/cr128/fisher_group_effective_rank_boxplots.png
- {BASE}/fisher_boxplots/cr128/data/refinedweb_full_fisher_both_1500_seq2048_cr128.csv
- {TREE}/fisher_boxplots/cr128/data/jsonl

### cr64
- {BASE}/fisher_boxplots/cr64/fisher_group_overlap_boxplots.png
- {BASE}/fisher_boxplots/cr64/data/refinedweb_full_fisher_both_1500_seq2048_cr64.csv
- {TREE}/fisher_boxplots/cr64/data/jsonl

### topk_softmax_cr64
- {BASE}/fisher_boxplots/topk_softmax_cr64/fisher_group_overlap_boxplots.png
- {BASE}/fisher_boxplots/topk_softmax_cr64/data/refinedweb_full_fisher_topk_softmax_1500_seq2048_cr64.csv
- {TREE}/fisher_boxplots/topk_softmax_cr64/data/jsonl

### all_datasets (combined plots)
- {BASE}/fisher_boxplots/all_datasets/fisher_group_overlap_boxplots.png
- {BASE}/fisher_boxplots/all_datasets/fisher_group_mass_boxplots.png
- {BASE}/fisher_boxplots/all_datasets/fisher_group_effective_rank_boxplots.png

---

## Spectrum boxplots

Folder: {TREE}/spectrum_boxplots

### Graphs
- {s}/spectrum_group_energy_boxplots.png
- {s}/spectrum_group_effective_rank_boxplots.png
- {s}/spectrum_group_threshold_rank_boxplots.png
- {s}/spectrum_group_angle_boxplots.png
- {s}/spectrum_group_adaptive_rank_boxplots.png

### Data
- {s}/data/spectrum_all_layers_summary.csv
- {s}/data/spectrum_all_layers.csv
"""
    (poster_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(POSTER))
    args = parser.parse_args()
    poster_dir = Path(args.out_dir)
    poster_dir.mkdir(parents=True, exist_ok=True)

    stage_baselines(poster_dir)
    stage_comparison(poster_dir)
    _run([sys.executable, str(ROOT / "scripts/plot_poster.py"), "--out-dir", str(poster_dir)])
    build_fisher_boxplots(poster_dir)
    build_spectrum_boxplots(poster_dir)
    cleanup_legacy(poster_dir)
    write_readme(poster_dir)
    print(f"\nDone: {poster_dir}")
    print(f"Share after push: {TREE}")


if __name__ == "__main__":
    main()
