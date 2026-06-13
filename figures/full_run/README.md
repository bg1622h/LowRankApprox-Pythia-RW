# Full-run figures

Generated from `streaming_v2_results/clean/*_1500clean_fp16.csv`
(extracted from `lowrank_experiments_full.tar`).

Settings: TinyLlama 1.1B, rank 8, batch 16, seq 2048, 1500 steps, fp16.

Regenerate:
```bash
python scripts/generate_full_run_figures.py
```

## Generated figures

- `article/experiment_loss_curves.pdf`
- `article/experiment_vram_components.pdf`
- `baselines/baseline_final_bars.png`
- `baselines/baseline_val_curves.png`
- `comparison/all_methods_final_bars.png`
- `comparison/all_methods_val_curves.png`
- `comparison/comparison_baselines_vs_stochastic.png`

## Source CSV logs

- **galore2**: `streaming_v2_results/clean/adammini_galore2_r8_1500clean_fp16.csv`
- **lotus**: `streaming_v2_results/clean/adammini_lotus_r8_1500clean_fp16.csv`
- **adaptive_stochastic**: `streaming_v2_results/clean/adammini_adaptive_stochastic_r8_1500clean_fp16.csv`
- **stochastic**: `streaming_v2_results/clean/adammini_stochastic_r8_1500clean_fp16.csv`
- **stochastic_old**: `streaming_v2_results/clean/adammini_stochastic_old_r8_1500clean_fp16.csv`

## Not regenerated (no full-run diagnostics in archive)

The archive contains training CSV logs only. Fisher JSONL, spectrum JSONL,
and NPZ diagnostics were not included. Existing pilot figures remain at:

- `figures/fisher_overlap_boxplots.png` — from `runs/fisher_*_jsonl/` (pilot)
- `figures/spectrum_energy_boxplots.png` — from `runs/spectrum_all_layers.csv` (pilot)
- `figures/spectrum_group_threshold_rank_boxplots.png` — same pilot spectrum data
- `comparison/comparison_baselines_vs_fisher.png` — Fisher runs not in archive
- `runs/poster/fisher_boxplots/**` — pilot Fisher JSONL only
