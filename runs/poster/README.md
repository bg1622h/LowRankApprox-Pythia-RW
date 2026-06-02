# Poster bundle (graphs + raw data)

Regenerate: `python scripts/build_poster_assets.py`

---

## Baselines — GaLore2 + Lotus

Folder: https://github.com/bg1622h/LowRankApprox-Pythia-RW/tree/main/runs/poster/baselines

### Graphs
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/baselines/baseline_val_curves.png
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/baselines/baseline_final_bars.png

### Data
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/baselines/data/refinedweb_full_galore2_1500_seq2048.csv
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/baselines/data/refinedweb_full_lotus_1500_seq2048.csv

---

## Comparison — all methods

Folder: https://github.com/bg1622h/LowRankApprox-Pythia-RW/tree/main/runs/poster/comparison

### Graphs
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/comparison/all_methods_val_curves.png
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/comparison/all_methods_final_bars.png
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/comparison/comparison_baselines_vs_fisher.png

### Data
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/comparison/data/refinedweb_full_adaptive_stochastic_1500_seq2048.csv
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/comparison/data/refinedweb_full_fisher_both_1500_seq2048_cr128.csv
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/comparison/data/refinedweb_full_fisher_topk_softmax_1500_seq2048_cr64.csv
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/comparison/data/summary_methods.csv
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/comparison/data/summary_memory.csv

---

## Fisher boxplots

Folder: https://github.com/bg1622h/LowRankApprox-Pythia-RW/tree/main/runs/poster/fisher_boxplots

Each subfolder has `data/training_*.csv`, `data/jsonl/`, and PNG boxplots.

### cr128
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/fisher_boxplots/cr128/fisher_group_overlap_boxplots.png
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/fisher_boxplots/cr128/fisher_group_mass_boxplots.png
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/fisher_boxplots/cr128/fisher_group_effective_rank_boxplots.png
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/fisher_boxplots/cr128/data/refinedweb_full_fisher_both_1500_seq2048_cr128.csv
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/tree/main/runs/poster/fisher_boxplots/cr128/data/jsonl

### cr64
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/fisher_boxplots/cr64/fisher_group_overlap_boxplots.png
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/fisher_boxplots/cr64/data/refinedweb_full_fisher_both_1500_seq2048_cr64.csv
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/tree/main/runs/poster/fisher_boxplots/cr64/data/jsonl

### topk_softmax_cr64
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/fisher_boxplots/topk_softmax_cr64/fisher_group_overlap_boxplots.png
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/fisher_boxplots/topk_softmax_cr64/data/refinedweb_full_fisher_topk_softmax_1500_seq2048_cr64.csv
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/tree/main/runs/poster/fisher_boxplots/topk_softmax_cr64/data/jsonl

### all_datasets (combined plots)
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/fisher_boxplots/all_datasets/fisher_group_overlap_boxplots.png
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/fisher_boxplots/all_datasets/fisher_group_mass_boxplots.png
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/fisher_boxplots/all_datasets/fisher_group_effective_rank_boxplots.png

---

## Spectrum boxplots

Folder: https://github.com/bg1622h/LowRankApprox-Pythia-RW/tree/main/runs/poster/spectrum_boxplots

### Graphs
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/spectrum_boxplots/spectrum_group_energy_boxplots.png
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/spectrum_boxplots/spectrum_group_effective_rank_boxplots.png
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/spectrum_boxplots/spectrum_group_threshold_rank_boxplots.png
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/spectrum_boxplots/spectrum_group_angle_boxplots.png
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/spectrum_boxplots/spectrum_group_adaptive_rank_boxplots.png

### Data
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/spectrum_boxplots/data/spectrum_all_layers_summary.csv
- https://github.com/bg1622h/LowRankApprox-Pythia-RW/blob/main/runs/poster/spectrum_boxplots/data/spectrum_all_layers.csv
