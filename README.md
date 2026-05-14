# Student 4 Individual Submission
**Project:** Diffusion Policy Imitation Learning for Dual-Arm Manipulation
**Role:** Full experiment pipeline, visualization, statistical analysis, final report + theoretical discussion

**GitHub:** https://github.com/<YOUR_USERNAME>/dual-arm-diffusion-ablation
*(replace `<YOUR_USERNAME>` after you push --- see `PUSH_TO_GITHUB.md`)*

---

## Folder Structure

| Folder | Contents | Authorship |
|--------|----------|------------|
| `ablation_code/` | `run_ablation.py` (12-experiment driver), `model.py` (network) | `run_ablation.py` is mine; `model.py` is Student 3's, unchanged |
| `ablation_results/` | 12 experiment subfolders + aggregated plots + `all_results.json` | All mine (raw outputs from my training runs) |
| `visualization_code/` | `student4_visualizations.py` | Mine |
| `figures/` | 12 final PNG figures used in the team report | Mine |
| `figures_pdf/` | Same figures in PDF format for LaTeX inclusion | Mine |
| `report/student4_report.tex` | This individual report | Mine |
| `presentation.pptx` / `presentation_v2.pptx` | Final presentation slides | Mine |

---

## My Contributions in Detail

### 1. Ablation Experiments (12 runs)
Built an automated experiment driver that runs three ablation studies back-to-back:

- **Demonstration count** (5 runs): `N ∈ {50, 100, 200, 300, 500}`
- **Prediction horizon** (4 runs): `H_p ∈ {4, 8, 16, 32}`
- **Observation horizon** (3 runs): `H_o ∈ {1, 2, 4}`

Each run trains the Diffusion Policy for 100 epochs at batch size 256 on an
NVIDIA RTX PRO 6000 (96GB), records the per-epoch loss to `loss.csv`, saves
the configuration to `config.json`, and stores the final EMA checkpoint.

**Key finding:** Demonstration count dominates --- 8.5× loss reduction from
50→500 demos --- while either horizon axis varies by less than 1.2×.

### 2. Visualization Pipeline
Implemented a single-file pipeline (`student4_visualizations.py`) that
applies a uniform style and renders all 12 figures used in the team report:

- Aggregate comparisons: success-rate bar chart, radar plot, summary table
- Training dynamics: BC and Diffusion training curves, log-scale comparison
- Diagnostics: DAgger iteration improvement, average episode length
- Conceptual: covariate-shift illustration, action-chunking schematic
- Distribution analysis: t-SNE of states visited by Expert / BC / DAgger

### 3. Statistical Analysis
- Aggregated raw outputs from Students 1–3 into the team's main results table.
- Computed Maximum Mean Discrepancy (MMD) between expert and learner state
  distributions to quantitatively confirm covariate shift.

### 4. Final Report and Theoretical Discussion
Authored the team's main 12-page LaTeX report (`report.tex` in the team
repository), covering the BC compounding-error bound, DAgger's regret
analysis, the Diffusion Policy formulation, sim-to-real considerations,
and the integrated cross-method discussion.

---

## How to Reproduce My Ablations

```bash
# From inside ablation_code/ with the dataset placed at ./data/demo_robomimic_compatible.hdf5
python run_ablation.py
```

This will produce all artifacts under `./ablation_results/`. Total wall-clock
time on a single RTX PRO 6000: ~90 minutes.

---

## Inputs from Other Team Members

| From | What I used |
|------|-------------|
| Student 1 | `demo_robomimic_compatible.hdf5` (500 expert demonstrations) |
| Student 2 | BC test MSE (0.0058), success rate (44%), DAgger iteration logs and final 86% |
| Student 3 | `ConditionalUnet1D` model, training loss curve, final loss 4.4×10⁻⁴ |
