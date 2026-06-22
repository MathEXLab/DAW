<div align="center">

# DAW: Dimension-Aware Weighting for Stable Long-Term Forecasting

<!-- Decorative badges — edit the URLs/labels as needed -->
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)
<!-- Optional extras:
![Conda](https://img.shields.io/badge/conda-342B029.svg?logo=anaconda&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white)
![Code Style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
-->

*A data-centric, plug-and-play objective-reweighting framework that mitigates catastrophic error accumulation and topological degradation during long-term autoregressive rollouts of chaotic spatiotemporal systems.*

</div>

---

## Overview

Machine-learning surrogates are powerful tools for forecasting chaotic dynamical systems, but they suffer from catastrophic error accumulation and loss of topological fidelity during long autoregressive rollouts. Standard sample-wise uniform objectives (e.g., MSE) implicitly assume all training samples are equally informative, driving surrogates to *regress toward the conditional mean* on imbalanced state distributions and to under-represent rare, dynamically complex regimes (e.g., wave-merging events).

**Dimension-Aware Weighting (DAW)** addresses this bottleneck by using the **local dimension** $d$ — a measure of a system's instantaneous active degrees of freedom, derived from extreme value theory (EVT) and the Poincaré recurrence theorem — as an *a priori* indicator of dynamical complexity. DAW adaptively reshapes the loss landscape to force the network to allocate representational capacity to sparse, high-$d$ regimes, thereby suppressing localized error jumps, delaying structural collapse, and improving relative long-term stability.

Key properties:

- **Knowledge-informed:** weights are derived from phase-space geometry ($d$), not from target-space statistics.
- **Plug-and-play:** acts purely on the training objective; architecture-agnostic with **zero inference overhead**.

> 📄 **Paper:** *DAW: Dimension-Aware Weighting for Stable Long-Term Forecasting* — \<TODO: link / arXiv ID\>

---

## Method at a Glance

DAW computes a per-sample weight in four steps and optimizes a weighted MSE objective:

1. **Local dimension & density estimation.** Estimate $d_i$ for each sample via EVT/GPD ($d = 1/\sigma$), then estimate the empirical density $P(d)$ with Gaussian-kernel KDE (min-max normalized).
2. **Base density weighting.** Inverse-density weight $w_i^{\text{base}} = \max\!\big(1 - \alpha\, P'(d_i),\, \epsilon\big)$ (DenseWeight-style), which symmetrically elevates *both* tails of $P(d)$.
3. **Right-tail (high-$d$) reshaping.** Break the symmetry with a linear tilt on the normalized indicator $\tilde{d}_i \in [0,1]$: $w_i^{\text{reshaped}} = w_i^{\text{base}} \cdot \tilde{d}_i$, suppressing the trivial low-$d$ tail and amplifying rare high-$d$ states.
4. **Normalization.** Rescale so the expected weight over the batch equals 1.0, preserving the global gradient scale.

$$\mathcal{L}_{\mathrm{DAW}}(\theta) = \frac{1}{N}\sum_{i=1}^{N} w_i \,\big\lVert f_\theta(x_i) - y_i \big\rVert_2^2$$

<!-- Optional: add a method/teaser figure here -->
<!-- <div align="center"><img src="assets/daw_overview.png" width="80%"></div> -->

---

## Installation

```bash
# Clone the repository
git clone <TODO: repo URL>
cd <TODO: repo name>

# (Recommended) create an environment
# TODO: fill in your preferred environment manager / Python version
conda create -n daw python=3.11
conda activate daw

# Install dependencies
pip install -r requirements.txt
```


---

## Repository Structure

```text
.
├── ckpts/                       # trained model checkpoints (one dir per weighting scheme)
│   ├── DAW/
│   ├── DenseWeight/
│   ├── RandomWeight/
│   └── Standard/
├── data/
│   └── generation/
│       ├── generate_ks_dataset.py   # build the KS forecasting dataset (samples, normalization)
│       └── KS.py                    # KS equation integrator / solver
├── notebooks/                   # analysis & figure-generation notebooks
├── scripts/
│   ├── experiments.sh           # launch training/eval across all weighting schemes
│   └── generate_ks.sh           # generate the KS dataset end-to-end
├── run_experiments.py           # main entry point (training + rollout evaluation)
├── requirements.txt
└── README.md
```

---

## Data Generation

We evaluate on the **1D Kuramoto–Sivashinsky (KS)** equation, a canonical model of spatiotemporal chaos:

$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + \frac{\partial^2 u}{\partial x^2} + \frac{\partial^4 u}{\partial x^4} = 0$$

**Configuration used in the paper:**

| Parameter | Value |
|---|---|
| Domain size $L$ | 22 |
| Spatial grid $N_x$ | 64 |
| Integration step $dt$ | 0.01 |
| Transient discarded | first 10,000 steps |
| Downsampled step (forecast task) | 0.25 |
| Total samples | 2,500,000 |
| Normalization | Z-score (train stats applied to val/test) |

Generate the dataset:

```bash
# TODO: fill in your actual script name and arguments
python data/generation/ks_gen.py 
```

Compute the local dimension $d$ for each state (EVT/GPD, $d = 1/\sigma$):

```bash
# TODO: fill in your actual script name and arguments
python src/local_dim.py \
    --input data/ks_L22.npy \
    --quantile 0.98 \
    --traj-length-LT 850 \
    --out data/ks_L22_dim.npy
```

> **Note on trajectory length:** As shown in Appendix A, the standardized $d$ distribution converges and stabilizes around **850 LT**; this is the configuration used throughout the experiments.

---

## Training

All methods share an identical MLP backbone, optimizer, learning-rate schedule, batch size, and training budget — **only the per-sample weight $w_i$ differs**.

```bash
# DAW (ours)
# TODO: fill in your actual entry point and arguments
python src/train.py \
    --config configs/daw.yaml \
    --weighting daw \
    --alpha 1.0
```

Reproduce the baselines:

```bash
# Standard (uniform weighting, w_i = 1)
python src/train.py --config configs/base.yaml --weighting standard

# DenseWeight (target-space density weighting)
python src/train.py --config configs/base.yaml --weighting denseweight --alpha 1.0

# RandomWeight (shuffled DAW weights ablation)
python src/train.py --config configs/base.yaml --weighting randomweight
```

| Method | Weighting | Tests |
|---|---|---|
| **DAW** (ours) | inverse-density of $P(d)$ × $\tilde{d}$-tilt | — |
| Standard | $w_i = 1$ | whether uniform training already suffices |
| DenseWeight | inverse-density of target $L_2$-norm | whether statistical rarity ≈ dynamical complexity |
| RandomWeight | DAW weights, randomly permuted | whether gains stem from weight–topology alignment vs. gradient variance |

---

## Evaluation

The trained surrogate is queried in closed loop (autoregressive rollout). Forecast horizons are reported in **Lyapunov times (LT)**.

```bash
# TODO: fill in your actual entry point and arguments
python src/eval.py \
    --ckpt <path/to/checkpoint> \
    --data data/ks_L22.npy \
    --dim data/ks_L22_dim.npy \
    --horizon-steps 80
```

**Metrics**

- **MAE** — point-wise absolute error over the full spatiotemporal domain.
- **Spatial Pearson Correlation** — structural / phase fidelity (0.5 = predictability threshold).
- **Cumulative Error (CE)** — sum of absolute errors over a ±3-step window centered on peak-$d$ within high-$d$ event windows (>75th percentile).
- **Overall Win Rate** — fraction of high-$d$ event windows in which a method attains the lowest CE.

**Representative results** (mean ± std over 3 runs; see paper Table 1 for the full table):

| Horizon | Method | MAE ↓ | Correlation ↑ |
|---|---|---|---|
| 0.5 LT (Step 40) | **DAW** | **0.2993 ± 0.0233** | **0.8874 ± 0.0170** |
| 0.5 LT (Step 40) | Standard | 0.7049 ± 0.0835 | 0.5503 ± 0.0690 |
| 1.0 LT (Step 80) | **DAW** | **0.7779 ± 0.0306** | **0.4153 ± 0.0353** |
| 1.0 LT (Step 80) | Standard | 1.0635 ± 0.0184 | 0.1419 ± 0.0468 |

DAW is the last method to cross the 0.5 correlation threshold (≈0.85 LT) and achieves the highest Overall Win Rate (42.4%) across high-$d$ event windows. Consistent with the limits of chaotic predictability, all methods decay by 1.0 LT — DAW provides a systematic **delay** of structural collapse rather than its elimination.

<!-- Optional: add results figures here -->
<!-- <div align="center"><img src="assets/rollout_metrics.png" width="90%"></div> -->

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{TODO_citekey,
  title   = {DAW: Dimension-Aware Weighting for Stable Long-Term Forecasting},
  author  = {TODO: Author One and Author Two and Author Three},
  journal = {TODO: venue / arXiv preprint},
  year    = {2026},
  note    = {TODO: arXiv ID / DOI}
}
```

---

## Acknowledgments

<!-- TODO: funding, compute, collaborators -->
This work was supported in part by \<TODO\>.

## License

Released under the \<TODO: e.g., MIT\> License. See [`LICENSE`](LICENSE) for details.