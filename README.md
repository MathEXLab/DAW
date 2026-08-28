<div align="center">

# DAW: Dynamics-Aware Weighting for Deep Learning Forecasts of Chaotic Systems

<!-- Decorative badges — edit the URLs/labels as needed -->
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2608.22277-b31b1b.svg)](https://arxiv.org/abs/2608.22277)
<!-- Optional extras:
![Conda](https://img.shields.io/badge/conda-342B029.svg?logo=anaconda&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white)
![Code Style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
-->

*A data-centric, plug-and-play objective-reweighting framework that mitigates catastrophic error accumulation and topological degradation during long-term autoregressive rollouts of chaotic spatiotemporal systems.*

</div>

---

## Abstract

Deep learning surrogates have become powerful tools for simulating and forecasting complex dynamical systems, yet their utility remains limited by catastrophic error accumulation during long-term autoregressive rollouts.
This behavior is partly tied to the nature of the underlying systems: chaotic spatiotemporal systems, such as the Kuramoto-Sivashinsky (KS) equation, visit phase space unevenly, with dynamics dominated by recurrent, low-dimensional quiescent states (e.g., near-laminar flows) and characterized by rare and dynamically complex topological transitions (e.g., wave-merging events).
Trained under a sample-wise uniform objective, standard neural surrogates allocate their finite capacity to the statistically more numerous low-dimensional quiescent states, systematically under-representing the transient regimes that trigger disproportionate, localized errors.
Existing imbalanced-regression methods attempt to tackle this challenge by reweighting samples according to target-space density. 
However, statistical target-space rarity need not coincide with the intrinsic dynamical rarity -- the recurrence geometry of the attractor that contributes directly to the source of the imbalance. 
To address this, we introduce **Dynamics-Aware Weighting (DAW)**, a data-centric objective reweighting framework. 
Using the **local dimension $d$** from dynamical systems theory as an a priori measure of a state's active degrees of freedom or complexity, DAW reshapes the loss landscape to allocate representational capacity toward the sparse, high-$d$ regimes where forecast errors are systematically large. 
Evaluated on the chaotic KS equation, DAW consistently outperforms uniform training, purely statistical density weighting, and its randomly permuted ablation.
In particular, DAW reduces the long-term error in autoregressive forecasting relative to all baselines. 
Event-level analysis shows that DAW achieves this by suppressing the localized error amplifications incurred during sharp jumps in the local dimension $d$, which typically accompany complex physical processes such as wave-merging in the KS system.

Key properties:

- **Knowledge-informed:** weights are derived from phase-space geometry ($d$), not from target-space statistics.
- **Plug-and-play:** acts purely on the training objective; architecture-agnostic with **zero inference overhead**.

> 📄 **Paper:** *DAW: Dynamics-Aware Weighting for Deep Learning Forecasts of Chaotic Systems* — [arXiv:2608.22277](https://arxiv.org/abs/2608.22277)

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
git clone https://github.com/ZhousLab/Dimension-Aware-Weighting.git
cd Dimension-Aware-Weighting

# (Recommended) create an environment
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
│   │   └── run_<timestamp>_DAW_alpha<alpha>/          # hparams.json, best_model.pth, last_model.pth, loss_history.json
│   ├── DenseWeight/
│   ├── RandomWeight/
│   └── Standard/
├── data/
│   └── ks/                      # dataset produced by generate_ks.sh / generate_ks_dataset.py
│       ├── generation/
│       │   ├── generate_ks_dataset.py   # build the KS forecasting dataset (integrate, split, normalize, downsample)
│       │   └── KS.py                    # KS equation spectral integrator / solver
│       ├── mean.npy / std.npy   # train-set normalization stats (applied to val/test)
│       ├── t.npy / u.npy        # raw integrated trajectory (only with --save_raw)
│       ├── train/                       # data.npy + precomputed d_sample_pair_*.npy / theta_sample_pair_*.npy
│       ├── val/                         # data.npy
│       └── test/                        # data.npy
├── notebooks/
│   └── results_visualization.ipynb      # analysis & figure-generation notebook
├── pypardi/                     # local/global dynamical-indices library (EVT/GPD estimation of the local dimension d)
│   ├── local_indices.py         # local_indices.compute(...) -> local dimension d, extremal index theta
│   ├── global_indices.py
│   ├── attractors.py
│   ├── di_evaluate.py / di_evaluate_par.py
│   └── utils.py
├── scripts/
│   ├── generate_ks.sh            # wrapper around generate_ks_dataset.py ("paper" preset or custom args)
│   ├── calculate_di.sh           # wrapper around calculate_di_sample_pair.py
│   ├── experiments_run.sh        # train Standard / DenseWeight / DAW / RandomWeight across seeds
│   └── experiments_test.sh       # autoregressive rollout evaluation for every trained checkpoint
├── calculate_di_sample_pair.py  # compute the local dimension d for each (input, output) sample pair via pypardi
├── run_experiments.py           # main training entry point (Standard / DenseWeight / DAW / RandomWeight)
├── forecast_test.py             # autoregressive rollout evaluation of a trained checkpoint on the test set
├── requirements.txt
├── LICENSE
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

Generate the dataset (writes `mean.npy`, `std.npy`, and `train/`, `val/`, `test/` splits under `<save_dir>/<name>/`; the paper preset sets `save_dir=data`, `name=ks`, so output lands in `data/ks/`):

```bash
# Reproduce the exact paper configuration (L=3.5, N=64, dt=0.01, 2.5M steps, ...)
chmod +x scripts/generate_ks.sh #(run only once)
./scripts/generate_ks.sh paper

# Or pass your own arguments through; unset flags fall back to the script's own defaults
./scripts/generate_ks.sh custom --L 8.0 --N 128 --name L8_N128

# Equivalently, invoke the generator directly. It uses a local `from KS import KS`
# import, so run it from within its own directory:
cd data/ks/generation
python generate_ks_dataset.py \
    --L 3.5 --N 64 --dt 0.01 --diffusion 1.0 \
    --length 2500000 --start_from 10000 \
    --train_ratio 0.7 --val_ratio 0.15 \
    --downsample 25 \
    --save_dir ../.. --name ks \
    --dtype float32 --seed 0 --save_raw
```

Compute the local dimension $d$ for each `(input, output)` sample pair (EVT/GPD via `pypardi`, $d = 1/\sigma$). The input and output windows are embedded jointly before estimating $d$, and results are saved as `d_sample_pair_in<input_len>_out<output_len>_q<quantile>...npy` / `theta_sample_pair_...npy` next to the input data:

```bash
python calculate_di_sample_pair.py \
    --data_path data/ks/train/data.npy \
    --input_len 3 --output_len 1 \
    --quantile 0.99 \
    --save_dir data/ks/train

# Or use the wrapper script (edit the paths inside to match your checkout):
./scripts/calculate_di.sh
```

> **Note on trajectory length:** As shown in Appendix A, the standardized $d$ distribution converges and stabilizes around **850 LT**; this is the configuration used throughout the experiments.

---

## Run Experiments

All methods share an identical MLP backbone, optimizer, learning-rate schedule, batch size, and training budget — **only the per-sample weight $w_i$ differs**.

```bash
# DAW (ours)
python run_experiments.py \
    --method DAW --alpha 1.0 \
    --data_dir data/ks \
    --d_path data/ks/train/d_sample_pair_in3_out1_q0.99_None.npy \
    --base_save_dir ./ckpts/DAW \
    --device cuda:0
```

Reproduce the baselines (`--data_dir` must contain `train/`, `val/`, `test/` splits as produced by `generate_ks_dataset.py`; `--d_path` defaults to `<data_dir>/train/d_sample_pair_in3_out1_q0.99.npy` if omitted, so pass it explicitly if `calculate_di_sample_pair.py` wrote a different filename):

```bash
# Standard (uniform weighting, w_i = 1)
python run_experiments.py --method Standard --data_dir data/ks --base_save_dir ./ckpts/Standard

# DenseWeight (target-space density weighting on the output L2-norm)
python run_experiments.py --method DenseWeight --alpha 0.5 --data_dir data/ks --base_save_dir ./ckpts/DenseWeight

# RandomWeight (shuffled DAW weights ablation; also requires --d_path)
python run_experiments.py --method RandomWeight --alpha 1.0 \
    --data_dir data/ks \
    --d_path data/ks/train/d_sample_pair_in3_out1_q0.99_None.npy \
    --base_save_dir ./ckpts/RandomWeight
```

Each run writes `hparams.json`, `best_model.pth`, `last_model.pth`, and `loss_history.json` to a timestamped subfolder of `--base_save_dir` (e.g. `ckpts/DAW/run_<timestamp>_DAW_alpha1.0/`). `scripts/experiments_run.sh` chains all four methods across several seeds to reproduce the paper's mean ± std results in one go. See `python run_experiments.py --help` for the full set of model/optimization flags (model type, hidden width, depth, batch size, learning rate, warmup epochs, etc.).

| Method | Weighting | Tests |
|---|---|---|
| **DAW** (ours) | inverse-density of $P(d)$ × $\tilde{d}$-tilt | — |
| Standard | $w_i = 1$ | whether uniform training already suffices |
| DenseWeight | inverse-density of target $L_2$-norm | whether statistical rarity ≈ dynamical complexity |
| RandomWeight | DAW weights, randomly permuted | whether gains stem from weight–topology alignment vs. gradient variance |

---

## Evaluation

`forecast_test.py` rolls a trained checkpoint forward autoregressively on the KS test set and reports rollout metrics:

```bash
python forecast_test.py \
    --model_path ckpts/DAW/run_<timestamp>_DAW_alpha1.0/best_model.pth \
    --test_data_path data/ks/test/data.npy \
    --ar_steps 80   # 80 steps ≈ 1 Lyapunov time (LT) at the paper's downsampled dt
```

To evaluate every checkpoint under `ckpts/` in one pass (writing `ar_test_results.nc` next to each `best_model.pth`), use the wrapper script:

```bash
./scripts/experiments_test.sh
```

---


## Citation

If you find this work useful, please cite:

```bibtex
@article{fang2026daw,
  title   = {DAW: Dynamics-Aware Weighting for Deep Learning Forecasts of Chaotic Systems},
  author  = {Zhou Fang and Gianmarco Mengaldo},
  journal = {arXiv preprint arXiv:2608.22277},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.22277}
}
```

---



## License

Released under the MIT License. 