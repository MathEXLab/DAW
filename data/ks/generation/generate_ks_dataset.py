#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_ks_dataset.py
======================
Unified Kuramoto-Sivashinsky (KS) data generation + preprocessing pipeline.

Pipeline
--------
1. Generate   : integrate the 1D KS equation for ``--length`` steps at the fine
                integration step ``--dt`` on the periodic domain [0, 2*pi*L).
2. Truncate   : discard the first ``--start_from`` steps (transient burn-in) so
                the trajectory has settled onto the attractor.
3. Split      : partition the remaining trajectory into train/val/test by ratio
                (contiguous in time, no shuffling -- this is a dynamical system).
4. Normalize  : per-spatial-channel z-score using *train-split* statistics only;
                the same statistics are applied to val/test (and saved for
                de-normalizing model predictions at inference time).
5. Downsample : keep every ``--downsample``-th step, which defines the forecasting
                time grid (effective dt = dt * downsample).

This ordering reproduces the original scripts exactly: statistics are computed on
the full-resolution train split, then every split is thinned.

Default configuration matches the paper
----------------------------------------
    L = 3.5  (domain = 2*pi*L ~= 22)    N = 64        dt = 0.01
    downsample = 25  -> effective dt = 0.25           start_from = 10000

Output layout (under ``<save_dir>/<name>/``)
--------------------------------------------
    mean.npy            train-split per-channel mean  (shape: [N])
    std.npy             train-split per-channel std   (shape: [N])
    train/data.npy      normalized, downsampled train split
    val/data.npy        (only if --val_ratio > 0)
    test/data.npy       (only if 1 - train_ratio - val_ratio > 0)
    u.npy, t.npy        raw trajectory + times (only if --save_raw)
"""

import os
import time
import argparse

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; fall back to a no-op wrapper.
    def tqdm(iterable, *args, **kwargs):
        return iterable

from KS import KS  # spectral solver, left untouched


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description="Generate and preprocess a 1D Kuramoto-Sivashinsky dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- KS solver / generation ----
    g = p.add_argument_group("generation")
    g.add_argument("--L", type=float, default=3.5,
                   help="KS domain parameter; physical domain is [0, 2*pi*L). "
                        "L=3.5 gives a domain ~= 22.")
    g.add_argument("--N", type=int, default=64,
                   help="Number of Fourier collocation (spatial grid) points.")
    g.add_argument("--dt", type=float, default=0.01,
                   help="Fine integration time step.")
    g.add_argument("--diffusion", type=float, default=1.0,
                   help="Hyperdiffusion coefficient on the u_xxxx term.")
    g.add_argument("--length", type=int, default=2_500_000,
                   help="Number of integration steps to generate (before truncation).")
    g.add_argument("--members", type=int, default=1,
                   help="Number of independent ensemble members to integrate in parallel.")
    g.add_argument("--seed", type=int, default=None,
                   help="RNG seed for the initial condition (None = nondeterministic).")

    # ---- Truncation / splitting ----
    s = p.add_argument_group("split")
    s.add_argument("--start_from", type=int, default=10_000,
                   help="Number of leading steps to discard as transient burn-in.")
    s.add_argument("--train_ratio", type=float, default=0.7,
                   help="Fraction of the post-transient trajectory used for training.")
    s.add_argument("--val_ratio", type=float, default=0.15,
                   help="Fraction used for validation. Test = 1 - train - val "
                        "(0.15 with the defaults).")

    # ---- Downsampling ----
    d = p.add_argument_group("downsample")
    d.add_argument("--downsample", type=int, default=25,
                   help="Keep every k-th step. Effective dt = dt * downsample.")
    d.add_argument("--target_dt", type=float, default=None,
                   help="If set, overrides --downsample with round(target_dt / dt).")

    # ---- I/O ----
    o = p.add_argument_group("output")
    o.add_argument("--save_dir", type=str, default="./data/ks",
                   help="Root directory for outputs.")
    o.add_argument("--name", type=str, default="AllPass",
                   help="Subfolder name created under --save_dir.")
    o.add_argument("--dtype", type=str, default="float32",
                   choices=["float32", "float64"],
                   help="Storage dtype for the saved arrays.")
    o.add_argument("--no_normalize", action="store_true",
                   help="Skip z-score normalization (still computes/saves stats).")
    o.add_argument("--save_raw", action="store_true",
                   help="Also save the raw, full-resolution trajectory (u.npy, t.npy).")

    return p


# --------------------------------------------------------------------------- #
# Stage 1: generation
# --------------------------------------------------------------------------- #
def generate(args, rng):
    """Integrate the KS equation; return (us, ts).

    us has shape (length, N) when members == 1, else (length, members, N).
    """
    ks = KS(L=args.L, N=args.N, dt=args.dt,
            diffusion=args.diffusion, members=args.members, rs=rng)

    # Noisy, zero-mean initial condition. advance() recomputes the spectral
    # state from ks.x on every step, so setting ks.x is sufficient.
    u0 = 0.01 * rng.standard_normal(size=(args.members, args.N))
    ks.x = u0 - u0.mean(axis=-1, keepdims=True)

    us = np.empty((args.length, args.members, args.N), dtype=args.dtype)
    t0 = time.time()
    for i in tqdm(range(args.length), desc="integrating KS"):
        ks.advance()
        us[i] = ks.x
    print(f"[generate] {args.length} steps in {time.time() - t0:.1f}s")

    if args.members == 1:
        us = us[:, 0, :]  # (length, N)

    ts = (np.arange(args.length, dtype=args.dtype) * args.dt)
    return us, ts


# --------------------------------------------------------------------------- #
# Stages 2-5: truncate -> split -> normalize -> downsample
# --------------------------------------------------------------------------- #
def split_indices(n, train_ratio, val_ratio):
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    if n_train + n_val > n:
        raise ValueError("train_ratio + val_ratio exceeds 1.0")
    return {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n),
    }


def process(us, args):
    """Truncate transient, split, normalize on train stats, downsample.

    Returns (splits_dict, mean, std). Empty splits are dropped.
    """
    # --- Stage 2: discard transient ---
    data = us[args.start_from:]
    print(f"[process] post-transient shape {data.shape} "
          f"(discarded first {args.start_from} steps)")

    # --- Stage 3: contiguous time split ---
    n = data.shape[0]
    bounds = split_indices(n, args.train_ratio, args.val_ratio)

    # --- Stage 4: z-score stats from train only ---
    lo, hi = bounds["train"]
    train_full = data[lo:hi]
    if train_full.shape[0] == 0:
        raise ValueError("Empty train split; check ratios and start_from.")
    mean = train_full.mean(axis=0)
    std = train_full.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)  # guard against dead channels

    splits = {}
    for name, (lo, hi) in bounds.items():
        chunk = data[lo:hi]
        if chunk.shape[0] == 0:
            continue
        if not args.no_normalize:
            chunk = (chunk - mean) / std
        # --- Stage 5: downsample (defines the forecasting dt) ---
        chunk = chunk[::args.downsample]
        splits[name] = chunk.astype(args.dtype, copy=False)
        print(f"[process] {name:5s} -> {splits[name].shape}")

    return splits, mean.astype(args.dtype), std.astype(args.dtype)


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def save_outputs(splits, mean, std, ts, us, args):
    out_root = os.path.join(args.save_dir, args.name)
    os.makedirs(out_root, exist_ok=True)

    np.save(os.path.join(out_root, "mean.npy"), mean)
    np.save(os.path.join(out_root, "std.npy"), std)

    for name, arr in splits.items():
        split_dir = os.path.join(out_root, name)
        os.makedirs(split_dir, exist_ok=True)
        np.save(os.path.join(split_dir, "data.npy"), arr)

    if args.save_raw:
        np.save(os.path.join(out_root, "u.npy"), us)
        np.save(os.path.join(out_root, "t.npy"), ts)

    print(f"[save] wrote outputs to {out_root}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = build_parser().parse_args()
    args.dtype = np.float32 if args.dtype == "float32" else np.float64

    # Resolve downsample factor from target_dt if provided.
    if args.target_dt is not None:
        factor = int(round(args.target_dt / args.dt))
        if factor < 1:
            raise ValueError("target_dt must be >= dt.")
        if not np.isclose(factor * args.dt, args.target_dt):
            print(f"[warn] target_dt={args.target_dt} is not an integer multiple "
                  f"of dt={args.dt}; using factor {factor} (dt_eff="
                  f"{factor * args.dt:g}).")
        args.downsample = factor

    print(f"[config] domain=2*pi*{args.L:g} (~{2 * np.pi * args.L:.2f}), "
          f"N={args.N}, dt={args.dt:g}, downsample={args.downsample} "
          f"-> dt_eff={args.dt * args.downsample:g}")

    rng = np.random.RandomState(args.seed)

    us, ts = generate(args, rng)
    splits, mean, std = process(us, args)
    save_outputs(splits, mean, std, ts, us, args)


if __name__ == "__main__":
    main()