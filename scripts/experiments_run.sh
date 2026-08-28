#!/usr/bin/env bash
#
# Reproduce the main KS results (README "Representative results" table) with
# an MLP backbone, input_len=3 / output_len=1, repeated over several seeds
# so that mean +/- std can be computed across runs.
#
# Usage:
#   ./scripts/experiments_run.sh
#
# Edit PY / SEEDS / DEVICE below to match your environment.

set -euo pipefail

PY=python                                        # or: path to your venv python
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT/run_experiments.py"

DATA_DIR="$ROOT/data/ks"
D_PATH="$DATA_DIR/train/d_sample_pair_in3_out1_q0.99_None.npy"

MODEL_TYPE="MLP"
INPUT_LEN=3
OUTPUT_LEN=1
DEVICE="cuda:0"

# Repeat every method across these seeds to get the mean +/- std reported in the README
SEEDS=(0 1 2)

for SEED in "${SEEDS[@]}"; do

  # Standard
  "$PY" "$SCRIPT" \
      --method Standard \
      --model_type "$MODEL_TYPE" \
      --input_len "$INPUT_LEN" --output_len "$OUTPUT_LEN" \
      --data_dir "$DATA_DIR" \
      --base_save_dir "$ROOT/ckpts/Standard" \
      --random_seed \
      --device "$DEVICE"

  # DenseWeight
  "$PY" "$SCRIPT" \
      --method DenseWeight --alpha 0.5 \
      --model_type "$MODEL_TYPE" \
      --input_len "$INPUT_LEN" --output_len "$OUTPUT_LEN" \
      --data_dir "$DATA_DIR" \
      --base_save_dir "$ROOT/ckpts/DenseWeight" \
      --random_seed \
      --device "$DEVICE"

  # DAW (ours)
  "$PY" "$SCRIPT" \
      --method DAW --alpha 1.0 \
      --model_type "$MODEL_TYPE" \
      --input_len "$INPUT_LEN" --output_len "$OUTPUT_LEN" \
      --data_dir "$DATA_DIR" \
      --d_path "$D_PATH" \
      --base_save_dir "$ROOT/ckpts/DAW" \
      --random_seed \
      --device "$DEVICE"

  # RandomWeight (shuffled-DAW ablation)
  "$PY" "$SCRIPT" \
      --method RandomWeight --alpha 1.0 \
      --model_type "$MODEL_TYPE" \
      --input_len "$INPUT_LEN" --output_len "$OUTPUT_LEN" \
      --data_dir "$DATA_DIR" \
      --d_path "$D_PATH" \
      --base_save_dir "$ROOT/ckpts/RandomWeight" \
      --random_seed \
      --device "$DEVICE"

done

echo "[experiments_run] all methods finished for ${SEEDS[*]} runs"
