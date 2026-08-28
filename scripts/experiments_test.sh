#!/usr/bin/env bash
#
# Autoregressive rollout evaluation for every checkpoint produced by
# experiments_run.sh. For each ckpts/<Method>/run_*/best_model.pth, roll the
# model forward AR_STEPS steps on the KS test set (via forecast_test.py) and
# save predictions/metrics to <run_dir>/ar_test_results.nc.
#
# Usage:
#   ./scripts/experiments_test.sh

set -euo pipefail

PY=python                                   # or: path to your venv python
AR_STEPS=80 # 1 LT (covers the 0.5 LT / 1.0 LT horizons reported in the README)

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT/forecast_test.py"
TEST_DATA_PATH="$ROOT/data/ks/test/data.npy"

METHODS=(Standard DenseWeight DAW RandomWeight)

for METHOD in "${METHODS[@]}"; do
  METHOD_DIR="$ROOT/ckpts/$METHOD"
  [ -d "$METHOD_DIR" ] || continue

  for RUN_DIR in "$METHOD_DIR"/run_*/; do
    [ -d "$RUN_DIR" ] || continue
    MODEL_PATH="${RUN_DIR}best_model.pth"
    if [ ! -f "$MODEL_PATH" ]; then
      echo "skip $RUN_DIR (no best_model.pth yet)"
      continue
    fi

    echo "=== Evaluating $MODEL_PATH ==="
    "$PY" "$SCRIPT" \
        --model_path "$MODEL_PATH" \
        --test_data_path "$TEST_DATA_PATH" \
        --ar_steps "$AR_STEPS"
  done
done

echo "[experiments_test] all checkpoints evaluated."
