#!/usr/bin/env bash
#
# Usage:
#   ./generate_ks.sh paper
#       Run the fixed full paper configuration.
#
#   ./generate_ks.sh custom [args...]
#       Run with your own parameters; everything after "custom" is passed
#       straight through to generate_ks_dataset.py. Any flag not supplied
#       falls back to the script's own default.
#
# Examples:
#   ./generate_ks.sh custom --L 8.0 --N 128 --name L8_N128
#   ./generate_ks.sh custom --length 500000 --target_dt 0.5 --name coarse
#   ./generate_ks.sh custom --train_ratio 0.8 --val_ratio 0.1 --seed 1
#
# Edit PY / SCRIPT below to match your environment.

set -euo pipefail

PY=python                                   # or: path to your venv python
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT/data/ks/generation/generate_ks_dataset.py"

PRESET="${1:-paper}"

case "$PRESET" in

  # ---- fixed full paper configuration --------------------------------
  paper)
    "$PY" "$SCRIPT" \
      --L 3.5 --N 64 --dt 0.01 --diffusion 1.0 \
      --length 2500000 --start_from 10000 \
      --train_ratio 0.7 --val_ratio 0.15 \
      --downsample 25 \
      --save_dir "$ROOT/data" --name ks \
      --dtype float32 --seed 0 --save_raw
    ;;

  # ---- user-controlled: forward all remaining args -------------------
  custom)
    "$PY" "$SCRIPT" "${@:2}"
    ;;

  *)
    echo "Unknown preset: '$PRESET'" >&2
    echo "Usage: $0 {paper | custom [args...]}" >&2
    exit 1
    ;;
esac

echo "[generate_ks] preset '$PRESET' finished."