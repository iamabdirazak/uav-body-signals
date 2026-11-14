#!/usr/bin/env bash
# Every experiment in the paper, in order. Finished training runs are skipped, so this can be restarted.
# Needs data/UAV-GESTURE (scripts/get_data.sh) and, for the last step, data/uavhuman_skeletons.npz
# (scripts/prepare_uavhuman.py).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m experiments.train
python -m experiments.baselines
python -m experiments.dataset_stats
python -m experiments.robustness
python -m experiments.viewpoint
python -m experiments.safety_gate
if [ -f data/uavhuman_skeletons.npz ]; then python -m experiments.uavhuman; fi
python -m experiments.benchmark
python -m experiments.summary
