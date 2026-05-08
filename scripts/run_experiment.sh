#!/usr/bin/env bash
# End-to-end experiment runner.
set -e

# 1. Download dataset (idempotent)
python scripts/download_data.py

# 2. Run all models and produce Tables 4.1 / 4.2
python main.py --model all --config configs/config.yaml

echo
echo "Results saved in outputs/results.csv and outputs/results.json"
