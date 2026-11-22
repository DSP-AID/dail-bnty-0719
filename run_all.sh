#!/usr/bin/env bash
# Reproduce the whole pipeline: train one SAC policy on two source buildings,
# then zero-shot evaluate it on 13 held-out building/weather combinations.
#
# Prerequisites (see README.md): EnergyPlus >= 24.1 and a Python env with
# sinergym==3.12.2 + stable-baselines3. Point EPLUS_PATH (or
# ENERGYPLUS_INSTALLATION_FOLDER) at the EnergyPlus installation directory.
set -euo pipefail
cd "$(dirname "$0")"

: "${EPLUS_PATH:?set EPLUS_PATH to the EnergyPlus installation folder}"
export EPLUS_PATH ENERGYPLUS_INSTALLATION_FOLDER="$EPLUS_PATH"
export PYTHONPATH="$EPLUS_PATH${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PY="${PYTHON:-python3}"
# Training is checkpoint-resumable (artifacts/last.zip every 20k steps). The
# EnergyPlus in-process restart at episode boundaries segfaults occasionally,
# so relaunch until the final model exists (bounded to avoid an endless loop).
for i in $(seq 1 60); do
  [ -f code/artifacts/sac_transfer.zip ] && break
  "$PY" code/train.py --timesteps "${TIMESTEPS:-400000}" || true
done
[ -f code/artifacts/sac_transfer.zip ] || { echo "training did not finish" >&2; exit 1; }

# Evaluation is likewise resumable (results JSON written per episode).
for i in $(seq 1 5); do
  "$PY" code/evaluate.py --model code/artifacts/sac_transfer.zip | tee /dev/stderr | grep -q EVAL_DONE && break
done
