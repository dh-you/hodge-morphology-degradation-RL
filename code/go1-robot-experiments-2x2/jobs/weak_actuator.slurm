#!/usr/bin/env bash

#SBATCH --job-name=go1_weak_actuator
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=11
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/network/dy0130/go1_weak_actuator_%j.out
#SBATCH --error=/scratch/network/dy0130/go1_weak_actuator_%j.err

set -euo pipefail

REPO=/scratch/network/dy0130/go1-robot-experiments
PLAYGROUND=/scratch/network/dy0130/upstream/mujoco_playground
MENAGERIE="$PLAYGROUND/mujoco_playground/external_deps/mujoco_menagerie"
ENVIRONMENT=/scratch/network/dy0130/.go1-robot-experiments
PLAYGROUND_COMMIT=220abb9d6a41eaa258047dacf9ca70a92e008b78
MENAGERIE_COMMIT=1b86ece576591213e2b666ebf59508454200ca97

: "${EXPERIMENT_COMMIT:?required full clean experiment implementation SHA}"
: "${GO1_EXPERIMENT_OUTPUT_ROOT:?required external experiment output root}"

MODEL="${MODEL:-multirank_hodge_lower}"
SEED="${SEED:-0}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-22937600}"
NUM_EVALS="${NUM_EVALS:-2}"
EVALUATION_GRID="${EVALUATION_GRID:-full}"

test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPERIMENT_COMMIT"
test -z "$(git -C "$REPO" status --porcelain)"
test "$(git -C "$PLAYGROUND" rev-parse HEAD)" = "$PLAYGROUND_COMMIT"
test -z "$(git -C "$PLAYGROUND" status --porcelain)"
test "$(git -C "$MENAGERIE" rev-parse HEAD)" = "$MENAGERIE_COMMIT"
test -z "$(git -C "$MENAGERIE" status --porcelain)"
test -x "$ENVIRONMENT/bin/python"

module purge
module load cudatoolkit/12.8
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CUDA_HOME/targets/x86_64-linux/lib/libcusparse.so.12${LD_PRELOAD:+:$LD_PRELOAD}"
export GO1_PLAYGROUND_SOURCE="$PLAYGROUND"
export GO1_MENAGERIE_SOURCE="$MENAGERIE"
export JAX_COMPILATION_CACHE_DIR=/scratch/network/dy0130/.jax-cache
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

nvidia-smi
cd "$REPO"
"$ENVIRONMENT/bin/python" -m experiments.weak_actuator.train \
  --output-root "$GO1_EXPERIMENT_OUTPUT_ROOT" \
  --model "$MODEL" \
  --seed "$SEED" \
  --num-timesteps "$NUM_TIMESTEPS" \
  --num-evals "$NUM_EVALS" \
  --evaluation-grid "$EVALUATION_GRID" \
  --require-gpu
