#!/usr/bin/env bash

#SBATCH --job-name=go1_weak_hodge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --cpus-per-task=11
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=/scratch/network/dy0130/go1_weak_hodge_%A_%a.out
#SBATCH --error=/scratch/network/dy0130/go1_weak_hodge_%A_%a.err

set -euo pipefail

REPO=/scratch/network/dy0130/go1-robot-experiments
PLAYGROUND=/scratch/network/dy0130/upstream/mujoco_playground
MENAGERIE="$PLAYGROUND/mujoco_playground/external_deps/mujoco_menagerie"
ENVIRONMENT=/scratch/network/dy0130/.go1-robot-experiments
PLAYGROUND_COMMIT=220abb9d6a41eaa258047dacf9ca70a92e008b78
MENAGERIE_COMMIT=1b86ece576591213e2b666ebf59508454200ca97
REQUESTED_TIMESTEPS=400000000
NUM_EVALS=19
ACTUAL_TIMESTEPS=412876800
EVALUATION_GRID=full

: "${SLURM_ARRAY_TASK_ID:?this launcher requires the Slurm array}"

case "$SLURM_ARRAY_TASK_ID" in
  0) MODEL=multirank_hodge_lower; SEED=1 ;;
  1) MODEL=multirank_hodge_lower; SEED=2 ;;
  2) MODEL=multirank_hodge_lower; SEED=3 ;;
  3) MODEL=multirank_hodge_lower; SEED=4 ;;
  4) MODEL=multirank_hodge_lower; SEED=5 ;;
  5) MODEL=multirank_hodge_full;  SEED=1 ;;
  6) MODEL=multirank_hodge_full;  SEED=2 ;;
  7) MODEL=multirank_hodge_full;  SEED=3 ;;
  8) MODEL=multirank_hodge_full;  SEED=4 ;;
  9) MODEL=multirank_hodge_full;  SEED=5 ;;
  *)
    echo "array task must be in 0-9, got $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '{"task_id":%s,"model":"%s","seed":%s,"requested_steps":%s,"actual_steps":%s,"num_evals":%s,"evaluation_grid":"%s"}\n' \
    "$SLURM_ARRAY_TASK_ID" "$MODEL" "$SEED" "$REQUESTED_TIMESTEPS" \
    "$ACTUAL_TIMESTEPS" "$NUM_EVALS" "$EVALUATION_GRID"
  exit 0
fi

: "${EXPERIMENT_COMMIT:?required full clean experiment implementation SHA}"
: "${GO1_EXPERIMENT_OUTPUT_ROOT:?required external experiment output root}"
[[ "$EXPERIMENT_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  echo "EXPERIMENT_COMMIT must be a full lowercase SHA" >&2
  exit 2
}

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

echo "campaign_task=$SLURM_ARRAY_TASK_ID model=$MODEL seed=$SEED requested_steps=$REQUESTED_TIMESTEPS actual_steps=$ACTUAL_TIMESTEPS num_evals=$NUM_EVALS grid=$EVALUATION_GRID"
nvidia-smi
cd "$REPO"
"$ENVIRONMENT/bin/python" -m experiments.weak_actuator.train \
  --output-root "$GO1_EXPERIMENT_OUTPUT_ROOT" \
  --model "$MODEL" \
  --seed "$SEED" \
  --num-timesteps "$REQUESTED_TIMESTEPS" \
  --num-evals "$NUM_EVALS" \
  --evaluation-grid "$EVALUATION_GRID" \
  --require-gpu
