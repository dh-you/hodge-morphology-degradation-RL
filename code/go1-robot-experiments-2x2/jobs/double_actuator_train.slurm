#!/usr/bin/env bash
#SBATCH --job-name=go1_double_train
#SBATCH --partition=gpu
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-14%4
#SBATCH --output=/scratch/network/dy0130/go1_double_train_%A_%a.out
#SBATCH --error=/scratch/network/dy0130/go1_double_train_%A_%a.err

set -euo pipefail
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED=$((11 + SLURM_ARRAY_TASK_ID % 3))
case "$MODEL_INDEX" in
  0) MODEL=multirank_hodge_full_w150 ;;
  1) MODEL=multirank_hodge_lower_w177 ;;
  2) MODEL=gcn_w1107 ;;
  3) MODEL=pointwise_w252 ;;
  4) MODEL=native_mlp_w480_240_120 ;;
  *) exit 2 ;;
esac
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '{"campaign":"double-actuator-train","task_id":%s,"model":"%s","seed":%s,"requested_steps":400000000,"actual_steps":412876800,"num_evals":19}\n' "$SLURM_ARRAY_TASK_ID" "$MODEL" "$SEED"
  exit 0
fi
set -euo pipefail
REPO=/scratch/network/dy0130/go1-robot-experiments
PLAYGROUND=/scratch/network/dy0130/upstream/mujoco_playground
MENAGERIE="$PLAYGROUND/mujoco_playground/external_deps/mujoco_menagerie"
ENVIRONMENT=/scratch/network/dy0130/.go1-robot-experiments
PLAYGROUND_COMMIT=220abb9d6a41eaa258047dacf9ca70a92e008b78
MENAGERIE_COMMIT=1b86ece576591213e2b666ebf59508454200ca97
: "${EXPERIMENT_COMMIT:?required full clean campaign implementation SHA}"
: "${GO1_EXPERIMENT_OUTPUT_ROOT:?required external artifact root}"
[[ "$EXPERIMENT_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 2
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPERIMENT_COMMIT"
test -z "$(git -C "$REPO" status --porcelain)"
test "$(git -C "$PLAYGROUND" rev-parse HEAD)" = "$PLAYGROUND_COMMIT"
test -z "$(git -C "$PLAYGROUND" status --porcelain)"
test "$(git -C "$MENAGERIE" rev-parse HEAD)" = "$MENAGERIE_COMMIT"
test -z "$(git -C "$MENAGERIE" status --porcelain)"
test -x "$ENVIRONMENT/bin/python"
"$ENVIRONMENT/bin/python" "$REPO/scripts/check_campaign_storage.py" --output-root "$GO1_EXPERIMENT_OUTPUT_ROOT"
module purge
module load cudatoolkit/12.8
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CUDA_HOME/targets/x86_64-linux/lib/libcusparse.so.12${LD_PRELOAD:+:$LD_PRELOAD}"
export GO1_PLAYGROUND_SOURCE="$PLAYGROUND"
export GO1_MENAGERIE_SOURCE="$MENAGERIE"
export JAX_COMPILATION_CACHE_DIR=/scratch/network/dy0130/.jax-cache
export UV_CACHE_DIR=/scratch/network/dy0130/.uv-cache
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$REPO"
nvidia-smi
"$ENVIRONMENT/bin/python" -m experiments.double_actuator.train \
  --output-root "$GO1_EXPERIMENT_OUTPUT_ROOT" --model "$MODEL" --seed "$SEED" \
  --num-timesteps 400000000 --num-evals 19 --require-gpu
