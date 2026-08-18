#!/usr/bin/env bash
#SBATCH --job-name=go1_double_eval
#SBATCH --partition=gpu
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-14%4
#SBATCH --output=/scratch/network/dy0130/go1_double_eval_%A_%a.out
#SBATCH --error=/scratch/network/dy0130/go1_double_eval_%A_%a.err

set -euo pipefail
POLICY_INDEX=$SLURM_ARRAY_TASK_ID
: "${EVALUATION_SHARD:?required held-out shard 0 or 1}"
SHARD=$EVALUATION_SHARD
[[ "$SHARD" == 0 || "$SHARD" == 1 ]] || exit 2
MODEL_INDEX=$((POLICY_INDEX / 3))
SEED=$((11 + POLICY_INDEX % 3))
case "$MODEL_INDEX" in
  0) MODEL=multirank_hodge_full_w150 ;;
  1) MODEL=multirank_hodge_lower_w177 ;;
  2) MODEL=gcn_w1107 ;;
  3) MODEL=pointwise_w252 ;;
  4) MODEL=native_mlp_w480_240_120 ;;
  *) exit 2 ;;
esac
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf "{\"campaign\":\"double-actuator-heldout-shard\",\"task_id\":%s,\"policy_index\":%s,\"model\":\"%s\",\"seed\":%s,\"shard\":%s,\"pairs\":11,\"conditions\":891,\"rows\":80190}\\n" \
    "$SLURM_ARRAY_TASK_ID" "$POLICY_INDEX" "$MODEL" "$SEED" "$SHARD"
  exit 0
fi

EVALUATOR_REPO=/scratch/network/dy0130/go1-robot-experiments-eval-shards
PLAYGROUND=/scratch/network/dy0130/upstream/mujoco_playground
MENAGERIE="$PLAYGROUND/mujoco_playground/external_deps/mujoco_menagerie"
ENVIRONMENT=/scratch/network/dy0130/.go1-robot-experiments
PLAYGROUND_COMMIT=220abb9d6a41eaa258047dacf9ca70a92e008b78
MENAGERIE_COMMIT=1b86ece576591213e2b666ebf59508454200ca97
: "${EVALUATOR_COMMIT:?required full clean evaluator implementation SHA}"
: "${TRAINING_COMMIT:?required frozen training implementation SHA}"
: "${GO1_EXPERIMENT_OUTPUT_ROOT:?required external artifact root}"
[[ "$EVALUATOR_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$TRAINING_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 2
test "$(git -C "$EVALUATOR_REPO" rev-parse HEAD)" = "$EVALUATOR_COMMIT"
test -z "$(git -C "$EVALUATOR_REPO" status --porcelain)"
test "$(git -C "$PLAYGROUND" rev-parse HEAD)" = "$PLAYGROUND_COMMIT"
test -z "$(git -C "$PLAYGROUND" status --porcelain)"
test "$(git -C "$MENAGERIE" rev-parse HEAD)" = "$MENAGERIE_COMMIT"
test -z "$(git -C "$MENAGERIE" status --porcelain)"
test -x "$ENVIRONMENT/bin/python"
"$ENVIRONMENT/bin/python" "$EVALUATOR_REPO/scripts/check_campaign_storage.py" \
  --output-root "$GO1_EXPERIMENT_OUTPUT_ROOT"
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
cd "$EVALUATOR_REPO"
nvidia-smi
TRAINING_ROOT="$GO1_EXPERIMENT_OUTPUT_ROOT/$TRAINING_COMMIT/double-actuator/$MODEL/seed-$SEED/steps-400000000/evals-19"
"$ENVIRONMENT/bin/python" -m experiments.double_actuator.evaluate \
  --training-root "$TRAINING_ROOT" --training-commit "$TRAINING_COMMIT" \
  --model "$MODEL" --seed "$SEED" --shard "$SHARD" --require-gpu
