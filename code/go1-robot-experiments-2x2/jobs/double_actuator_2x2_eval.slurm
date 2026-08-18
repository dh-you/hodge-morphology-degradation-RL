#!/usr/bin/env bash
#SBATCH --job-name=go1_2x2_eval
#SBATCH --partition=gpu
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/network/dy0130/go1_2x2_eval_%A_%a.out
#SBATCH --error=/scratch/network/dy0130/go1_2x2_eval_%A_%a.err

set -euo pipefail
: "${POLICY_GROUP:?damage_all, healthy_short, or healthy_hodge_full required}"
: "${EVAL_KIND:?nominal or heldout required}"
: "${CELL:?A, B, or C required}"
: "${IMPLEMENTATION_COMMIT:?required clean 2x2 implementation SHA}"
: "${GO1_2X2_OUTPUT_ROOT:?required external 2x2 artifact root}"
case "$POLICY_GROUP" in
  damage_all)
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
    TRAINING_COMMIT=9a125fd245874b1d3be50238f3cdddc510832ce0
    if [[ "$MODEL" == multirank_hodge_full_w150 ]]; then
      TRAINING_BASE=/scratch/network/dy0130/go1-robot-experiments-artifacts-retries/3348141-timeout
    else
      TRAINING_BASE=/scratch/network/dy0130/go1-robot-experiments-artifacts
    fi
    TRAINING_ROOT="$TRAINING_BASE/$TRAINING_COMMIT/double-actuator/$MODEL/seed-$SEED/steps-400000000/evals-19"
    ;;
  healthy_short)
    MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
    SEED=$((11 + SLURM_ARRAY_TASK_ID % 3))
    case "$MODEL_INDEX" in
      0) MODEL=multirank_hodge_lower_w177 ;;
      1) MODEL=gcn_w1107 ;;
      2) MODEL=pointwise_w252 ;;
      3) MODEL=native_mlp_w480_240_120 ;;
      *) exit 2 ;;
    esac
    TRAINING_COMMIT="$IMPLEMENTATION_COMMIT"
    TRAINING_ROOT="$GO1_2X2_OUTPUT_ROOT/$TRAINING_COMMIT/double-actuator-healthy/$MODEL/seed-$SEED/steps-400000000/evals-19"
    ;;
  healthy_hodge_full)
    MODEL=multirank_hodge_full_w150
    SEED=$((11 + SLURM_ARRAY_TASK_ID))
    TRAINING_COMMIT="$IMPLEMENTATION_COMMIT"
    TRAINING_ROOT="$GO1_2X2_OUTPUT_ROOT/$TRAINING_COMMIT/double-actuator-healthy/$MODEL/seed-$SEED/steps-400000000/evals-19"
    ;;
  *) exit 2 ;;
esac
if [[ "$EVAL_KIND" == nominal ]]; then
  [[ "$CELL" == A || "$CELL" == C ]] || exit 2
elif [[ "$EVAL_KIND" == heldout ]]; then
  [[ "$CELL" == B ]] || exit 2
  : "${EVALUATION_SHARD:?heldout evaluation requires shard 0 or 1}"
  [[ "$EVALUATION_SHARD" == 0 || "$EVALUATION_SHARD" == 1 ]] || exit 2
else
  exit 2
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '{"campaign":"double-actuator-2x2-eval","task_id":%s,"policy_group":"%s","cell":"%s","kind":"%s","model":"%s","seed":%s,"training_root":"%s"}\n' "$SLURM_ARRAY_TASK_ID" "$POLICY_GROUP" "$CELL" "$EVAL_KIND" "$MODEL" "$SEED" "$TRAINING_ROOT"
  exit 0
fi
REPO=/scratch/network/dy0130/go1-robot-experiments-2x2
PLAYGROUND=/scratch/network/dy0130/upstream/mujoco_playground
MENAGERIE="$PLAYGROUND/mujoco_playground/external_deps/mujoco_menagerie"
ENVIRONMENT=/scratch/network/dy0130/.go1-robot-experiments
PLAYGROUND_COMMIT=220abb9d6a41eaa258047dacf9ca70a92e008b78
MENAGERIE_COMMIT=1b86ece576591213e2b666ebf59508454200ca97
test "$(git -C "$REPO" rev-parse HEAD)" = "$IMPLEMENTATION_COMMIT"
test -z "$(git -C "$REPO" status --porcelain)"
test "$(git -C "$PLAYGROUND" rev-parse HEAD)" = "$PLAYGROUND_COMMIT"
test -z "$(git -C "$PLAYGROUND" status --porcelain)"
test "$(git -C "$MENAGERIE" rev-parse HEAD)" = "$MENAGERIE_COMMIT"
test -z "$(git -C "$MENAGERIE" status --porcelain)"
"$ENVIRONMENT/bin/python" "$REPO/scripts/check_campaign_storage.py" --output-root "$GO1_2X2_OUTPUT_ROOT"
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
if [[ "$EVAL_KIND" == nominal ]]; then
  "$ENVIRONMENT/bin/python" -m experiments.double_actuator.evaluate_nominal --training-root "$TRAINING_ROOT" --output-root "$GO1_2X2_OUTPUT_ROOT" --model "$MODEL" --seed "$SEED" --cell "$CELL" --training-commit "$TRAINING_COMMIT" --require-gpu
else
  "$ENVIRONMENT/bin/python" -m experiments.double_actuator.evaluate_heldout_control --training-root "$TRAINING_ROOT" --output-root "$GO1_2X2_OUTPUT_ROOT" --model "$MODEL" --seed "$SEED" --shard "$EVALUATION_SHARD" --training-commit "$TRAINING_COMMIT" --require-gpu
fi
