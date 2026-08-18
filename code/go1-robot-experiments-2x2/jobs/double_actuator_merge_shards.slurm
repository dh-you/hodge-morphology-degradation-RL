#!/usr/bin/env bash
#SBATCH --job-name=go1_double_merge
#SBATCH --partition=all
#SBATCH --qos=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --array=0-14
#SBATCH --output=/scratch/network/dy0130/go1_double_merge_%A_%a.out
#SBATCH --error=/scratch/network/dy0130/go1_double_merge_%A_%a.err

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
  printf "{\"campaign\":\"double-actuator-heldout-merge\",\"task_id\":%s,\"model\":\"%s\",\"seed\":%s,\"shards\":2,\"rows\":160380}\\n" \
    "$SLURM_ARRAY_TASK_ID" "$MODEL" "$SEED"
  exit 0
fi

EVALUATOR_REPO=/scratch/network/dy0130/go1-robot-experiments-eval-shards
ENVIRONMENT=/scratch/network/dy0130/.go1-robot-experiments
: "${EVALUATOR_COMMIT:?required full clean evaluator implementation SHA}"
: "${TRAINING_COMMIT:?required frozen training implementation SHA}"
: "${GO1_EXPERIMENT_OUTPUT_ROOT:?required external artifact root}"
[[ "$EVALUATOR_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$TRAINING_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 2
test "$(git -C "$EVALUATOR_REPO" rev-parse HEAD)" = "$EVALUATOR_COMMIT"
test -z "$(git -C "$EVALUATOR_REPO" status --porcelain)"
test -x "$ENVIRONMENT/bin/python"
cd "$EVALUATOR_REPO"
TRAINING_ROOT="$GO1_EXPERIMENT_OUTPUT_ROOT/$TRAINING_COMMIT/double-actuator/$MODEL/seed-$SEED/steps-400000000/evals-19"
"$ENVIRONMENT/bin/python" -m experiments.double_actuator.merge_shards \
  --training-root "$TRAINING_ROOT" --training-commit "$TRAINING_COMMIT" \
  --model "$MODEL" --seed "$SEED"
