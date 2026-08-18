#!/usr/bin/env bash
#SBATCH --job-name=go1_2x2_merge
#SBATCH --partition=all
#SBATCH --qos=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/network/dy0130/go1_2x2_merge_%A_%a.out
#SBATCH --error=/scratch/network/dy0130/go1_2x2_merge_%A_%a.err

set -euo pipefail
: "${POLICY_GROUP:?healthy_short or healthy_hodge_full required}"
: "${IMPLEMENTATION_COMMIT:?required clean 2x2 implementation SHA}"
: "${GO1_2X2_OUTPUT_ROOT:?required external 2x2 artifact root}"
case "$POLICY_GROUP" in
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
    ;;
  healthy_hodge_full)
    MODEL=multirank_hodge_full_w150
    SEED=$((11 + SLURM_ARRAY_TASK_ID))
    ;;
  *) exit 2 ;;
esac
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '{"campaign":"double-actuator-2x2-merge","task_id":%s,"policy_group":"%s","model":"%s","seed":%s,"rows":160380}\n' "$SLURM_ARRAY_TASK_ID" "$POLICY_GROUP" "$MODEL" "$SEED"
  exit 0
fi
REPO=/scratch/network/dy0130/go1-robot-experiments-2x2
ENVIRONMENT=/scratch/network/dy0130/.go1-robot-experiments
test "$(git -C "$REPO" rev-parse HEAD)" = "$IMPLEMENTATION_COMMIT"
test -z "$(git -C "$REPO" status --porcelain)"
cd "$REPO"
"$ENVIRONMENT/bin/python" -m experiments.double_actuator.merge_heldout_control --output-root "$GO1_2X2_OUTPUT_ROOT" --model "$MODEL" --seed "$SEED" --training-commit "$IMPLEMENTATION_COMMIT"
