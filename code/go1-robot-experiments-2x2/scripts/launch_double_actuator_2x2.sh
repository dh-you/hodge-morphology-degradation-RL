#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?smoke or campaign}
IMPLEMENTATION_COMMIT=${2:?frozen implementation commit}
GO1_2X2_OUTPUT_ROOT=${3:-/scratch/network/dy0130/go1-robot-experiments-2x2-artifacts}
REPO=/scratch/network/dy0130/go1-robot-experiments-2x2
ENVIRONMENT=/scratch/network/dy0130/.go1-robot-experiments
test "$(git -C "$REPO" rev-parse HEAD)" = "$IMPLEMENTATION_COMMIT"
test -z "$(git -C "$REPO" status --porcelain)"
export IMPLEMENTATION_COMMIT GO1_2X2_OUTPUT_ROOT

if [[ "$MODE" == smoke ]]; then
  sbatch --parsable --export=ALL,IMPLEMENTATION_COMMIT,GO1_2X2_OUTPUT_ROOT "$REPO/jobs/double_actuator_healthy_smoke.slurm"
  exit
fi
[[ "$MODE" == campaign ]] || exit 2
"$ENVIRONMENT/bin/python" "$REPO/scripts/check_double_actuator_2x2_smokes.py" \
  --artifact-root "$GO1_2X2_OUTPUT_ROOT" \
  --implementation-commit "$IMPLEMENTATION_COMMIT"
"$ENVIRONMENT/bin/python" "$REPO/scripts/check_campaign_storage.py" \
  --output-root "$GO1_2X2_OUTPUT_ROOT"

EXPORTS="ALL,IMPLEMENTATION_COMMIT=$IMPLEMENTATION_COMMIT,GO1_2X2_OUTPUT_ROOT=$GO1_2X2_OUTPUT_ROOT"
C_JOB=$(sbatch --parsable --array=0-14%4 --export="$EXPORTS,POLICY_GROUP=damage_all,EVAL_KIND=nominal,CELL=C" "$REPO/jobs/double_actuator_2x2_eval.slurm")
SHORT_TRAIN=$(sbatch --parsable --export="$EXPORTS" "$REPO/jobs/double_actuator_healthy_train_short.slurm")
HF_TRAIN=$(sbatch --parsable --export="$EXPORTS" "$REPO/jobs/double_actuator_healthy_train_hodge_full.slurm")

A_SHORT=$(sbatch --parsable --array=0-11%4 --dependency="aftercorr:$SHORT_TRAIN" --export="$EXPORTS,POLICY_GROUP=healthy_short,EVAL_KIND=nominal,CELL=A" "$REPO/jobs/double_actuator_2x2_eval.slurm")
A_HF=$(sbatch --parsable --array=0-2%2 --dependency="aftercorr:$HF_TRAIN" --export="$EXPORTS,POLICY_GROUP=healthy_hodge_full,EVAL_KIND=nominal,CELL=A" "$REPO/jobs/double_actuator_2x2_eval.slurm")
B0_SHORT=$(sbatch --parsable --array=0-11%4 --dependency="aftercorr:$SHORT_TRAIN" --export="$EXPORTS,POLICY_GROUP=healthy_short,EVAL_KIND=heldout,CELL=B,EVALUATION_SHARD=0" "$REPO/jobs/double_actuator_2x2_eval.slurm")
B1_SHORT=$(sbatch --parsable --array=0-11%4 --dependency="aftercorr:$SHORT_TRAIN" --export="$EXPORTS,POLICY_GROUP=healthy_short,EVAL_KIND=heldout,CELL=B,EVALUATION_SHARD=1" "$REPO/jobs/double_actuator_2x2_eval.slurm")
B0_HF=$(sbatch --parsable --array=0-2%2 --dependency="aftercorr:$HF_TRAIN" --export="$EXPORTS,POLICY_GROUP=healthy_hodge_full,EVAL_KIND=heldout,CELL=B,EVALUATION_SHARD=0" "$REPO/jobs/double_actuator_2x2_eval.slurm")
B1_HF=$(sbatch --parsable --array=0-2%2 --dependency="aftercorr:$HF_TRAIN" --export="$EXPORTS,POLICY_GROUP=healthy_hodge_full,EVAL_KIND=heldout,CELL=B,EVALUATION_SHARD=1" "$REPO/jobs/double_actuator_2x2_eval.slurm")
MERGE_SHORT=$(sbatch --parsable --array=0-11 --dependency="aftercorr:$B0_SHORT:$B1_SHORT" --export="$EXPORTS,POLICY_GROUP=healthy_short" "$REPO/jobs/double_actuator_2x2_merge.slurm")
MERGE_HF=$(sbatch --parsable --array=0-2 --dependency="aftercorr:$B0_HF:$B1_HF" --export="$EXPORTS,POLICY_GROUP=healthy_hodge_full" "$REPO/jobs/double_actuator_2x2_merge.slurm")

LAUNCH_ROOT="$GO1_2X2_OUTPUT_ROOT/$IMPLEMENTATION_COMMIT/double-actuator-2x2"
mkdir -p "$LAUNCH_ROOT"
test ! -e "$LAUNCH_ROOT/LAUNCH.json"
printf '{"implementation_commit":"%s","output_root":"%s","jobs":{"C":"%s","healthy_short_training":"%s","healthy_hodge_full_training":"%s","A_short":"%s","A_hodge_full":"%s","B0_short":"%s","B1_short":"%s","B0_hodge_full":"%s","B1_hodge_full":"%s","B_merge_short":"%s","B_merge_hodge_full":"%s"}}\n' \
  "$IMPLEMENTATION_COMMIT" "$GO1_2X2_OUTPUT_ROOT" "$C_JOB" "$SHORT_TRAIN" "$HF_TRAIN" \
  "$A_SHORT" "$A_HF" "$B0_SHORT" "$B1_SHORT" "$B0_HF" "$B1_HF" \
  "$MERGE_SHORT" "$MERGE_HF" > "$LAUNCH_ROOT/LAUNCH.json"
cat "$LAUNCH_ROOT/LAUNCH.json"
