#!/usr/bin/env bash
#SBATCH --job-name=go1_2x2_htrain_hf
#SBATCH --partition=gpu
#SBATCH --qos=gpu-medium
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:30:00
#SBATCH --array=0-2%2
#SBATCH --output=/scratch/network/dy0130/go1_2x2_htrain_hf_%A_%a.out
#SBATCH --error=/scratch/network/dy0130/go1_2x2_htrain_hf_%A_%a.err

set -euo pipefail
SEED=$((11 + SLURM_ARRAY_TASK_ID))
MODEL=multirank_hodge_full_w150
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '{"campaign":"double-actuator-healthy","task_id":%s,"model":"%s","seed":%s,"requested_steps":400000000,"actual_steps":412876800,"num_evals":19}\n' "$SLURM_ARRAY_TASK_ID" "$MODEL" "$SEED"
  exit 0
fi
REPO=/scratch/network/dy0130/go1-robot-experiments-2x2
PLAYGROUND=/scratch/network/dy0130/upstream/mujoco_playground
MENAGERIE="$PLAYGROUND/mujoco_playground/external_deps/mujoco_menagerie"
ENVIRONMENT=/scratch/network/dy0130/.go1-robot-experiments
DAMAGE_ROOT=/scratch/network/dy0130/go1-robot-experiments-artifacts-retries/3348141-timeout
DAMAGE_COMMIT=9a125fd245874b1d3be50238f3cdddc510832ce0
PLAYGROUND_COMMIT=220abb9d6a41eaa258047dacf9ca70a92e008b78
MENAGERIE_COMMIT=1b86ece576591213e2b666ebf59508454200ca97
: "${IMPLEMENTATION_COMMIT:?required clean 2x2 implementation SHA}"
: "${GO1_2X2_OUTPUT_ROOT:?required external 2x2 artifact root}"
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
REFERENCE="$DAMAGE_ROOT/$DAMAGE_COMMIT/double-actuator/$MODEL/seed-$SEED/steps-400000000/evals-19"
cd "$REPO"
nvidia-smi
"$ENVIRONMENT/bin/python" -m experiments.double_actuator.train_healthy --output-root "$GO1_2X2_OUTPUT_ROOT" --damage-reference-root "$REFERENCE" --model "$MODEL" --seed "$SEED" --num-timesteps 400000000 --num-evals 19 --require-gpu
