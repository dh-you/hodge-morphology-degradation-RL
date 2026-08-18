# Weak-actuator PPO experiment

This experiment trains a fresh canonical `go1_core` policy under the original
weak/dead-actuator task. Model construction comes only from the exact-pinned
core release.

The training curriculum assigns each vectorized environment member to healthy
(`0.30`), weak (`0.50`), or dead (`0.20`). Nonhealthy members damage one
uniformly selected actuator. Weak strength is log-uniform on `[0.05, 0.20)`;
dead strength is zero. The assignment is fixed for that vectorized member and
is not resampled at reset.

## Profiles

The practical default is:

```text
requested steps: 22,937,600
evaluations:      2
actual steps:     22,937,600
evaluation grid: full (9,810 rows)
```

The original 400M campaign accounting is available without changing code:

```text
requested steps: 400,000,000
evaluations:      19
actual steps:     412,876,800
evaluation grid: full (9,810 rows)
```

The runner saves only the terminal checkpoint. Evaluation/checkpoint cadence
does not alter the curriculum or PPO optimizer configuration, but it does
affect Brax batch accounting and is therefore recorded in the immutable run
path and metadata.

## Run

```bash
export GO1_EXPERIMENT_OUTPUT_ROOT=/scratch/network/$USER/go1-results

python -m experiments.weak_actuator.train \
  --output-root "$GO1_EXPERIMENT_OUTPUT_ROOT" \
  --model multirank_hodge_lower \
  --seed 0 \
  --num-timesteps 22937600 \
  --num-evals 2 \
  --evaluation-grid full \
  --require-gpu
```

`--evaluation-grid smoke` evaluates only `stand`, reset key 0, over all 109
ordered conditions. `full` evaluates the original nine commands and ten reset
keys for 9,810 ordered rows.

The checkout must be clean and committed. Existing output paths are never
overwritten. A failure after path creation is recorded in `failure.json`.

For the fixed two-model, five-seed campaign, submit
`jobs/weak_actuator_hodge.slurm`. Its ten immutable array tasks map
`multirank_hodge_lower` and `multirank_hodge_full` to seeds 1 through 5.
Each task requests 400,000,000 steps with 19 evaluations, realizes exactly
412,876,800 environment steps, and performs the full 9,810-row terminal
evaluation.

## Outputs

```text
<output-root>/<experiment-sha>/weak-actuator/<model>/seed-<seed>/
  steps-<requested>/evals-<count>/grid-<grid>/
    run.json
    curriculum.json
    ppo_config.json
    progress.jsonl
    checkpoints/<actual-step>/
    condition_metrics.csv
    evaluation.json
    result.json
```

The restored checkpoint must match the terminal parameters, fixed-probe
logits, and deterministic actions exactly. Every rollout metric must be finite
and termination bookkeeping must be valid.
