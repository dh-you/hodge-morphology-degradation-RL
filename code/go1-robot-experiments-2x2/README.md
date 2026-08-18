# Go1 Robot Experiments

Scientific robot-failure experiments by Derek You, built on the shared
[`go1-morphology-core`](https://github.com/dh-you/go1-morphology-core) model
library.

This repository owns task-specific environments, perturbations, curricula,
training entry points, evaluation protocols, and scientific outputs. Model
architectures, Go1 feature construction, topology, and checkpoint
reconstruction stay in the exact-pinned core package.

## Weak-actuator experiment

`experiments/weak_actuator` trains any of the six canonical core policies in
`Go1JoystickFlatTerrain`, `impl="jax"`, under the original healthy/weak/dead
actuator curriculum:

- category probabilities: healthy `0.30`, weak `0.50`, dead `0.20`;
- one uniformly selected actuator per nonhealthy vectorized environment;
- weak strength sampled log-uniformly from `[0.05, 0.20)`;
- healthy and dead strengths fixed at `1.0` and `0.0`;
- assignments fixed per vectorized environment member, without reset
  resampling.

The default evaluation is the original complete grid: 9 commands × 10 reset
keys × 109 healthy/damage conditions, each with a 1,000-step horizon. Use
`--evaluation-grid smoke` for the stand/reset-0 109-row grid.

The practical default training profile uses one original campaign interval:
22,937,600 requested steps and 2 evaluations. To reproduce the original 400M
step accounting, use 400,000,000 requested steps and 19 evaluations; the PPO
batch accounting then yields 412,876,800 actual environment steps.

## Run

Runs require a clean committed checkout because the external output path is
bound to the exact experiment commit. The output root must not be inside this
repository.

Create the locked environment and point its Menagerie assets at the pinned
source checkout:

```bash
uv sync --frozen
uv run python scripts/link_pinned_menagerie.py \
  --source /scratch/network/dy0130/upstream/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie \
  --replace-installed-copy
```

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

Original campaign cadence:

```bash
python -m experiments.weak_actuator.train \
  --output-root "$GO1_EXPERIMENT_OUTPUT_ROOT" \
  --model multirank_hodge_lower \
  --seed 0 \
  --num-timesteps 400000000 \
  --num-evals 19 \
  --evaluation-grid full \
  --require-gpu
```

For Slurm, export `EXPERIMENT_COMMIT`, `GO1_EXPERIMENT_OUTPUT_ROOT`, and any
optional model/profile overrides, then submit `jobs/weak_actuator.slurm`.

## Paper-ready campaign

The frozen six-model main study and paired Hodge extension are specified in
[`docs/PAPER_WEAK_ACTUATOR_PROTOCOL.md`](docs/PAPER_WEAK_ACTUATOR_PROTOCOL.md).
The existing Hodge-L/Hodge-F seeds 1-5 remain under implementation commit
`771c3d73b1fb2a97a417eb58a945266c1770cd65` and are reused without copying or
modification.

After committing a clean campaign implementation, submit the two additive arrays:

```bash
export EXPERIMENT_COMMIT="$(git rev-parse HEAD)"
export GO1_EXPERIMENT_OUTPUT_ROOT=/scratch/network/$USER/go1-robot-experiments-artifacts

sbatch --export=ALL jobs/weak_actuator_baselines.slurm
sbatch --export=ALL jobs/weak_actuator_hodge_extension.slurm
```

The first array trains Native MLP, Central MLP, Joint MLP, and GCN at seeds
1-5. The second trains Hodge-L and Hodge-F at seeds 6-10. Valid runs are never
rerun; technical retries require a separate, explicitly registered artifact
root.

After all 30 new tasks pass, install the analysis group and generate the
all-or-nothing paper pack twice for reproducibility:

```bash
uv sync --frozen --group analysis
python -m experiments.weak_actuator.paper generate \
  --artifact-root "$GO1_EXPERIMENT_OUTPUT_ROOT" \
  --campaign-commit "$EXPERIMENT_COMMIT" \
  --verify-reproducible
```

Outputs are immutable and include every run-defining input:

```text
$GO1_EXPERIMENT_OUTPUT_ROOT/<experiment-sha>/weak-actuator/
  <model>/seed-<seed>/steps-<requested-steps>/
    evals-<num-evals>/grid-<smoke-or-full>/
      run.json
      curriculum.json
      ppo_config.json
      progress.jsonl
      checkpoints/<actual-step>/
      condition_metrics.csv
      evaluation.json
      result.json
```

The terminal checkpoint is restored through `go1_core.load_checkpoint` and
must preserve parameters, probe logits, and deterministic actions exactly.

## Scientific continuity

The task mechanics come from pre-split source commit
`1659809766f6aba43a1074cc332922317e9bec5a`. The permanent mapping and the
intentional interface differences are documented in
[`docs/WEAK_ACTUATOR_PROTOCOL.md`](docs/WEAK_ACTUATOR_PROTOCOL.md).

Historical extraction and GPU-certification evidence is archived in
[`docs/HISTORICAL_CERTIFICATION.md`](docs/HISTORICAL_CERTIFICATION.md); none of
its reproduction harnesses or reference fixtures are active code.

## Repository layout

```text
experiments/weak_actuator/       training protocol and runner
jobs/weak_actuator.slurm         GPU launcher
jobs/weak_actuator_{baselines,hodge_extension}.slurm analysis campaign arrays
src/go1_robot_experiments/       reusable task protocol and rollout logic
tests/                            CPU scientific-contract tests
docs/                             protocol continuity and archived provenance
```

Run the CPU checks with:

```bash
uv lock --check
uv run pytest -q
```
