# Contributing to Go1 Morphology Core

`go1-morphology-core` is the shared, task-independent model library used by
the robot-failure and terrain research tracks. Its `main` branch represents
the reviewed canonical core. Experiment-specific environments, curricula,
metrics, evaluators, and campaigns belong in their respective experiment
repositories.

GitHub Free does not enforce the desired ruleset for this private repository,
so the requirements below are a maintainer agreement. In particular, never
push semantic changes directly to `main`.

## Default: consume the released core

Experiment repositories should normally pin the released core at an exact
commit. The v1.0.0 implementation is distributed from:

```text
debc05563fdc0d8dd1befde4426781ce611c7386
```

The `v1.0.0` tag is immutable and must always peel to that commit. A task that
does not require shared model machinery should not modify this repository.

Use this placement test when deciding where code belongs:

> Would both research tracks need this capability if the other track's
> experiment did not exist?

If not, keep the change in the experiment repository. Actuator and sensor
faults, latency, payloads, terrain generation, terrain curricula, and
task-specific evaluation are not core concerns.

## Temporary core experiments

When shared model machinery must be explored, create a short-lived branch
from the current `main`:

```text
<maintainer>/<short-change>
```

Examples include `derek/hodge-neural-update` and
`friend/terrain-observation-hook`. Do not maintain permanent branches named
for individual contributors.

During the experiment:

1. Make unrestricted experimental changes on the temporary branch.
2. Pin the consuming experiment repository to the branch's exact commit, not
   to the moving branch name.
3. Leave the other research track on its existing released-core pin.
4. Record the experimental core commit with the resulting evidence.

If the experiment is unsuccessful, restore the consumer's released-core pin
and delete the temporary branch. Canonical `main` remains unchanged.

## Promoting a successful experiment

Promote a change only when it is task-neutral and should become part of the
shared core. Before opening a pull request:

1. Reduce the branch to the smallest reviewable shared-core change.
2. Document public API and versioning consequences.
3. Preserve the meaning of existing model IDs and the frozen Go1 contract.
4. Run the self-contained test suite and all seven reference tests without
   failures or skips.
5. For changes affecting models, features, topology, or checkpoint loading,
   attach evidence for parameter counts, fingerprints, checkpoint
   reconstruction, logits, and deterministic actions.
6. Ask the other maintainer to review and approve the pull request.
7. Squash-merge the approved change and delete its source branch.

Because repository settings cannot currently enforce this workflow, both
maintainers are responsible for following it. The absence of technical branch
protection is not permission to bypass review for semantic changes.

## Compatibility and model identities

Do not silently redefine a canonical model such as
`multirank_hodge_lower`. Prefer a new model identity for new semantics. A
change that intentionally alters an existing canonical model or public
contract requires a major release and explicit migration documentation.

Every released change must retain enough provenance to identify the exact
core commit used by each consuming experiment. Consumers may upgrade their
pins independently; the robot and terrain repositories do not need to move to
a new core commit at the same time.

## Versioning

| Change | Version policy |
|---|---|
| Documentation or packaging only | No release, or a patch release when distribution requires one |
| Bug fix preserving intended behavior | Patch, for example `1.0.1` |
| Backward-compatible capability or new model | Minor, for example `1.1.0` |
| Existing canonical semantics or contract change | Major, for example `2.0.0` |

Documentation-only changes do not alter the scientific identity of an
existing release, and release tags never move.
