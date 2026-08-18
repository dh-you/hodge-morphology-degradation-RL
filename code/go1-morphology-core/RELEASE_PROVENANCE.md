# Go1 Morphology Core v1.0.0 Release Provenance

This record binds the v1.0.0 distribution to the completed extraction and
training certifications. The release changes documentation and packaging
metadata only; `src/go1_core`, the committed fixtures, and
`EXTRACTION_MANIFEST.md` are byte-identical to certified core commit
`652a05e3d7f446d2cd770a8361a5b3dac79e37e8`.

## Authority chain

| Stage | Immutable identity |
|---|---|
| Historical research source | `1659809766f6aba43a1074cc332922317e9bec5a` |
| Extraction inventory commit | `419b7118ba378e7881cb09ad36675e70bc9339c4` |
| Extraction manifest SHA-256 | `62aff4b4abf8332ba2ab1e2b490761b417c967e5d16e51bcc663c2c842041a82` |
| Certified standalone core | `652a05e3d7f446d2cd770a8361a5b3dac79e37e8` |
| Job 4 consumer implementation | `213897ce9ca63cfe57278155966eaad428b34fc9` |
| Job 4 certification record | `5835e182df1b0093591b412b5a4ce619cf8aa249` |
| Job 4 report SHA-256 | `02a5a38eb046102a368bccf8b4c9e4437c6abedc8504906e731463aa0863559c` |
| Playground | `220abb9d6a41eaa258047dacf9ca70a92e008b78` |
| Menagerie | `1b86ece576591213e2b666ebf59508454200ca97` |

## Job 1 — frozen inventory

Status: **PASS**.

The extraction inventory pinned the complete Go1 observation/action contract,
topology, feature construction, six canonical actors, PPO/checkpoint routing,
historical artifacts, tree signatures, and functional fingerprints. Its
manifest remains the authority for historical context, including legacy
operators deliberately excluded from the runtime package.

## Job 2 — standalone core parity

Status: **PASS**.

All six terminal checkpoints from training array `3343705`, seed 0, step
`412876800` loaded through `go1_core` alone on CPU. Parameter counts, actor and
full parameter-tree signatures, serialized factory kwargs, checkpoint values,
24D logits, and 12D deterministic actions matched their frozen references.

| Model | Params | Actor tree | Restore | Logits | Actions |
|---|---:|---|---|---|---|
| `native_mlp` | 192,408 | PASS | PASS | PASS | PASS |
| `mlp_match_lower` | 93,162 | PASS | PASS | PASS | PASS |
| `pointwise_match_lower` | 93,244 | PASS | PASS | PASS | PASS |
| `gcn_match_lower` | 92,987 | PASS | PASS | PASS | PASS |
| `multirank_hodge_lower` | 92,962 | PASS | PASS | PASS | PASS |
| `multirank_hodge_full` | 126,570 | PASS | PASS | PASS | PASS |

## Job 3 and Job 3A — evaluator limitation

Job 3 status: **BLOCKED** by historical GPU non-repeatability. Its exact
historical CSV criteria and `rtol=atol=1e-6` were not weakened.

**Job 3A classification: `inconclusive`. There were 0/28 Holm-corrected
rejections for the predeclared F/N > F/F differential tests, while 11/28
fields exceeded the finite observed F/F envelope. Deterministic boundaries
were exact. No claim of evaluator equivalence is made.**

Job 3A found no pre-MJX implementation mismatch and no corrected statistical
rejection, but its predeclared finite-envelope rule did not permit an
`indistinguishable` classification. Version 1.0.0 therefore makes no claim of
historical long-horizon GPU trajectory or CSV equivalence.

## Job 4 — standalone PPO training

Status: **PASS**.

A fresh seed-0 `multirank_hodge_lower` policy trained for exactly `22,937,600`
environment steps with the frozen weak-actuator curriculum. The bounded run
used 140 rollout batches and 17,920 optimizer steps on an A100 `3g.20gb` MIG.

| Gate | Certified result |
|---|---|
| Actor parameters | 92,962 |
| Evaluation reward delta | `+6.0444832254` |
| Average episode length delta | `+491.5390625` |
| Changed actor leaves | 51 |
| Parameter-tree/value restore | exact PASS |
| Logits restore | exact PASS |
| Deterministic actions restore | exact PASS |
| Restored 109-condition rollout | PASS |
| Historical learning-trajectory claim | none |

The only saved checkpoint was `000022937600`. The restored policy produced 109
ordered, finite, valid stand-command evaluation rows without comparison to the
historical campaign CSV.

## Release boundary

Version 1.0.0 owns only the frozen Go1 contract, topology dependency closure,
six model identities, PPO network construction, checkpoint reconstruction,
and deterministic inference machinery. It does not own environments, terrain,
reward definitions, curricula, evaluators, launchers, checkpoint saving, or
training loops.

The final wheel SHA-256 is recorded in `dist/SHA256SUMS`, the annotated
`v1.0.0` tag message, and the Job 5 handoff. Distribution version verification
uses `importlib.metadata.version("go1-morphology-core")`; no
`go1_core.__version__` symbol is introduced.
