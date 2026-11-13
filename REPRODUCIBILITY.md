# Reproducibility notes

## Included

- CABLE, four baselines, and paper ablation method presets.
- Portable configurations and evaluation records for 175 reported continual runs.
- Seed-level aggregation, pointwise bootstrap learning curves, and a fixed single-task SAC reference pooled over seeds 1--5.
- Actual per-run optimizer and training settings, with model dimensions derived by the environment.
- Source-file hashes tracing this release to the development implementation.

## Protocol

Training seeds are 1–5. Environment and replay sampling seed derivations are implemented in `src/training/sac_trainer.py`, `src/training/continual_experiment.py`, `src/envs/cw_env.py`, and the replay/memory classes. Configurations select the run seed. The released code specifies derivations rather than asserting that all random streams share the same seed.

Evaluation uses stochastic actions. Final performance uses five final evaluation rounds. Task-end performance uses the last five checkpoints during that task. Forgetting is task-end minus final success. Bootstrapping resamples seeds, not individual correlated evaluations. The first task is included in the paper FT convention.

Normalized forward transfer uses the same fixed task-specific single-task reference for every method and seed. This reference is obtained by checkpoint-wise averaging the single-task SAC success curves from seeds 1--5 and is stored in \`results/reference_seeds1_5.json\`.

## Release limitations

The archived runs used different execution devices; the result index preserves this metadata. Portable configurations default to CPU. Identical seeds do not guarantee bitwise identical results across platforms.

The development repository contains additional analyses and checkpoints that are not necessary to recompute the included performance curves. This release provides the core implementation and compact evaluation evidence, not every exploratory experiment.

Source code and evidence are available here. A persistent archival DOI and trained model downloads have not been established. No new dataset is introduced; Meta-World supplies the environments. The release does not claim new statistical hypothesis tests or a comprehensive hyperparameter search.
