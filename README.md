# CABLE

PyTorch implementation of CABLE for task-incremental continual reinforcement learning on Continual World task sequences using Meta-World v3.

## Installation

Tested with Python 3.11. Use a fresh virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

CPU execution is supported. Select CUDA only with a compatible PyTorch installation. MuJoCo rendering may require a supported graphics backend; training does not require rendering.

## Train and evaluate

```bash
python scripts/train.py --config configs/cable_cw10.json --seed 1
```

Evaluation is part of training: five stochastic episodes per task every 20,000 training steps, followed by five final evaluation rounds. Each task has 500,000 training environment steps, including exploration. Evaluation rollouts are excluded from that budget.

A short execution check (not a paper experiment):

```bash
python scripts/train.py --config configs/cable_cw10.json --steps-per-task 200 --run-name smoke
```

Run seeds 1–5 separately. Configurations for every archived CW10/CW3 run and ablation are in `configs/archived/`; `results/run_index.json` maps configurations to evaluation records. CW3 sequences are named `cw3_0` through `cw3_7` internally (CW3-1 through CW3-8 in the paper).

Paper method names map to implementation IDs as follows:

| Paper method | Implementation ID |
|---|---|
| CABLE | `success_replay_best_pcgrad` |
| No projection | `success_replay_best` |
| BC-priority projection | `success_replay_best_bc_priority_pcgrad` |
| All replay states | `all_replay_best_pcgrad` |
| Final teacher | `success_replay_final_pcgrad` |
| Fine-tuning | `fine_tuning` |
| ClonEx-SAC | `clonex_sac` |
| RECALL | `recall` |
| PackNet | `packnet` |

## Recompute results without training

```bash
python scripts/aggregate_results.py
python scripts/plot_results.py
```

These commands use the included evaluation records. Aggregation reports final average performance and forgetting, with 4,000 seed bootstrap resamples and two-sided 90% percentile intervals. CW3 averages the eight sequences within each seed before bootstrapping.

Forward transfer is omitted by default while additional single-task references are being completed. The included `results/reference_seeds1_2.json` is an explicitly labelled two-seed reference, not a five-seed reference. To evaluate all runs against that same reference:

```bash
python scripts/aggregate_results.py --reference results/reference_seeds1_2.json
```

Reported normalized FT uses the mean of 25 equally spaced checkpoint success rates and includes every sequence position, including the first. The training engine also writes legacy metrics that exclude the first position; use `scripts/aggregate_results.py` for the paper reporting convention. Final FT tables will be updated after the five-seed reference is available.

## Single-task references

```bash
python scripts/train.py --config configs/single_task.json --seed 1 --run-name single_task_seed1
PYTHONPATH=src python scripts/aggregate_single_task_seeds.py \
  --batch-directories outputs/single_task/single_task_seed1 outputs/single_task/single_task_seed2 \
  --output-directory outputs/single_task/reference
```

Repeat the training command for the required seeds and supply all completed batch directories to the aggregation command.

## Implementation and reproducibility

- Actor and twin critics: separate four-hidden-layer MLPs, 256 units per layer, task-specific heads except for single-head fine-tuning and independent SAC.
- First hidden layer: LayerNorm and tanh; subsequent layers: LeakyReLU with slope 0.2.
- Task indicators select heads and are excluded from backbone inputs.
- Reward configuration: `cw10_v1`, including the local v1-compatible Stick Pull implementation in `src/envs/cw_env.py`.
- CABLE: successful-state reservoir, best observed teacher selected by evaluation success with return tie-breaking, fixed teacher-to-current Gaussian KL targets.
- Gradient coordination acts on shared actor parameters and the coefficient-weighted BC gradient; critics use current-task transitions.
- The executed development trainer was renamed from `continual_experiment_copy.py` to `continual_experiment.py`; numerical training logic is retained. Internal compatibility helpers are retained where required by this shared trainer, but unpublished research methods are not registered.
- Archived configurations explicitly specify seeds and optimizer reset behavior. CW10 RECALL seed 1 retains optimizer state; seeds 2–5 reset it. CPU is the portable default; the original recorded device is retained in the result index. Recorded runtimes are not a controlled hardware comparison.

The release does not include checkpoints, private credentials, development notes, or full training logs. Evaluation records support recomputation of aggregate metrics and learning curves; they do not replace retraining for independent validation. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Tests

```bash
PYTHONPATH=src python -m pytest -q tests
```

## License

Original project code is released under the MIT license. Third-party dependencies retain their respective licenses; see the notices file.
