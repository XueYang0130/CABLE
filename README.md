# CABLE

PyTorch implementation of CABLE for continual reinforcement learning on **CW10** and **CW3**.

CABLE combines successful-state memory, best-teacher relabeling, and SAC-priority gradient projection to coordinate new-task SAC learning with old-task BC regularization.

## Installation

Requires Python 3.11.

```bash
git clone https://github.com/XueYang0130/CABLE.git
cd CABLE
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
python scripts/train.py --config configs/cable_cw10.json --seed 1 --device cpu
```

Evaluation runs automatically during training. Outputs are saved to `outputs/`; changing `--seed` selects a separate run directory. Add `--smoke` for a short installation check.

| Method | Configuration |
|---|---|
| CABLE | `configs/cable_cw10.json` |
| Fine-tuning | `configs/fine_tuning_cw10.json` |
| ClonEx-SAC | `configs/clonex_sac_cw10.json` |
| RECALL | `configs/recall_cw10.json` |
| PackNet | `configs/packnet_cw10.json` |

CW3 and per-run configurations are in [`configs/archived/`](configs/archived/). Ablation presets are in `configs/`.

## Results

Recompute average performance, forgetting, and learning curves from the included evaluation records:

```bash
python scripts/aggregate_results.py
python scripts/plot_results.py
```

Results are written to `results/recomputed_metrics.json` and `results/figures/`. Run-to-configuration mappings are listed in [`results/run_index.json`](results/run_index.json).

## Repository Structure

```text
src/        Agents, environments, training, and evaluation
methods/    CABLE, baselines, and ablations
configs/    Experiment configurations
scripts/    Training, aggregation, and plotting
results/    Evaluation records and generated results
tests/      Implementation tests
```

## Tests

```bash
PYTHONPATH=src python -m pytest -q tests
```

## Documentation and License

- [Reproducibility](REPRODUCIBILITY.md)
- [Baseline sources and third-party notices](THIRD_PARTY_NOTICES.md)
- [Citation](CITATION.cff)
- [MIT License](LICENSE)
