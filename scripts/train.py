"""Train and evaluate one experiment configuration."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def build_command(args):
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open() as stream:
        config = json.load(stream) if config_path.suffix == '.json' else yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a mapping.')
    seed = args.seed if args.seed is not None else config.get('seed', 1)
    run_name = args.run_name or config.get('run_name')
    if args.run_name is None and run_name:
        run_name = re.sub(r'_seed\d+(?=_|$)', f'_seed{seed}', run_name)
        if args.seed is not None and not re.search(r'_seed\d+(?:_|$)', run_name):
            run_name += f'_seed{seed}'
    if args.smoke:
        run_name = (run_name or config.get('method', 'experiment')) + '_smoke'
    command = [sys.executable, str(ROOT / 'scripts/run.py'), '--config', str(config_path)]
    overrides = {
        'seed': seed,
        'device': args.device,
        'output_dir': str(Path(args.output_dir).expanduser().resolve()) if args.output_dir else None,
        'run_name': run_name,
        'baseline_curves': str(Path(args.baseline_curves).expanduser().resolve()) if args.baseline_curves else None,
        'steps_per_task': args.steps_per_task,
    }
    if args.smoke:
        overrides.update(steps_per_task=40, total_steps=40, max_episode_steps=10,
                         eval_every=40, stoch_eval_episodes=1, det_eval_episodes=0,
                         start_steps=8, update_after=8, update_every=8,
                         batch_size=8, episodic_batch_size=8, episodic_memory_per_task=16,
                         packnet_retrain_steps=2)
        if config.get('mode') == 'single-batch':
            overrides['num_tasks'] = 2
        elif config.get('mode') == 'continual' and config.get('task_sequence', 'cw10') == 'cw10':
            overrides['sequence_task_count'] = 2
        command.append('--no-gradient-diagnostics')
    for name, value in overrides.items():
        if value is not None:
            command += ['--' + name.replace('_', '-'), str(value)]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--output-dir')
    parser.add_argument('--run-name')
    parser.add_argument('--baseline-curves')
    parser.add_argument('--steps-per-task', type=int)
    parser.add_argument('--smoke', action='store_true', help='Run a short installation check, not a paper experiment.')
    args = parser.parse_args()
    try:
        command = build_command(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    result = subprocess.run(command, cwd=ROOT)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
