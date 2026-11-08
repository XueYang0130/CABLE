"""Run one archived paper configuration with explicit portable overrides."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--run-name')
    parser.add_argument('--baseline-curves')
    parser.add_argument('--steps-per-task', type=int)
    args = parser.parse_args()
    command = [sys.executable, str(ROOT / 'scripts/run.py'), '--config', args.config,
               '--device', args.device, '--output-dir', args.output_dir]
    for name in ['seed','run_name','baseline_curves','steps_per_task']:
        value = getattr(args, name)
        if value is not None:
            command += ['--'+name.replace('_','-'), str(value)]
    subprocess.run(command, cwd=ROOT, check=True)
if __name__ == '__main__':
    main()
