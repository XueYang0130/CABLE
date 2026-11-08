"""Check reader-facing entry points and paper method defaults."""
from argparse import Namespace
from pathlib import Path
import os
import subprocess
import sys

from scripts.train import build_command
from methods import get_method

ROOT = Path(__file__).resolve().parents[1]


def test_changing_seed_selects_a_distinct_output_directory():
    args = Namespace(config=str(ROOT/'configs/cable_cw10.json'), seed=3, device=None,
                     output_dir=None, run_name=None, baseline_curves=None,
                     steps_per_task=None, smoke=False)
    command = build_command(args)
    assert command[command.index('--run-name')+1] == 'cable_cw10_seed3'
    assert command[command.index('--seed')+1] == '3'
    assert '--device' not in command


def test_reference_aggregation_runs_without_pythonpath(tmp_path):
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, str(ROOT/'scripts/aggregate_single_task_seeds.py'), '--help'],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_method_defaults_match_paper_projection_and_packnet_settings():
    assert get_method('cable').defaults['bc_max_norm_ratio'] == 1_000_000
    assert get_method('all_states').defaults['bc_max_norm_ratio'] == 1_000_000
    assert get_method('packnet').defaults['gradient_clip_norm'] == 2e-5
    assert get_method('packnet').defaults['packnet_retrain_steps'] == 100_000
