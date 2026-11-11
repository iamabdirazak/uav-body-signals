"""Paths and helpers shared by the experiment scripts. Run every script from the repository root."""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS = os.path.join(ROOT, 'runs', 'checkpoints')
OUTPUT = os.path.join(ROOT, 'runs', 'results')


def checkpoint_path(name, k):
    os.makedirs(CHECKPOINTS, exist_ok=True)
    return os.path.join(CHECKPOINTS, f'{name}_fold{k}.pt')


def save_json(obj, filename):
    os.makedirs(OUTPUT, exist_ok=True)
    path = os.path.join(OUTPUT, filename)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)
    print(f'Saved {os.path.relpath(path, ROOT)}')


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--data', default=None, help='UAVGESTURE folder (default: data/UAV-GESTURE/UAVGESTURE or $UAVG_ROOT)')
    return p


def load(args):
    from uavsignals.data import DEFAULT_ROOT, load_dataset
    return load_dataset(args.data or DEFAULT_ROOT)


def mean_std(values):
    import numpy as np
    return f'{100 * np.mean(values):.1f} ± {100 * np.std(values):.1f}'
