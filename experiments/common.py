"""Paths and helpers shared by the experiment scripts. Run every script from the repository root."""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS = os.path.join(ROOT, 'runs', 'checkpoints')
OUTPUT = os.path.join(ROOT, 'runs', 'results')


def checkpoint_path(name, k, seed=0):
    os.makedirs(CHECKPOINTS, exist_ok=True)
    return os.path.join(CHECKPOINTS, f'{name}_fold{k}_seed{seed}.pt')


def save_json(obj, filename, quiet=False):
    os.makedirs(OUTPUT, exist_ok=True)
    path = os.path.join(OUTPUT, filename)
    with open(path + '.tmp', 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(path + '.tmp', path)
    if not quiet:
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


def trained_runs(name):
    """(seed, fold) pairs with a saved checkpoint for a configuration."""
    import glob
    import re
    runs = []
    for path in glob.glob(os.path.join(CHECKPOINTS, f'{name}_fold*_seed*.pt')):
        k, s = map(int, re.search(r'_fold(\d+)_seed(\d+)\.pt$', path).groups())
        runs.append((s, k))
    return sorted(runs)


def load_nets(name):
    """{(seed, fold): network} for every trained run of a configuration."""
    from uavsignals.training import CONFIGS, load_model
    return {(s, k): load_model(CONFIGS[name], checkpoint_path(name, k, s)) for s, k in trained_runs(name)}


def per_seed_video_accuracy(nets, cfg, videos, perturb=None):
    """Video accuracy for each seed, pooled over the test videos of all folds."""
    from uavsignals.protocol import fold
    from uavsignals.training import evaluate
    correct = {}
    for (s, k), net in nets.items():
        _, test_names = fold(videos, k)
        _, _, predictions = evaluate(net, cfg, videos, test_names, perturb)
        hits = [predictions[n] == videos[n].label for n in test_names]
        correct.setdefault(s, []).extend(hits)
    return {s: float(sum(h) / len(h)) for s, h in correct.items()}
