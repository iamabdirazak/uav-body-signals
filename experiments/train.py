"""Train and evaluate every configuration on the three actor-disjoint folds, with several seeds (Table I).

Each (configuration, fold, seed) run saves a checkpoint and its per-video predictions. Finished runs
are skipped, so an interrupted job can simply be started again.

    python -m experiments.train                          # all configurations, seeds 0-4
    python -m experiments.train lstm tcn --seeds 0 1 2   # selected configurations and seeds
"""
import json
import os
import time

import torch

from uavsignals.protocol import FOLDS, fold
from uavsignals.training import CONFIGS, evaluate, train

from .common import OUTPUT, checkpoint_path, load, parser, save_json

SEEDS = [0, 1, 2, 3, 4]


def results_file(name):
    return os.path.join(OUTPUT, f'train_{name}.json')


def main():
    p = parser(__doc__)
    p.add_argument('configs', nargs='*', help=f'any of: {", ".join(CONFIGS)} (default: all)')
    p.add_argument('--seeds', nargs='*', type=int, default=SEEDS)
    p.add_argument('--folds', nargs='*', type=int, default=list(FOLDS))
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    names = args.configs or list(CONFIGS)
    unknown = set(names) - set(CONFIGS)
    if unknown:
        p.error(f'unknown configuration(s): {", ".join(sorted(unknown))}')
    videos = load(args)

    for name in names:
        cfg = CONFIGS[name]
        path = results_file(name)
        runs = json.load(open(path))['runs'] if os.path.exists(path) else []
        done = {(r['seed'], r['fold']) for r in runs}
        for seed in args.seeds:
            for k in args.folds:
                if (seed, k) in done:
                    continue
                start = time.time()
                train_names, test_names = fold(videos, k)
                net = train(cfg, videos, train_names, seed=seed, threads=args.threads)
                torch.save(net.state_dict(), checkpoint_path(name, k, seed))
                window_acc, video_acc, predictions = evaluate(net, cfg, videos, test_names)
                runs.append(dict(seed=seed, fold=k, window=window_acc, video=video_acc, predictions=predictions))
                save_json(dict(config=name, runs=runs), os.path.basename(path), quiet=True)
                print(f'{name} seed {seed} fold {k}: window {100 * window_acc:.1f}%  video {100 * video_acc:.1f}%'
                      f'  ({time.time() - start:.0f} s)', flush=True)


if __name__ == '__main__':
    main()
