"""Train and evaluate models on the three actor-disjoint folds (Table I, Fig. 3).

    python -m experiments.train                 # all six configurations
    python -m experiments.train lstm tcn        # selected configurations
    python -m experiments.train lstm --folds 2  # one fold
"""
import time

import numpy as np
import torch

from uavsignals.protocol import FOLDS, fold
from uavsignals.training import CONFIGS, evaluate, train

from .common import checkpoint_path, load, mean_std, parser, save_json


def main():
    p = parser(__doc__)
    p.add_argument('configs', nargs='*', help=f'any of: {", ".join(CONFIGS)} (default: all)')
    p.add_argument('--folds', nargs='*', type=int, default=list(FOLDS))
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    names = args.configs or list(CONFIGS)
    unknown = set(names) - set(CONFIGS)
    if unknown:
        p.error(f'unknown configuration(s): {", ".join(sorted(unknown))}')
    videos = load(args)

    results = {}
    for name in names:
        cfg = CONFIGS[name]
        r = dict(window=[], video=[], confusion=np.zeros((13, 13), int))
        start = time.time()
        for k in args.folds:
            train_names, test_names = fold(videos, k)
            net = train(cfg, videos, train_names, epochs=cfg.epochs[k - 1], threads=args.threads)
            torch.save(net.state_dict(), checkpoint_path(name, k))
            window_acc, video_acc, confusion = evaluate(net, cfg, videos, test_names)
            r['window'].append(float(window_acc))
            r['video'].append(float(video_acc))
            r['confusion'] += confusion
            print(f'  {name} fold {k}: window {100 * window_acc:.1f}%  video {100 * video_acc:.1f}%', flush=True)
        print(f'{name}: window {mean_std(r["window"])}  video {mean_std(r["video"])}  ({time.time() - start:.0f} s)', flush=True)
        r['confusion'] = r['confusion'].tolist()
        results[name] = r
        save_json(results, f'train_{"_".join(names)}.json')


if __name__ == '__main__':
    main()
