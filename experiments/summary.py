"""Gather every number the paper reports into results/paper.json, from the experiment outputs in runs/results.

    python -m experiments.summary                      # writes results/paper.json
    python -m experiments.summary --out my_runs.json   # somewhere else

Accuracy is pooled over the 93 test videos for each seed, then summarised as the mean over seeds with a 95%
confidence interval. Models are compared with a paired sign-flip test on per-video accuracy averaged over seeds.
"""
import json
import os

import numpy as np

from uavsignals.constants import CLASSES

from .common import OUTPUT, ROOT, load as load_videos, parser
from .stats import mean_ci, paired_test

MODELS = ['stacked_lstm_image', 'stacked_lstm_body', 'lstm_no_aug', 'tcn', 'stgcn', 'lstm']
PAIRS = [('lstm', 'tcn'), ('lstm', 'stgcn'), ('lstm', 'lstm_no_aug'), ('stacked_lstm_body', 'stacked_lstm_image'),
         ('lstm_no_aug', 'stacked_lstm_body')]


def load(name):
    path = os.path.join(OUTPUT, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def summarise(runs, videos):
    """Per-seed pooled video accuracy, mean window accuracy, and each video's accuracy averaged over seeds."""
    seeds = sorted({r['seed'] for r in runs})
    video, window, per_video = [], [], {}
    for s in seeds:
        rs = [r for r in runs if r['seed'] == s]
        preds = {n: p for r in rs for n, p in r['predictions'].items()}
        video.append(np.mean([preds[n] == videos[n].label for n in preds]))
        window.append(np.mean([r['window'] for r in rs]))
        for n, p in preds.items():
            per_video.setdefault(n, []).append(p == videos[n].label)
    v, vci = mean_ci(video)
    w, wci = mean_ci(window)
    return dict(seeds=len(seeds), video=v, video_ci=vci, window=w, window_ci=wci,
                per_seed_video=[float(x) for x in video]), {n: float(np.mean(c)) for n, c in per_video.items()}


def main():
    p = parser(__doc__)
    p.add_argument('--out', default=os.path.join(ROOT, 'results', 'paper.json'))
    args = p.parse_args()
    videos = load_videos(args)

    R, per_video = {'models': {}}, {}
    for name in MODELS:
        t = load(f'train_{name}.json')
        if t:
            R['models'][name], per_video[name] = summarise(t['runs'], videos)
    for name, d in (load('baselines.json') or {}).items():
        R['models'][name], per_video[name] = summarise(d['runs'], videos)
    R['seeds'] = R['models'].get('lstm', {}).get('seeds', 0)

    R['paired'] = {}
    for a, b in PAIRS:
        if a in per_video and b in per_video:
            names = sorted(per_video[a])
            R['paired'][f'{a}_vs_{b}'] = paired_test([per_video[a][n] for n in names], [per_video[b][n] for n in names])

    if 'lstm' in per_video:
        per_class = {}
        for n, c in per_video['lstm'].items():
            per_class.setdefault(CLASSES[videos[n].label], []).append(c)
        R['per_class'] = {c: dict(mean=float(np.mean(v)), videos=len(v)) for c, v in per_class.items()}

    for key, name in [('robustness', 'robustness.json'), ('viewpoint', 'viewpoint.json'), ('gate', 'safety_gate.json'),
                      ('dataset', 'dataset_stats.json'), ('uavhuman', 'uavhuman.json'), ('timing', 'timing.json'),
                      ('benchmark', 'benchmark.json')]:
        R[key] = load(name)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(R, f, indent=1)
    for k, v in R['models'].items():
        print(f'{k:20s} video {100 * v["video"]:5.1f} ± {100 * v["video_ci"]:.1f}%   ({v["seeds"]} seeds)')
    for k, v in R['paired'].items():
        print(f'{k:40s} difference {100 * v["difference"]:+.1f} points, p = {v["p"]:.3f}')
    print(f'Saved {os.path.relpath(args.out, ROOT)}')


if __name__ == '__main__':
    main()
