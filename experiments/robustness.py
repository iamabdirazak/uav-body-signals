"""Accuracy under simulated deployment conditions, applied to test keypoints without retraining.

Uses every seed's checkpoints from experiments.train and reports the mean over seeds with a 95%
confidence interval.

    python -m experiments.robustness
    python -m experiments.robustness lstm --conditions elevation rotation
"""
from uavsignals.augment import perturbation
from uavsignals.training import CONFIGS

from .common import load, load_nets, parser, per_seed_video_accuracy, save_json
from .stats import mean_ci

MODELS = ['stacked_lstm_body', 'lstm_no_aug', 'tcn', 'stgcn', 'lstm']
CONDITIONS = {
    'elevation': [0, 20, 40, 60, 70],          # camera pitch, degrees
    'rotation': [0, 10, 20, 30],               # camera roll, degrees
    'dropout': [0, .1, .2, .3, .5, .7],        # fraction of joints removed at random
    'arm_occlusion': [0, 1],                   # probability of hiding one arm for 10-30 frames
    'jitter': [0, .05, .1, .2, .3],            # keypoint noise, torso lengths
    'fps': [25, 12.5, 8.33, 5, 3],             # pose estimator rate
}


def main():
    p = parser(__doc__)
    p.add_argument('configs', nargs='*', default=MODELS)
    p.add_argument('--conditions', nargs='*', default=list(CONDITIONS), choices=list(CONDITIONS))
    args = p.parse_args()
    videos = load(args)
    results = {}
    for name in args.configs:
        cfg = CONFIGS[name]
        nets = load_nets(name)
        results[name] = {}
        for kind in args.conditions:
            results[name][kind] = {}
            for value in CONDITIONS[kind]:
                accs = per_seed_video_accuracy(nets, cfg, videos, perturbation(kind, value))
                mean, ci = mean_ci(list(accs.values()))
                results[name][kind][str(float(value))] = dict(video=mean, ci=ci, per_seed=accs)
                print(f'{name:18s} {kind:13s} {value:>5}   video {100 * mean:5.1f} ± {100 * ci:.1f}%', flush=True)
            save_json(results, 'robustness.json', quiet=True)
    save_json(results, 'robustness.json')


if __name__ == '__main__':
    main()
