"""Accuracy under simulated deployment conditions, applied to test keypoints without retraining (Fig. 4).

Needs the fold checkpoints from experiments.train.

    python -m experiments.robustness                          # proposed model and both baselines
    python -m experiments.robustness lstm --conditions elevation rotation
"""
import numpy as np

from uavsignals.augment import perturbation
from uavsignals.protocol import FOLDS, fold
from uavsignals.training import CONFIGS, evaluate, load_model

from .common import checkpoint_path, load, parser, save_json

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
    p.add_argument('configs', nargs='*', default=['stacked_lstm_body', 'lstm_no_aug', 'lstm'])
    p.add_argument('--conditions', nargs='*', default=list(CONDITIONS), choices=list(CONDITIONS))
    args = p.parse_args()
    videos = load(args)

    results = {}
    for name in args.configs:
        cfg = CONFIGS[name]
        nets = {k: load_model(cfg, checkpoint_path(name, k)) for k in FOLDS}
        results[name] = {}
        for kind in args.conditions:
            results[name][kind] = {}
            for value in CONDITIONS[kind]:
                window, video = [], []
                for k in FOLDS:
                    _, test_names = fold(videos, k)
                    w, v, _ = evaluate(nets[k], cfg, videos, test_names, perturbation(kind, value))
                    window.append(w)
                    video.append(v)
                results[name][kind][str(float(value))] = dict(window=float(np.mean(window)), video=float(np.mean(video)))
                print(f'{name:18s} {kind:13s} {value:>5}   video {100 * np.mean(video):5.1f}%', flush=True)
        save_json(results, 'robustness.json')


if __name__ == '__main__':
    main()
