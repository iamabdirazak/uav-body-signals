"""Viewpoint test with camera models that training never saw.

Each test clip is viewed from a camera raised to a given elevation, in three ways:

  squash       the first-order vertical foreshortening used in training (augment.elevation)
  perspective  the skeleton kept flat, lifted to 3D and seen through a pinhole camera 10 m away
  limb depth   as perspective, but each limb segment also gets the depth that restores its length,
               with the arms toward the camera (uavsignals.viewpoint)

    python -m experiments.viewpoint
"""
from uavsignals.augment import perturbation
from uavsignals.training import CONFIGS
from uavsignals.viewpoint import drone_view

from .common import load, load_nets, parser, per_seed_video_accuracy, save_json
from .stats import mean_ci

MODELS = ['stacked_lstm_body', 'lstm_no_aug', 'tcn', 'stgcn', 'lstm']
ELEVATIONS = [0, 15, 30, 45, 60, 70]
RANGE_M = 10
VIEWS = {
    'squash': lambda e: perturbation('elevation', e),
    'perspective': lambda e: drone_view(RANGE_M, e, limb_depth=False),
    'limb_depth': lambda e: drone_view(RANGE_M, e, limb_depth=True),
}


def main():
    p = parser(__doc__)
    p.add_argument('configs', nargs='*', default=MODELS)
    p.add_argument('--views', nargs='*', default=list(VIEWS), choices=list(VIEWS))
    args = p.parse_args()
    videos = load(args)
    results = {'range_m': RANGE_M, 'elevations': ELEVATIONS}
    for name in args.configs:
        cfg = CONFIGS[name]
        nets = load_nets(name)
        results[name] = {}
        for view in args.views:
            results[name][view] = {}
            for e in ELEVATIONS:
                accs = per_seed_video_accuracy(nets, cfg, videos, VIEWS[view](e))
                mean, ci = mean_ci(list(accs.values()))
                results[name][view][str(float(e))] = dict(video=mean, ci=ci, per_seed=accs)
                print(f'{name:18s} {view:12s} {e:2d}°   video {100 * mean:5.1f} ± {100 * ci:.1f}%', flush=True)
            save_json(results, 'viewpoint.json', quiet=True)
    save_json(results, 'viewpoint.json')


if __name__ == '__main__':
    main()
