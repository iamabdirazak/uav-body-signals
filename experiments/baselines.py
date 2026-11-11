"""Non-temporal baselines (Table I, top rows).

  frame_rf    random forest on single frames; a window's prediction is the mean of its frames
  frame_mlp   the same with a 128-64 multilayer perceptron
  window_rf   random forest on the flattened 30-frame window

    python -m experiments.baselines
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from uavsignals.constants import WINDOW
from uavsignals.features import normalise, to_features
from uavsignals.protocol import FOLDS, fold

from .common import load, parser, save_json


def windows(video, stride):
    """(num_windows, 30, 28) body-centred positions, no velocities."""
    f = to_features(normalise(video.keypoints, video.visible, 'body'), velocity=False)
    if len(f) < WINDOW:
        f = np.concatenate([f, np.repeat(f[-1:], WINDOW - len(f), 0)])
    return np.stack([f[i:i + WINDOW] for i in range(0, len(f) - WINDOW + 1, stride)])


def classifier(name, seed):
    if name == 'frame_mlp':
        return MLPClassifier((128, 64), max_iter=400, random_state=seed)
    return RandomForestClassifier(300, n_jobs=-1, random_state=seed)


def main():
    p = parser(__doc__)
    p.add_argument('--seeds', nargs='*', type=int, default=[0, 1, 2, 3, 4])
    args = p.parse_args()
    videos = load(args)
    results = {}
    for name in ['frame_rf', 'frame_mlp', 'window_rf']:
        runs = []
        for seed, k in [(s, k) for s in args.seeds for k in FOLDS]:
            train_names, test_names = fold(videos, k)
            X = [windows(videos[n], 5) for n in train_names]
            y = np.concatenate([np.full(len(w), videos[n].label) for n, w in zip(train_names, X)])
            X = np.concatenate(X)
            clf = classifier(name, seed)
            if name.startswith('frame'):
                clf.fit(X[:, WINDOW // 2], y)            # one frame per training window
            else:
                clf.fit(X.reshape(len(X), -1), y)
            correct_w = total_w = 0
            predictions = {}
            for n in test_names:
                W = windows(videos[n], 15)
                if name.startswith('frame'):
                    P = clf.predict_proba(W.reshape(-1, W.shape[-1])).reshape(len(W), WINDOW, -1).mean(1)
                else:
                    P = clf.predict_proba(W.reshape(len(W), -1))
                label = videos[n].label
                correct_w += (P.argmax(1) == label).sum()
                total_w += len(P)
                predictions[n] = int(P.mean(0).argmax())
            video = sum(predictions[n] == videos[n].label for n in test_names) / len(test_names)
            runs.append(dict(seed=seed, fold=k, window=float(correct_w / total_w), video=float(video),
                             predictions=predictions))
            print(f'{name:10s} seed {seed} fold {k}: video {100 * video:.1f}%', flush=True)
        results[name] = dict(runs=runs)
        save_json(results, 'baselines.json', quiet=True)
    save_json(results, 'baselines.json')


if __name__ == '__main__':
    main()
