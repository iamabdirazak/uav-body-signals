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

from .common import load, mean_std, parser, save_json


def windows(video, stride):
    """(num_windows, 30, 28) body-centred positions, no velocities."""
    f = to_features(normalise(video.keypoints, video.visible, 'body'), velocity=False)
    if len(f) < WINDOW:
        f = np.concatenate([f, np.repeat(f[-1:], WINDOW - len(f), 0)])
    return np.stack([f[i:i + WINDOW] for i in range(0, len(f) - WINDOW + 1, stride)])


def classifier(name):
    if name == 'frame_mlp':
        return MLPClassifier((128, 64), max_iter=400, random_state=0)
    return RandomForestClassifier(300, n_jobs=-1, random_state=0)


def main():
    args = parser(__doc__).parse_args()
    videos = load(args)
    results = {}
    for name in ['frame_rf', 'frame_mlp', 'window_rf']:
        window_acc, video_acc = [], []
        for k in FOLDS:
            train_names, test_names = fold(videos, k)
            X = [windows(videos[n], 5) for n in train_names]
            y = np.concatenate([np.full(len(w), videos[n].label) for n, w in zip(train_names, X)])
            X = np.concatenate(X)
            clf = classifier(name)
            if name.startswith('frame'):
                clf.fit(X[:, WINDOW // 2], y)            # one frame per training window
            else:
                clf.fit(X.reshape(len(X), -1), y)
            correct_w = total_w = correct_v = 0
            for n in test_names:
                W = windows(videos[n], 15)
                if name.startswith('frame'):
                    P = clf.predict_proba(W.reshape(-1, W.shape[-1])).reshape(len(W), WINDOW, -1).mean(1)
                else:
                    P = clf.predict_proba(W.reshape(len(W), -1))
                label = videos[n].label
                correct_w += (P.argmax(1) == label).sum()
                total_w += len(P)
                correct_v += P.mean(0).argmax() == label
            window_acc.append(float(correct_w / total_w))
            video_acc.append(float(correct_v / len(test_names)))
        results[name] = dict(window=window_acc, video=video_acc)
        print(f'{name:10s} window {mean_std(window_acc)}  video {mean_std(video_acc)}', flush=True)
    save_json(results, 'baselines.json')


if __name__ == '__main__':
    main()
