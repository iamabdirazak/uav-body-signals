"""Streaming evaluation of the safety gate on command streams from unseen actors (Fig. 5).

Each test actor's `have command` clip is joined to each of that actor's other clips to form an
armed stream (the operator asks for attention, then signals X). Each other clip alone forms an
unarmed stream, in which nothing should be executed except `wave off`. Decisions every 0.2 s.

Needs the fold checkpoints from experiments.train.

    python -m experiments.safety_gate
"""
import numpy as np

from uavsignals.constants import FPS, HAVE_COMMAND, WAVE_OFF, WINDOW
from uavsignals.gate import run
from uavsignals.protocol import FOLDS, fold
from uavsignals.training import CONFIGS, load_model, predict

from .common import checkpoint_path, load, parser, save_json

STRIDE = 5   # 0.2 s per decision at 25 fps

POLICIES = {
    'Act on every window': dict(mode='naive'),
    'Vote 3/4, p >= 0.8': dict(mode='vote', k=3, n=4, tau=0.8),
    'Gate 3/4, p >= 0.8': dict(mode='gate', k=3, n=4, tau=0.8),
    'Gate 6/8, p >= 0.7': dict(mode='gate', k=6, n=8, tau=0.7),
}


def build_streams(videos, nets, cfg):
    """[(probabilities, intended class, onset step, armed?)] for every stream in every fold."""
    streams = []
    for k in FOLDS:
        _, test_names = fold(videos, k)
        by_actor = {}
        for n in test_names:
            by_actor.setdefault(videos[n].actor, {})[videos[n].label] = n
        for clips in by_actor.values():
            if HAVE_COMMAND not in clips:
                continue
            for label, clip in clips.items():
                if label == HAVE_COMMAND:
                    continue
                for armed in (True, False):
                    parts = [clips[HAVE_COMMAND], clip] if armed else [clip]
                    x = np.concatenate([videos[n].keypoints for n in parts])
                    m = np.concatenate([videos[n].visible for n in parts])
                    seg = np.concatenate([np.full(len(videos[n].keypoints), videos[n].label) for n in parts])
                    P = predict(nets[k], cfg, x, m, stride=STRIDE)
                    window_end_labels = seg[np.arange(len(P)) * STRIDE + WINDOW - 1]
                    onset = int(np.argmax(window_end_labels == label)) if armed else 0
                    streams.append((P, label, onset, armed))
    return streams


def score(executed, target, onset, armed):
    """(wrong actions, intended command executed?, latency in seconds) for one stream."""
    actions = [(t, c) for t, c in executed if c != HAVE_COMMAND]
    if not armed:
        return sum(1 for _, c in actions if c != WAVE_OFF or target != WAVE_OFF), 0, None
    grace = 6   # windows overlapping the onset may already show the new signal
    early = [a for a in actions if a[0] < onset - grace]
    late = [a for a in actions if a[0] >= onset - grace]
    wrong = len(early) + sum(1 for _, c in late if c != target)
    hits = [t for t, c in late if c == target]
    latency = (hits[0] - onset) * STRIDE / FPS + WINDOW / FPS if hits else None
    return wrong, int(bool(hits)), latency


def main():
    p = parser(__doc__)
    p.add_argument('--config', default='lstm')
    args = p.parse_args()
    videos = load(args)
    cfg = CONFIGS[args.config]
    nets = {k: load_model(cfg, checkpoint_path(args.config, k)) for k in FOLDS}
    streams = build_streams(videos, nets, cfg)
    n_armed = sum(s[3] for s in streams)
    n_unarmed = len(streams) - n_armed
    print(f'{len(streams)} streams ({n_armed} armed, {n_unarmed} unarmed)\n')

    results = {}
    for label, policy in POLICIES.items():
        wrong = wrong_unarmed = executed = 0
        latencies = []
        for P, target, onset, armed in streams:
            w, hit, latency = score(run(P, **policy), target, onset, armed)
            wrong += w
            if armed:
                executed += hit
                if latency is not None:
                    latencies.append(latency)
            else:
                wrong_unarmed += w
        results[label] = dict(executed=executed / n_armed, wrong_per_stream=wrong / len(streams),
                              wrong_per_unarmed_stream=wrong_unarmed / n_unarmed,
                              median_latency_s=float(np.median(latencies)), streams=len(streams))
        r = results[label]
        print(f'{label:22s} executed {100 * r["executed"]:5.1f}%   wrong/stream {r["wrong_per_stream"]:.3f}   '
              f'latency {r["median_latency_s"]:.1f} s')
    save_json(results, 'safety_gate.json')


if __name__ == '__main__':
    main()
