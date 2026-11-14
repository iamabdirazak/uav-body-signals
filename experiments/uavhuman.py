"""Cross-dataset evaluation on UAV-Human: real aerial footage of other people.

  signals    the 13 UAV-GESTURE signals as performed in UAV-Human, classified by the models trained on
             UAV-GESTURE. Every seed's three fold models are used, and accuracy is averaged per seed.
  everyday   UAV-Human actions that are not UAV-control gestures (walking, waving, calling for help and
             so on), streamed through the models and the safety gate. Any executed command is false. Up to
             --per-action clips of every action are drawn at random with a fixed seed, so every action
             counts equally.

    python -m experiments.uavhuman --archive data/uavhuman_skeletons.npz
"""
import json
import os

import numpy as np
import torch

from uavsignals.constants import CLASSES, FPS, HAVE_COMMAND, WINDOW
from uavsignals.features import pad_to_window
from uavsignals.gate import SafetyGate, run
from uavsignals.training import CONFIGS
from uavsignals.uavhuman import UAV_CONTROL, load_uavhuman

from .common import OUTPUT, load_nets, parser, save_json
from .stats import mean_ci

STRIDE = 5


def windows(clip, cfg, stride):
    x, m = pad_to_window(clip['keypoints'], clip['visible'])
    return torch.tensor(np.stack([cfg.features(x[i:i + WINDOW], m[i:i + WINDOW])
                                  for i in range(0, len(x) - WINDOW + 1, stride)]))


@torch.no_grad()
def probabilities(nets, X):
    return {key: torch.softmax(net(X), -1).numpy() for key, net in nets.items()}


def main():
    p = parser(__doc__)
    p.add_argument('--archive', default=os.path.join('data', 'uavhuman_skeletons.npz'))
    p.add_argument('--config', default='lstm')
    p.add_argument('--per-action', type=int, default=30)
    args = p.parse_args()
    cfg = CONFIGS[args.config]
    nets = load_nets(args.config)
    seeds = sorted({s for s, _ in nets})
    clips = load_uavhuman(args.archive)
    with open(os.path.join(OUTPUT, 'safety_gate.json')) as f:
        chosen = {(c['seed'], c['fold']): {k: c[k] for k in ('mode', 'k', 'n', 'tau')}
                  for c in json.load(f)['selected_settings']['False']}

    # --- signals: clip-level accuracy, per seed (mean over that seed's three fold models)
    signal_clips = {n: c for n, c in clips.items() if c['signal'] is not None}
    correct = {s: [] for s in seeds}
    per_signal = {s: {} for s in seeds}
    confusion = np.zeros((len(CLASSES), len(CLASSES)))
    for c in signal_clips.values():
        X = windows(c, cfg, 15)
        for (s, k), P in probabilities(nets, X).items():
            pred = int(P.mean(0).argmax())
            hit = pred == c['signal']
            correct[s].append(hit)
            per_signal[s].setdefault(CLASSES[c['signal']], []).append(hit)
            confusion[c['signal'], pred] += 1
    acc = [np.mean(correct[s]) for s in seeds]
    signals = dict(clips=len(signal_clips), people=len({c['person'] for c in signal_clips.values()}),
                   accuracy=dict(zip(['mean', 'ci'], mean_ci(acc))),
                   per_signal={c: float(np.mean([np.mean(per_signal[s][c]) for s in seeds])) for c in per_signal[seeds[0]]},
                   confusion=confusion.tolist())
    print(f'Signals: {signals["clips"]} clips from {signals["people"]} people, accuracy '
          f'{100 * signals["accuracy"]["mean"]:.1f} ± {100 * signals["accuracy"]["ci"]:.1f}% (chance 7.7%)')
    for c, a in sorted(signals['per_signal'].items(), key=lambda kv: -kv[1]):
        print(f'   {c:18s} {100 * a:5.1f}%')

    # --- everyday motion: false commands per minute, naive vs the selected gate
    rng = np.random.default_rng(0)
    by_action = {}
    for n, c in sorted(clips.items()):
        if c['action'] not in UAV_CONTROL:
            by_action.setdefault(c['action'], []).append(n)
    everyday = {n: clips[n] for names in by_action.values()
                for n in rng.permutation(names)[:args.per_action]}
    minutes = sum(len(c['keypoints']) for c in everyday.values()) / FPS / 60
    counts = {s: dict(naive=0, gate=0, gate_armed=0) for s in seeds}
    for c in everyday.values():
        X = windows(c, cfg, STRIDE)
        for (s, k), P in probabilities(nets, X).items():
            counts[s]['naive'] += sum(1 for _, cls in run(P, mode='naive') if cls != HAVE_COMMAND) / 3
            executed = run(P, **chosen[(s, k)])
            counts[s]['gate'] += sum(1 for _, cls in executed if cls != HAVE_COMMAND) / 3
            gate = SafetyGate(**chosen[(s, k)])
            events = [gate.step(p) for p in P]
            counts[s]['gate_armed'] += any(e and e[0] == 'armed' for e in events) / 3
    rate = lambda key: mean_ci([counts[s][key] / minutes for s in seeds])    # noqa: E731
    clips_armed = mean_ci([counts[s]['gate_armed'] / len(everyday) for s in seeds])
    everyday_result = dict(clips=len(everyday), minutes=minutes, actions=len({c['action'] for c in everyday.values()}),
                           naive_per_minute=dict(zip(['mean', 'ci'], rate('naive'))),
                           gate_per_minute=dict(zip(['mean', 'ci'], rate('gate'))),
                           clips_arming_gate=dict(zip(['mean', 'ci'], clips_armed)))
    print(f'Everyday motion: {len(everyday)} clips, {everyday_result["actions"]} actions, {minutes:.0f} min. '
          f'False commands per minute: every window {everyday_result["naive_per_minute"]["mean"]:.2f}, '
          f'gate {everyday_result["gate_per_minute"]["mean"]:.3f}. '
          f'Clips that arm the gate: {100 * clips_armed[0]:.1f}%')
    save_json(dict(signals=signals, everyday=everyday_result), 'uavhuman.json')


if __name__ == '__main__':
    main()
