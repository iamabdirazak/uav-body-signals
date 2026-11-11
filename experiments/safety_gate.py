"""Streaming evaluation of the safety gate on command streams from unseen actors (Fig. 5).

Each test actor's `have command` clip is joined to each of that actor's other clips to form an armed
stream (the operator asks for attention, then signals X). Each other clip alone forms an unarmed
stream, in which nothing should be executed except `wave off`. Decisions are made every 0.2 s, with
every seed's fold models from experiments.train.

Besides fixed example policies, one gate setting is chosen for each fold using only that fold's
training actors: their streams, scored by the models of the other folds, which never saw them. The
rule is fixed in advance. Among settings with at most 0.05 wrong actions per stream, take the one that
executes the most intended commands, then the lowest latency.

The script also reports false wave-offs, a lighter vote for wave off (2 of the last 3 windows), and
which wrong commands were executed, grouped by what they would make the drone do.

    python -m experiments.safety_gate
"""
import numpy as np

from uavsignals.constants import CLASSES, FPS, HAVE_COMMAND, WAVE_OFF, WINDOW
from uavsignals.gate import run
from uavsignals.protocol import FOLDS, fold
from uavsignals.training import CONFIGS, predict

from .common import load, load_nets, parser, save_json
from .stats import mean_ci

STRIDE = 5          # 0.2 s per decision at 25 fps
GRACE = 6           # windows overlapping the onset of X may already show X
MAX_WRONG = 0.05    # selection rule: wrong actions per stream allowed

FIXED = {
    'Act on every window': dict(mode='naive'),
    'Vote 3/4, p ≥ 0.8': dict(mode='vote', k=3, n=4, tau=0.8),
    'Gate 3/4, p ≥ 0.8': dict(mode='gate', k=3, n=4, tau=0.8),
    'Gate 6/8, p ≥ 0.7': dict(mode='gate', k=6, n=8, tau=0.7),
}
GRID = [dict(mode='gate', k=k, n=n, tau=tau)
        for n in (4, 6, 8, 10) for k in range(n // 2 + 1, n + 1) for tau in (0.5, 0.6, 0.7, 0.8, 0.9)]

# What the drone would do if the command were executed by mistake.
RISK = {
    'high': ['land', 'move_downward', 'move_ahead', 'landing_direction'],   # descends, or moves toward the person
    'medium': ['move_to_left', 'move_to_right', 'move_upward'],             # moves sideways or up
    'low': ['hover', 'slow_down', 'all_clear', 'not_clear', 'wave_off'],    # holds, slows, stops or acknowledges
}
RISK_OF = {CLASSES.index(c): level for level, cs in RISK.items() for c in cs}


def build_streams(videos, nets, cfg):
    """{(seed, fold): [stream]} with window probabilities from that seed's model for that fold."""
    streams = {}
    for (s, k), net in nets.items():
        _, test_names = fold(videos, k)
        by_actor = {}
        for n in test_names:
            by_actor.setdefault(videos[n].actor, {})[videos[n].label] = n
        out = []
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
                    P = predict(net, cfg, x, m, stride=STRIDE)
                    labels = seg[np.arange(len(P)) * STRIDE + WINDOW - 1]
                    onset = int(np.argmax(labels == label)) if armed else 0
                    out.append(dict(P=P, target=label, onset=onset, armed=armed))
        streams[(s, k)] = out
    return streams


def score(executed, stream):
    """Outcome of one stream: wrong commands executed, whether the intended one was, and its latency."""
    target, onset, armed = stream['target'], stream['onset'], stream['armed']
    actions = [(t, c) for t, c in executed if c != HAVE_COMMAND]
    if armed:
        wrong = [c for t, c in actions if t < onset - GRACE or c != target]
        hits = [t for t, c in actions if t >= onset - GRACE and c == target]
    else:
        wrong = [c for t, c in actions if c != WAVE_OFF or target != WAVE_OFF]
        hits = [t for t, c in actions if c == WAVE_OFF and target == WAVE_OFF]
    latency = (hits[0] - onset) * STRIDE / FPS + WINDOW / FPS if hits else None
    return dict(wrong=wrong, hit=bool(hits), latency=latency, armed=armed, target=target)


def outcomes(streams, policy):
    return [score(run(st['P'], **policy), st) for st in streams]


def summarise(records):
    """Metrics for one seed's streams."""
    armed = [r for r in records if r['armed']]
    unarmed = [r for r in records if not r['armed']]
    aborts = [r for r in records if r['target'] == WAVE_OFF]
    latencies = [r['latency'] for r in armed if r['latency'] is not None]
    abort_lat = [r['latency'] for r in aborts if r['latency'] is not None]
    wrong = [c for r in records for c in r['wrong']]
    return dict(
        executed=float(np.mean([r['hit'] for r in armed])),
        wrong_per_stream=len(wrong) / len(records),
        wrong_per_unarmed_stream=sum(len(r['wrong']) for r in unarmed) / len(unarmed),
        median_latency_s=float(np.median(latencies)) if latencies else None,
        false_wave_off_armed=sum(r['wrong'].count(WAVE_OFF) for r in armed) / len(armed),
        false_wave_off_unarmed=sum(r['wrong'].count(WAVE_OFF) for r in unarmed) / len(unarmed),
        abort_executed=float(np.mean([r['hit'] for r in aborts])),
        abort_median_latency_s=float(np.median(abort_lat)) if abort_lat else None,
        wrong_by_risk={level: sum(RISK_OF[c] == level for c in wrong) / len(records) for level in RISK},
        wrong_commands={CLASSES[c]: wrong.count(c) for c in sorted(set(wrong))},
    )


def across_seeds(per_seed):
    """Mean and 95% interval across seeds for each scalar metric; counts are summed."""
    out = {}
    for key in per_seed[0]:
        values = [m[key] for m in per_seed]
        if key == 'wrong_by_risk':
            out[key] = {lvl: mean_ci([v[lvl] for v in values])[0] for lvl in RISK}
        elif key == 'wrong_commands':
            total = {}
            for v in values:
                for c, n in v.items():
                    total[c] = total.get(c, 0) + n
            out[key] = total
        elif all(v is not None for v in values):
            mean, ci = mean_ci(values)
            out[key] = dict(mean=mean, ci=ci)
    return out


def select(streams, seed, test_fold):
    """Gate setting chosen on the training actors of `test_fold`, by the fixed rule."""
    selection = [st for k in FOLDS if k != test_fold for st in streams[(seed, k)]]
    best = None
    for policy in GRID:
        m = summarise(outcomes(selection, policy))
        latency = m['median_latency_s'] if m['median_latency_s'] is not None else np.inf
        key = (m['wrong_per_stream'] <= MAX_WRONG, m['executed'] if m['wrong_per_stream'] <= MAX_WRONG
               else -m['wrong_per_stream'], -latency)
        if best is None or key > best[0]:
            best = (key, policy)
    return best[1]


def main():
    p = parser(__doc__)
    p.add_argument('--config', default='lstm')
    args = p.parse_args()
    videos = load(args)
    cfg = CONFIGS[args.config]
    streams = build_streams(videos, load_nets(args.config), cfg)
    seeds = sorted({s for s, _ in streams})
    n_streams = sum(len(streams[(seeds[0], k)]) for k in FOLDS)
    print(f'{len(seeds)} seeds, {n_streams} streams per seed\n')

    results, chosen = {}, {}
    policies = dict(FIXED)
    policies['Gate 6/8, p ≥ 0.7, wave off 2/3'] = dict(mode='gate', k=6, n=8, tau=0.7, wave_off=(2, 3))
    for label, policy in policies.items():
        per_seed = [summarise([r for k in FOLDS for r in outcomes(streams[(s, k)], policy)]) for s in seeds]
        results[label] = across_seeds(per_seed)

    # held-out selection: a separate choice for every (seed, test fold)
    for with_light_abort in (False, True):
        per_seed = []
        for s in seeds:
            records = []
            for k in FOLDS:
                policy = dict(select(streams, s, k))
                if with_light_abort:
                    policy['wave_off'] = (2, 3)
                chosen.setdefault(str(with_light_abort), []).append(dict(seed=s, fold=k, **policy))
                records += outcomes(streams[(s, k)], policy)
            per_seed.append(summarise(records))
        label = 'Gate selected on training actors' + (', wave off 2/3' if with_light_abort else '')
        results[label] = across_seeds(per_seed)

    for label, r in results.items():
        print(f'{label:38s} executed {100 * r["executed"]["mean"]:5.1f}%  wrong/stream '
              f'{r["wrong_per_stream"]["mean"]:.3f}  latency {r["median_latency_s"]["mean"]:.1f} s  '
              f'false wave-off {r["false_wave_off_armed"]["mean"]:.3f}/{r["false_wave_off_unarmed"]["mean"]:.3f}  '
              f'abort {100 * r["abort_executed"]["mean"]:.0f}% in {r["abort_median_latency_s"]["mean"]:.1f} s')
    save_json(dict(policies=results, selected_settings=chosen, risk_groups=RISK, streams_per_seed=n_streams),
              'safety_gate.json')


if __name__ == '__main__':
    main()
