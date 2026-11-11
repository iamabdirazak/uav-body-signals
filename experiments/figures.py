"""Draw the README figures in assets/ from results.

    python -m experiments.figures                 # from the paper's numbers (results/paper.json)
    python -m experiments.figures --from-runs     # from your own runs (runs/results/*.json)
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from uavsignals import augment as A
from uavsignals.constants import EDGES

from .common import OUTPUT, ROOT
from .style import (ACCENT, ACCENT_SOFT, G1, G2, G3, GRID, INK, INK2, PANEL, card, rounded_bar, x_grid)

ASSETS = os.path.join(ROOT, 'assets')
WIDE = 7.16                # inches; every figure shares this width so they align at full width
RGB_FLOW_BASELINE = 91.9   # P-CNN, Perera et al. 2018, official splits


def load_results(from_runs):
    if not from_runs:
        with open(os.path.join(ROOT, 'results', 'paper.json')) as f:
            return json.load(f)
    merged = {'train': {}}
    for path in glob.glob(os.path.join(OUTPUT, '*.json')):
        with open(path) as f:
            data = json.load(f)
        key = os.path.basename(path)[:-5]
        if key.startswith('train'):
            merged['train'].update(data)
        else:
            merged[key] = data
    return merged


def video_pct(entry):
    return 100 * (np.mean(entry['video']) if 'video' in entry else entry['video_mean'])


def reference_pose():
    """One body-centred frame of `move to left`, y pointing up."""
    with open(os.path.join(ROOT, 'demo', 'reference_signals.json')) as f:
        seq = json.load(f)['move_to_left']
    return np.array(seq[len(seq) // 3], dtype=np.float32)


def draw_skeleton(ax, k, cx, cy, sc, col=INK, lw=1.0, ms=3, z=4, hidden=()):
    for a, b in EDGES:
        if a in hidden or b in hidden:
            continue
        ax.plot([cx + sc * k[a, 0], cx + sc * k[b, 0]], [cy + sc * k[a, 1], cy + sc * k[b, 1]],
                color=col, lw=lw, solid_capstyle='round', zorder=z)
    shown = [j for j in range(len(k)) if j not in hidden]
    ax.scatter(cx + sc * k[shown, 0], cy + sc * k[shown, 1], s=ms, color=col, zorder=z + 1, linewidths=0)


def save(fig, name):
    os.makedirs(ASSETS, exist_ok=True)
    fig.savefig(os.path.join(ASSETS, f'{name}.png'))
    plt.close(fig)
    print(f'Saved assets/{name}.png')


def pipeline(K):
    fig = plt.figure(figsize=(WIDE, 1.62))
    card(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 71.6)
    ax.set_ylim(0, 16.2)
    ax.axis('off')
    W, H, Y = 11.8, 12.4, 1.9
    xs = [2.4 + i * 13.45 for i in range(5)]
    steps = [('Camera stream', '25 fps RGB video'), ('Pose estimation', '14 body keypoints'),
             ('Body-centred features', 'position + velocity'), ('Temporal model', 'LSTM over 1.2 s'),
             ('Safety gate', 'vote + confidence')]
    for i, (x, (title, sub)) in enumerate(zip(xs, steps)):
        ax.add_patch(FancyBboxPatch((x, Y), W, H, boxstyle='round,pad=0,rounding_size=1.2', fc=PANEL, ec='none', zorder=2))
        ax.text(x + W / 2, Y + 2.75, title, ha='center', fontsize=6.9, fontweight='semibold', zorder=3)
        ax.text(x + W / 2, Y + 1.15, sub, ha='center', fontsize=6.0, color=INK2, zorder=3)
        if i < 4:
            ax.add_patch(FancyArrowPatch((x + W + 0.25, Y + 7.6), (xs[i + 1] - 0.25, Y + 7.6),
                                         arrowstyle='-|>,head_length=2.6,head_width=1.7', color=G2, lw=0.8, zorder=1))
    gy = Y + 7.7
    for j in range(3):   # stacked video frames
        ax.add_patch(FancyBboxPatch((xs[0] + 2.6 + j * 0.9, gy - 2.4 + j * 0.7), 4.6, 3.2,
                                    boxstyle='round,pad=0,rounding_size=0.5',
                                    fc=['#3A3A3C', '#48484A', '#5A5A5E'][j], ec=PANEL, lw=0.8, zorder=3 + j))
    draw_skeleton(ax, K, xs[0] + 6.25, gy + 0.3, 0.42, col=G3, lw=0.5, ms=0.6, z=7)
    draw_skeleton(ax, K, xs[1] + W / 2, gy + 0.9, 0.95, col=INK, lw=0.9, ms=2.2)
    cx = xs[2] + W / 2
    ax.plot([cx - 3.6, cx + 3.6], [gy + 0.2] * 2, color=GRID, lw=0.8, zorder=3)
    ax.plot([cx, cx], [gy - 2.9, gy + 2.6], color=GRID, lw=0.8, zorder=3)
    draw_skeleton(ax, K, cx, gy + 0.2, 0.95, col=ACCENT, lw=0.9, ms=2.2)
    for j in range(7):   # frames of the window, entering the LSTM
        ax.add_patch(FancyBboxPatch((xs[3] + 2.0 + j * 1.0, gy - 1.7), 0.6, 3.6, boxstyle='round,pad=0,rounding_size=0.3',
                                    fc=ACCENT, alpha=0.25 + 0.1 * j, ec='none', zorder=3))
    for j in range(4):   # four decisions, three agree
        ax.add_patch(Circle((xs[4] + 3.0 + j * 1.65, gy + 0.1), 0.55, fc=ACCENT if j else G1, ec='none', zorder=4))
    save(fig, 'pipeline')


def augmentation(K):
    fig = plt.figure(figsize=(WIDE, 1.9))
    card(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 71.6)
    ax.set_ylim(0, 19)
    ax.axis('off')
    rng = np.random.default_rng(3)
    x0 = K[None].repeat(30, 0)
    m0 = np.ones((30, 14), bool)
    variants = [('Recorded', 'as captured', K, ()),
                ('Elevation 60°', 'drone looking down', A.elevation(x0, m0, 60)[0][0], ()),
                ('Roll 10°', 'airframe tilted', A.rotate(x0, m0, 10)[0][0], ()),
                ('Dropout and noise', 'joints missed, jitter', A.jitter(x0, m0, 0.06, rng)[0][0], (4, 7, 10))]
    W, gap, left = 15.5, 1.6, 2.4
    for i, (title, sub, k, hidden) in enumerate(variants):
        x = left + i * (W + gap)
        cx = x + W / 2
        ax.add_patch(FancyBboxPatch((x, 1.9), W, 15.2, boxstyle='round,pad=0,rounding_size=1.2', fc=PANEL, ec='none'))
        if i:
            draw_skeleton(ax, K, cx, 12.6, 2.3, col=G1, lw=0.8, ms=1.6, z=3)
        draw_skeleton(ax, k, cx, 12.6, 2.3, col=INK if i == 0 else ACCENT, lw=1.0, ms=3, hidden=hidden)
        ax.text(cx, 4.65, title, ha='center', fontsize=6.9, fontweight='semibold')
        ax.text(cx, 3.05, sub, ha='center', fontsize=6.0, color=INK2)
    save(fig, 'augmentation')


def ablation(R):
    t, b = R['train'], R.get('baselines', {})
    rows = [('Frame classifier (random forest)', b.get('frame_rf'), G1, 'Without temporal modelling'),
            ('Window classifier (random forest)', b.get('window_rf'), G1, None),
            ('Stacked LSTM, image coordinates', t.get('stacked_lstm_image'), G2, 'Building the proposed model'),
            ('+ body-centred keypoints', t.get('stacked_lstm_body'), G2, None),
            ('+ velocities, 2 × 128 LSTM', t.get('lstm_no_aug'), G2, None),
            ('+ viewpoint augmentation (proposed)', t.get('lstm'), ACCENT, None),
            ('Temporal convolutional network', t.get('tcn'), G1, 'Alternative temporal models'),
            ('Spatio-temporal graph network', t.get('stgcn'), G1, None)]
    rows = [r for r in rows if r[1] is not None]
    fig = plt.figure(figsize=(WIDE, 2.9))
    card(fig)
    ax = fig.add_axes([0.335, 0.12, 0.6, 0.78])
    ys, y = [], 0
    for i, r in enumerate(rows):
        if r[3] and i:
            y += 0.8
        ys.append(y)
        y += 1
    ax.set_xlim(0, 100)
    ax.set_ylim(y - 0.35, -1.3)
    ax.set_yticks([])
    x_grid(ax, [0, 25, 50, 75, 100])
    ax.set_xticklabels(['0', '25', '50', '75', '100%'])
    fig_h = fig.get_figheight() * fig.dpi
    for (label, entry, col, group), yy in zip(rows, ys):
        v = video_pct(entry)
        rounded_bar(ax, yy, v, h=0.64, color=col, rpx=4)
        inside = v > 15
        ax.text(v - 1.0 if inside else v + 1.0, yy, f'{v:.1f}', va='center', ha='right' if inside else 'left',
                fontsize=6.6, color='white' if inside else INK, zorder=4,
                fontweight='semibold' if col == ACCENT else 'regular')
        fig.text(0.315, ax.transData.transform((0, yy))[1] / fig_h, label, ha='right', va='center', fontsize=6.9,
                 fontweight='semibold' if col == ACCENT else 'regular', color=INK if col == ACCENT else INK2)
        if group:
            fig.text(0.045, ax.transData.transform((0, yy - 0.95))[1] / fig_h, group.upper(), fontsize=5.6,
                     color=INK2, fontweight='semibold', va='center')
    ax.axvline(RGB_FLOW_BASELINE, color=G3, lw=0.7, ls=(0, (2, 2)), zorder=5)
    ax.text(RGB_FLOW_BASELINE - 0.6, -1.25, 'RGB + optical-flow baseline', fontsize=5.9, color=G3, ha='right', va='top')
    save(fig, 'ablation')


def robustness(R):
    rob = R['robustness']
    series = [('stacked_lstm_body', 'Stacked LSTM', G1, 1.0, 7), ('lstm_no_aug', 'No augmentation', G2, 1.0, 7),
              ('lstm', 'Proposed', ACCENT, 1.6, 12)]
    panels = [('elevation', 'Camera elevation', lambda v: f'{v:.0f}°', [0, 20, 40, 60]),
              ('rotation', 'Camera roll', lambda v: f'{v:.0f}°', [0, 10, 20, 30]),
              ('dropout', 'Missing joints', lambda v: f'{100 * v:.0f}%', [0, 0.3, 0.7]),
              ('jitter', 'Keypoint noise', lambda v: f'{v:g}', [0, 0.1, 0.2, 0.3]),
              ('fps', 'Pose rate (fps)', None, None)]
    fig = plt.figure(figsize=(WIDE, 2.0))
    card(fig)
    for i, (kind, title, fmt, ticks) in enumerate(panels):
        ax = fig.add_axes([0.065 + i * 0.187, 0.16, 0.15, 0.56])
        for name, label, col, lw, ms in series:
            if name not in rob or kind not in rob[name]:
                continue
            d = rob[name][kind]
            xs = sorted(float(k) for k in d)
            ys = [100 * d[next(k for k in d if float(k) == x)]['video'] for x in xs]
            if kind == 'fps':
                ys, xs = ys[::-1], list(range(len(xs)))
            ax.plot(xs, ys, color=col, lw=lw, solid_capstyle='round', zorder=3 if col == ACCENT else 2)
            ax.scatter(xs, ys, s=ms, color=col, zorder=4, edgecolor='#1C1C1E', linewidth=0.4)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 50, 100])
        ax.set_yticklabels(['0', '50', '100%'] if i == 0 else [], fontsize=6.0)
        for tk in [0, 25, 50, 75, 100]:
            ax.axhline(tk, color=GRID, lw=0.5, zorder=0)
        if kind == 'fps':
            ax.set_xticks(range(5))
            ax.set_xticklabels(['25', '12.5', '8.3', '5', '3'], fontsize=6.0)
        else:
            ax.set_xticks(ticks)
            ax.set_xticklabels([fmt(v) for v in ticks], fontsize=6.0)
        ax.set_title(title, fontsize=6.9, fontweight='semibold', loc='left', pad=5)
    handles = [plt.Line2D([], [], color=c, lw=lw, marker='o', ms=3.2, mec='#1C1C1E', mew=0.4) for _, _, c, lw, _ in series]
    fig.legend(handles, [s[1] for s in series], loc='upper left', bbox_to_anchor=(0.055, 0.95), ncol=3, fontsize=6.4,
               handlelength=1.8, columnspacing=2.2, labelcolor=INK)
    save(fig, 'robustness')


def safety_gate(R):
    colours = [G2, G1, ACCENT_SOFT, ACCENT]
    policies = [(label, 100 * r['executed'], r['wrong_per_stream'], c)
                for (label, r), c in zip(R['safety_gate'].items(), colours)]
    fig = plt.figure(figsize=(WIDE, 1.75))
    card(fig)
    ax_l = fig.add_axes([0.25, 0.17, 0.33, 0.58])
    ax_r = fig.add_axes([0.63, 0.17, 0.33, 0.58])
    for a in (ax_l, ax_r):
        a.set_ylim(len(policies) - 0.4, -0.6)
        a.set_yticks([])
    ax_l.set_xlim(0, 100)
    x_grid(ax_l, [0, 25, 50, 75, 100])
    ax_l.set_xticklabels(['0', '25', '50', '75', '100%'])
    ax_r.set_xlim(0, 5.2)
    x_grid(ax_r, [0, 1, 2, 3, 4, 5], fmt='{:.0f}')
    fig_h = fig.get_figheight() * fig.dpi
    for i, (label, executed, wrong, col) in enumerate(policies):
        rounded_bar(ax_l, i, executed, h=0.62, color=col, rpx=4)
        rounded_bar(ax_r, i, wrong, h=0.62, color=col, rpx=4)
        ax_l.text(executed - 1.2, i, f'{executed:.1f}', ha='right', va='center', fontsize=6.6, color='white', zorder=4)
        ax_r.text(wrong + 0.06 if wrong < 1 else wrong - 0.06, i, f'{wrong:.3f}' if wrong < 1 else f'{wrong:.2f}',
                  ha='left' if wrong < 1 else 'right', va='center', fontsize=6.6,
                  color=INK if wrong < 1 else 'white', zorder=4)
        fig.text(0.23, ax_l.transData.transform((0, i))[1] / fig_h, label.replace('>=', '≥'), ha='right', va='center',
                 fontsize=6.9, fontweight='semibold' if col == ACCENT else 'regular',
                 color=INK if col == ACCENT else INK2)
    ax_l.set_title('Intended command executed', loc='left', fontsize=6.9, pad=7)
    ax_r.set_title('Wrong actions per stream', loc='left', fontsize=6.9, pad=7)
    save(fig, 'safety_gate')


def highlights(R):
    """The four headline numbers as one strip."""
    lstm, no_aug = R['robustness']['lstm']['elevation'], R['robustness']['lstm_no_aug']['elevation']
    gate = list(R['safety_gate'].values())
    items = [(f"{video_pct(R['train']['lstm']):.1f}%", 'video accuracy on', 'unseen people'),
             (f"{100 * lstm['70.0']['video']:.1f}%", 'at 70° camera elevation', f"vs {100 * no_aug['70.0']['video']:.1f}% unaugmented"),
             (f"{10 * round(gate[0]['wrong_per_stream'] / gate[-1]['wrong_per_stream'] / 10)}×", 'fewer wrong actions', 'with the safety gate'),
             (f"{R['benchmark']['lstm']['ms_per_window']:.1f} ms", 'per decision on', 'one CPU thread')]
    fig = plt.figure(figsize=(WIDE, 1.15))
    card(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    ax.axis('off')
    for i, (value, line1, line2) in enumerate(items):
        x = i + 0.5
        ax.text(x, 0.60, value, ha='center', va='center', fontsize=17, fontweight='semibold',
                color=ACCENT if i == 0 else INK)
        ax.text(x, 0.33, line1, ha='center', va='center', fontsize=6.6, color=INK2)
        ax.text(x, 0.19, line2, ha='center', va='center', fontsize=6.6, color=INK2)
        if i:
            ax.plot([i, i], [0.2, 0.8], color=GRID, lw=0.6)
    save(fig, 'highlights')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from-runs', action='store_true')
    args = p.parse_args()
    R = load_results(args.from_runs)
    K = reference_pose()
    pipeline(K)
    augmentation(K)
    if 'train' in R and R['train']:
        ablation(R)
    if 'robustness' in R:
        robustness(R)
    if all(k in R for k in ('train', 'robustness', 'safety_gate', 'benchmark')):
        highlights(R)
    if 'safety_gate' in R:
        safety_gate(R)


if __name__ == '__main__':
    main()
