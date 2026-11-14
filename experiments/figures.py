"""Draw the pipeline figure in assets/ used by the README.

    python -m experiments.figures
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from uavsignals.constants import EDGES

from .common import ROOT
from .style import ACCENT, G1, G2, G3, GRID, INK, INK2, PANEL, card

ASSETS = os.path.join(ROOT, 'assets')
WIDE = 7.16   # inches, the width of a README image at full width


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


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    pipeline(reference_pose())


if __name__ == '__main__':
    main()
