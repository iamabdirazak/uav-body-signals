"""Figure style: dark, calm and minimal, matching the demo window. Inter typeface, one blue accent."""
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch

CARD = '#1C1C1E'      # figure card
PANEL = '#2C2C2E'     # raised element on the card
INK = '#F5F5F7'       # primary text
INK2 = '#98989D'      # secondary text
GRID = '#38383A'
ACCENT = '#3E9BC8'    # the proposed method (same blue as the demo)
ACCENT_SOFT = '#2A6585'
G1 = '#48484A'        # quiet comparison
G2 = '#7C7C80'        # stronger comparison
G3 = '#AEAEB2'

mpl.rcParams.update({
    'font.family': 'Inter', 'font.size': 7, 'text.color': INK,
    'axes.edgecolor': 'none', 'axes.labelcolor': INK2, 'axes.titlesize': 7.5, 'axes.titleweight': 'semibold',
    'axes.titlecolor': INK, 'axes.titlelocation': 'left', 'axes.titlepad': 5,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'xtick.major.size': 0, 'ytick.major.size': 0, 'axes.facecolor': 'none', 'figure.facecolor': 'none',
    'legend.frameon': False, 'legend.fontsize': 6.8, 'savefig.dpi': 400, 'savefig.transparent': True,
})


def card(fig, radius_in=0.14):
    """Rounded dark card behind the whole figure; the corners outside it stay transparent."""
    w, h = fig.get_size_inches()
    fig.patches.append(FancyBboxPatch((0, 0), 1, 1, boxstyle=f'round,pad=0,rounding_size={radius_in / w}',
                                      mutation_aspect=w / h, transform=fig.transFigure, fc=CARD, ec='none', zorder=-10))


def _aspect(ax):
    """y/x data units per pixel, so rounded corners stay circular."""
    ax.figure.canvas.draw_idle()
    bb = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return (abs(y1 - y0) / bb.height) / (abs(x1 - x0) / bb.width)


def rounded_bar(ax, y, width, h=0.62, color=ACCENT, rpx=3.2):
    """Horizontal bar with gently rounded corners. Call after the axis limits are final."""
    if width <= 0:
        return
    asp = _aspect(ax)
    x_per_px = abs(ax.get_xlim()[1] - ax.get_xlim()[0]) / ax.get_window_extent().width
    r = min(rpx * x_per_px * ax.figure.dpi / 100, width / 2, (h / 2) / asp)
    ax.add_patch(FancyBboxPatch((0, y - h / 2), width, h, boxstyle=f'round,pad=0,rounding_size={r}',
                                fc=color, ec='none', mutation_aspect=asp, zorder=3))


def x_grid(ax, ticks, fmt='{:.0f}%'):
    ax.set_xticks(ticks)
    ax.set_xticklabels([fmt.format(t) for t in ticks])
    for t in ticks:
        ax.axvline(t, color=GRID, lw=0.6, zorder=0)
