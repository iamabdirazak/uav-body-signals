"""Figure style: dark, calm and minimal, matching the demo window. Inter typeface, one blue accent."""
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch

CARD = '#1C1C1E'      # figure card
PANEL = '#2C2C2E'     # raised element on the card
INK = '#F5F5F7'       # primary text
INK2 = '#98989D'      # secondary text
GRID = '#38383A'
ACCENT = '#3E9BC8'    # the proposed method (same blue as the demo)
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
