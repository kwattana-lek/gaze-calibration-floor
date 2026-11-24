# fig_style.py — shared IEEE Access figure style (import in all six figure scripts)
"""Two-column IEEE Access. Column width 3.5in, full width 7.16in. Serif body-matching
font, 8pt floor / 9pt labels+ticks. Okabe-Ito colourblind-safe palette; meaning is
never colour-alone (always paired with marker shape or line style). Vector PDF + 300dpi
PNG per figure into figures/. No gridlines unless they aid value reading, no legend box,
no in-axes titles (captions live in LaTeX)."""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__)); _AG = os.path.dirname(_HERE)
FIGDIR = os.path.join(_AG, "figures"); os.makedirs(FIGDIR, exist_ok=True)

COL_W, FULL_W = 3.5, 7.16          # inches
COL_H_MAX, FULL_H_MAX = 3.0, 3.5

# Okabe-Ito (colourblind-safe)
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#E69F00", "#009E73", "#999999"
VERMILLION, SKY, PURPLE, BLACK = "#D55E00", "#56B4E9", "#F0E442", "#000000"
# tercile identity (colour + marker + linestyle, so never colour-alone)
TERCILE = [("#0072B2", "o", "-"), ("#E69F00", "s", "--"), ("#D55E00", "^", ":")]

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.4, "lines.markersize": 4.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "axes.grid": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,          # embed real fonts (TrueType), not Type-3
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def new_fig(full=False, h=None, ncols=1):
    w = FULL_W if full else COL_W
    h = h or (2.7 if not full else 3.0)
    fig, ax = plt.subplots(1, ncols, figsize=(w, h), constrained_layout=True,
                           sharey=(ncols > 1))
    return fig, ax


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf + .png")
