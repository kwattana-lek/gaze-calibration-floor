# fig5.py — two-channel triage schematic (full width, no data). Pure matplotlib.
import matplotlib.patches as mp
from matplotlib.patches import FancyArrowPatch
import fig_style as st

fig, ax = st.new_fig(full=True, h=3.3)
ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
GREYF = "#EDEDED"; ACC = st.BLUE

def box(x, y, w, h, text, fc=GREYF, ec=st.BLACK, fs=9, bold=False):
    ax.add_patch(FancyArrowPatch((0, 0), (0, 0)))  # no-op keep import used
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.4",
                                   linewidth=1.0, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")

def arrow(x0, y0, x1, y1, style="-|>", color=st.BLACK):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=11,
                                 lw=1.0, color=color, shrinkA=2, shrinkB=2))

# support set
box(2, 18, 17, 10, "K = 9\ncalibration\nsupport set", fc="white", bold=True)
# two parallel gates
box(30, 27, 20, 8, "image gate\n(learned detector)")
box(30, 11, 20, 8, "residual gate\n(resub. residual, τ=3.42°)")
# outcomes
box(66, 33, 22, 8, "KEEP", fc="white", bold=True)
box(66, 19, 22, 8, "RECAPTURE", fc="white", bold=True)
box(66, 5, 22, 8, "DISCARD", fc="white", bold=True)

# arrows support -> gates
arrow(19, 25, 30, 31); arrow(19, 21, 30, 15)
# gates -> outcomes
arrow(50, 31, 66, 37)             # image gate -> keep (clean image)
arrow(50, 30, 66, 23)             # image gate -> recapture (image corrupt)
arrow(50, 15, 66, 22)             # residual gate -> recapture /
arrow(50, 13, 66, 9)              # residual gate -> discard (label corrupt)

# cost annotations
ax.text(89.5, 37, "no cost", fontsize=8, va="center", ha="left", color="#3a7d3a")
ax.text(89.5, 23, "one prompt,\na few seconds", fontsize=8, va="center", ha="left", color="#8a6d00")
ax.text(89.5, 9, "one frame of nine,\npermanently", fontsize=8, va="center", ha="left", color=st.VERMILLION)

# channel damage-ratio annotations
ax.text(40, 24.2, "image damage ratio 0.40  (<1 → worth keeping)", fontsize=7.5, ha="center", va="top", color=st.GREEN)
ax.text(40, 8.2, "label damage ratio >1 above the 4.2–4.8° floor", fontsize=7.5, ha="center", va="top", color=ACC)

# ordering null note
ax.text(50, 1.5, "Gate ordering (parallel vs cascade) was measured and made no difference (d$_z$≈−0.25, n.s.).",
        fontsize=7.5, ha="center", va="center", style="italic", color=st.GREY)
st.save(fig, "fig5_triage_schematic")
