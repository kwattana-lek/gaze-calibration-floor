# fig6.py — the floor mechanism (column width, no data; illustrative schematic).
import numpy as np
import fig_style as st

def gauss(x, mu, sd): return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))

x = np.linspace(0, 16, 600); SIGMA = 3.0; SD = 1.7
fig, axs = st.new_fig(full=False, h=3.3, ncols=1)
import matplotlib.pyplot as plt
plt.close(fig)
fig, (a0, a1) = plt.subplots(2, 1, figsize=(st.COL_W, 3.3), constrained_layout=True, sharex=True)

for ax, delta, title, sep in [(a0, 2.0, r"$\delta <$ floor", False), (a1, 8.0, r"$\delta >$ floor", True)]:
    clean = gauss(x, SIGMA, SD); corr = gauss(x, SIGMA + delta, SD)
    ax.plot(x, clean, color=st.BLUE, lw=1.4, label="clean")
    ax.plot(x, corr, color=st.VERMILLION, lw=1.4, ls="--", label="corrupted")
    ax.fill_between(x, np.minimum(clean, corr), color=st.GREY, alpha=0.45, lw=0)   # overlap
    ax.axvline(SIGMA, color=st.BLACK, lw=0.6, ls=":")
    ax.text(SIGMA, ax.get_ylim()[1] * 0.92 if False else 0.245, r"$\sigma$", fontsize=8, ha="center", va="bottom")
    ax.set_ylim(0, 0.26); ax.set_yticks([])
    ax.text(0.02, 0.9, title, transform=ax.transAxes, fontsize=9, va="top")
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    if sep:
        thr = SIGMA + delta / 2
        ax.axvline(thr, color=st.GREEN, lw=1.2)
        ax.text(thr + 0.2, 0.20, "threshold", color=st.GREEN, fontsize=8, va="center", ha="left")
    else:
        ax.annotate("no threshold\nseparates these", xy=(SIGMA + delta / 2, 0.06), xytext=(9.6, 0.16),
                    fontsize=8, ha="center", arrowprops=dict(arrowstyle="->", lw=0.7, color=st.BLACK))
    ax.legend(loc="upper right", handlelength=1.6, fontsize=8)

a1.set_xlabel("residual (deg)")
for ext in ("pdf", "png"):
    import os; fig.savefig(os.path.join(st.FIGDIR, f"fig6_floor_mechanism.{ext}"), dpi=300, bbox_inches="tight")
plt.close(fig); print("  wrote figures/fig6_floor_mechanism.pdf + .png")
