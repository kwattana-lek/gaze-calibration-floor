# fig1.py — the collapse (full width, two panels). Contribution 3 evidence.
import numpy as np
import fig_style as st
from fig_data import fig1_collapse

D = fig1_collapse(); dl = D["deltas"]; P = D["pooled"]; f70 = D["floor70"]
fig, (axL, axR) = st.new_fig(full=True, h=3.2, ncols=2)
labels = ["low-σ", "mid-σ", "high-σ"]

for ax, xmode in [(axL, "raw"), (axR, "norm")]:
    for gi, (col, mk, ls) in enumerate(st.TERCILE):
        au = P["curves"][gi]
        x = np.array(dl, float) if xmode == "raw" else np.array(dl, float) / P["sig_t"][gi]
        ax.plot(x, au, marker=mk, ls=ls, color=col, label=f"{labels[gi]} (σ={P['sig_t'][gi]:.1f}°)")
        # mark the per-tercile 0.70 crossing
        c = P["cross"][gi]; xc = c if xmode == "raw" else c / P["sig_t"][gi]
        ax.plot([xc], [0.70], marker=mk, color=col, ms=7, mfc="white", mew=1.2, zorder=5)
    ax.axhline(0.70, color=st.GREY, ls="-", lw=0.8, zorder=0)
    ax.axhline(0.90, color=st.GREY, ls="--", lw=0.8, zorder=0)
    ax.set_ylim(0.52, 1.0)

axL.set_ylabel("detection ROC-AUC (LOO residual)")
axL.set_xlabel("label displacement δ (deg)")
axR.set_xlabel(r"δ / σ")
# reference-line labels only on the outer (right) edge, to avoid mid-figure clutter
axR.text(1.01, 0.70, "AUC 0.70", transform=axR.get_yaxis_transform(), va="center", ha="left", fontsize=8, color=st.GREY, clip_on=False)
axR.text(1.01, 0.90, "0.90", transform=axR.get_yaxis_transform(), va="center", ha="left", fontsize=8, color=st.GREY, clip_on=False)
axL.annotate(f"between-tercile\nCV = {P['cv_raw']:.3f}", xy=(0.04, 0.93), xycoords="axes fraction",
             fontsize=8, va="top", ha="left")
axR.annotate(f"between-tercile\nCV = {P['cv_norm']:.3f}", xy=(0.04, 0.93), xycoords="axes fraction",
             fontsize=8, va="top", ha="left")
axL.legend(loc="lower right", handlelength=1.8)
st.save(fig, "fig1_collapse")
