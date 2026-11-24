# fig3.py — damage ratio against severity (column width). Two thresholds, one interval.
import numpy as np
import fig_style as st
from fig_data import fig3_ratio

D = fig3_ratio(); dd = D["deltas"]; ratio = D["ratio"]; f70 = D["floor70"]; dstar = D["dstar"]
fig, ax = st.new_fig(full=False, h=2.95)

# detection floor band + ratio=1 reference
ax.axvspan(f70[0], f70[1], color=st.VERMILLION, alpha=0.15, zorder=0)
ax.axhline(1.0, color=st.GREY, ls="-", lw=0.8, zorder=1)
# label-channel ratio: markers + LIGHT STRAIGHT lines (no spline)
ax.plot(dd, ratio, marker="o", ls="-", lw=1.0, color=st.BLUE, label="label rotation")
# interpolated crossing
ax.plot([dstar], [1.0], marker="*", ms=11, color=st.BLACK, zorder=6)
ax.annotate(f"interpolated\nδ*≈{dstar:.1f}°", xy=(dstar, 1.0), xytext=(dstar + 0.5, 2.4),
            fontsize=8, ha="left", arrowprops=dict(arrowstyle="-", lw=0.7, color=st.BLACK))
# image-channel pooled ratio as its own reference + marker
ax.axhline(D["img_ratio"], color=st.GREEN, ls="--", lw=1.0, zorder=1)
ax.plot([10], [D["img_ratio"]], marker="D", ms=5, color=st.GREEN, zorder=5)
ax.text(10.2, D["img_ratio"], f"image (pooled {D['img_ratio']:.2f})", color=st.GREEN, fontsize=8, va="center", ha="left")
ax.text((f70[0]+f70[1])/2, 5.3, "det. floor\n4.2–4.8°", color=st.VERMILLION, fontsize=8, ha="center", va="top")

ax.set_xlabel("label displacement δ (deg)")
ax.set_ylabel("damage ratio  (misleading / lost-frame)")
ax.set_xlim(3, 11.4); ax.set_ylim(-0.8, 6.0)
ax.legend(loc="upper left", handlelength=1.8)
st.save(fig, "fig3_damage_ratio")

# --- Fig 3b: the two cost components in degrees (the inset did not fit in fig3 without crowding) ---
fig2, ax2 = st.new_fig(full=False, h=2.5)
ax2.axvspan(D["floor70"][0], D["floor70"][1], color=st.VERMILLION, alpha=0.15, zorder=0)
ax2.plot(dd, D["mis"], marker="o", ls="-", lw=1.1, color=st.BLUE, label="misleading cost (naive−oracle)")
ax2.plot(dd, D["lost"], marker="s", ls="--", lw=1.1, color=st.GREY, label="lost-frame cost (oracle−clean)")
ax2.axhline(0, color=st.BLACK, lw=0.5)
ax2.set_xlabel("label displacement δ (deg)")
ax2.set_ylabel("cost (deg)")
ax2.set_xlim(3, 10.6)
ax2.legend(loc="upper left", handlelength=1.8)
st.save(fig2, "fig3b_cost_components")
