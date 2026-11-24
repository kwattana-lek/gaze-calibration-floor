# fig2.py — severity anchored against the detection floor (column width).
import numpy as np
import fig_style as st
from fig_data import fig2_displacement

D = fig2_displacement(); d = D["disp"]; f70 = D["floor70"]; f90 = D["floor90"]
fig, ax = st.new_fig(full=False, h=2.9)
ax.hist(d, bins=40, color=st.BLUE, alpha=0.85, edgecolor="white", linewidth=0.3)
ymax = ax.get_ylim()[1]

# undetectable: area LEFT of the 0.70 floor; soft-edged band over the 4.2-4.8 range
ax.axvspan(0, f70[0], color=st.VERMILLION, alpha=0.20)
ax.axvspan(f70[0], f70[1], color=st.VERMILLION, alpha=0.10)          # soft edge = the range
# unreliably-detectable: 0.70 crossing up to the 0.90 crossing (lighter), soft-edged
ax.axvspan(f70[1], f90[0], color=st.ORANGE, alpha=0.14)
ax.axvspan(f90[0], f90[1], color=st.ORANGE, alpha=0.07)

# p10 tick
p10 = D["pct"][10]
ax.annotate(f"p10 = {p10:.1f}°", xy=(p10, 0), xytext=(p10, ymax * 0.12), ha="center", fontsize=8,
            arrowprops=dict(arrowstyle="-", lw=0.8, color=st.BLACK))
ax.plot([p10], [0], marker="^", color=st.BLACK, ms=6, clip_on=False, zorder=6)

ax.annotate("8–10%\nundetectable", xy=(2.3, ymax * 0.62), fontsize=8, ha="center", va="center", color=st.VERMILLION)
ax.annotate("27–31% unreliable", xy=(6.4, ymax * 0.88), fontsize=8, ha="center", va="center", color="#B26A00")
ax.set_xlabel("realised donor-swap displacement (deg)")
ax.set_ylabel("corrupted frames")
ax.set_xlim(0, d.max() * 1.02)
st.save(fig, "fig2_displacement")
