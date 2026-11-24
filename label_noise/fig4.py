# fig4.py — operating-point trade-off (column width). Currency is user time.
import numpy as np
import fig_style as st
from fig_data import fig4_tradeoff

D = fig4_tradeoff(); tau = D["tau"]
fig, ax = st.new_fig(full=False, h=2.9)
styles = [(st.BLUE, "-", "o"), (st.ORANGE, "--", "s"), (st.VERMILLION, ":", "^")]
for (pc, (col, ls, mk)) in zip(D["pcs"], styles):
    c = D["curves"][pc]
    ax.plot(c["wasted"], c["missed"], ls=ls, color=col, lw=1.3, label=f"$p_c$={pc}")
    sx, sy = c["star"]
    ax.plot([sx], [sy], marker="*", ms=12, color=col, mec=st.BLACK, mew=0.5, zorder=6)

# annotate the LOPO-selected point on the middle curve
sx, sy = D["curves"][0.2]["star"]
ax.annotate(f"τ=3.42°\n({sx:.2f}, {sy:.2f})", xy=(sx, sy), xytext=(sx + 0.25, sy + 0.55),
            fontsize=8, ha="left", arrowprops=dict(arrowstyle="->", lw=0.7, color=st.BLACK))
ax.set_xlabel("wasted recaptures / session")
ax.set_ylabel("corrupted frames missed / session")
ax.set_xlim(left=0); ax.set_ylim(bottom=0)
from matplotlib.lines import Line2D
star = Line2D([], [], marker="*", color=st.BLACK, ls="none", ms=10)   # real star marker (not a text glyph)
h, l = ax.get_legend_handles_labels()
ax.legend(h + [star], l + ["LOPO τ=3.42°"], loc="upper right", handlelength=2.0)
st.save(fig, "fig4_operating_point")
