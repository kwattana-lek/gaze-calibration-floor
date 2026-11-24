# fig2_severity_floor.py — realised donor-swap displacement vs the detection floor.
"""Pure plotting from the committed 15-subject severity re-analysis CSV
(logs/displacement_raw.csv, commit 14498a8). No new computation. Every annotated
number is recomputed from the CSV; if a computed value falls outside the stated
label range, the script prints a WARNING and does not hardcode the label."""
import csv, os, sys
import numpy as np
import fig_style as st

_HERE = os.path.dirname(os.path.abspath(__file__)); _AG = os.path.dirname(_HERE)
CSV = os.path.join(_AG, "logs", "displacement_raw.csv")
FLOOR70 = (4.2, 4.8)   # AUC-0.70 crossing range  -> undetectable region is left of this
FLOOR90 = (8.0, 8.7)   # AUC-0.90 crossing range

d = np.array([float(r["displacement_deg"]) for r in csv.DictReader(open(CSV))])
n = len(d)
p10 = float(np.percentile(d, 10))
f = {t: float(np.mean(d < t)) for t in (4.2, 4.8, 8.0, 8.7)}
print(f"n={n}  p10={p10:.2f}  <4.2={f[4.2]*100:.0f}%  <4.8={f[4.8]*100:.0f}%  <8.0={f[8.0]*100:.0f}%  <8.7={f[8.7]*100:.0f}%")

# --- verify the CSV supports each annotation before drawing it ---
def check(cond, msg):
    if not cond: print(f"WARNING: {msg}")
    return cond
und = check(round(f[4.2]*100) == 8 and round(f[4.8]*100) == 10,
            f"undetectable fraction {f[4.2]*100:.0f}-{f[4.8]*100:.0f}% != stated 8-10%")
unr = check(round(f[8.0]*100) == 27 and round(f[8.7]*100) == 31,
            f"unreliable fraction {f[8.0]*100:.0f}-{f[8.7]*100:.0f}% != stated 27-31%")
p10ok = check(round(p10, 1) == 4.8, f"p10={p10:.2f} rounds to {round(p10,1)}, not 4.8")
und_lbl = f"{f[4.2]*100:.0f}–{f[4.8]*100:.0f}%\nundetectable"
unr_lbl = f"{f[8.0]*100:.0f}–{f[8.7]*100:.0f}% unreliable"

fig, ax = st.new_fig(full=False, h=2.9)
ax.hist(d, bins=40, color=st.BLUE, alpha=0.85, edgecolor="white", linewidth=0.3)
ymax = ax.get_ylim()[1]

# undetectable: area LEFT of the 0.70 floor; the 4.2-4.8 range drawn as a soft-edged band
ax.axvspan(0, FLOOR70[0], color=st.VERMILLION, alpha=0.22)
ax.axvspan(FLOOR70[0], FLOOR70[1], color=st.VERMILLION, alpha=0.10)          # soft edge = the range, not a line
# unreliably-detectable: 0.70 crossing up to the 0.90 crossing (lighter), also soft-edged
ax.axvspan(FLOOR70[1], FLOOR90[0], color=st.ORANGE, alpha=0.15)
ax.axvspan(FLOOR90[0], FLOOR90[1], color=st.ORANGE, alpha=0.07)

# p10 labelled tick
ax.plot([p10], [0], marker="^", color=st.BLACK, ms=6, clip_on=False, zorder=6)
ax.annotate(f"p10 = {p10:.1f}°", xy=(p10, 0), xytext=(p10, ymax * 0.13), ha="center", fontsize=8,
            arrowprops=dict(arrowstyle="-", lw=0.8, color=st.BLACK))

ax.annotate(und_lbl, xy=(2.2, ymax * 0.60), fontsize=8, ha="center", va="center", color=st.VERMILLION)
ax.annotate(unr_lbl, xy=(6.4, ymax * 0.90), fontsize=8, ha="center", va="center", color="#B26A00")

ax.set_xlabel("realised displacement (deg)")
ax.set_ylabel("corrupted-frame count")          # y-axis is a raw count (not a density)
ax.set_xlim(0, d.max() * 1.02)
st.save(fig, "fig2_severity_floor")
print(f"annotations verified: undetectable={und} unreliable={unr} p10={p10ok}")
