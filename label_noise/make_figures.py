#!/usr/bin/env python3
# make_figures.py — the three figures for the Neurocomputing manuscript
"""The paper has ten tables and no figures, yet its two headline claims are both
curves: a detection curve with a plateau and two crossings, and a proportionality
that is only visible as a collapse. Tables cannot show either.

  fig_floor.pdf     the detectability floor, and its insensitivity to the
                    contamination rate p_c (Table 2 made visible)
  fig_collapse.pdf  the model-relative constant across seven configurations
                    (Tables 3 and 4 made visible)
  fig_ratio.pdf     the damage-ratio crossing R = 1 against the floor band ---
                    the coincidence the discussion turns on

WHAT FIGURE 2 CLAIMS, PRECISELY
The seven detection curves do NOT coincide once divided by e_hat, and the figure
must not pretend they do: their shapes differ above the floor, and Section 7.5
shows that shape difference is a property of the ridge penalty rather than of
the models. What IS invariant is the 0.7 CROSSING. So the figure draws the
curves recessively and the crossings prominently: scattered over a 2.0x range on
the raw axis, clustered in a 1.32x range on the model-relative one. That is the
claim, drawn exactly as strongly as the evidence supports.

AGGREGATION NOTE FOR FIGURE 3
The damage ratio is aggregated exactly as exp_b_analyze.py does it --- per
subject mean of each cost, then the ratio of the across-subject means --- and
NOT as the mean of per-cell ratios. The latter is what a naive reading of the
CSV gives and it is wrong: lost_cost passes through zero, 167 of 945 cells are
already NaN, and the mean of the survivors swings between -2.2 and +5.0 with no
relation to the reported sequence. A ratio of noisy quantities must be
aggregated as a ratio of means.

Design: validated categorical slots 1-3 (blue/orange/aqua); every series also
carries its own dash pattern and marker so the figures survive greyscale
printing and colour-vision deficiency without relying on hue. Serif type at the
document's size; reference lines recessive.

Usage:  python3 make_figures.py
"""
import csv
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "manuscript", "figs")

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
    "legend.fontsize": 8.2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.labelcolor": INK,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def roc_auc(scores, pos):
    s = np.asarray(scores, float); p = np.asarray(pos, bool)
    ok = ~np.isnan(s); s, p = s[ok], p[ok]
    a, b = int(p.sum()), int((~p).sum())
    if a == 0 or b == 0:
        return float("nan")
    u, inv, c = np.unique(s, return_inverse=True, return_counts=True)
    cs = np.cumsum(c); avg = (cs - c + cs + 1) / 2.0
    return float((avg[inv][p].sum() - a * (a + 1) / 2) / (a * b))


def crossing(ds, aa, t=0.7):
    """First sustained upward crossing, linearly interpolated."""
    if not (aa[-1] == aa[-1]) or aa[-1] < t:
        return float("nan")
    for i in range(1, len(ds)):
        if aa[i] >= t > aa[i - 1]:
            f = (t - aa[i - 1]) / (aa[i] - aa[i - 1])
            return ds[i - 1] + f * (ds[i] - ds[i - 1])
    return float("nan")


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.9)
    ax.set_axisbelow(True)


def jl(path, **keep):
    for line in open(os.path.join(HERE, path)):
        r = json.loads(line)
        if all(abs(r[k] - v) < 1e-9 if isinstance(v, float) else r.get(k) == v
               for k, v in keep.items()):
            yield r


# ------------------------------------------------------------------ figure 1
def fig_floor():
    rows = list(jl("exp_a_records.jsonl", model="random_offset"))
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["pc"]][r["delta"]].append(r)

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.axvspan(4.2, 4.8, color=INK, alpha=0.07, lw=0, zorder=0)
    ax.axvspan(8.0, 8.7, color=INK, alpha=0.07, lw=0, zorder=0)
    for y in (0.7, 0.9):
        ax.axhline(y, color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)

    style = {0.1: (BLUE, "-", "o"), 0.2: (ORANGE, (0, (5, 2)), "s"),
             0.4: (AQUA, (0, (1, 1.6)), "^")}
    for pc in sorted(by):
        ds = sorted(by[pc])
        aa = [roc_auc([x["score_loo"] for x in by[pc][d]],
                      [x["is_corrupted"] for x in by[pc][d]]) for d in ds]
        col, ls, mk = style[pc]
        ax.plot(ds, aa, color=col, ls=ls, marker=mk, lw=1.7, ms=4.4,
                mec="white", mew=0.7, zorder=3, clip_on=False,
                label=f"$p_c = {pc:g}$")

    ax.text(4.5, 0.972, "0.7 crossing\n4.2–4.8°", ha="center", va="top",
            fontsize=7.6, color=MUTED, linespacing=1.3)
    ax.text(8.35, 0.972, "0.9 crossing\n8.0–8.7°", ha="center", va="top",
            fontsize=7.6, color=MUTED, linespacing=1.3)
    ax.set_xlabel("label displacement $\\delta$ (degrees)")
    ax.set_ylabel("detection ROC-AUC")
    ax.set_xlim(2.7, 10.3); ax.set_ylim(0.5, 1.0)
    ax.set_xticks([3, 4, 5, 6, 8, 10])
    leg = ax.legend(loc="lower right", frameon=True, framealpha=0.95,
                    edgecolor=GRID, borderpad=0.5)
    leg.get_frame().set_linewidth(0.5)
    tidy(ax)
    fig.savefig(os.path.join(OUT, "fig_floor.pdf"))
    plt.close(fig)
    print("  fig_floor.pdf")


# ------------------------------------------------------------------ figure 2
def iso_crossing(ds, aa, ns, t=0.7):
    """Crossing read off an isotonic (monotone, frame-weighted) fit of pooled AUC on
    severity -- the estimator of exp_isotonic_floor.py (reviewer, major comment 7)."""
    from sklearn.isotonic import IsotonicRegression
    x, y, w = np.array(ds, float), np.array(aa, float), np.array(ns, float)
    ok = np.isfinite(y)
    yf = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x[ok], y[ok], sample_weight=w[ok]).predict(x)
    if yf[0] >= t or yf[-1] < t:
        return float("nan")
    i = int(np.argmax(yf >= t)); x0, x1, y0, y1 = x[i-1], x[i], yf[i-1], yf[i]
    return float(x0 + (t - y0) / (y1 - y0) * (x1 - x0)) if y1 > y0 else float(x1)


def cells_of(rows, lvkey=None):
    g = defaultdict(list)
    for r in rows:
        g[r.get(lvkey, "-") if lvkey else "-"].append(r)
    out = {}
    for k, rs in g.items():
        ds = sorted({r["delta"] for r in rs})
        aa = [roc_auc([x["loo_residual_deg"] for x in rs if x["delta"] == d],
                      [x["is_corrupted"] for x in rs if x["delta"] == d])
              for d in ds]
        per = defaultdict(list)
        for x in rs:
            if not x["is_corrupted"] and x["delta"] <= 10:
                per[x["subject"]].append(x["loo_residual_deg"])
        eh = float(np.mean([np.mean(v) for v in per.values()]))      # mean of subject means (Sec. 6)
        ns = [sum(1 for x in rs if x["delta"] == d) for d in ds]
        out[k] = (ds, aa, eh, iso_crossing(ds, aa, ns))
    return out


def fig_collapse():
    full = cells_of(list(jl("sigma_sweep_records.jsonl", model="random_offset")),
                    "sigma_level")
    full = {k: v for k, v in full.items() if k in ("L0", "L1", "L2")}
    froz = cells_of(list(jl("resnet_frozen_records.jsonl", pc=0.2, lam=0.01)),
                    "level")
    uni = cells_of(list(jl("unigaze_arm_records.jsonl", pc=0.2, lam=0.01)))
    # EYEDIAP: same frozen UniGaze model, and a ResNet-18 trained LOPO there under the
    # L0 protocol (exp_a driver, so the LOO residual is in score_loo). Hollow markers
    # = EYEDIAP; colour still identifies the model family.
    uni_e = cells_of(list(jl("eyediap_unigaze_arm_records.jsonl", pc=0.2, lam=0.01)))
    ft_e = cells_of([dict(r, loo_residual_deg=r["score_loo"]) for r in
                     jl("exp_a_records_eyediap.jsonl", model="random_offset", pc=0.2)
                     if r["score_loo"] is not None])
    RT = "s000,s001,s002,s003,s004,s005,s006,s007,s008,s009,s011,s012,s013,s016".split(",")
    rt_e = cells_of([r for r in jl("rtgene_unigaze_arm_records.jsonl", pc=0.2, lam=0.01) if r["subject"] in RT])
    groups = [("ResNet-18, full fine-tune", BLUE, "-", "o", full, False),
              ("ResNet-18, frozen $+$ ridge", ORANGE, (0, (5, 2)), "s", froz, False),
              ("UniGaze ViT-H/14, frozen $+$ ridge", AQUA, (0, (1, 1.6)), "^", uni, False),
              ("ResNet-18, full fine-tune, EYEDIAP", BLUE, "-", "o", ft_e, True),
              ("UniGaze ViT-H/14, frozen $+$ ridge, EYEDIAP", AQUA, (0, (1, 1.6)), "^", uni_e, True),
              ("UniGaze ViT-H/14, frozen $+$ ridge, RT-GENE", AQUA, (0, (1, 1.6)), "v", rt_e, True)]

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.1), sharey=True)
    fig.subplots_adjust(wspace=0.09, bottom=0.37)
    for ax, mode in zip(axes, ("raw", "rel")):
        ax.axhline(0.7, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=2)
        xs = []
        for label, col, ls, mk, cs, hollow in groups:
            for k in sorted(cs):
                ds, aa, eh, x7 = cs[k]
                x = ds if mode == "raw" else [d / eh for d in ds]
                ax.plot(x, aa, color=col, ls=ls, lw=1.0, alpha=0.45, zorder=3)
                cx = x7 if mode == "raw" else x7 / eh
                xs.append(cx)
                ax.plot([cx], [0.7], marker=mk, ms=6.0, color=col,
                        mfc="white" if hollow else col, mec=col if hollow else "white",
                        mew=1.1 if hollow else 0.9, zorder=5, clip_on=False)
        lo, hi = min(xs), max(xs)
        ax.annotate("", xy=(lo, 0.585), xytext=(hi, 0.585),
                    arrowprops=dict(arrowstyle="|-|,widthA=0.3,widthB=0.3",
                                    color=INK, lw=0.9, shrinkA=0, shrinkB=0))
        # label in axes fractions so a wide string can never clip the spine
        ax.text(0.5, 0.035,
                (f"crossings span {lo:.2f}–{hi:.2f}°  ({hi/lo:.2f}$\\times$)"
                 if mode == "raw" else
                 f"crossings span {lo:.2f}–{hi:.2f}  ({hi/lo:.2f}$\\times$)"),
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8, color=INK)
        tidy(ax)
        ax.set_ylim(0.45, 1.0)

    axes[0].set_xlabel("$\\delta$ (degrees)")
    axes[0].set_title("raw displacement", color=MUTED, pad=5)
    axes[0].set_ylabel("detection ROC-AUC")
    axes[0].set_xlim(0, 26)
    axes[1].set_xlabel("$\\delta \\,/\\, \\hat{e}$")
    axes[1].set_title("model-relative", color=MUTED, pad=5)
    axes[1].set_xlim(0, 4)

    handles = [plt.Line2D([], [], color=c, ls=ls, marker=mk, lw=1.2, ms=5,
                          mfc="white" if h else c, mec=c if h else "white",
                          mew=1.0 if h else 0.8, label=lab)
               for lab, c, ls, mk, _, h in groups]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False,
               bbox_to_anchor=(0.5, -0.22), handlelength=2.8, labelspacing=0.35)
    fig.savefig(os.path.join(OUT, "fig_collapse.pdf"))
    plt.close(fig)
    for lab, _, _, _, cs, _ in groups:
        print(f"  [{lab}] " + ", ".join(
            f"{k}: e={v[2]:.2f} floor={v[3]:.2f} c={v[3]/v[2]:.3f}"
            for k, v in sorted(cs.items())))
    print("  fig_collapse.pdf")


# ------------------------------------------------------------------ figure 3
def fig_ratio():
    """Damage ratio, aggregated exactly as exp_b_analyze.py does it."""
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "exp_b_records.csv")))
            if r["channel"] == "a_rot"]
    per = defaultdict(lambda: defaultdict(lambda: ([], [])))
    for r in rows:
        try:
            m, l = float(r["misleading_cost"]), float(r["lost_cost"])
        except (TypeError, ValueError):
            continue
        M, L = per[float(r["severity"])][r["subject"]]
        M.append(m); L.append(l)

    ds, ratio = [], []
    for d in sorted(per):
        sm = [np.nanmean(v[0]) for v in per[d].values()]
        sl = [np.nanmean(v[1]) for v in per[d].values()]
        num, den = np.nanmean(sm), np.nanmean(sl)
        ds.append(d); ratio.append(num / den if den else np.nan)

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.axvspan(4.2, 4.8, color=INK, alpha=0.08, lw=0, zorder=0)
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.plot(ds, ratio, color=BLUE, ls="-", marker="o", lw=1.8, ms=4.6,
            mec="white", mew=0.7, zorder=3)
    ax.axvline(4.84, color=ORANGE, lw=1.3, ls=(0, (1, 1.4)), zorder=2)

    top = np.nanmax(ratio)
    ax.annotate("$\\delta^{*} = 4.84°$", (4.84, top), xytext=(7, -2),
                textcoords="offset points", color=ORANGE, fontsize=8.5)
    ax.text(0.035, 0.95, "shaded: detectability floor 4.2–4.8°",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7.8, color=MUTED)
    ax.text(max(ds), 1.0, "  $R=1$", va="center", fontsize=8, color=MUTED)
    ax.set_xlabel("label displacement $\\delta$ (degrees)")
    ax.set_ylabel("damage ratio $R$")
    ax.set_xlim(3.6, 10.6)
    tidy(ax)
    fig.savefig(os.path.join(OUT, "fig_ratio.pdf"))
    plt.close(fig)
    print("  fig_ratio.pdf  " +
          ", ".join(f"{d:g}:{v:.2f}" for d, v in zip(ds, ratio)))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("writing figures ->", OUT)
    fig_floor(); fig_collapse(); fig_ratio()
    print("done")
