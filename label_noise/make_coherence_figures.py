#!/usr/bin/env python3
# make_coherence_figures.py -- the three figures of the coherence manuscript.
"""Reuses fig_style.py, the house figure style already used by the earlier
manuscript: Okabe-Ito, colour never alone (every series also carries a marker and
a line style, so the figures survive greyscale printing and colour-vision
deficiency), vector PDF plus 300 dpi PNG, Type-42 embedded fonts, no legend box,
captions in LaTeX rather than in-axes titles.

The palette subset used here was checked against the adjacent-pair CVD and
normal-vision floors: worst adjacent deltaE 11.0 (deutan) and 24.2 (normal).
Orange falls below 3:1 contrast on white, which is why every series is labelled
directly rather than through a legend box.

Figures are written single-column at the elsarticle text width, not the
two-column IEEE width fig_style defaults to.

Usage:  python3 make_coherence_figures.py [--fig 1|2|3|all]
"""
import argparse, json, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import fig_style as F                                   # noqa: E402
import matplotlib.pyplot as plt                         # noqa: E402
import corruption_temporal as ct                        # noqa: E402
import exp_coherence as E                               # noqa: E402
import exp_prior_tiebreak as PT                         # noqa: E402
import gaze_math as gm                                  # noqa: E402
from exp_prior_detector import resub_functional         # noqa: E402
from unigaze_arm import SUBJ, roc_auc                   # noqa: E402

OUT = os.path.join(_HERE, "manuscript_coherence", "figs")
CACHE = os.path.join(_HERE, "manuscript_coherence", "figdata.json")
W = 5.4                                    # elsarticle single-column text width
LAM, PC, SEEDS = 1e-2, 0.2, 10
GRID = E.DELTAS_BASE + E.DELTAS_EXT
# Okabe-Ito, house order; colour is never the only channel
SERIES = [(F.BLUE, "o", "-"), (F.ORANGE, "s", "--"),
          (F.GREEN, "^", "-."), (F.VERMILLION, "D", ":")]


def cache(key, fn):
    d = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    if key not in d:
        d[key] = fn()
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        json.dump(d, open(CACHE, "w"), indent=1)
        print(f"  computed and cached: {key}", flush=True)
    return d[key]


def floors(feats, funcs, pc, loc, scorer=None, res_f=None, pop=None):
    out = []
    for kap in E.KAPPAS:
        aucs = []
        for d in GRID:
            v, p = [], []
            for s in SUBJ:
                Fm, Y = feats[s]
                for sd in range(SEEDS):
                    c = ct.apply_coherent(Y, s, d, pc, sd, kap, locality=loc)
                    Yc = c["corrupted_labels"]
                    if scorer == "resub":
                        r = gm.angular_error_deg(res_f[s] @ Yc, Yc)
                    elif scorer == "prior":
                        r = gm.angular_error_deg(Yc, pop[s])
                    else:
                        r = E.loo_resid(funcs[s], Yc)
                    v.extend(np.atleast_1d(r).tolist())
                    p.extend(c["is_corrupted"].tolist())
            aucs.append(roc_auc(v, p))
        out.append(E.crossing(GRID, aucs, 0.7)[0])
    return out


def label_end(ax, x, y, text, colour, dy=0.0):
    ax.annotate(text, xy=(x, y), xytext=(4, dy), textcoords="offset points",
                color=colour, fontsize=8, va="center", ha="left")


def fig1():
    """c as a function of kappa, four model configurations."""
    d = json.load(open(os.path.join(_HERE, "c_vs_kappa.json")))
    fig, ax = plt.subplots(figsize=(W, 3.0), constrained_layout=True)
    for (name, rec), (col, mk, ls) in zip(d.items(), SERIES):
        ys = [rec["c"][str(k)] for k in E.KAPPAS]
        ax.plot(E.KAPPAS, ys, color=col, marker=mk, linestyle=ls, label=name)
        label_end(ax, E.KAPPAS[-1], ys[-1], f" {name}", col)
    ax.axhspan(1.02, 1.12, color=F.GREY, alpha=0.18, lw=0)
    ax.annotate("published band", xy=(0.02, 1.07), fontsize=7.5,
                color=F.GREY, va="center")
    ax.set_xlabel(r"corruption coherence $\kappa$")
    ax.set_ylabel(r"$c=\delta_{\mathrm{floor}}@0.7\,/\,\hat{e}$")
    ax.set_xlim(-0.03, 1.42); ax.set_xticks(E.KAPPAS)
    F.save(fig, "fig_c_vs_kappa") if hasattr(F, "save") else None
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_c_vs_kappa.{ext}"))
    plt.close(fig); print("  wrote fig_c_vs_kappa")


def fig2(feats, funcs):
    """Locality x coherence: the interaction."""
    sc = cache("loc_scattered", lambda: floors(feats, funcs, PC, "scattered"))
    cl = cache("loc_clustered", lambda: floors(feats, funcs, PC, "clustered"))
    fig, ax = plt.subplots(figsize=(W, 3.0), constrained_layout=True)
    for ys, nm, (col, mk, ls) in ((sc, "scattered", SERIES[0]),
                                  (cl, "clustered", SERIES[1])):
        ax.plot(E.KAPPAS, ys, color=col, marker=mk, linestyle=ls)
        label_end(ax, E.KAPPAS[-1], ys[-1], f" {nm}", col)
    ax.set_xlabel(r"corruption coherence $\kappa$")
    ax.set_ylabel(r"$\delta_{\mathrm{floor}}@0.7$  (deg)")
    ax.set_xlim(-0.03, 1.38); ax.set_xticks(E.KAPPAS)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_interaction.{ext}"))
    plt.close(fig); print("  wrote fig_interaction")


def fig3(feats, funcs):
    """Detector floors: residual scorers move with kappa, the prior does not."""
    W_, full = PT.population_maps(1.0)
    res_f = {s: resub_functional(feats[s][0], LAM) for s in SUBJ}
    pop = {s: PT.predict(W_[s], feats[s][0]) for s in SUBJ}
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.9), constrained_layout=True,
                             sharey=True)
    for ax, pc in zip(axes, (0.2, 0.4)):
        for sc, nm, (col, mk, ls) in (("resub", "resubstitution", SERIES[0]),
                                      ("loo", "leave-one-out", SERIES[1]),
                                      ("prior", "population prior", SERIES[2])):
            ys = cache(f"det_{pc}_{sc}",
                       lambda sc=sc, pc=pc: floors(feats, funcs, pc, "clustered",
                                                   scorer=sc, res_f=res_f, pop=pop))
            xs = [k for k, y in zip(E.KAPPAS, ys) if y]
            yy = [y for y in ys if y]
            ax.plot(xs, yy, color=col, marker=mk, linestyle=ls)
            if pc == 0.4:
                label_end(ax, xs[-1], yy[-1], f" {nm}", col)
            if len(yy) < len(ys):
                ax.annotate("no crossing", xy=(E.KAPPAS[len(yy)], yy[-1]),
                            xytext=(2, 6), textcoords="offset points",
                            fontsize=7, color=col)
        ax.set_xlabel(r"$\kappa$"); ax.set_xticks(E.KAPPAS)
        ax.set_title(rf"$p_c={pc}$", fontsize=9)
    axes[0].set_ylabel(r"$\delta_{\mathrm{floor}}@0.7$  (deg)")
    axes[1].set_xlim(-0.03, 1.75)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_detector.{ext}"))
    plt.close(fig); print("  wrote fig_detector")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fig", default="all")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    feats = E.load_feats("unigaze", "L0")
    funcs = {s: E.loo_functionals(feats[s][0], LAM) for s in SUBJ}
    if a.fig in ("1", "all"): fig1()
    if a.fig in ("2", "all"): fig2(feats, funcs)
    if a.fig in ("3", "all"): fig3(feats, funcs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
