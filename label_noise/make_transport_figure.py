#!/usr/bin/env python3
# make_transport_figure.py -- the one figure of the transport section.
"""c(kappa) on three corpora with bootstrap bands, plus each corpus's ratio to MPIIFaceGaze.

Style follows make_coherence_figures.py: fig_style.py's Okabe-Ito palette, colour
never the only channel (each corpus also carries its own marker and line style),
direct labels rather than a legend box, vector PDF plus 300 dpi PNG, elsarticle
single-column width.

Two panels because the figure answers two different questions and a reader should
not have to infer the second from the first. Left: c(kappa) per corpus, which
shows the DIRECTION replicating. Right: the ratio between corpora with its
interval, which shows how much the data can actually say about their EQUALITY --
the point of it is the width of the band, not the closeness of the line to 1.

Recomputed from cached features via the same functions exp_transport_ci uses;
reads no summary JSON.

Usage:  python3 make_transport_figure.py
"""
import json, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import fig_style as F                       # noqa: E402
import matplotlib.pyplot as plt             # noqa: E402
import exp_coherence as E                   # noqa: E402
import exp_transport_ci as TC               # noqa: E402
from exp_transport import load              # noqa: E402

OUT = os.path.join(_HERE, "manuscript_coherence", "figs")
W, B = 5.4, 2000
SUBJ_A = [f"p{i:02d}" for i in range(15)]
SUBJ_E = "e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e14,e15,e16".split(",")   # EYEDIAP, 14 participants


def subj_b():
    p = os.path.join(_HERE, "..", "data", "RTGENE_Normalized", "participants.json")
    if not os.path.exists(p):
        sys.exit("ABORT: participants.json missing -- run prepare_rtgene_labels.py.")
    return json.load(open(p))["unique_participants"]


def arm(feat, subj, rng):
    clean, cells = TC.precompute(load(feat, subj), subj)
    _, pt = TC.one_replicate(subj, clean, cells)
    reps = [TC.one_replicate([subj[i] for i in rng.integers(0, len(subj), len(subj))],
                             clean, cells) for _ in range(B)]
    return pt, reps


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    A = arm("unigaze_features", SUBJ_A, rng)
    Bm = arm("rtgene_unigaze_features", subj_b(), rng)
    Em = arm("eyediap_unigaze_features", SUBJ_E, rng)
    ks = list(E.KAPPAS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, 2.5), constrained_layout=True)

    # Direct labels rather than a legend box (house style). Placed at kappa=0.5,
    # where the two curves are furthest apart -- at either end they converge and
    # a label there sits on top of the other series.
    mid = ks[len(ks) // 2]
    for (pt, reps), col, mk, ls, name, off in (
            (A, F.BLUE, "o", "-", "MPIIFaceGaze", (-4, 11)),
            (Bm, F.VERMILLION, "s", "--", "RT-GENE", (-4, -17)),
            (Em, F.GREEN, "D", ":", "EYEDIAP", (30, -6))):
        c = [pt[k][1] for k in ks]
        band = [TC.pct([r[1][k][1] for r in reps]) for k in ks]
        ax1.fill_between(ks, [b[0] for b in band], [b[1] for b in band],
                         color=col, alpha=0.13, linewidth=0)
        ax1.plot(ks, c, color=col, marker=mk, linestyle=ls)
        ax1.annotate(name, (mid, pt[mid][1]), xytext=off, textcoords="offset points",
                     color=col, fontsize=8, ha="center")
    ax1.set_xlabel(r"coherence $\kappa$")
    ax1.set_ylabel(r"$c = \delta_{\mathrm{floor}} / \hat{e}$")
    ax1.set_xticks(ks)

    ax2.axhline(1.0, color=F.GREY, linewidth=0.8, linestyle=":")
    rats, rbands = {}, {}
    for Xm, col, mk, ls, name, off in ((Bm, F.VERMILLION, "s", "--", "RT-GENE", (-4, 9)),
                                       (Em, F.GREEN, "D", ":", "EYEDIAP", (-4, -14))):
        rat = [Xm[0][k][1] / A[0][k][1] for k in ks]
        rband = [TC.pct([rb[1][k][1] / ra[1][k][1] for ra, rb in zip(A[1], Xm[1])
                         if ra[1][k][1] and rb[1][k][1]]) for k in ks]
        ax2.fill_between(ks, [b[0] for b in rband], [b[1] for b in rband],
                         color=col, alpha=0.13, linewidth=0)
        ax2.plot(ks, rat, color=col, marker=mk, linestyle=ls)
        ax2.annotate(name, (ks[-1], rat[-1]), xytext=off, textcoords="offset points",
                     color=col, fontsize=8, ha="right")
        rats[name], rbands[name] = rat, rband
    rat = rats["RT-GENE"] + rats["EYEDIAP"]; rband = rbands["RT-GENE"] + rbands["EYEDIAP"]
    ax2.set_xlabel(r"coherence $\kappa$")
    ax2.set_ylabel(r"ratio of $c$ to MPIIFaceGaze")
    ax2.set_xticks(ks)
    # mid-grid: the curve dips well below 1 there, so the reference line is the
    # only thing at y=1 and the label sits in clear space rather than on a marker
    ax2.annotate("equal", (mid, 1.0), xytext=(0, 5), textcoords="offset points",
                 color=F.GREY, fontsize=8, ha="center")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_transport.{ext}"))
    plt.close(fig)
    print(f"  wrote figs/fig_transport.pdf + .png")
    print(f"  ratio {min(rat):.3f}--{max(rat):.3f}; widest band "
          f"{max(b[1]-b[0] for b in rband):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
