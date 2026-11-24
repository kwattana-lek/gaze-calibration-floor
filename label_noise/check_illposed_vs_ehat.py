#!/usr/bin/env python3
"""Does I at kappa=1 order with the arm's own ehat, WITHIN MPIIFaceGaze?

The EYEDIAP divergence (I above 1 at kappa=1, where MPIIFaceGaze and RT-GENE are
below it) has an explanation from Section 6.4: I is the corrupted subset's
within-fit residual over the clean subset's, a rigid rotation is exactly
consistent with a rotated mapping, so a more accurate model RAISES I. But across
corpora ehat is confounded with everything else. Here the corpus, subjects and
protocol are fixed and only the model changes, across a 2.3x range of ehat.
A negative correlation earns the mechanistic scoping; anything else does not.
"""
import sys, numpy as np
sys.path.insert(0, '.')
import exp_coherence as E, corruption_temporal as ct, exp_identifiability as ID
from unigaze_arm import SUBJ

LAM, SEEDS, DELTAS = 1e-2, 10, (6, 10, 16, 20)
ARMS = [("resnet", "L0", "ResNet-18 L0"), ("resnet", "L1", "ResNet-18 L1"),
        ("resnet", "L2", "ResNet-18 L2"), ("unigaze", None, "UniGaze ViT-H/14")]

print(f"{'arm':<18} {'ehat':>6}   " + "".join(f"{'d='+str(d):>8}" for d in DELTAS)
      + f"{'  mean I':>9}")
print("-" * 70)
rows = []
for arm, lvl, name in ARMS:
    feats = E.load_feats(arm, lvl or "L0")
    fu = {s: E.loo_functionals(feats[s][0], LAM) for s in SUBJ}
    eh = float(np.mean([np.mean(E.loo_resid(fu[s], feats[s][1])) for s in SUBJ]))
    Is = []
    for d in DELTAS:
        rs = []
        for s in SUBJ:
            F, Y = feats[s]
            for sd in range(SEEDS):
                c = ct.apply_coherent(Y, s, d, 0.2, sd, 1.0, locality="clustered")
                m = c["is_corrupted"]
                wc = ID.within(F[~m], Y[~m], LAM)
                wx = ID.within(F[m], c["corrupted_labels"][m], LAM)
                if wc and wx and wc > 0:
                    rs.append(wx / wc)
        Is.append(np.mean(rs) if rs else np.nan)
    rows.append((name, eh, Is))
    print(f"{name:<18} {eh:>6.2f}   " + "".join(f"{v:>8.2f}" for v in Is)
          + f"{np.mean(Is):>9.2f}")

eh = [r[1] for r in rows]; mi = [float(np.mean(r[2])) for r in rows]
r = float(np.corrcoef(eh, mi)[0, 1])
print(f"\ncorrelation(ehat, mean I at kappa=1) within corpus: r = {r:+.3f}")
print("Section 6.4 predicts NEGATIVE (higher ehat -> lower I)")
print("verdict:", "CONSISTENT with the mechanism -> scope by ehat" if r < -0.5
      else "NOT explained by ehat -> hold the claim to the two corpora")