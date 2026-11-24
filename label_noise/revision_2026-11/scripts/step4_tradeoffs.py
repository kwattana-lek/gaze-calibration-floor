"""Step 4: LOO vs resubstitution gate, detection/cost trade-offs.

Conditions
  controlled : Exp A random offset, delta in {3,4,5,6} (near floor) and {8,10}; L0; seeds 0-2.
  structured : donor swap from revision gate_transfer_scores.jsonl; L0-L3; seeds 0-9.
Thresholds are chosen per scorer by LOPO (on the other subjects) to minimise expected cost
  cost/session = 9(1-pc)FPR + w * 9 pc (1-recall),  c_fp = 1, w in {1,2,4,8}
and evaluated on the held-out subject. Paired LOO - resub differences, bootstrap over subjects.
"""
import collections, csv, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REV = os.path.dirname(HERE); LN = os.path.dirname(REV)
OUT = os.path.join(REV, "results"); RS = np.random.RandomState(0); B = 2000
WS = [1, 2, 4, 8]; GRID = np.round(np.arange(0.0, 15.0001, 0.02), 2)


def rates(score, lab, tau):
    f = score >= tau; c = lab
    rec = f[c].mean() if c.any() else np.nan; fpr = f[~c].mean() if (~c).any() else np.nan
    return rec, fpr


def subj_curves(by):  # by[s] = (score, lab) -> per-subject recall/fpr on GRID
    out = {}
    for s, (sc, lb) in by.items():
        sc = np.asarray(sc); lb = np.asarray(lb, bool)
        f = sc[None, :] >= GRID[:, None]
        rec = f[:, lb].mean(1) if lb.any() else np.full(len(GRID), np.nan)
        fpr = f[:, ~lb].mean(1) if (~lb).any() else np.full(len(GRID), np.nan)
        out[s] = (rec, fpr)
    return out


def cost(rec, fpr, pc, w): return 9 * (1 - pc) * fpr + w * 9 * pc * (1 - rec)


def lopo(curves, pc, w):
    S = sorted(curves); res = {}
    for s in S:
        R = np.nanmean([curves[t][0] for t in S if t != s], 0); F = np.nanmean([curves[t][1] for t in S if t != s], 0)
        i = int(np.nanargmin(cost(R, F, pc, w))); rec, fpr = curves[s][0][i], curves[s][1][i]
        res[s] = dict(tau=float(GRID[i]), rec=rec, fpr=fpr, cost=cost(rec, fpr, pc, w),
                      recap=9 * (1 - pc) * fpr + 9 * pc * rec, missed=9 * pc * (1 - rec))
    return res


def load():
    data = collections.defaultdict(lambda: collections.defaultdict(lambda: {"LOO": ([], []), "resub": ([], [])}))
    for l in open(os.path.join(LN, "exp_a_records.jsonl")):
        r = json.loads(l)
        if r["model"] != "random_offset": continue
        cond = "controlled δ3–6" if r["delta"] <= 6 else "controlled δ8–10"
        for k, nm in (("score_loo", "LOO"), ("score_small_loss", "resub")):
            d = data[("L0", cond, r["pc"])][r["subject"]][nm]; d[0].append(r[k]); d[1].append(r["is_corrupted"])
    p = os.path.join(OUT, "gate_transfer_scores.jsonl")
    for l in open(p):
        r = json.loads(l)
        if r["kind"] != "cell": continue
        for nm, k in (("LOO", "loo"), ("resub", "resub")):
            d = data[(r["level"], "structured", r["pc"])][r["subject"]][nm]; d[0].extend(r[k]); d[1].extend(r["is_corrupted"])
    return data


def main():
    data = load(); rows = []
    for (lvl, cond, pc), subj in sorted(data.items()):
        if len(subj) < 15: continue                       # only complete levels
        cur = {nm: subj_curves({s: v[nm] for s, v in subj.items()}) for nm in ("LOO", "resub")}
        for w in WS:
            res = {nm: lopo(cur[nm], pc, w) for nm in cur}; S = sorted(res["LOO"])
            dcost = np.array([res["LOO"][s]["cost"] - res["resub"][s]["cost"] for s in S])
            bs = [np.nanmean(dcost[RS.randint(0, len(S), len(S))]) for _ in range(B)]
            for nm in ("LOO", "resub"):
                m = lambda k: float(np.nanmean([res[nm][s][k] for s in S]))
                rows.append(dict(level=lvl, condition=cond, pc=pc, w=w, scorer=nm, tau_median=float(np.median([res[nm][s]["tau"] for s in S])),
                                 recall=m("rec"), fpr=m("fpr"), recaptures=m("recap"), missed=m("missed"), cost=m("cost"),
                                 dcost_LOO_minus_resub=float(np.nanmean(dcost)), dcost_lo=float(np.percentile(bs, 2.5)), dcost_hi=float(np.percentile(bs, 97.5))))
    with open(os.path.join(OUT, "04_cost_sensitivity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); [w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()}) for r in rows]
    for r in rows:
        if r["w"] == 4: print(f"{r['level']} {r['condition']:18s} pc={r['pc']} {r['scorer']:5s} τ={r['tau_median']:.2f} rec={r['recall']:.3f} fpr={r['fpr']:.3f} recap={r['recaptures']:.2f} missed={r['missed']:.2f} cost={r['cost']:.3f}  Δcost(LOO−resub)={r['dcost_LOO_minus_resub']:+.3f} [{r['dcost_lo']:+.3f},{r['dcost_hi']:+.3f}]")


if __name__ == "__main__":
    main()
