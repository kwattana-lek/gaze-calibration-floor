import json, os, numpy as np
from collections import defaultdict
AG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    from scipy.stats import wilcoxon; HAVE=True
except Exception:
    HAVE=False
recs=[json.loads(l) for l in open(os.path.join(AG,"label_noise/exp_a_records.jsonl"))]

def roc_auc(sc,pos):
    sc=np.asarray(sc,float);pos=np.asarray(pos,bool);ok=~np.isnan(sc);sc,pos=sc[ok],pos[ok]
    a,b=int(pos.sum()),int((~pos).sum())
    if a==0 or b==0: return np.nan
    u,inv,cnt=np.unique(sc,return_inverse=True,return_counts=True);cs=np.cumsum(cnt);av=(cs-cnt+cs+1)/2.0
    return float((av[inv][pos].sum()-a*(a+1)/2)/(a*b))

def per_subject_auc(model, delta, pc, scorer):
    by=defaultdict(lambda:([], []))
    for r in recs:
        if r["model"]==model and r["delta"]==delta and abs(r["pc"]-pc)<1e-9:
            v=r.get("score_"+scorer); by[r["subject"]][0].append(np.nan if v is None else v); by[r["subject"]][1].append(r["is_corrupted"])
    return {s: roc_auc(a,b) for s,(a,b) in by.items()}

def paired(model, delta, pc, A, B):
    aa,bb=per_subject_auc(model,delta,pc,A),per_subject_auc(model,delta,pc,B)
    subs=sorted(set(aa)&set(bb))
    x=np.array([aa[s] for s in subs]);y=np.array([bb[s] for s in subs])
    m=~(np.isnan(x)|np.isnan(y)); x,y=x[m],y[m]; d=x-y
    dz=d.mean()/d.std(ddof=1) if len(d)>1 and d.std(ddof=1)>0 else np.nan
    if HAVE and len(d)>=1 and np.any(d!=0):
        try: p=wilcoxon(x,y,zero_method="wilcox").pvalue
        except Exception: p=np.nan
    else: p=np.nan
    return len(d), d.mean(), dz, p, x.mean(), y.mean()

print("Paired per-subject test (fold=subject, seeds pooled -> 27 frames/subject/cell).")
print("d_z=Cohen's d_z on paired AUC diffs; p=Wilcoxon signed-rank (scipy=%s).\n"%HAVE)
for A,B in [("loo","small_loss"),("loo","source")]:
    print(f"===== {A.upper()} vs {B} =====")
    print(f"{'cond':10s} {'pc':>4s} {'n':>3s} {'meanA':>6s} {'meanB':>6s} {'meanΔ':>7s} {'d_z':>6s} {'wilcoxon_p':>11s}")
    for cond,(model,delta) in [("delta=6",("random_offset",6)),("structured",("structured_swap",None))]:
        for pc in (0.1,0.2,0.4):
            n,md,dz,p,ma,mb=paired(model,delta,pc,A,B)
            ps = "   nan" if (p!=p) else f"{p:.4f}"
            print(f"{cond:10s} {pc:>4.1f} {n:>3d} {ma:>6.3f} {mb:>6.3f} {md:>+7.3f} {dz:>+6.2f} {ps:>11s}")
    print()
