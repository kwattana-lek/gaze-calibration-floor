import json, os, numpy as np
from collections import defaultdict
AG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
recs=[json.loads(l) for l in open(os.path.join(AG,"label_noise/exp_a_records.jsonl"))]
sigma={r["subject"]: r["calib_full_mae"] for r in json.load(open(os.path.join(AG,"experiments/lopo_backbone/results_regen_20260809.json")))}
DELTAS=[3,4,5,6,8,10]
def roc_auc(sc,pos):
    sc=np.asarray(sc,float);pos=np.asarray(pos,bool);ok=~np.isnan(sc);sc,pos=sc[ok],pos[ok]
    a,b=int(pos.sum()),int((~pos).sum())
    if a==0 or b==0: return np.nan
    u,inv,cnt=np.unique(sc,return_inverse=True,return_counts=True);cs=np.cumsum(cnt);av=(cs-cnt+cs+1)/2.0
    return float((av[inv][pos].sum()-a*(a+1)/2)/(a*b))
def cross(ds,a,level=0.7):
    d=np.array(ds,float);a=np.array(a,float);o=np.argsort(d);d,a=d[o],a[o]
    if np.all(a<level): return None,"never"
    if a[0]>=level: return d[0],"<=min"
    for i in range(1,len(d)):
        if a[i-1]<level<=a[i]:
            t=(level-a[i-1])/(a[i]-a[i-1]); return float(d[i-1]+t*(d[i]-d[i-1])),"ok"
    return None,"na"
# per-subject AUC vs delta (random_offset, pooled seeds+pc); + corrupted counts
subj_auc={}; subj_ncorr={}
for s in sorted(sigma):
    au=[]; nc=[]
    for d in DELTAS:
        sel=[r for r in recs if r["subject"]==s and r["model"]=="random_offset" and r["delta"]==d]
        y=[r["is_corrupted"] for r in sel]; sc=[r["score_loo"] for r in sel]
        au.append(roc_auc([np.nan if v is None else v for v in sc],y)); nc.append(int(sum(y)))
    subj_auc[s]=au; subj_ncorr[s]=nc
print("=== per-subject: sigma, corrupted@δ (min-max), δ@AUC=0.7 (raw) and (δ/σ)@0.7 ===")
print(f"{'s':4s} {'sigma':>6s} {'nc_range':>9s} {'d@0.7':>7s} {'(d/σ)@0.7':>10s}")
raw=[]; ratio=[]; sig_used=[]
for s in sorted(sigma, key=lambda x: sigma[x]):
    c,st=cross(DELTAS,subj_auc[s])
    ncr=f"{min(subj_ncorr[s])}-{max(subj_ncorr[s])}"
    if c is None:
        print(f"{s:4s} {sigma[s]:6.2f} {ncr:>9s} {'  '+st:>7s} {'-':>10s}"); continue
    raw.append(c); ratio.append(c/sigma[s]); sig_used.append(sigma[s])
    print(f"{s:4s} {sigma[s]:6.2f} {ncr:>9s} {c:7.2f} {c/sigma[s]:10.3f}")
raw=np.array(raw); ratio=np.array(ratio)
print(f"\nsubjects with interior 0.7 crossing: {len(raw)}/15")
print(f"RAW  δ@0.7 :  mean {raw.mean():.2f}°  std {raw.std(ddof=1):.3f}°  CV {raw.std(ddof=1)/raw.mean():.3f}  range [{raw.min():.2f},{raw.max():.2f}]")
print(f"σ-NORM (δ/σ):  mean {ratio.mean():.3f}   std {ratio.std(ddof=1):.4f}   CV {ratio.std(ddof=1)/ratio.mean():.3f}  range [{ratio.min():.3f},{ratio.max():.3f}]")
print(f"collapse: CV drops {raw.std(ddof=1)/raw.mean():.3f} -> {ratio.std(ddof=1)/ratio.mean():.3f}  ({'YES smaller' if ratio.std(ddof=1)/ratio.mean()<raw.std(ddof=1)/raw.mean() else 'NO, not smaller'})")
r=np.corrcoef(sig_used,raw)[0,1]; print(f"corr(sigma, δ@0.7) = {r:+.3f}  (collapse predicts strong positive)")

# --- sigma terciles (robust) ---
print("\n=== σ-terciles: mean LOO AUC vs δ, and δ@0.7 per tercile ===")
order=sorted(sigma, key=lambda x: sigma[x]); ter=[order[:5],order[5:10],order[10:]]
for gi,g in enumerate(ter):
    au=[np.nanmean([subj_auc[s][i] for s in g]) for i in range(len(DELTAS))]
    c,_=cross(DELTAS,au); smean=np.mean([sigma[s] for s in g])
    print(f"T{gi+1} (σ~{smean:.2f}, {g}): AUC={[round(x,3) for x in au]}  δ@0.7={c if c else 'na'}  (δ/σ)@0.7={c/smean if c else float('nan'):.3f}")

# --- §6.4 severity anchoring at 15 subjects (structured) ---
print("\n=== §6.4 severity anchoring @15 subjects (structured donor-swap displacements) ===")
off=np.array([r["actual_offset_deg"] for r in recs if r["model"]=="structured_swap" and r["is_corrupted"]])
pcts=[5,10,25,50,75,90,95]
print("displacement percentiles (deg):", {f"p{p}":round(float(np.percentile(off,p)),2) for p in pcts})
floor=raw.mean()  # 15-subj LOO δ@0.7 (raw mean)
print(f"fraction of structured displacements below LOO δ@0.7 (~{floor:.2f}°): {100*(off<floor).mean():.1f}%  (n={off.size})")
