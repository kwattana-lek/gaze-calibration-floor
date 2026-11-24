"""Does the L0 e_hat gap (Exp A 3.84 vs sweep 3.43) come from adaptation steps or support split?
Clean-support LOO residual (pc=0) for 4 subjects x {split seed 42, split seed 0} x {20, 150 steps}."""
import os,sys,json,numpy as np
AG=os.getcwd(); sys.path[:0]=[os.path.join(AG,"label_noise"),os.path.join(AG,"scripts"),AG]
import backbone_api as api, loo_residual as lr
out={}
for s in ["p00","p03","p07","p11"]:
    bb=api.load_backbone(s,ckpt_path=os.path.join(AG,"models_sigma","L0",f"backbone_mpiifacegaze_{s}.pt"))
    for sp in (42,0):
        sup=api.load_subject_support(s,K=9,seed=sp)
        for st in (20,150):
            v=float(np.mean(lr.loo_residuals(bb,sup["images"],sup["labels"],steps=st))); out[f"{s}|split{sp}|{st}"]=v; print(s,sp,st,round(v,3),flush=True)
json.dump(out,open(os.path.join(AG,"label_noise","revision_2026-11","results","check_steps_vs_support.json"),"w"),indent=1)
for sp in (42,0):
    for st in (20,150): print("mean split",sp,"steps",st, round(np.mean([v for k,v in out.items() if f"split{sp}|{st}" in k]),3))
