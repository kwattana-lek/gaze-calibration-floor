import os,sys,time,json
AG=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0,os.path.join(AG,"label_noise"))
import numpy as np, backbone_api as api, exp_a_scorers as sc, loo_residual as lr, corruption as corr, hashlib
dev=api.get_device(); print("device",dev)
def rng(sid,model,delta,pc,seed):
    k=f"{sid}|{model}|{delta}|{pc}|{seed}".encode(); return np.random.RandomState(int.from_bytes(hashlib.sha256(k).digest()[:8],"big")%(2**32))
# reproduce exp_a record: p00 seed0 random_offset delta3 pc0.1 (Exp A: models/, 20 steps)
m=api.load_backbone("p00"); s=api.load_subject_support("p00",K=9,seed=42)
rec=[json.loads(l) for l in open(os.path.join(AG,"label_noise","exp_a_records.jsonl"))][:9]
c=corr.apply_corruption(s["labels"],model="random_offset",rng=rng("p00","random_offset",3,0.1,0),pc=0.1,delta_deg=3,pool_labels=s["pool_labels"])
t=time.time(); sm=sc.small_loss_scores(m,s["images"],c["corrupted_labels"]); t1=time.time()-t
t=time.time(); lo=lr.loo_residuals(m,s["images"],c["corrupted_labels"]); t2=time.time()-t
print("mask match", list(c["is_corrupted"])==[r["is_corrupted"] for r in rec])
print("resub new ", np.round(sm,4)); print("resub rec ", np.round([r["score_small_loss"] for r in rec],4))
print("loo new   ", np.round(lo,3)); print("loo rec   ", np.round([r["score_loo"] for r in rec],3))
print(f"time resub {t1:.1f}s  loo {t2:.1f}s")
