"""Step 4 runtime: wall-clock per session of the resubstitution and LOO scorers on this machine (CPU), L0, 3 subjects x 5 repeats,
after one warm-up. Writes results/04_runtime.json."""
import json, os, sys, time, platform, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_transfer_run as G
out = {"machine": platform.platform(), "processor": platform.processor(), "torch": torch.__version__,
       "threads": torch.get_num_threads(), "device": "cpu", "subjects": ["p00", "p05", "p10"], "repeats": 5}
t_res, t_loo, t_ad = [], [], []
for s in out["subjects"]:
    S = G.Subj(s); bb = G.load_bb("L0", s); obs, _, _ = S.corrupt(0.2, 0)
    G.sc.small_loss_scores(bb, S.sup_imgs, obs)  # warm-up
    for _ in range(out["repeats"]):
        t = time.perf_counter(); G.sc.small_loss_scores(bb, S.sup_imgs, obs); t_res.append(time.perf_counter() - t)
        t = time.perf_counter(); G.lr.loo_residuals(bb, S.sup_imgs, obs); t_loo.append(time.perf_counter() - t)
        t = time.perf_counter(); G.api.adapt(bb, S.sup_imgs, obs); t_ad.append(time.perf_counter() - t)
f = lambda v: {"median_s": round(float(np.median(v)), 3), "p10_s": round(float(np.percentile(v, 10)), 3), "p90_s": round(float(np.percentile(v, 90)), 3), "n": len(v)}
out.update(resub=f(t_res), loo=f(t_loo), adapt_only=f(t_ad))
out["loo_over_resub"] = round(out["loo"]["median_s"] / out["resub"]["median_s"], 2)
json.dump(out, open(os.path.join(G.REV, "results", "04_runtime.json"), "w"), indent=1); print(json.dumps(out, indent=1))
