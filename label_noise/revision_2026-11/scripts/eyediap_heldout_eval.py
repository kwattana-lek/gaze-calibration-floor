"""Re-evaluate the 14 surviving EYEDIAP LOPO checkpoints with the fixed Table-2 protocol.
cross = MAE on all held-out frames (no adaptation); personalized = adapt on the K=9 support
(calibration_split seed 42, as in every experiment) with backbone_api.adapt defaults
(full net, lr 1e-5, 20 steps, BN frozen), MAE on the remaining frames. Read-only on models/."""
import os, sys, json, numpy as np
AG = os.getcwd(); sys.path[:0] = [os.path.join(AG, "label_noise"), os.path.join(AG, "scripts"), AG]
import backbone_api as api, gaze_math as gm, train_backbone as tb
S = os.environ["GAZE_SUBJECTS"].split(","); rows = {}
def mae(m, X, V):
    return float(np.concatenate([gm.angular_error_vec_deg(api.predict(m, X[i:i+500]), V[i:i+500]) for i in range(0, len(X), 500)]).mean())
for s in S:
    bb = api.load_backbone(s, ckpt_path=os.path.join(AG, "models", f"backbone_mpiifacegaze_{s}.pt"))
    X, Y = tb.load_images(s, None, np.random.default_rng(42)); n = len(X)
    V = gm.pitchyaw_to_vector(gm.vector_to_pitchyaw(Y).reshape(-1, 2))
    sup = np.asarray(api.calibration_split(n, 9, seed=42)[0]); rest = np.setdiff1d(np.arange(n), sup)
    m = api.adapt(bb, X[sup], gm.vector_to_pitchyaw(Y[sup]).reshape(-1, 2))
    rows[s] = dict(n=n, cross_all=mae(bb, X, V), base_eval=mae(bb, X[rest], V[rest]), personalized=mae(m, X[rest], V[rest]))
    print(s, {k: round(v, 3) for k, v in rows[s].items()}, flush=True)
mean = {k: float(np.mean([r[k] for r in rows.values()])) for k in ("cross_all", "base_eval", "personalized")}
print("MEAN", {k: round(v, 3) for k, v in mean.items()}, "improved", sum(r["personalized"] < r["base_eval"] for r in rows.values()), "/", len(rows))
json.dump(dict(rows=rows, mean=mean), open(os.path.join(AG, "label_noise", "revision_2026-11", "results", "eyediap_heldout_eval.json"), "w"), indent=1)
