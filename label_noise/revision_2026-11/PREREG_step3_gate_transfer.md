# Pre-registration: gate transfer across accuracy levels L0–L3 (Step 3)

Written 2026-11-23, **before any L1–L3 resubstitution score has been computed**.
No resubstitution residual exists for L1–L3 anywhere in the project
(`sigma_sweep_records.jsonl` holds LOO and source residuals only). The SHA-256 of
this file is recorded in `logs/prereg_hash.txt` at creation; the runner refuses to start if it changes.

## Question
Does the optimal resubstitution-residual threshold scale with the model's own error ê
(H_scale), or does a fixed angle transfer (H_fixed)?

## Fixed design
- Checkpoints: `models_sigma/{L0,L1,L2,L3}` (L0 is byte-identical to `models/`, the Exp A checkpoints).
- Adaptation: `backbone_api.adapt` defaults: full network, lr 1e-5, **20 steps**, BN frozen,
  i.e. the recipe that produced τ = 3.42° (Exp A / seed extension). Not the 150-step sweep recipe.
- Support: `load_subject_support(s, K=9, seed=42)`; 15 subjects.
- Corruption: structured donor swap, `corruption.apply_corruption`, seeds 0–9, p_c ∈ {0.1, 0.2, 0.4},
  RNG key `SHA-256(subject|structured_swap|None|pc|seed)`, which is identical to Exp A / seed extension,
  so the L0 rows are a reproduction check against existing records.
- Scores per frame: resubstitution residual and LOO residual.
- Per subject and level (clean support, p_c = 0): clean LOO residuals, clean resubstitution residuals,
  held-out personalized MAE (adapt on clean support, evaluate on a fixed 1000-frame non-support subsample,
  `RandomState(1000)`), and cross-subject MAE on the same frames.

## ê definitions (all reported; one is primary)
- **ê_clean (primary):** the mean over subjects of the mean clean-support LOO residual (p_c = 0 fit, 20 steps).
- ê_dep (deployment-computable): the mean over sessions and subjects of the mean LOO residual over all nine
  frames of the *observed, contaminated* support, per p_c. A deployment can compute this without clean labels.
- ê_heldout: held-out personalized MAE. Reported only; a deployment cannot compute it.

## Policies (frozen)
- **A, fixed:** τ = 3.42°.
- **B, scaled (primary):** τ = 0.96 · ê_clean(L). 0.96 is the paper's ratio (3.42/3.57).
- B′, scaled with the ratio re-derived under this protocol: τ = (3.42 / ê_clean(L0)) · ê_clean(L).
  L0 is the reference level, so B′ uses no information from L1–L3 outcomes.
- B_dep: τ = 0.96 · ê_dep(L, p_c).
- **C, oracle (reference only):** τ minimizing expected cost on that level's own records, pooled over subjects.
- D, tuned in place: LOPO Youden's J on that level (as in Table 11).

Cost per session = c_fp · 9(1−p_c)·FPR + c_fn · 9p_c·(1−recall), with c_fp = 1.
The primary weight is **c_fn = 4**, the middle of the paper's 3–5.5 band. Weights {1, 2, 4, 8} are also reported.

## Metrics
Recall, FPR, precision, F1, ROC-AUC (threshold-free), recaptures per session, and expected cost.
These are computed per subject; means and 95% bootstrap CIs are taken over subjects (B = 2000), and differences
are paired on identical draws. Downstream: held-out MAE after each policy, where flagged frames are replaced by
clean substitutes (non-support, non-eval frames, RNG key `subject|subst|pc|seed`) and the model is re-adapted.
Naive (no gate) and true-corrupted-replaced oracle are reported for reference.

## Predictions and decision rule
- H_fixed predicts τ_C ≈ 3.42° at every level. H_scale predicts τ_C ∝ ê_clean.
- **Scaling supported** if B is within 0.02 recall *and* 0.02 FPR of D at every level, *and* A is not
  (it fails at L3 or L2 by more than that).
- **Fixed supported** (positioning Outcome C) if A is within the tolerance at every level, and B is not better.
- **Both within tolerance:** τ is insensitive to ê over this range, and "self-calibrating" is not warranted.
- **Neither:** the gate is model-specific, and the paper keeps only the floor claims.
- L3 (ê ≈ 8.9° under 150 steps) is the decisive level. If its resubstitution ROC-AUC is near chance, the
  comparison is uninformative there and will be reported as such.
