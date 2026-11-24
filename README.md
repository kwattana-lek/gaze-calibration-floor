# When can a corrupted calibration frame be caught?

Code for the paper *When can a corrupted calibration frame be caught? A model-relative detectability floor and a
fixed-threshold gate for few-shot gaze personalization* (W. Kurdthongmee, M. E. E. Alahi, P. Kurdthongmee;
submitted to Image and Vision Computing).

This repository holds every script that produced the paper's numbers, tables and figures: backbone training, the
corruption simulator, the detectors, all experiment drivers, the statistical analyses and the revision analyses. It holds
**no data**: the gaze datasets must be obtained from their providers under their own licences, and the trained
checkpoints and per-frame record files are released separately (see *Checkpoints and records*).

## Licence

Code: MIT (see `LICENSE`). Datasets and third-party checkpoints remain under their providers' licences.

## Layout

| path | contents |
|---|---|
| `label_noise/` | corruption simulator (`corruption.py`), residual scorers (`loo_residual.py`, `exp_a_scorers.py`), backbone facade (`backbone_api.py`), experiment drivers (`exp_*.py`, `*_arm.py`), analyses and figure scripts |
| `label_noise/revision_2026-11/scripts/` | analyses added in the 2026-11 revision (resubstitution floor, gate transfer across backbones, deployment trade-offs, damage-ratio stability, per-cell deployment tables, protocol inventory) |
| `label_noise/revision_2026-11/PREREG_step3_gate_transfer.md` | the protocol of the gate-transfer test, frozen before it was run; its SHA-256 is in `logs/prereg_hash.txt` and the runner refuses to start if the file has changed |
| `scripts/` | backbone training (`train_backbone.py`) and detector utilities |
| `src/` | image-quality scores used for the image-corruption channel |

## Environment

Python 3.10–3.12. Versions used for the paper:

```bash
pip install -r requirements.txt
```

All commands run from the repository root with

```bash
export PYTHONPATH=.:scripts:label_noise PYTHONHASHSEED=0
```

## Data

| dataset | where to get it | notes |
|---|---|---|
| MPIIFaceGaze | from its authors (CC BY-NC-SA 4.0) | normalized with the GazeHub procedure to 112×112 face crops; point `GAZE_DATA_ROOT` at the normalized folder (default `data/MPIIFaceGaze_Normalized`) |
| EYEDIAP | from the Idiap Research Institute under its End-User Licence Agreement | **not redistributable**; `label_noise/prepare_eyediap_labels.py` rebuilds the labels from a licensee's own copy |
| RT-GENE | from its authors (CC BY-NC-SA 4.0) | `label_noise/prepare_rtgene.py`, `prepare_rtgene_labels.py` |
| UniGaze ViT-H/14 checkpoint | from its authors under their research licence | used by the frozen arms (`unigaze_arm*.py`) |

## Reproducing the results

1. **Backbones.** `scripts/train_backbone.py` trains the leave-one-person-out ResNet-18 backbones; the four accuracy
   levels L0–L3 differ only in epochs and frames per subject (Section 8 of the paper).
2. **Experiments.** Each driver in `label_noise/` writes an append-only, resume-safe record file. Every corruption
   draw is keyed by a SHA-256 hash of (subject, corruption model, severity, contamination rate, seed), so a re-run
   reproduces the same masks and displacements. Every simulated-corruption experiment uses
   `corruption.draw_corruption_mask` (each frame corrupted independently with probability `p_c`, at least one frame
   kept clean); the real-lapse experiment (`exp_real_lapse.py`) instead substitutes exactly two of the nine frames
   per trial.
3. **Analyses.** The analysis scripts read the record files and regenerate every reported number, table and figure.
   For the revision analyses, in dependency order:

```bash
python label_noise/revision_2026-11/scripts/gate_transfer_run.py score
python label_noise/revision_2026-11/scripts/step3_thresholds.py
python label_noise/revision_2026-11/scripts/gate_transfer_run.py mae --thresholds label_noise/revision_2026-11/results/step3_thresholds.json
python label_noise/revision_2026-11/scripts/step2_resub_floor.py
python label_noise/revision_2026-11/scripts/step3_analysis.py
python label_noise/revision_2026-11/scripts/step3_mae_analysis.py
python label_noise/revision_2026-11/scripts/step4_tradeoffs.py
python label_noise/revision_2026-11/scripts/step7_damage_ratio_denominator.py
python label_noise/revision_2026-11/scripts/step9_deployment_by_level.py
python label_noise/revision_2026-11/scripts/step11_protocol_inventory.py
```

**Reproducibility checks done before submission.** Re-running the ten analysis scripts above from the record files
regenerated all 19 of their outputs byte-for-byte, figures included. Re-running the gate-transfer scoring from the
checkpoints for a sample of subjects, backbones, seeds and contamination rates reproduced every corruption mask,
displacement and per-frame residual exactly (CPU, PyTorch 2.5.1). GPU or MPS runs and other library versions can
differ in the last floating-point digits.

## Checkpoints and records

The trained checkpoints (about 2.5 GB) and the per-frame record files for MPIIFaceGaze and RT-GENE will be deposited
with a DOI on acceptance and linked here. Records derived from EYEDIAP cannot be redistributed under its licence;
licensees can regenerate them with the drivers above.

## All scripts

| file | purpose (first line of its docstring) |
|---|---|
| `label_noise/analyze_finetuned_coherence.py` | The sweep records carry an e_hat measured as the mean clean-frame LOO residual |
| `label_noise/analyze_floor.py` | Regenerate report sections 2-4 for a given floor-sweep jsonl. |
| `label_noise/analyze_sigma.py` | Analyses 1-7 on the emitted split-42 clean residuals (P1 support-LOO primary, |
| `label_noise/arm_shape_compare.py` | The finding this script exists to test. |
| `label_noise/backbone_api.py` | Phase-1 backbone facade, ported to run INSIDE the ES 112px MPIIFaceGaze pipeline. |
| `label_noise/check_eyediap_target_frame.py` | prepare_eyediap_labels.py verifies the eyeball/head geometry by reprojection. It |
| `label_noise/check_illposed_vs_ehat.py` | Does I at kappa=1 order with the arm's own ehat, WITHIN MPIIFaceGaze? |
| `label_noise/corruption.py` | Label-corruption simulator for Phase 1 (HANDOFF §4.2). |
| `label_noise/corruption_temporal.py` | Every corruption model in corruption.py displaces each corrupted frame |
| `label_noise/cpe_by_level.py` | Contamination at rate p_c is a MIXTURE, not a rescaling. On the calibration set |
| `label_noise/cpe_floor_diagnostic.py` | unc_calib_probe.py --selftest reports calibrated CPE 0.076-0.102 on WELL-SPECIFIED |
| `label_noise/cpe_mechanism.py` | Corrupted calibration labels inflate /residual/ on the calibration set. A |
| `label_noise/data_processing_rt_test.py` | Acquire the list of files |
| `label_noise/data_processing_rt_train.py` | Acquire the list of files |
| `label_noise/detector_lopo_save.py` | Prereq (c): reproduce detector_lopo's 0.9512 AND persist per-fold weights. |
| `label_noise/dump_lopo_predictions.py` | The population-prior scorer needs, for every support frame, what a model that |
| `label_noise/emit_sigma.py` | Emit clean (p_c=0) per-frame residuals on the DETECTION support |
| `label_noise/exp_a_analyze.py` | Experiment A analysis -> logs/EXP_A_REPORT.md. |
| `label_noise/exp_a_family_stats.py` | Experiment A's headline claim is that scorers separate by the kind of quantity |
| `label_noise/exp_a_run.py` | Experiment A sweep: six per-frame scorers over 15 subjects, deltas |
| `label_noise/exp_a_scorers.py` | Experiment A — six per-frame corruption scorers (higher score = more likely |
| `label_noise/exp_b.py` | s0 backbone only, seeds 0-2, clean arm cached per subject. Per cell |
| `label_noise/exp_b_analyze.py` | Aggregate exp_b_records.csv. Tests the §3.2 pre-registered prediction (25cde1c): |
| `label_noise/exp_b_delta5.py` | Adds ONE severity level (channel a_rot, δ=5) to narrow the ratio-crossing bracket |
| `label_noise/exp_b_finegrid.py` | Adds severities 4.25, 4.5, 4.75 (channel a_rot) to MEASURE the damage-ratio |
| `label_noise/exp_c.py` | s0 backbone only, seeds 0-2, budgets B∈{0,1,2}. Scorer = resubstitution residual |
| `label_noise/exp_c_analyze.py` | Aggregate exp_c_records.csv. Oracle in the SAME table. Per cell: post-adaptation |
| `label_noise/exp_coherence.py` | The published floor (c = delta_floor@0.7 / e_hat ~ 1.03, invariant across two |
| `label_noise/exp_displacement.py` | Pure re-analysis, no retraining. Recover the realised angular displacement of |
| `label_noise/exp_finetuned_detector.py` | On the frozen arms the population prior beat both residual scorers at every |
| `label_noise/exp_identifiability.py` | At p_c = 0.4 with kappa = 1 the ROC-AUC never sustains 0.7 out to delta = 50, and |
| `label_noise/exp_interaction_ci.py` | exp_transport.py answered whether c(kappa) transports, but it runs at |
| `label_noise/exp_isotonic_floor.py` | Crossing.  ROC-AUC is fitted as a MONOTONE function of severity by isotonic |
| `label_noise/exp_k_sweep.py` | Repeats the fine-tuned ResNet-18 L0 sweep at K = 5, 9 and 16 support frames with ONE |
| `label_noise/exp_operating_point.py` | Turn §6.5's ROC-AUC into a threshold. Pure re-analysis of existing residual |
| `label_noise/exp_paired_stats.py` |  |
| `label_noise/exp_prior_detector.py` | exp_prior_tiebreak.py asked: GIVEN the true partition, can a population prior say |
| `label_noise/exp_prior_tiebreak.py` | exp_identifiability.py shows that under coherent corruption the corrupted subset |
| `label_noise/exp_purification.py` | Does adaptation recover under contamination? Compare purified-LOO (drop the |
| `label_noise/exp_real_lapse.py` | EXPLORATORY. Declared as such in PREREG_eyediap.md section 4 (commit 86235d4), |
| `label_noise/exp_real_lapse_timesplit.py` | POST-HOC, and labelled so wherever it is reported. Written AFTER |
| `label_noise/exp_seed_extension.py` | Extend structured donor-swap AND random-offset δ=6 to seeds 3-9 (seeds 0-2 |
| `label_noise/exp_sigma_coherent.py` | The coherence result is established on four frozen arms, where every adaptation |
| `label_noise/exp_sigma_scaling.py` | per-subject AUC vs delta (random_offset, pooled seeds+pc); + corrupted counts |
| `label_noise/exp_sigma_sweep.py` | sigma-sweep: is the detection floor a FIXED MULTIPLE of the model's own error? |
| `label_noise/exp_sigma_threshold.py` | Pure re-analysis. Test whether τ_s = k·σ_s equalizes the per-subject FPR that |
| `label_noise/exp_transport.py` | Everything in the manuscript is measured on one dataset. The claim that the |
| `label_noise/exp_transport_ci.py` | exp_transport.py reports ten point estimates and no uncertainty. The headline -- |
| `label_noise/fig1.py` | fig1.py — the collapse (full width, two panels). Contribution 3 evidence. |
| `label_noise/fig2.py` | fig2.py — severity anchored against the detection floor (column width). |
| `label_noise/fig2_severity_floor.py` | Pure plotting from the committed 15-subject severity re-analysis CSV |
| `label_noise/fig3.py` | fig3.py — damage ratio against severity (column width). Two thresholds, one interval. |
| `label_noise/fig4.py` | fig4.py — operating-point trade-off (column width). Currency is user time. |
| `label_noise/fig5.py` | fig5.py — two-channel triage schematic (full width, no data). Pure matplotlib. |
| `label_noise/fig6.py` | fig6.py — the floor mechanism (column width, no data; illustrative schematic). |
| `label_noise/fig_data.py` | Every figure's numbers come from committed CSVs / records, reproducing the exact |
| `label_noise/fig_style.py` | Two-column IEEE Access. Column width 3.5in, full width 7.16in. Serif body-matching |
| `label_noise/gaze_math.py` | Gaze geometry for the label-noise Phase-1 study. Pure numpy, no repo dependency. |
| `label_noise/loo_residual.py` | Label-consistency scoring (HANDOFF §4.3): per support frame, how far is the |
| `label_noise/make_coherence_figures.py` | Reuses fig_style.py, the house figure style already used by the earlier |
| `label_noise/make_figures.py` | The paper has ten tables and no figures, yet its two headline claims are both |
| `label_noise/make_tables.py` | Single source of truth for the manuscript's tables. |
| `label_noise/make_transport_figure.py` | c(kappa) on three corpora with bootstrap bands, plus each corpus's ratio to MPIIFaceGaze. |
| `label_noise/make_transport_tables.py` | Sibling of make_tables.py for everything that spans two corpora. |
| `label_noise/overnight_analyze.py` | Overnight batch analyzer -> logs/MONDAY_REPORT.md (new tooling; reads outputs |
| `label_noise/overnight_cpu_shim.py` | CPU-forcing shim for the Step-1 sanity run (new tooling; modifies NO files). |
| `label_noise/overnight_gate.py` | STEP-1 MPS sanity gate (new tooling; does not modify experiment code). |
| `label_noise/phase2_prereq_report.py` | Verify Phase-2 prerequisites (a)(b)(c) -> logs/PHASE2_PREREQ_REPORT.md. |
| `label_noise/prepare_eyediap_labels.py` | EYEDIAP is the second (third, with RT-GENE) corpus for the transport of c. Unlike |
| `label_noise/prepare_rtgene.py` | The transport experiment is only meaningful if RT-GENE is normalised the same way |
| `label_noise/prepare_rtgene_labels.py` | WHY THIS EXISTS (and why GazeHub's own scripts are not run as shipped) |
| `label_noise/prove_seeding.py` | Proof that the seeding fix makes corruption process-independent. |
| `label_noise/resnet_frozen_arm.py` | The UniGaze arm found a different curve SHAPE: R = floor@0.9 / floor@0.7 is |
| `label_noise/run_phase1.py` | Phase-1 sweep driver (HANDOFF §4.3-4.5). |
| `label_noise/test_smoke.py` | Validates the ported Phase-1 pipeline. Synthetic path needs no data/checkpoint: |
| `label_noise/unc_calib_probe.py` | The CVPR 2026 uncertainty-calibration work for gaze fits a post-hoc calibrator |
| `label_noise/unigaze_arm.py` | Tests whether the model-relative floor constant c = delta_floor / e_hat |
| `label_noise/unigaze_arm_ci.py` | The arm's driver reports a single pooled point estimate, c = delta_floor / |
| `label_noise/unigaze_arm_extend.py` | The first UniGaze run leaves three things unmeasured, and all three are free |
| `label_noise/unigaze_clean_ehat.py` | Why this exists. |
| `label_noise/unigaze_probe.py` | Answers one question: can this machine run a second-architecture arm at all? |
| `label_noise/revision_2026-11/scripts/bench.py` | reproduce exp_a record: p00 seed0 random_offset delta3 pc0.1 (Exp A: models/, 20 steps) |
| `label_noise/revision_2026-11/scripts/check_steps_vs_support.py` | Does the L0 e_hat gap (Exp A 3.84 vs sweep 3.43) come from adaptation steps or support split? |
| `label_noise/revision_2026-11/scripts/eyediap_heldout_eval.py` | Re-evaluate the 14 surviving EYEDIAP LOPO checkpoints with the fixed Table-2 protocol. |
| `label_noise/revision_2026-11/scripts/gate_transfer_run.py` | Step 3 runner: gate transfer across L0-L3 (see ../PREREG_step3_gate_transfer.md). |
| `label_noise/revision_2026-11/scripts/step11_protocol_inventory.py` | Review item 11: unified protocol inventory, read from the record files themselves (read-only). |
| `label_noise/revision_2026-11/scripts/step2_resub_floor.py` | Step 2: detectability floor of the resubstitution residual vs LOO, same trials. |
| `label_noise/revision_2026-11/scripts/step3_analysis.py` | Step 3 detection analysis. Inputs: results/gate_transfer_scores.jsonl, results/step3_thresholds.json. |
| `label_noise/revision_2026-11/scripts/step3_figure.py` | 03_threshold_comparison.png: (a) thresholds vs e_clean per level; (b) pooled per-subject ROC-AUC; (c) expected cost (w=4). |
| `label_noise/revision_2026-11/scripts/step3_mae_analysis.py` | Downstream held-out MAE after each policy (results/gate_transfer_mae.jsonl). Per-subject means over seeds, |
| `label_noise/revision_2026-11/scripts/step3_thresholds.py` | Step 3: compute frozen policy thresholds from gate_transfer_scores.jsonl, exactly per PREREG_step3_gate_transfer.md. |
| `label_noise/revision_2026-11/scripts/step4_figure.py` | 04_deployment_tradeoffs.png: (a) paired cost difference LOO - resub (w=4) per condition and p_c; (b) the same across w; |
| `label_noise/revision_2026-11/scripts/step4_runtime.py` | Step 4 runtime: wall-clock per session of the resubstitution and LOO scorers on this machine (CPU), L0, 3 subjects x 5 repeats, |
| `label_noise/revision_2026-11/scripts/step4_tradeoffs.py` | Step 4: LOO vs resubstitution gate, detection/cost trade-offs. |
| `label_noise/revision_2026-11/scripts/step7_damage_ratio_denominator.py` | Review item 8: stability of the damage ratio R = misleading / lost = (naive-oracle)/(oracle-clean). |
| `label_noise/revision_2026-11/scripts/step9_deployment_by_level.py` | Review item 9: fixed (A, tau = 3.42) vs scaled (B, tau = 0.96 e_hat) per backbone level and contamination rate, |
| `scripts/cqs_detection.py` | RQ1 (de-circularized, full-face-correct): does CQS detect corrupted calibration |
| `scripts/detector_lopo.py` | Solidify the detector's headline number: full leave-one-person-out x seeds. |
| `scripts/train_backbone.py` | Linchpin experiment: train a proper MPIIFaceGaze gaze backbone (leave-one-person-out) |
| `scripts/train_detector.py` | Learned calibration-quality detector — the reframed contribution. |
| `src/cqs.py` | Calibration Quality Score (CQS) |
| `src/config.py` | CQS parameters (to be calibrated in Phase 3) |
| `label_noise/revision_2026-11/scripts/make_graphical_abstract.py` | Graphical abstract: pooled leave-one-out ROC-AUC against δ/ê for accuracy levels L0–L3 (from `sigma_sweep_records.jsonl`). |
