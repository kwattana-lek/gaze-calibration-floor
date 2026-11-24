#!/usr/bin/env python
# unigaze_probe.py — feasibility probe for a UniGaze arm of the sigma sweep
"""Answers one question: can this machine run a second-architecture arm at all?

The sigma sweep established that the detectability floor tracks the model's own
error proportionally across a 2.6x range of ACCURACY (L0-L3, one ResNet-18 at
four training budgets). The remaining axis is ARCHITECTURE.

TWO CONSTRAINTS, both established before writing this:

  1. LEAKAGE. unigaze_b16_joint, unigaze_l16_joint and unigaze_h14_joint are
     supervised-trained on a joint set that INCLUDES MPIIFaceGaze Train (UniGaze
     paper, arXiv:2502.02307). Using any of them leave-one-person-out on
     MPIIFaceGaze leaks the held-out subject and invalidates the protocol.
     The HF repo UniGaze/UniGaze-models contains exactly four checkpoints, and
     unigaze_h14_cross_X (ETH-XGaze only) is the ONLY admissible one. There is
     no smaller clean checkpoint to fall back to.
     This script refuses anything not allow-listed.

  2. COMPUTE. unigaze_h14_cross_X is ViT-H/14: 632.0M parameters against the
     ResNet-18's 11.2M, a 56x ratio, at 224x224 rather than 112x112. A measured
     forward pass on 9 frames takes ~1.34 s. Since a sweep cell costs 1
     resubstitution + K leave-one-out adaptations, full fine-tuning is unlikely
     to be viable, so this script times TWO regimes:
         full      — every parameter trainable (the ResNet protocol's analogue)
         head-only — backbone frozen, only the output head adapts
     Head-only is the realistic fallback and is what the DAP line already does
     via UniGazeAdapterWrapper.

What this does NOT measure: gaze accuracy. The timing loop regresses the model's
own output against a perturbed copy of itself, because forward+backward cost
through the backbone dominates and does not depend on the loss being the real
one. Treat the numbers as compute feasibility only.

Usage:  python unigaze_probe.py [--k 9] [--steps 20]
"""
import argparse
import os
import sys
import time
import traceback

ADMISSIBLE = {"unigaze_h14_cross_X"}
KNOWN_LEAKY = {"unigaze_b16_joint", "unigaze_l16_joint", "unigaze_h14_joint"}
RESNET_PARAMS = 11.2e6
RESNET_CELL_S = 8.4          # measured ResNet-18 cost for one sweep cell

FACE_DIR = os.path.expanduser(
    "~/Desktop/Documents Since 29-01-2569/Datasets/MPIIFaceGaze_Normalized/Image")


def hr(t=""):
    print(f"\n{'=' * 68}\n{t}\n{'=' * 68}" if t else "=" * 68, flush=True)


def as_tensor(out, torch, batch):
    """UniGaze returns a dict; pull the prediction tensor out of it."""
    if torch.is_tensor(out):
        return out, "(tensor)"
    if isinstance(out, dict):
        for k in ("pred_gaze", "gaze", "pred", "output", "out", "logits"):
            v = out.get(k)
            if torch.is_tensor(v):
                return v, k
        cands = [(k, v) for k, v in out.items()
                 if torch.is_tensor(v) and v.is_floating_point()
                 and v.ndim <= 2 and v.shape[0] == batch]
        if cands:
            k, v = min(cands, key=lambda kv: kv[1].numel())
            return v, k
        for k, v in out.items():
            if torch.is_tensor(v) and v.is_floating_point():
                return v, k
    if isinstance(out, (list, tuple)):
        for v in out:
            if torch.is_tensor(v) and v.is_floating_point():
                return v, "(seq[0])"
    raise TypeError(f"cannot find a prediction tensor in {type(out).__name__}: "
                    f"{list(out) if hasattr(out, 'keys') else out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unigaze_h14_cross_X")
    ap.add_argument("--k", type=int, default=9)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--subject", default="p00")
    ap.add_argument("--allow-leaky", action="store_true")
    a = ap.parse_args()

    hr("0. LEAKAGE GUARD")
    if a.model in KNOWN_LEAKY and not a.allow_leaky:
        sys.exit(f"REFUSED: {a.model} is trained on a joint set including "
                 f"MPIIFaceGaze.\n         Admissible: {sorted(ADMISSIBLE)}")
    if a.model not in ADMISSIBLE and not a.allow_leaky:
        sys.exit(f"REFUSED: {a.model} not allow-listed.\n"
                 f"         Admissible: {sorted(ADMISSIBLE)}")
    print(f"  {a.model}: admissible (not trained on MPIIFaceGaze)")
    print("  note: the UniGaze release has no smaller clean checkpoint —")
    print("        b16/l16/h14 joint are all MPIIFaceGaze-contaminated.")

    hr("1. ENVIRONMENT")
    try:
        import torch
    except ImportError:
        sys.exit("torch not importable — run inside the agentgaze env.")
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    have_mps = bool(mps and mps.is_available())
    dev = torch.device("mps" if have_mps else "cpu")
    print(f"  torch {torch.__version__}   MPS {have_mps}   device {dev}")
    try:
        import unigaze
    except ImportError:
        sys.exit("unigaze not installed.  pip install unigaze")
    print(f"  unigaze {getattr(unigaze, '__version__', '?')}")

    hr("2. LOAD")
    t = time.time()
    try:
        model = unigaze.load(a.model, device=str(dev))
    except Exception:
        traceback.print_exc()
        sys.exit(f"\nFAILED to load {a.model}.")
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  loaded in {time.time()-t:.1f}s")
    print(f"  parameters: {n_par/1e6:.1f}M   vs ResNet-18 "
          f"{RESNET_PARAMS/1e6:.1f}M  ->  {n_par/RESNET_PARAMS:.0f}x")

    hr("3. INPUT")
    import numpy as np
    d = os.path.join(FACE_DIR, a.subject, "face")
    imgs = None
    if os.path.isdir(d):
        try:
            from PIL import Image
            fs = sorted(os.listdir(d))[:a.k]
            imgs = np.stack([np.asarray(Image.open(os.path.join(d, f)).convert("RGB"))
                             for f in fs])
            print(f"  {len(imgs)} real face crops from {a.subject}, "
                  f"{imgs.shape[1]}x{imgs.shape[2]}")
        except Exception as e:
            print(f"  could not read crops ({e}); using random")
    if imgs is None:
        imgs = (np.random.rand(a.k, 224, 224, 3) * 255).astype("uint8")
        print(f"  random 224x224 input ({a.k} frames)")
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().div(255).to(dev)

    sync = (lambda: torch.mps.synchronize()) if have_mps else (lambda: None)

    def mem_gb():
        try:
            return torch.mps.driver_allocated_memory() / 1e9
        except Exception:
            return float("nan")

    hr("4. FORWARD")
    model.eval()
    with torch.no_grad():
        raw = model(x); sync()
        t = time.time(); raw = model(x); sync(); fwd = time.time() - t
    pred, key = as_tensor(raw, torch, a.k)
    print(f"  forward on {a.k} frames: {fwd*1000:.0f} ms")
    print(f"  output: {type(raw).__name__}"
          + (f" with keys {list(raw)}" if isinstance(raw, dict) else ""))
    print(f"  prediction tensor taken from '{key}', shape {tuple(pred.shape)}")
    print(f"  memory: {mem_gb():.2f} GB")

    with torch.no_grad():
        tgt = as_tensor(model(x), torch, a.k)[0].detach().clone()
    tgt = tgt + torch.randn_like(tgt) * 0.05

    def time_regime(name, params):
        n = sum(p.numel() for p in params)
        if n == 0:
            print(f"\n  {name}: no trainable parameters found — skipped")
            return float("nan")
        print(f"\n  {name}: {n/1e6:.2f}M trainable ({100*n/n_par:.1f}% of model)")
        try:
            opt = torch.optim.Adam(params, lr=a.lr)
            sync(); t = time.time()
            for _ in range(a.steps):
                opt.zero_grad(set_to_none=True)
                loss = ((as_tensor(model(x), torch, a.k)[0] - tgt) ** 2).mean()
                loss.backward()
                opt.step()
            sync(); el = time.time() - t
            print(f"    {a.steps} steps: {el:.2f}s  ({el/a.steps*1000:.0f} ms/step)"
                  f"   peak {mem_gb():.2f} GB")
            return el
        except RuntimeError as e:
            print(f"    FAILED: {str(e)[:160]}")
            return float("nan")

    hr(f"5. ADAPTATION COST ({a.steps} steps @ lr={a.lr})")
    print("  NOTE: dummy target — times compute, not accuracy.")
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    full = time_regime("full fine-tune", [p for p in model.parameters()])

    head_names = [n for n, _ in model.named_parameters()
                  if any(s in n.lower() for s in ("head", "fc", "regress", "mlp_head"))]
    for n, p in model.named_parameters():
        p.requires_grad_(n in head_names)
    head = time_regime(f"head-only ({len(head_names)} tensors matched)",
                       [p for p in model.parameters() if p.requires_grad])

    hr("6. EXTRAPOLATION")
    print(f"  a sweep cell = 1 resubstitution + {a.k} leave-one-out adaptations\n")
    print(f"  {'regime':<16}{'cell':>10}{'vs ResNet':>12}{'one level':>12}{'4 levels':>11}")
    for name, secs in (("full fine-tune", full), ("head-only", head)):
        if secs != secs:
            print(f"  {name:<16}{'failed':>10}{'—':>12}{'—':>12}{'—':>11}")
            continue
        cell = secs * (1 + a.k)
        print(f"  {name:<16}{cell/60:>9.1f}m{cell/RESNET_CELL_S:>11.0f}x"
              f"{495*cell/3600:>11.1f}h{1980*cell/3600:>10.1f}h")
    print(f"\n  (ResNet-18 reference: {RESNET_CELL_S}s per cell, "
          f"{495*RESNET_CELL_S/3600:.1f}h per level)")
    print("\n  Guide: >24h for a single level makes the arm impractical as a"
          "\n  like-for-like replication. Head-only changes the adaptation"
          "\n  protocol, so it answers a related but not identical question —"
          "\n  say so explicitly if it is what gets reported.")

    hr("DONE")


if __name__ == "__main__":
    main()
