#!/usr/bin/env python
"""CPU-forcing shim for the Step-1 sanity run (new tooling; modifies NO files).

run_phase1.py has no --device flag and src/config.py hardcodes DEVICE=mps when
MPS is available. To run the identical pipeline on CPU for the MPS-vs-CPU sanity
gate WITHOUT editing experiment code, we disable MPS at runtime *before* the
import chain (run_phase1 -> backbone_api -> `from src.config import DEVICE`)
evaluates the device, then hand off to run_phase1 unchanged.

Usage: PYTHONHASHSEED=0 python label_noise/overnight_cpu_shim.py <run_phase1 args...>
"""
import sys, runpy, os
import torch

torch.backends.mps.is_available = lambda: False          # force CPU in src.config
if hasattr(torch.backends, "cuda"):
    torch.cuda.is_available = lambda: False

_HERE = os.path.dirname(os.path.abspath(__file__))
target = os.path.join(_HERE, "run_phase1.py")
sys.argv = ["run_phase1.py"] + sys.argv[1:]
runpy.run_path(target, run_name="__main__")
