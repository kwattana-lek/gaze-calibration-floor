import torch

# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Paths
DATA_ROOT      = "data"
MODELS_ROOT    = "models"
EXPERIMENTS    = "experiments"
FIGURES        = "figures"
LOGS           = "logs"

# CQS parameters (to be calibrated in Phase 3)
CQS_LAMBDA     = 0.000770   # q_hp=0.5 at 30 deg (calibrated for MPIIGaze range)   # head pose tolerance
CQS_SIGMA_MAX  = 40.0     # sharpness normaliser — calibrate this
CQS_W1         = 0.35      # landmark weight
CQS_W2         = 0.40      # sharpness weight
CQS_W3         = 0.25      # head pose weight
CQS_TAU        = 0.60      # decision threshold

# MAML adaptation
MAML_LR        = 1e-2   # calibrated: best MAE on MPIIGaze p00
MAML_STEPS     = 20     # calibrated: lr=1e-2, steps=20 → MAE=6.41 deg
K_TARGET       = 9         # support set size
R_MAX          = 5         # max recapture attempts per point

# Experiment
SEEDS          = [1, 2, 3, 4, 5]
NOISE_LEVELS   = [0.0, 0.2, 0.3, 0.4]
