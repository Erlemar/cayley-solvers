# --- Configuration (edit here) ---
# Hyperparameters below are the sweep-selected optimum for IHES (lnZ = 56.02).
# lambda (REG_COEF) is the one puzzle-specific knob: a clean single-peaked optimum
# at 5e-8 (5e-7 too strong -> over-regularizes; 5e-9 too weak). EPS_EXPLORE MUST be
# 0 for the cube: at this graph size any exploration collapses the policy (an
# eps=0.1 control gave a total 0/8 solve failure). See the paper's tuning tips.

CFG = dict(
    batch_global = 2048,     # trajectories per step (sharded over the 8 TPU cores)
    nmax         = 30,        # length of on-policy training trajectories
    iters        = 30000,    # SHORT demo (~20 min on v3-8) that produces a working
                             # (under-trained) checkpoint. The reference run used
                             # 800k (~4 h on a v6e-8) for near-converged lengths.
    hidden       = 2048,     # ResMLP width
    blocks       = 6,        # residual blocks
    emb_dim      = 16,       # per-facelet embedding
    lr           = 3e-4,
    clip         = 100.0,
    reg_coef     = 5e-8,     # lambda for the exact flow regularizer (paper eq. 8)
    eps_explore  = 0.0,      # MUST be 0 for cubes (see note above)
    log_every    = 2000,
    eval_every   = 20000,
    ckpt_every   = 20000,
    seed         = 0,
    save_dir     = "/kaggle/working",
)

# IHES Picture Cube group order (Schreier-Sims); lnZ is fixed, not learned.
IHES_ORDER = 2125922464947725402112000  # 2.1259e24  -> lnZ = 56.0162
for k, v in CFG.items():
    print(f"{k:14s} = {v}")
