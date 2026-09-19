### CONFIG - the whole notebook is driven by this dict ###
import json
import os

CFG = {
    # ---- manage ----
    # Any contiguous subset of ['pretrain','curriculum','bellman','bellman_dd','az'].
    # Default = the FULL from-scratch chain (~25-30h on a T4 - several sessions!).
    # Set ['az'] to just reproduce AZ v4 from the shipped stage-4 checkpoint (~30 min).
    'stages_to_run': ['pretrain', 'curriculum', 'bellman', 'bellman_dd', 'az'],
    'quick_test': False,      # True = tiny epochs/samples everywhere (smoke test)
    # Experiment label. Part of the chain identity: runs resume ONLY a chain whose
    # (experiment, model, seed) all match. Give every new training/recipe its own
    # label to start a fresh chain - no need to touch the seed or the inputs.
    'experiment': 'az4-baseline',
    'seed': 71,
    'precision': 'auto',      # 'auto' | 'off' (fp32) | 'bf16'
    'compile': 'auto',        # 'auto' (Ampere+ only) | True | False

    # Multi-session chaining: a Kaggle run that hits the session limit loses ALL its
    # output, so training stops GRACEFULLY at this wall budget (checkpoints +
    # models/chain_state.json land in the output), and the next run of this notebook
    # continues from its own previous output automatically (auto_resume; the notebook's
    # own output must be attached as an input - the published version already has it).
    # Just re-run the notebook until the chain reports all stages complete.
    'max_wall_hours': 8.5,        # keep safely under the Kaggle session limit
                                  # (normal value: 8.5 - keep safely under the session limit)
    'eval_reserve_minutes': 20,   # left aside for export + canary + bench + saving
    'auto_resume': True,

    # ---- MODEL POOL (the selected backbone is used by ALL stages) ----
    # 'model_type' picks the architecture; each registered model keeps its own parameter
    # block below. Add your own model in the model-pool cell and list its params here.
    'model': {
        'model_type': 'ResMLPDistance',
        'ResMLPDistance': {           # our architecture; these defaults = the AZ v4 6M trunk
            'hidden_dims': (2048, 512),   # see the bigger-trunk warning in the Params cell
            'num_res_blocks': 2,
            'embed_dim': 16,
        },
        'PilgrimAttnRes': {           # community baseline family (@ogurtsov's notebook)
            'hd1': 1024,
            'hd2': 256,
            'nrd': 4,                     # number of plain residual blocks
            'n_attn_blocks': 0,           # used when block_type='attn_res'
            'block_type': 'residual',     # 'residual' | 'attn_res'
            'dropout_rate': 0.0,
        },
    },

    # ---- stage 1: random-walk regression pretrain (from scratch) ----
    'pretrain': {
        'warmstart': None,            # from scratch
        'epochs': 4000,
        'samples_per_epoch': 1_000_000,
        'batch_size': 16384,
        'k_max': 80,                  # walk length. NOT "diameter x 3" (that was the 3x3x3
                                      # config): megaminx diameter is >= 50 by counting, and
                                      # the walk mixes by ~60 - past that, larger k inflates
                                      # the label, not the reach. 80 is empirical (40/80/100).
        'n_back': 1,                  # non-backtracking depth
        'lr': 2e-3,
        'optimizer': {'name': 'AdamW', 'params': {}},   # any torch.optim name; lr comes from 'lr'
        'loss': {'name': 'MSELoss', 'params': {}},      # torch.nn loss or PinballLoss/LogCoshLoss
        'checkpoint_every': 200,
        'resume': None,               # path to a checkpoint to continue from
    },

    # ---- stage 2: mixed-k curriculum + EMA + val early-stop ----
    'curriculum': {
        'warmstart': 'auto',
        'budget_epochs': 50_000,      # early stopping usually fires near ~6k
        'warmup_epochs': 5_000,
        'warmup_k_max': 35,
        'mix_k_list': (50, 70, 80, 100),
        'samples_per_epoch': 300_000,
        'batch_size': 8192,
        'lr': 5e-4,
        'n_back': 1,
        'optimizer': {'name': 'AdamW', 'params': {}},
        'loss': {'name': 'MSELoss', 'params': {}},      # training loss; val metric stays MSE
        'ema_tau': 1e-3,              # per-step EMA decay; best_ema.pt is the output
        'val_every': 100,
        'patience': 1000,             # early stop: no val improvement for this many epochs
        'val_k_max': 80,
        'val_n_walks': 2000,
        'checkpoint_every': 500,
        'resume': None,
    },

    # ---- stage 3: Bellman bootstrap + frontier + BFS-d6 mixins ----
    'bellman': {
        'warmstart': 'auto',
        'epochs': 500,
        'samples_per_epoch': 500_000,
        'batch_size': 8192,
        'k_max': 80,
        'n_back': 1,
        'lr': 5e-4,
        'optimizer': {'name': 'AdamW', 'params': {}},
        'target_update_every': 10,    # copy model -> frozen target net every N epochs
        'target_net_chunk': 4096,
        'clip_upper': True,           # clip target at walk depth (a provable upper bound)
        'clip_lower': True,           # clip target at 0
        'frontier_fraction': 0.25,    # beam-frontier states mixin (Bellman-bootstrap targets)
        'frontier_walk_depth_cap': 200.0,
        'bfs_d6_fraction': 0.10,      # exact-distance states (d<=6) mixin
        'n_anchor_v0': 0,             # stage 3 has NO anchors (stage 4 adds them)
        'n_anchor_d1': 0,
        'checkpoint_every': 25,
        'resume': None,
    },

    # ---- stage 4: + exact anchors (V(solved)=0, V(d=1)=1 every batch) ----
    'bellman_dd': {
        'warmstart': 'auto',
        'epochs': 50,
        'samples_per_epoch': 500_000,
        'batch_size': 8192,
        'k_max': 80,
        'n_back': 1,
        'lr': 5e-4,
        'optimizer': {'name': 'AdamW', 'params': {}},
        'target_update_every': 10,
        'target_net_chunk': 4096,
        'clip_upper': True,
        'clip_lower': True,
        'frontier_fraction': 0.25,
        'frontier_walk_depth_cap': 200.0,
        'bfs_d6_fraction': 0.10,
        'n_anchor_v0': 32,            # 32 copies of solved with target 0, every batch
        'n_anchor_d1': 4,             # 4 copies of each of the 24 depth-1 states, target 1
        'checkpoint_every': 10,
        'resume': None,
        # Note: the original stage-4 run also used a pattern-database lower-bound penalty
        # (lambda_pdb=5). Omitted here (PDB files are ~1.8GB and the effect is secondary);
        # the shipped stage-4 checkpoint IS the original lambda=5 artifact.
    },

    # ---- stage 5: dual-head AZ fine-tune (policy CE + Bellman value) ----
    'az': {
        'warmstart': 'auto',
        # Policy data: a solutions CSV (initial_state_id,path). 'auto' = the shipped
        # community-best submission_73731.csv. Point this at any stronger CSV to retrain
        # on better paths - more-consistent merged paths converge FASTER (stop earlier!).
        'policy_csv': 'auto',
        'policy_dataset': 'auto',     # 'auto' = prebuilt tensor for the default CSV, else built from policy_csv
        'epochs': 30,                 # original budget was 200; the good model is ~ep24
        'select_epoch': 24,           # checkpoint used for export/eval ('last' = final epoch)
        'lr_t_max': 200,              # cosine schedule length - keep 200 (see Params cell)
        'samples_per_epoch': 500_000,
        'rw_batch_size': 8192,        # value-batch size (random walks + anchors + BFS-d6)
        'policy_batch_size': 1024,    # policy CE batch per step
        'alpha': 1.0,                 # policy CE weight
        'beta': 1.0,                  # value MSE weight
        'k_max': 80,
        'n_back': 1,
        'lr': 5e-4,
        'optimizer': {'name': 'AdamW', 'params': {}},
        'anchor_v0': 32,
        'anchor_d1': 4,
        'bfs_d6_fraction': 0.10,
        'target_update_every': 10,
        'checkpoint_every': 5,        # ep 4,9,14,19,24,29 - explore stops near 24
        'resume': None,
    },

    # ---- eval ----
    'eval': {
        'canary': True,               # V(solved), V(d=1), exact-distance MAE, saturation probe
        'canary_bfs_samples': 200_000,
        'compare_reference': True,    # also canary the shipped m_az_v4_v_only for side-by-side
        'bench': True,                # small cayleypy beam benchmark on a few puzzles
        'bench_pids': (0, 100, 300),
        'bench_beam_width': 2 ** 12,
        'bench_max_steps': 200,
        'history_depth': 10,          # non-backtracking depth for cayleypy beam
    },

    # ---- solve + submission.csv (off by default) ----
    'solve': {
        'enabled': False,
        'checkpoint': 'auto',         # 'auto' = this run's exported value head, else shipped reference
        'list_states_to_solve': list(range(0, 20)),  # SHORT list by default; [] = ALL 1001 (hours!)
        'beam_width': 2 ** 14,
        'max_steps': 400,
    },
}

# Env-var overrides (used by the local smoke harness; harmless on Kaggle).
if os.environ.get('AZ4_QUICK_TEST') == '1':
    CFG['quick_test'] = True
if os.environ.get('AZ4_STAGES'):
    CFG['stages_to_run'] = [s for s in os.environ['AZ4_STAGES'].split(',') if s]
if os.environ.get('AZ4_EXPERIMENT'):
    CFG['experiment'] = os.environ['AZ4_EXPERIMENT']
if os.environ.get('AZ4_MODEL_TYPE'):
    CFG['model']['model_type'] = os.environ['AZ4_MODEL_TYPE']
if os.environ.get('AZ4_SOLVE') == '1':
    CFG['solve']['enabled'] = True

STAGE_ORDER = ['pretrain', 'curriculum', 'bellman', 'bellman_dd', 'az']
assert all(s in STAGE_ORDER for s in CFG['stages_to_run']), CFG['stages_to_run']

if CFG['quick_test']:
    print('QUICK TEST MODE: tiny epochs/samples everywhere - results are meaningless.')
    CFG['compile'] = False
    CFG['max_wall_hours'] = 99.0
    CFG['pretrain'].update(epochs=2, samples_per_epoch=20_000, batch_size=4096,
                           checkpoint_every=2)
    CFG['curriculum'].update(budget_epochs=3, warmup_epochs=1, patience=2, val_every=1,
                             samples_per_epoch=20_000, val_n_walks=50, checkpoint_every=2)
    CFG['bellman'].update(epochs=2, samples_per_epoch=20_000, batch_size=2048,
                          checkpoint_every=2)
    CFG['bellman_dd'].update(epochs=2, samples_per_epoch=20_000, batch_size=2048,
                             checkpoint_every=2)
    CFG['az'].update(epochs=2, select_epoch=1, samples_per_epoch=20_000,
                     rw_batch_size=2048, policy_batch_size=256, checkpoint_every=1)
    CFG['eval'].update(canary_bfs_samples=20_000, bench_pids=(0,),
                       bench_beam_width=2 ** 8, bench_max_steps=60)
    CFG['solve'].update(list_states_to_solve=[0], beam_width=2 ** 8, max_steps=60)

if os.environ.get('AZ4_MAX_WALL_HOURS'):        # local smoke-test hook
    CFG['max_wall_hours'] = float(os.environ['AZ4_MAX_WALL_HOURS'])

with open('cfg.json', 'w', encoding='utf-8') as f:
    json.dump(CFG, f, indent=2, default=list)
print('stages_to_run:', CFG['stages_to_run'])
