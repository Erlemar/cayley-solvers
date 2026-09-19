### Run the selected stages (in pipeline order, chaining warm-starts) ###
PREV_STAGE = {'pretrain': None, 'curriculum': 'pretrain', 'bellman': 'curriculum',
              'bellman_dd': 'bellman', 'az': 'bellman_dd'}
STAGE_OUTPUTS = {}

# The shipped warm-start checkpoints are for the DEFAULT model only.
_DEFAULT_MODEL = ('ResMLPDistance',
                  {'hidden_dims': (2048, 512), 'num_res_blocks': 2, 'embed_dim': 16})


def model_is_default():
    t = CFG['model']['model_type']
    if t != _DEFAULT_MODEL[0]:
        return False
    p = model_params()
    ref = _DEFAULT_MODEL[1]
    return (tuple(p.get('hidden_dims', ())) == ref['hidden_dims']
            and p.get('num_res_blocks') == ref['num_res_blocks']
            and p.get('embed_dim') == ref['embed_dim'])


def resolve_warmstart(stage):
    w = CFG[stage].get('warmstart')
    if w not in ('auto', None):
        return Path(w)
    if w is None:
        return None
    prev = PREV_STAGE[stage]
    if prev in STAGE_OUTPUTS:
        print(f'[{stage}] warm-start from freshly trained {prev}: {STAGE_OUTPUTS[prev]}')
        return STAGE_OUTPUTS[prev]
    if not model_is_default():
        raise AssertionError(
            f"[{stage}] the shipped warm-start checkpoints are for the default 6M "
            f"ResMLPDistance, but model_type={CFG['model']['model_type']} with "
            f'params={model_params()}. Either include the earlier stages in '
            f"stages_to_run (train this model from 'pretrain' up), or set an explicit "
            f"'warmstart' path for this stage.")
    asset = ASSETS / STAGE_INPUT_ASSET[stage]
    assert asset.exists(), (
        f'{stage}: no fresh {prev} output and shipped warm-start missing: {asset}')
    print(f'[{stage}] warm-start from shipped asset: {asset.name}')
    return asset


# ---- multi-session chain state (models/chain_state.json in the run output) ----
CHAIN_STATE_PATH = MODELS_DIR / 'chain_state.json'
_FINGERPRINT = json.loads(json.dumps(
    {'experiment': CFG.get('experiment', 'az4-baseline'),
     'model_type': CFG['model']['model_type'], 'params': model_params(),
     'seed': CFG['seed']}, default=list))


def _rel_models(p):
    parts = Path(p).parts
    return Path(*parts[parts.index('models'):]).as_posix()


def _write_chain_state(completed, partial=None):
    state = {'fingerprint': _FINGERPRINT,
             'completed_rel': {s: _rel_models(p) for s, p in completed.items()},
             'partial_rel': ({'stage': partial['stage'],
                              'resume': _rel_models(partial['resume'])}
                             if partial else None)}
    CHAIN_STATE_PATH.write_text(json.dumps(state, indent=2), encoding='utf-8')


def _load_prev_chain_state():
    if not CFG.get('auto_resume', True) or PREV_OUTPUT is None:
        return {}, None
    try:
        state = json.loads((PREV_OUTPUT / 'models' / 'chain_state.json')
                           .read_text(encoding='utf-8'))
    except Exception as e:
        print('could not read previous chain_state:', e)
        return {}, None
    prev_fp = dict(state.get('fingerprint') or {})
    # chains written before the 'experiment' field existed belong to the default label
    prev_fp.setdefault('experiment', 'az4-baseline')
    if prev_fp != _FINGERPRINT:
        print('previous run was a different experiment/model/seed - ignoring its progress')
        print('  previous:', prev_fp)
        print('  current :', _FINGERPRINT)
        return {}, None
    completed = {s: str(PREV_OUTPUT / rel)
                 for s, rel in (state.get('completed_rel') or {}).items()}
    partial = state.get('partial_rel')
    if partial:
        partial = {'stage': partial['stage'],
                   'resume': str(PREV_OUTPUT / partial['resume'])}
    return completed, partial


completed_paths, prev_partial = _load_prev_chain_state()
if completed_paths:
    print('continuing chain: previously completed stages:', sorted(completed_paths))
if prev_partial:
    print(f"continuing chain: '{prev_partial['stage']}' resumes from "
          f"{prev_partial['resume']}")

# Kaggle mounts only the LATEST successful run's output, and each run's output only
# contains what that run wrote - so copy the chain's live artifacts forward into THIS
# run's output, making every successful output self-contained. Entries whose file is
# missing (outputs produced before this fix) are kept as stale pointers with a warning;
# they only matter if a later stage tries to load them.
import shutil


def _pull_forward(path_str, what):
    src = Path(path_str)
    dst = WORK / _rel_models(src)
    if src == dst or dst.exists():
        return str(dst) if dst.exists() else path_str
    if not src.exists():
        print(f'WARNING: {what} checkpoint missing from previous output ({src}) - '
              f'keeping a stale pointer; only a problem if something loads it.')
        return path_str
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


for _s in list(completed_paths):
    completed_paths[_s] = _pull_forward(completed_paths[_s], f'completed {_s}')
if prev_partial:
    prev_partial['resume'] = _pull_forward(prev_partial['resume'],
                                           f"partial {prev_partial['stage']}")

chain_stopped = False
current_partial = None

for stage in STAGE_ORDER:
    if stage not in CFG['stages_to_run']:
        continue
    if stage in completed_paths:
        STAGE_OUTPUTS[stage] = Path(completed_paths[stage])
        print(f'[{stage}] already completed in a previous run -> {STAGE_OUTPUTS[stage]}')
        continue
    if training_time_up():
        print(f"[{stage}] wall budget exhausted before this stage - re-run the "
              f'notebook to continue the chain.', flush=True)
        chain_stopped = True
        current_partial = prev_partial if (prev_partial
                                           and prev_partial['stage'] == stage) else None
        break
    if (prev_partial and prev_partial['stage'] == stage
            and CFG[stage].get('resume') is None):
        CFG[stage]['resume'] = prev_partial['resume']
    # A resuming stage restores its full weights from the resume checkpoint - the
    # warm-start would be loaded and immediately overwritten, so skip it (this also
    # avoids depending on the previous stage's file at all when resuming).
    resuming = CFG[stage].get('resume') is not None
    print('\n' + '=' * 70)
    print(f'STAGE: {stage}')
    print('resolved config:', json.dumps(CFG[stage], default=str))  # echo before running
    print('=' * 70, flush=True)
    t_stage = time.time()
    if stage == 'pretrain':
        out, done = run_pretrain(CFG['pretrain'], warmstart=None)
    elif stage == 'curriculum':
        out, done = run_curriculum(CFG['curriculum'],
                                   None if resuming else resolve_warmstart('curriculum'))
    elif stage == 'bellman':
        out, done = run_bellman(CFG['bellman'],
                                None if resuming else resolve_warmstart('bellman'),
                                'bellman')
    elif stage == 'bellman_dd':
        out, done = run_bellman(CFG['bellman_dd'],
                                None if resuming else resolve_warmstart('bellman_dd'),
                                'bellman_dd')
    elif stage == 'az':
        out, done = run_az(CFG['az'], None if resuming else resolve_warmstart('az'))
    print(f'[{stage}] {"done" if done else "PAUSED"} in '
          f'{(time.time() - t_stage) / 60:.1f} min -> {out}', flush=True)
    if done:
        STAGE_OUTPUTS[stage] = out
        completed_paths[stage] = str(out)
        _write_chain_state(completed_paths)
    else:
        chain_stopped = True
        current_partial = {'stage': stage, 'resume': str(out)}
        break

_write_chain_state(completed_paths, partial=current_partial)
todo = [s for s in CFG['stages_to_run'] if s not in completed_paths]
if chain_stopped:
    print(f'\nCHAIN PAUSED at the wall budget. Remaining stages: {todo}. '
          f'Re-run this notebook (with its own output attached as an input) to continue.',
          flush=True)
else:
    print('\nCHAIN COMPLETE: all selected stages finished.', flush=True)
print('stage outputs:', {k: str(v) for k, v in STAGE_OUTPUTS.items()})
