### Inputs: competition data + assets dataset + working dirs ###
from pathlib import Path

ON_KAGGLE = Path('/kaggle/input').exists()

if ON_KAGGLE:
    _comp_candidates = [
        Path('/kaggle/input/competitions/cayley-py-megaminx'),
        Path('/kaggle/input/cayley-py-megaminx'),
    ]
    COMP = next((p for p in _comp_candidates if (p / 'puzzle_info.json').exists()), None)
    assert COMP is not None, (
        'Competition data not found. Add Input -> Competitions -> cayley-py-megaminx')
    _asset_candidates = [
        Path('/kaggle/input/megaminx-az4-training-assets'),
        Path('/kaggle/input/datasets/artgor/megaminx-az4-training-assets'),
        Path('/kaggle/input/datasets/megaminx-az4-training-assets'),
    ]
    ASSETS = next((p for p in _asset_candidates if p.exists()), None)
    if ASSETS is None:
        print('contents of /kaggle/input:',
              sorted(str(p) for p in Path('/kaggle/input').glob('**/*'))[:40])
        raise AssertionError(
            'Assets dataset not mounted. Add Input -> Datasets -> '
            'artgor/megaminx-az4-training-assets (and re-run).')
    WORK = Path('/kaggle/working')
else:
    # Local run: point these env vars at directories with the same files.
    COMP = Path(os.environ.get('AZ4_COMP_DIR', './comp'))
    ASSETS = Path(os.environ.get('AZ4_ASSETS_DIR', './assets'))
    WORK = Path(os.environ.get('AZ4_WORK_DIR', './az4_work'))
    WORK.mkdir(parents=True, exist_ok=True)

PUZZLE_JSON = COMP / 'puzzle_info.json'
TEST_CSV = COMP / 'test.csv'
SAMPLE_SUB = COMP / 'sample_submission.csv'
MODELS_DIR = WORK / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Shipped warm-start checkpoints: stage name -> the file that stage warm-starts FROM
# (i.e. the previous stage's output, from our original AZ v4 run).
STAGE_INPUT_ASSET = {
    'curriculum': 'm07_big_k80_epoch3999.pt',
    'bellman': 'm_curr_v0_best_ema.pt',
    'bellman_dd': 'm_curr_v3_epoch0499.pt',
    'az': 'm_dd_v0_epoch0049.pt',
}
REFERENCE_V_ONLY = ASSETS / 'm_az_v4_v_only.pt'       # our original exported value head
BFS_D6_PATH = ASSETS / 'bfs_d6_train.pt'              # 19.35M exact-distance states, ~2.3GB
FRONTIER_PATH = ASSETS / 'frontier_states.pt'         # 300K beam-frontier states
DEFAULT_POLICY_CSV = ASSETS / 'submission_73731.csv'  # community-best public solutions
PREBUILT_POLICY_DATASET = ASSETS / 'az_dataset_73731.pt'

for p in (PUZZLE_JSON, TEST_CSV, SAMPLE_SUB):
    assert p.exists(), f'missing input file: {p}'

# Output of a PREVIOUS run of this notebook (for auto-continuation of multi-session
# chains). Found via its models/chain_state.json marker.
PREV_OUTPUT = None
if ON_KAGGLE:
    _prev_candidates = [
        Path('/kaggle/input/notebooks/artgor/cayleypy-az4-trainer-megaminx'),
        Path('/kaggle/input/cayleypy-az4-trainer-megaminx'),
        Path('/kaggle/input/kernels/artgor/cayleypy-az4-trainer-megaminx'),
    ]
    PREV_OUTPUT = next((c for c in _prev_candidates
                        if (c / 'models' / 'chain_state.json').exists()), None)
    if PREV_OUTPUT is None:  # mount conventions move around - scan shallowly
        for _pat in ('*', '*/*', '*/*/*'):
            _hits = [d for d in Path('/kaggle/input').glob(_pat)
                     if d.is_dir() and (d / 'models' / 'chain_state.json').exists()]
            if _hits:
                PREV_OUTPUT = _hits[0]
                break
elif os.environ.get('AZ4_PREV_OUTPUT'):
    _c = Path(os.environ['AZ4_PREV_OUTPUT'])
    PREV_OUTPUT = _c if (_c / 'models' / 'chain_state.json').exists() else None

print('competition data:', COMP)
print('assets:', ASSETS, '| assets present:',
      all(p.exists() for p in (BFS_D6_PATH, FRONTIER_PATH, DEFAULT_POLICY_CSV)))
print('previous run output:', PREV_OUTPUT if PREV_OUTPUT else 'none found')
