### Policy dataset builder: solutions CSV -> (state, action, remaining) tuples ###
# Every move of every solution contributes one AlphaZero-style training tuple:
# the state before the move, the move taken, and the number of moves remaining.

def build_policy_dataset(submission_csv, out_path):
    move_to_idx = {name: i for i, name in enumerate(PUZZLE.move_names)}
    with open(submission_csv, encoding='utf-8') as f:
        sub_rows = list(csv.DictReader(f))
    all_states, all_actions, all_values = [], [], []
    n_bad = 0
    for row in sub_rows:
        pid = int(row['initial_state_id'])
        path = PUZZLE.parse_path(row['path'])
        if not path:
            continue
        cur = [int(x) for x in TEST_ROWS[pid]['initial_state'].split(',')]
        for i, move in enumerate(path):
            all_states.append(np.array(cur, dtype=np.int8))
            all_actions.append(move_to_idx[move])
            all_values.append(len(path) - i)
            gen = PUZZLE.generators[move]
            cur = [cur[g] for g in gen]
        if tuple(cur) != PUZZLE.solved_state:
            n_bad += 1
    if n_bad:
        print(f'WARNING: {n_bad} paths did not end at solved - check the CSV')
    data = {
        'states': torch.tensor(np.stack(all_states), dtype=torch.int8),
        'actions': torch.tensor(all_actions, dtype=torch.int8),
        'values': torch.tensor(all_values, dtype=torch.float32),
        'source': str(submission_csv),
    }
    torch.save(data, out_path)
    print(f'policy dataset: {len(all_states):,} tuples from {len(sub_rows)} paths '
          f'-> {out_path}', flush=True)
    return out_path


def resolve_policy_dataset():
    """Return the path of the policy dataset the az stage should train on."""
    ds = CFG['az']['policy_dataset']
    if ds not in ('auto', None):
        return Path(ds)
    csv_path = CFG['az']['policy_csv']
    if csv_path in ('auto', None):
        if PREBUILT_POLICY_DATASET.exists():
            return PREBUILT_POLICY_DATASET
        csv_path = DEFAULT_POLICY_CSV
    out = MODELS_DIR / (Path(csv_path).stem + '_az_dataset.pt')
    if not out.exists():
        build_policy_dataset(csv_path, out)
    return out


if 'az' in CFG['stages_to_run']:
    POLICY_DATASET_PATH = resolve_policy_dataset()
    print('policy dataset for the az stage:', POLICY_DATASET_PATH)
