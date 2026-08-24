### Optional: solve puzzles and write submission.csv ###
# Off by default (CFG['solve']['enabled']). Solves `list_states_to_solve` (short list by
# default; set to [] to solve ALL 1001 - hours at useful beam widths). Unsolved or
# un-attempted puzzles fall back to the sample submission's path.
SOLVE_STATS = None

_solve_ok = CFG['solve']['enabled']
if _solve_ok:
    try:
        import cayleypy  # noqa: F401
    except Exception as _e:
        print('cayleypy not available - skipping solve:', _e)
        _solve_ok = False

if _solve_ok:
    import pandas as pd

    ckpt = CFG['solve']['checkpoint']
    if ckpt in ('auto', None):
        ckpt = V_ONLY_PATH if V_ONLY_PATH is not None else REFERENCE_V_ONLY
    assert ckpt is not None and Path(ckpt).exists(), f'no solve checkpoint available: {ckpt}'
    solve_model = load_v_model(ckpt)
    solve_model.eval()
    print(f'solving with {ckpt}')

    df_sample = pd.read_csv(SAMPLE_SUB, index_col='initial_state_id')
    todo = CFG['solve']['list_states_to_solve'] or list(range(len(df_sample)))
    print(f"solving {len(todo)} puzzles at beam_width={CFG['solve']['beam_width']}")

    new_paths = dict(df_sample['path'])  # pid -> path, prefilled with sample fallback
    n_solved = 0
    n_improved = 0
    t_all = time.time()
    for i, pid in enumerate(todo):
        t0 = time.time()
        path_str, length = beam_solve_pid(pid, solve_model,
                                          CFG['solve']['beam_width'],
                                          CFG['solve']['max_steps'])
        sample_len = len(str(df_sample['path'].loc[pid]).split('.'))
        if length is not None:
            n_solved += 1
            if length < sample_len:
                new_paths[pid] = path_str
                n_improved += 1
        if i < 20 or pid % 50 == 0:
            print(f'  pid {pid:4d} | solved {length is not None} | len {length} '
                  f'(sample {sample_len}) | {time.time() - t0:.1f}s', flush=True)

    df_out = df_sample.copy()
    df_out['path'] = [new_paths[pid] for pid in df_out.index]
    df_out.to_csv(WORK / 'submission.csv')
    total = int(df_out['path'].apply(lambda x: len(str(x).split('.'))).sum())
    SOLVE_STATS = {'attempted': len(todo), 'solved': n_solved,
                   'improved_vs_sample': n_improved, 'total_moves': total,
                   'wall_min': (time.time() - t_all) / 60}
    print(f'\nsolved {n_solved}/{len(todo)} attempted ({n_improved} beat the sample path)')
    print(f'submission.csv written - total moves over all 1001 rows: {total:,}')
    print('(rows outside list_states_to_solve keep the sample path - '
          'solve everything with list_states_to_solve=[] before submitting for real)')
