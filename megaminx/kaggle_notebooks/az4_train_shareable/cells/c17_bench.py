### cayleypy graph helper + small benchmark ###
_CAYLEY = {}


def get_cayley_graph():
    if 'graph' not in _CAYLEY:
        from cayleypy import CayleyGraphDef, CayleyGraph
        with open(PUZZLE_JSON, encoding='utf-8') as f:
            info = json.load(f)
        gens_names = list(info['generators'].keys())
        graph_def = CayleyGraphDef.create(
            generators=[np.array(info['generators'][n]) for n in gens_names],
            generator_names=gens_names,
            central_state=np.array(info['central_state']))
        _CAYLEY['graph'] = CayleyGraph(graph_def, dtype=torch.int8,
                                       bit_encoding_width=None,
                                       batch_size=2 ** 16, hash_chunk_size=2 ** 16)
    return _CAYLEY['graph']


def beam_solve_pid(pid, model, beam_width, max_steps):
    """Beam-search one test puzzle; returns (path_string or None, length or None)."""
    from cayleypy import Predictor
    graph = get_cayley_graph()
    initial = np.array([int(x) for x in TEST_ROWS[pid]['initial_state'].split(',')])
    res = graph.beam_search(
        start_state=initial, beam_width=beam_width, max_steps=max_steps,
        predictor=Predictor(graph, model), beam_mode='iterated',
        history_depth=CFG['eval']['history_depth'], hashed_neigbourhood=0,
        memory_cleanup=False, return_path=True, verbose=0, path_device='cuda')
    if not res.path_found:
        return None, None
    path_str = res.get_path_as_string()
    # Independent verification with our own puzzle code.
    final = PUZZLE.apply_path(tuple(initial), PUZZLE.parse_path(path_str))
    assert PUZZLE.is_solved(final), f'pid {pid}: cayleypy path does not verify!'
    return path_str, len(path_str.split('.'))


BENCH_RESULTS = {}
if CFG['eval']['bench'] and V_ONLY_PATH is not None:
    try:
        import cayleypy  # noqa: F401
        _have_cayleypy = True
    except Exception as e:
        _have_cayleypy = False
        print('cayleypy not available - skipping bench:', e)
    if _have_cayleypy:
        bench_model = load_v_model(V_ONLY_PATH)
        bench_model.eval()
        bw = CFG['eval']['bench_beam_width']
        print(f'benchmark: beam_width={bw}, max_steps={CFG["eval"]["bench_max_steps"]}, '
              f'model={EVAL_LABEL}')
        for pid in CFG['eval']['bench_pids']:
            t0 = time.time()
            path_str, length = beam_solve_pid(pid, bench_model, bw,
                                              CFG['eval']['bench_max_steps'])
            BENCH_RESULTS[pid] = length
            print(f'  pid {pid:4d} | solved: {length is not None} | '
                  f'len: {length} | {time.time() - t0:.1f}s', flush=True)
        solved = [v for v in BENCH_RESULTS.values() if v is not None]
        if solved:
            print(f'bench: {len(solved)}/{len(BENCH_RESULTS)} solved, '
                  f'mean len {sum(solved) / len(solved):.1f}')
