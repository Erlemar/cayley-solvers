### Value-calibration canary ###
# Fast checks that predict whether a V model will work in beam search. Thresholds come
# from our experiment history - several architectures matched training loss but failed
# ALL of them in beam search.
#
#   1. V(solved) ~ 0 and V(depth-1) ~ 1 (exact anchors).
#   2. Exact-distance MAE on the BFS-d6 shell (d = 0..6).
#   3. SATURATION: on deep random walks V must flatten near the puzzle's effective
#      diameter (~25-32), NOT keep growing with walk depth (V@80 - V@40 <= ~10) and
#      NOT collapse (V@80 >= ~20).
#   4. Mid-depth V std (walk depth 15-25) should stay small (~<4): high variance there
#      predicted beam collapse even when the means looked fine.

CANARY_RESULTS = {}


@torch.no_grad()
def run_canary(v_model, label):
    v_model.eval()
    res = {}
    v0 = float(v_model(SOLVED_T.unsqueeze(0)).item())
    d1_states = apply_all_generators(SOLVED_T.unsqueeze(0), GENERATORS_T).squeeze(0)
    v1 = float(v_model(d1_states).mean().item())
    res['V_solved'] = v0
    res['V_d1'] = v1

    # Exact-distance calibration on a BFS-d6 sample.
    if BFS_D6_PATH.exists():
        b_states, b_dists = load_bfs6()
        n = min(CFG['eval']['canary_bfs_samples'], b_states.size(0))
        gen = torch.Generator().manual_seed(0)
        idx = torch.randperm(b_states.size(0), generator=gen)[:n]
        s = b_states[idx].to(DEVICE)
        d = b_dists[idx]
        preds = []
        for i in range(0, n, 8192):
            preds.append(v_model(s[i:i + 8192]).float().cpu())
        pred = torch.cat(preds)
        mae_by_d = {}
        for depth in range(0, 7):
            m = d == depth
            if int(m.sum()) > 0:
                mae_by_d[depth] = float((pred[m] - d[m]).abs().mean())
        res['bfs_mae_by_depth'] = mae_by_d

    # Saturation probe on deep random walks.
    ws, wd = generate_walks_torch(PUZZLE, n_walks=1500, k_max=80,
                                  seed=CFG['seed'] + 999, device=DEVICE, n_back=1)
    preds = []
    for i in range(0, ws.size(0), 8192):
        preds.append(v_model(ws[i:i + 8192]).float())
    pv = torch.cat(preds)
    wd = wd.float()

    def bucket(lo, hi):
        m = (wd >= lo) & (wd < hi)
        return pv[m]

    v40 = float(bucket(35, 45).mean())
    v80 = float(bucket(75, 81).mean())
    std20 = float(bucket(15, 25).std())
    res.update(V_at_d40=v40, V_at_d80=v80, sat_gap=v80 - v40, V_std_d20=std20)

    print(f'--- canary: {label} ---')
    print(f'  V(solved) = {v0:+.3f}   (want ~0)     '
          f'{"PASS" if abs(v0) < 0.5 else "WARN"}')
    print(f'  V(d=1)    = {v1:+.3f}   (want ~1)     '
          f'{"PASS" if 0.5 < v1 < 1.5 else "WARN"}')
    if 'bfs_mae_by_depth' in res:
        mae_str = '  '.join(f'd{k}:{v:.2f}' for k, v in res['bfs_mae_by_depth'].items())
        print(f'  exact-distance MAE: {mae_str}')
    print(f'  V@walk40 = {v40:.1f}  V@walk80 = {v80:.1f}  gap = {v80 - v40:+.1f}  '
          f'{"PASS" if (v80 - v40) <= 10 else "WARN: V grows with walk depth - beam will likely fail"}')
    print(f'  V@walk80 level: {"PASS" if 20 <= v80 <= 40 else "WARN: outside [20,40] - drifting or collapsed"}')
    print(f'  V std @ walk 15-25 = {std20:.2f}  '
          f'{"PASS" if std20 < 4 else "WARN: high mid-depth variance predicts beam collapse"}')
    CANARY_RESULTS[label] = res
    return res


if CFG['eval']['canary'] and V_ONLY_PATH is not None:
    run_canary(load_v_model(V_ONLY_PATH), EVAL_LABEL)
    if (CFG['eval']['compare_reference'] and REFERENCE_V_ONLY.exists()
            and Path(V_ONLY_PATH) != REFERENCE_V_ONLY):
        run_canary(load_v_model(REFERENCE_V_ONLY), 'reference m_az_v4_v_only')
