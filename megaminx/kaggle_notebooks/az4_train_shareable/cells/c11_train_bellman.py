### Stages 3+4 trainer: Bellman bootstrap with frontier / BFS-d6 / anchor mixins ###

_BFS6_CACHE = {}


def load_bfs6():
    """(states int8 (N,120), distances float32 (N,)) on CPU; cached across stages."""
    if 'data' not in _BFS6_CACHE:
        print(f'loading BFS-d6 dataset ({BFS_D6_PATH.name}, ~2.3GB - takes a while '
              f'from the input mount)...', flush=True)
        d = torch.load(BFS_D6_PATH, map_location='cpu', weights_only=False)
        _BFS6_CACHE['data'] = (d['states'], d['distances'].to(torch.float32))
        print(f"  {d['states'].size(0):,} exact-distance states loaded", flush=True)
    return _BFS6_CACHE['data']


def load_frontier():
    if 'frontier' not in _BFS6_CACHE:
        d = torch.load(FRONTIER_PATH, map_location='cpu', weights_only=False)
        _BFS6_CACHE['frontier'] = d['states']
        print(f"  {d['states'].size(0):,} frontier states loaded", flush=True)
    return _BFS6_CACHE['frontier']


@torch.no_grad()
def bellman_targets(forward_fn, states, walk_depths, chunk_size,
                    clip_upper=True, clip_lower=True):
    """target = clip(1 + min_a V_target(apply(s, a)), 0, walk_depth).

    A child equal to the solved state contributes exactly 0 (the boundary condition).
    `forward_fn(chunk) -> (B,) float` is the frozen target net's value forward.
    """
    B, S = states.shape
    children_flat = apply_all_generators(states, GENERATORS_T).reshape(-1, S)
    is_solved = (children_flat == SOLVED_T).all(dim=1)
    vals = torch.empty(children_flat.size(0), dtype=torch.float32, device=states.device)
    for i in range(0, children_flat.size(0), chunk_size):
        vals[i:i + chunk_size] = forward_fn(children_flat[i:i + chunk_size]).flatten().float()
    vals = torch.where(is_solved, torch.zeros_like(vals), vals).view(B, -1)
    target = 1.0 + vals.min(dim=1).values
    if clip_upper:
        target = torch.minimum(target, walk_depths)
    if clip_lower:
        target = torch.clamp(target, min=0.0)
    return target


def run_bellman(cfg, warmstart, stage_name):
    """Bellman refinement. stage_name in ('bellman', 'bellman_dd') - only the anchor
    counts (and epochs) differ between the two stages."""
    out_dir = MODELS_DIR / stage_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = new_v_model()
    if warmstart:  # None when resuming (the resume checkpoint restores everything)
        load_warmstart_strict(model, warmstart)
    target_model = copy.deepcopy(model).eval()
    for p in target_model.parameters():
        p.requires_grad = False
    optim = make_optimizer(model.parameters(), cfg)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg['epochs'])
    start_epoch = try_resume(model, optim, sched, cfg['resume'])
    if start_epoch > 0:
        target_model.load_state_dict(clean_state_dict(model.state_dict()))
    model_c = maybe_compile(model)
    autocast_ctx = make_autocast()
    batch_gen = torch.Generator(device=DEVICE)
    batch_gen.manual_seed(CFG['seed'])
    cpu_gen = torch.Generator(device='cpu')
    cpu_gen.manual_seed(CFG['seed'] + 12345)

    # Anchors (stage 4): V(solved)=0 and the 24 depth-1 children = 1, exact, every batch.
    anchor_v0 = SOLVED_T.unsqueeze(0)                                    # (1, S)
    anchor_d1 = apply_all_generators(anchor_v0, GENERATORS_T).squeeze(0)  # (24, S)
    if cfg['n_anchor_v0'] or cfg['n_anchor_d1']:
        print(f"[{stage_name}] anchor mixin: V0 x {cfg['n_anchor_v0']}, "
              f"d=1 children 24 x {cfg['n_anchor_d1']}", flush=True)

    bfs6_states = bfs6_dists = None
    bfs6_per_batch = 0
    if cfg['bfs_d6_fraction'] > 0:
        bfs6_states, bfs6_dists = load_bfs6()
        bfs6_per_batch = max(1, int(round(cfg['batch_size'] * cfg['bfs_d6_fraction'])))
    fr_states = None
    fr_per_batch = 0
    if cfg['frontier_fraction'] > 0:
        fr_states = load_frontier()
        fr_per_batch = max(1, int(round(cfg['batch_size'] * cfg['frontier_fraction'])))
    print(f"[{stage_name}] per batch: rw {cfg['batch_size'] - bfs6_per_batch - fr_per_batch}"
          f' + bfs6 {bfs6_per_batch} + frontier {fr_per_batch}'
          f" + anchors {cfg['n_anchor_v0'] + 24 * cfg['n_anchor_d1']}", flush=True)

    target_fwd = lambda x: target_model(x)
    last_path = None
    for epoch in range(start_epoch, cfg['epochs']):
        t0 = time.time()
        n_walks = max(1, cfg['samples_per_epoch'] // cfg['k_max'])
        states, depths = generate_walks_torch(PUZZLE, n_walks=n_walks, k_max=cfg['k_max'],
                                              seed=CFG['seed'] + epoch, device=DEVICE,
                                              n_back=cfg['n_back'])
        depths_f = depths.to(torch.float32)
        rw_per_batch = cfg['batch_size'] - bfs6_per_batch - fr_per_batch

        # Pre-sample this epoch's mixin rows (fresh shuffle per epoch).
        n_batches_est = states.size(0) // rw_per_batch + 1
        if bfs6_states is not None:
            take = min(bfs6_per_batch * n_batches_est, bfs6_states.size(0))
            perm = torch.randperm(bfs6_states.size(0), generator=cpu_gen)[:take]
            ep_bfs_states = bfs6_states[perm].to(DEVICE)
            ep_bfs_dists = bfs6_dists[perm].to(DEVICE)
        if fr_states is not None:
            take = min(fr_per_batch * n_batches_est, fr_states.size(0))
            perm = torch.randperm(fr_states.size(0), generator=cpu_gen)[:take]
            ep_fr_states = fr_states[perm].to(DEVICE)

        model_c.train()
        total, nb = 0.0, 0
        bfs_cur = fr_cur = 0
        for idx in iterate_batches(states.shape[0], rw_per_batch, batch_gen, DEVICE):
            bs_rw = states[idx]
            bd_rw = depths_f[idx]
            target_rw = bellman_targets(target_fwd, bs_rw, bd_rw, cfg['target_net_chunk'],
                                        cfg['clip_upper'], cfg['clip_lower'])
            bs_parts, tg_parts = [bs_rw], [target_rw]
            if cfg['n_anchor_v0'] > 0:
                bs_parts.append(anchor_v0.expand(cfg['n_anchor_v0'], -1))
                tg_parts.append(torch.zeros(cfg['n_anchor_v0'], device=DEVICE))
            if cfg['n_anchor_d1'] > 0:
                bs_parts.append(anchor_d1.repeat(cfg['n_anchor_d1'], 1))
                tg_parts.append(torch.ones(N_GENERATORS * cfg['n_anchor_d1'], device=DEVICE))
            if bfs6_states is not None and bfs_cur + bfs6_per_batch <= ep_bfs_states.size(0):
                bs_parts.append(ep_bfs_states[bfs_cur:bfs_cur + bfs6_per_batch])
                tg_parts.append(ep_bfs_dists[bfs_cur:bfs_cur + bfs6_per_batch])
                bfs_cur += bfs6_per_batch
            if fr_states is not None and fr_cur + fr_per_batch <= ep_fr_states.size(0):
                bs_fr = ep_fr_states[fr_cur:fr_cur + fr_per_batch]
                fr_cur += fr_per_batch
                # Frontier states have no walk-depth bound; use a synthetic cap so
                # clip_upper effectively no-ops while clip_lower still applies.
                bd_fr = torch.full((bs_fr.size(0),), float(cfg['frontier_walk_depth_cap']),
                                   device=DEVICE)
                tg_fr = bellman_targets(target_fwd, bs_fr, bd_fr, cfg['target_net_chunk'],
                                        cfg['clip_upper'], cfg['clip_lower'])
                bs_parts.append(bs_fr)
                tg_parts.append(tg_fr)
            bs = torch.cat([p.to(torch.int64) for p in bs_parts], dim=0)
            tg = torch.cat(tg_parts, dim=0)
            with autocast_ctx:
                pred = model_c(bs)
                loss = F.mse_loss(pred.float(), tg)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total += float(loss.item())
            nb += 1
        sched.step()
        avg = total / max(nb, 1)
        if epoch % 5 == 0 or epoch == cfg['epochs'] - 1:
            print(f'epoch {epoch:4d} | loss {avg:.4f} | lr {sched.get_last_lr()[0]:.2e} '
                  f'| {time.time() - t0:.1f}s', flush=True)
        if (epoch + 1) % cfg['target_update_every'] == 0:
            target_model.load_state_dict(clean_state_dict(model.state_dict()))
        if (epoch + 1) % cfg['checkpoint_every'] == 0 or epoch == cfg['epochs'] - 1:
            last_path = out_dir / f'epoch_{epoch:04d}.pt'
            save_v_ckpt(last_path, model, epoch, loss=avg, optim=optim, sched=sched)
        if training_time_up() and epoch < cfg['epochs'] - 1:
            last_path = out_dir / f'epoch_{epoch:04d}.pt'
            save_v_ckpt(last_path, model, epoch, loss=avg, optim=optim, sched=sched)
            print(f'*** wall budget reached at epoch {epoch} - stopping {stage_name}; '
                  f'the next run resumes from {last_path.name}', flush=True)
            return last_path, False
    return last_path, True
