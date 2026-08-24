### Stage 5 trainer: dual-head AZ fine-tune (policy CE + Bellman value) ###

@torch.no_grad()
def az_value_forward(model, states, chunk_size=4096):
    outs = []
    for i in range(0, states.size(0), chunk_size):
        h = model.features(states[i:i + chunk_size])
        outs.append(model.value_head(h).squeeze(-1))
    return torch.cat(outs, dim=0)


@torch.no_grad()
def _quick_value_canary(model):
    """V(solved) and mean V(depth-1) from the value head - printed at checkpoints."""
    model.eval()  # BatchNorm backbones cannot do a batch-of-1 forward in train mode
    v0 = az_value_forward(model, SOLVED_T.unsqueeze(0)).item()
    d1 = apply_all_generators(SOLVED_T.unsqueeze(0), GENERATORS_T).squeeze(0)
    v1 = az_value_forward(model, d1).mean().item()
    return v0, v1


def run_az(cfg, warmstart):
    """Reproduces the AZ v4 run: value head trained like stage 4 (minus frontier),
    policy head CE on solution paths, shared trunk. STOP EARLY - see select_epoch."""
    out_dir = MODELS_DIR / 'az'
    out_dir.mkdir(parents=True, exist_ok=True)
    model_type = CFG['model']['model_type']
    model = DualHeadModel(build_v_model(output_dim=1),
                          n_actions=N_GENERATORS).to(DEVICE)
    print(f'AZ model: {model_type} backbone, {model.num_parameters():,} params', flush=True)
    if warmstart:  # None when resuming (the resume checkpoint restores everything)
        warmstart_v_into_dual(model, warmstart, model_type)
    target_model = copy.deepcopy(model).eval()
    for p in target_model.parameters():
        p.requires_grad = False

    policy = torch.load(POLICY_DATASET_PATH, map_location='cpu', weights_only=False)
    states_p = policy['states'].to(DEVICE).long()
    actions_p = policy['actions'].to(DEVICE).long()
    Np = states_p.size(0)
    print(f'policy dataset: {Np:,} (state, action) pairs '
          f"from {policy.get('source', '?')}", flush=True)

    bfs6_states = bfs6_dists = None
    bfs6_per_batch = 0
    if cfg['bfs_d6_fraction'] > 0:
        bfs6_states, bfs6_dists = load_bfs6()
        bfs6_per_batch = max(1, int(round(cfg['rw_batch_size'] * cfg['bfs_d6_fraction'])))

    optim = make_optimizer(model.parameters(), cfg)
    # NOTE: cosine T_max is lr_t_max (=200), NOT epochs - reproduces the original
    # schedule even when running only the first ~30 epochs.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg['lr_t_max'])
    start_epoch = try_resume(model, optim, sched, cfg['resume'])
    if start_epoch > 0:
        target_model.load_state_dict(clean_state_dict(model.state_dict()))
    autocast_ctx = make_autocast()
    rng_p = torch.Generator(device=DEVICE)
    rng_p.manual_seed(CFG['seed'])
    cpu_gen = torch.Generator(device='cpu')
    cpu_gen.manual_seed(CFG['seed'] + 1)

    anchor_v0 = SOLVED_T.unsqueeze(0)
    anchor_d1 = apply_all_generators(anchor_v0, GENERATORS_T).squeeze(0)
    target_fwd = lambda x: az_value_forward(target_model, x)

    print(f"epochs: {cfg['epochs']}  rw_batch: {cfg['rw_batch_size']}  "
          f"policy_batch: {cfg['policy_batch_size']}  alpha={cfg['alpha']}  "
          f"beta={cfg['beta']}", flush=True)

    def save_az_ckpt(epoch, avg_p, avg_v, acc):
        path = out_dir / f'epoch_{epoch:04d}.pt'
        torch.save({
            'epoch': epoch, 'state_dict': clean_state_dict(model.state_dict()),
            'model_config': {'model_type': model_type, 'params': model_params(),
                             'n_actions': N_GENERATORS},
            'p_loss': avg_p, 'v_loss': avg_v, 'top1_acc': acc,
            'optim_state': optim.state_dict(), 'sched_state': sched.state_dict(),
        }, path)
        return path

    ckpts = {}
    completed = True
    for epoch in range(start_epoch, cfg['epochs']):
        t0 = time.time()
        n_walks = max(1, cfg['samples_per_epoch'] // cfg['k_max'])
        rw_states, rw_depths = generate_walks_torch(
            PUZZLE, n_walks=n_walks, k_max=cfg['k_max'],
            seed=CFG['seed'] + epoch * 1000, device=DEVICE, n_back=cfg['n_back'])
        rw_depths_f = rw_depths.to(torch.float32)
        N_rw = rw_states.size(0)

        rw_per_batch = (cfg['rw_batch_size'] - bfs6_per_batch
                        - cfg['anchor_v0'] - N_GENERATORS * cfg['anchor_d1'])
        if bfs6_states is not None:
            take = min(bfs6_per_batch * (N_rw // rw_per_batch + 1), bfs6_states.size(0))
            perm = torch.randperm(bfs6_states.size(0), generator=cpu_gen)[:take]
            ep_bfs_states = bfs6_states[perm].to(DEVICE).long()
            ep_bfs_dists = bfs6_dists[perm].to(DEVICE)

        n_rw_batches = max(1, N_rw // rw_per_batch)
        bfs_cur = 0
        tot_p = tot_v = tot_acc = 0.0
        model.train()
        for b in range(n_rw_batches):
            bs_rw = rw_states[b * rw_per_batch:(b + 1) * rw_per_batch]
            bd_rw = rw_depths_f[b * rw_per_batch:(b + 1) * rw_per_batch]
            target_rw = bellman_targets(target_fwd, bs_rw, bd_rw, chunk_size=4096)

            bs_parts, tg_parts = [bs_rw], [target_rw]
            if cfg['anchor_v0'] > 0:
                bs_parts.append(anchor_v0.expand(cfg['anchor_v0'], -1))
                tg_parts.append(torch.zeros(cfg['anchor_v0'], device=DEVICE))
            if cfg['anchor_d1'] > 0:
                bs_parts.append(anchor_d1.repeat(cfg['anchor_d1'], 1))
                tg_parts.append(torch.ones(N_GENERATORS * cfg['anchor_d1'], device=DEVICE))
            if bfs6_states is not None and bfs_cur + bfs6_per_batch <= ep_bfs_states.size(0):
                bs_parts.append(ep_bfs_states[bfs_cur:bfs_cur + bfs6_per_batch])
                tg_parts.append(ep_bfs_dists[bfs_cur:bfs_cur + bfs6_per_batch])
                bfs_cur += bfs6_per_batch
            bs_v = torch.cat(bs_parts, dim=0)
            tg_v = torch.cat(tg_parts, dim=0)

            idx_p = torch.randint(0, Np, (cfg['policy_batch_size'],), generator=rng_p,
                                  device=DEVICE)
            bs_p = states_p[idx_p]
            ba_p = actions_p[idx_p]

            with autocast_ctx:
                h_v = model.features(bs_v)
                pred_v = model.value_head(h_v).squeeze(-1)
                v_loss = F.mse_loss(pred_v.float(), tg_v)
                h_p = model.features(bs_p)
                logits_p = model.policy_head(h_p)
                p_loss = F.cross_entropy(logits_p, ba_p)
                loss = cfg['alpha'] * p_loss + cfg['beta'] * v_loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            tot_p += float(p_loss.item())
            tot_v += float(v_loss.item())
            with torch.no_grad():
                tot_acc += float((logits_p.argmax(dim=1) == ba_p).float().mean().item())
        sched.step()

        avg_p, avg_v = tot_p / n_rw_batches, tot_v / n_rw_batches
        acc = tot_acc / n_rw_batches
        print(f'epoch {epoch:4d} | p_loss {avg_p:.4f} | v_loss {avg_v:.4f} | '
              f'top-1 acc {acc:.4f} | lr {sched.get_last_lr()[0]:.2e} '
              f'| {time.time() - t0:.1f}s', flush=True)
        if acc > 0.40:
            print('  NOTE: top-1 acc > 40% - likely PAST the value-quality sweet spot; '
                  'prefer an earlier checkpoint.', flush=True)

        if (epoch + 1) % cfg['target_update_every'] == 0:
            target_model.load_state_dict(model.state_dict())

        if (epoch + 1) % cfg['checkpoint_every'] == 0 or epoch == cfg['epochs'] - 1:
            path = save_az_ckpt(epoch, avg_p, avg_v, acc)
            ckpts[epoch] = path
            v0, v1 = _quick_value_canary(model)
            model.train()
            print(f'  checkpoint {path.name} | V(solved)={v0:+.3f} V(d=1)={v1:+.3f} '
                  f'(want ~0 and ~1)', flush=True)
        if training_time_up() and epoch < cfg['epochs'] - 1:
            ckpts[epoch] = save_az_ckpt(epoch, avg_p, avg_v, acc)
            print(f'*** wall budget reached at epoch {epoch} - stopping az; '
                  f'the next run resumes from {ckpts[epoch].name}', flush=True)
            completed = False
            break

    sel = cfg['select_epoch']
    if sel == 'last' or sel not in ckpts:
        sel_epoch = max(ckpts)
        if sel != 'last' and completed:
            print(f'select_epoch {sel} was not checkpointed; using last ({sel_epoch})')
    else:
        sel_epoch = sel
    if not completed:
        sel_epoch = max(ckpts)  # incomplete stage: the resume point, not a selection
    print(f'selected AZ checkpoint: epoch {sel_epoch}', flush=True)
    return ckpts[sel_epoch], completed
