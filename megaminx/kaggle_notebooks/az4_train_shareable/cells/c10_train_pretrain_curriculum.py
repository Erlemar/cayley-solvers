### Stage 1 (pretrain) + stage 2 (curriculum) trainers ###
import copy


def new_v_model():
    return build_v_model(output_dim=1).to(DEVICE)


# Loss factory for the regression stages (same pattern as the community notebook,
# including its two custom losses). Bellman/az losses are structural and stay fixed.
class PinballLoss(nn.Module):
    """Quantile (pinball) loss; tau < 0.5 targets the lower quantile."""

    def __init__(self, tau=0.2):
        super().__init__()
        self.tau = tau

    def forward(self, pred, target):
        e = target - pred
        return torch.mean(torch.maximum(self.tau * e, (self.tau - 1.0) * e))


class LogCoshLoss(nn.Module):
    def forward(self, pred, target):
        ax = torch.abs(pred - target)
        return torch.mean(ax + F.softplus(-2.0 * ax) - 0.6931471805599453)


CUSTOM_LOSSES = {'PinballLoss': PinballLoss, 'LogCoshLoss': LogCoshLoss}


def build_loss(loss_cfg):
    name = (loss_cfg or {}).get('name') or 'MSELoss'
    params = dict((loss_cfg or {}).get('params') or {})
    cls = CUSTOM_LOSSES.get(name) or getattr(nn, name, None)
    if cls is None or not (isinstance(cls, type) and issubclass(cls, nn.Module)):
        raise ValueError(f'unknown loss {name!r} (torch.nn loss or {sorted(CUSTOM_LOSSES)})')
    return cls(**params)


def load_warmstart_strict(model, path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(clean_state_dict(ck['state_dict']))
    print(f'  warm-start (strict) from {Path(path).name}', flush=True)


def save_v_ckpt(path, model, epoch, loss=None, optim=None, sched=None, extra=None):
    ck = {'epoch': epoch, 'state_dict': clean_state_dict(model.state_dict()),
          'model_config': v_model_config(), 'loss': loss}
    if optim is not None:
        ck['optim_state'] = optim.state_dict()
    if sched is not None:
        ck['sched_state'] = sched.state_dict()
    if extra:
        ck.update(extra)
    torch.save(ck, path)


def try_resume(model, optim, sched, resume_path):
    """Returns the epoch to start from (0 if no resume)."""
    if not resume_path:
        return 0
    ck = torch.load(resume_path, map_location='cpu', weights_only=False)
    model.load_state_dict(clean_state_dict(ck['state_dict']))
    if optim is not None and 'optim_state' in ck:
        optim.load_state_dict(ck['optim_state'])
    if sched is not None and 'sched_state' in ck:
        sched.load_state_dict(ck['sched_state'])
    start = int(ck.get('epoch', -1)) + 1
    print(f'  resumed from {resume_path} at epoch {start}', flush=True)
    return start


def iterate_batches(n, batch_size, generator, device):
    idx = torch.randperm(n, generator=generator, device=device)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]


def run_pretrain(cfg, warmstart=None):
    """Plain random-walk MSE regression, fresh walks every epoch."""
    out_dir = MODELS_DIR / 'pretrain'
    out_dir.mkdir(parents=True, exist_ok=True)
    model = new_v_model()
    if warmstart:
        load_warmstart_strict(model, warmstart)
    optim = make_optimizer(model.parameters(), cfg)
    loss_fn = build_loss(cfg.get('loss'))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg['epochs'])
    start_epoch = try_resume(model, optim, sched, cfg['resume'])
    model_c = maybe_compile(model)
    autocast_ctx = make_autocast()
    batch_gen = torch.Generator(device=DEVICE)
    batch_gen.manual_seed(CFG['seed'])
    last_path = None
    for epoch in range(start_epoch, cfg['epochs']):
        t0 = time.time()
        n_walks = max(1, cfg['samples_per_epoch'] // cfg['k_max'])
        states, depths = generate_walks_torch(PUZZLE, n_walks=n_walks, k_max=cfg['k_max'],
                                              seed=CFG['seed'] + epoch, device=DEVICE,
                                              n_back=cfg['n_back'])
        depths_f = depths.to(torch.float32)
        model_c.train()
        total, nb = 0.0, 0
        for idx in iterate_batches(states.shape[0], cfg['batch_size'], batch_gen, DEVICE):
            with autocast_ctx:
                pred = model_c(states[idx])
                loss = loss_fn(pred, depths_f[idx])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total += float(loss.item())
            nb += 1
        sched.step()
        avg = total / max(nb, 1)
        if epoch % 25 == 0 or epoch == cfg['epochs'] - 1:
            print(f'epoch {epoch:4d} | loss {avg:.4f} | lr {sched.get_last_lr()[0]:.2e} '
                  f'| {time.time() - t0:.1f}s', flush=True)
        if (epoch + 1) % cfg['checkpoint_every'] == 0 or epoch == cfg['epochs'] - 1:
            last_path = out_dir / f'epoch_{epoch:04d}.pt'
            save_v_ckpt(last_path, model, epoch, loss=avg, optim=optim, sched=sched)
        if training_time_up() and epoch < cfg['epochs'] - 1:
            last_path = out_dir / f'epoch_{epoch:04d}.pt'
            save_v_ckpt(last_path, model, epoch, loss=avg, optim=optim, sched=sched)
            print(f'*** wall budget reached at epoch {epoch} - stopping pretrain; '
                  f'the next run resumes from {last_path.name}', flush=True)
            return last_path, False
    return last_path, True


@torch.no_grad()
def _ema_update(ema_model, source, tau):
    for ep_, sp in zip(ema_model.parameters(), source.parameters()):
        ep_.mul_(1.0 - tau).add_(sp.detach(), alpha=tau)
    for eb, sb in zip(ema_model.buffers(), source.buffers()):
        if eb.dtype.is_floating_point:
            eb.mul_(1.0 - tau).add_(sb.detach(), alpha=tau)
        else:
            eb.copy_(sb)


@torch.no_grad()
def _eval_val_mse(model, val_states, val_depths, batch_size=8192):
    model.eval()
    total, seen = 0.0, 0
    for i in range(0, val_states.size(0), batch_size):
        bs = val_states[i:i + batch_size]
        bd = val_depths[i:i + batch_size]
        total += float(F.mse_loss(model(bs), bd, reduction='sum').item())
        seen += bs.size(0)
    return total / max(seen, 1)


def run_curriculum(cfg, warmstart):
    """Mixed-k curriculum fine-tune with per-step EMA and val-based early stopping.

    Output = best_ema.pt (the EMA snapshot at the best validation epoch).
    """
    out_dir = MODELS_DIR / 'curriculum'
    out_dir.mkdir(parents=True, exist_ok=True)
    model = new_v_model()
    if warmstart:  # None when resuming (the resume checkpoint restores everything)
        load_warmstart_strict(model, warmstart)
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad = False
    optim = make_optimizer(model.parameters(), cfg)
    loss_fn = build_loss(cfg.get('loss'))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg['budget_epochs'])
    start_epoch = try_resume(model, optim, sched, cfg['resume'])
    if cfg['resume']:
        ck = torch.load(cfg['resume'], map_location='cpu', weights_only=False)
        if 'ema_state_dict' in ck:
            ema_model.load_state_dict(clean_state_dict(ck['ema_state_dict']))
    model_c = maybe_compile(model)
    autocast_ctx = make_autocast()
    base_model = getattr(model_c, '_orig_mod', model_c)
    batch_gen = torch.Generator(device=DEVICE)
    batch_gen.manual_seed(CFG['seed'])
    k_gen = torch.Generator(device=DEVICE)
    k_gen.manual_seed(CFG['seed'] + 7777)

    mix_k = list(cfg['mix_k_list'])
    print(f"curriculum: warmup k={cfg['warmup_k_max']} for {cfg['warmup_epochs']} epochs, "
          f'then mix from {mix_k}', flush=True)
    val_states, val_depths = generate_walks_torch(
        PUZZLE, n_walks=cfg['val_n_walks'], k_max=cfg['val_k_max'],
        seed=CFG['seed'] - 1, device=DEVICE, n_back=cfg['n_back'])
    val_depths = val_depths.to(torch.float32)
    print(f'  fixed val set: {val_states.size(0):,} states', flush=True)

    best_val, best_epoch = float('inf'), -1
    best_path = out_dir / 'best_ema.pt'
    for epoch in range(start_epoch, cfg['budget_epochs']):
        t0 = time.time()
        if epoch < cfg['warmup_epochs']:
            k_for_epoch = cfg['warmup_k_max']
        else:
            k_for_epoch = mix_k[int(torch.randint(0, len(mix_k), (1,), generator=k_gen,
                                                  device=DEVICE).item())]
        n_walks = max(1, cfg['samples_per_epoch'] // k_for_epoch)
        states, depths = generate_walks_torch(PUZZLE, n_walks=n_walks, k_max=k_for_epoch,
                                              seed=CFG['seed'] + epoch, device=DEVICE,
                                              n_back=cfg['n_back'])
        depths_f = depths.to(torch.float32)
        model_c.train()
        total, nb = 0.0, 0
        for idx in iterate_batches(states.shape[0], cfg['batch_size'], batch_gen, DEVICE):
            with autocast_ctx:
                pred = model_c(states[idx])
                loss = loss_fn(pred, depths_f[idx])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            _ema_update(ema_model, base_model, cfg['ema_tau'])
            total += float(loss.item())
            nb += 1
        sched.step()

        if (epoch + 1) % cfg['val_every'] == 0 or epoch == cfg['budget_epochs'] - 1:
            val_ema = _eval_val_mse(ema_model, val_states, val_depths)
            print(f'epoch {epoch:5d} | k={k_for_epoch:<3} | train_mse {total / max(nb, 1):7.3f} '
                  f'| val_mse ema {val_ema:7.3f} | lr {sched.get_last_lr()[0]:.2e} '
                  f'| {time.time() - t0:.1f}s', flush=True)
            if val_ema < best_val:
                best_val, best_epoch = val_ema, epoch
                save_v_ckpt(best_path, ema_model, epoch, loss=val_ema,
                            extra={'val_mse': val_ema})
                print(f'  -> new best EMA (val_mse={val_ema:.4f}) saved', flush=True)
            elif (epoch - best_epoch >= cfg['patience']
                  and epoch >= cfg['warmup_epochs'] + cfg['patience']):
                print(f'*** early stop at epoch {epoch}: best val_mse {best_val:.4f} '
                      f'at epoch {best_epoch}', flush=True)
                break
        if (epoch + 1) % cfg['checkpoint_every'] == 0:
            save_v_ckpt(out_dir / f'epoch_{epoch:05d}.pt', base_model, epoch,
                        optim=optim, sched=sched,
                        extra={'ema_state_dict': clean_state_dict(ema_model.state_dict())})
        if training_time_up() and epoch < cfg['budget_epochs'] - 1:
            stop_path = out_dir / f'epoch_{epoch:05d}.pt'
            save_v_ckpt(stop_path, base_model, epoch, optim=optim, sched=sched,
                        extra={'ema_state_dict': clean_state_dict(ema_model.state_dict())})
            print(f'*** wall budget reached at epoch {epoch} - stopping curriculum; '
                  f'the next run resumes from {stop_path.name} '
                  f'(note: best-val/patience tracking restarts on resume)', flush=True)
            return stop_path, False
    assert best_path.exists(), 'curriculum produced no best_ema.pt (increase budget?)'
    return best_path, True
