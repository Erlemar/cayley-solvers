### Library code: MODEL POOL + checkpoint helpers (no knobs here) ###
# Two model families are registered out of the box, selected by CFG['model']['model_type']:
#   'ResMLPDistance'  - our architecture (embedding + LayerNorm ResMLP; AZ v4 = the 6M default)
#   'PilgrimAttnRes'  - the community baseline family from @ogurtsov's modular notebook
#                       (one-hot + BatchNorm; block_type 'residual' or 'attn_res')
# Every registered model works in EVERY stage: stages 1-4 train it as a plain V model
# (native head), stage 5 wraps the same backbone with policy+value heads (DualHeadModel).
#
# To add your own model: define a class with
#     forward(x) -> (B,) [or (B, output_dim)],  features(x) -> (B, feature_dim),
#     a feature_dim attribute, and an output head whose parameter name you register.
# Then:  MODEL_REGISTRY['MyNet'] = {'cls': MyNet, 'head_attr': 'my_head'}
# and put its constructor kwargs under CFG['model']['MyNet'].
import torch.nn as nn
import torch.nn.functional as F


# ---------------- our family: embedding + LayerNorm ResMLP ----------------

class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.lin2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        h = F.relu(self.ln1(self.lin1(x)))
        h = self.ln2(self.lin2(h))
        return F.relu(x + h)


class ResMLPDistance(nn.Module):
    """Distance predictor: int state (B, 120) -> (B,) scalar (or (B, output_dim)).

    `inference_chunk_size` bounds eval-mode VRAM when beam search hands us 100k+ states.
    """

    def __init__(self, state_size=120, num_classes=120, hidden_dims=(2048, 512),
                 num_res_blocks=2, inference_chunk_size=4096, encoding='embedding',
                 embed_dim=16, output_dim=1):
        super().__init__()
        self.state_size = state_size
        self.num_classes = num_classes
        self.inference_chunk_size = inference_chunk_size
        self.encoding = encoding
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        if encoding == 'onehot':
            in_dim = state_size * num_classes
            self.embedding = None
        elif encoding == 'embedding':
            in_dim = state_size * embed_dim
            self.embedding = nn.Embedding(num_classes, embed_dim)
        else:
            raise ValueError(f'unsupported encoding {encoding!r}')
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(inplace=True)]
            prev = h
        self.input_stack = nn.Sequential(*layers)
        self.res_blocks = nn.ModuleList([ResBlock(prev) for _ in range(num_res_blocks)])
        self.head = nn.Linear(prev, output_dim)
        self.feature_dim = prev

    def encode(self, x):
        target_dtype = self.input_stack[0].weight.dtype
        if self.encoding == 'onehot':
            return F.one_hot(x.long(), self.num_classes).to(target_dtype).flatten(start_dim=-2)
        return self.embedding(x.long()).to(target_dtype).flatten(start_dim=-2)

    def features(self, x):
        h = self.encode(x)
        h = self.input_stack(h)
        for block in self.res_blocks:
            h = block(h)
        return h

    def _forward_single(self, x):
        out = self.head(self.features(x))
        return out.squeeze(-1) if self.output_dim == 1 else out

    def forward(self, x):
        if self.training or self.inference_chunk_size is None or x.shape[0] <= self.inference_chunk_size:
            return self._forward_single(x)
        outs = []
        for i in range(0, x.shape[0], self.inference_chunk_size):
            outs.append(self._forward_single(x[i:i + self.inference_chunk_size]))
        return torch.cat(outs, dim=0)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ------- community family: one-hot + BatchNorm MLP (from @ogurtsov's notebook) -------

class RMSNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class BNResidualBlock(nn.Module):
    """Two-layer residual block with BatchNorm (their 'residual' block type)."""

    def __init__(self, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        return self.relu(out + x)


class AttnResidualBlock(nn.Module):
    """Residual MLP block with attention over previous block outputs (their 'attn_res')."""

    def __init__(self, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.query = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.rmsnorm = RMSNorm(hidden_dim)
        nn.init.zeros_(self.query)

    def forward(self, x, prev_outputs):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        if len(prev_outputs) > 0:
            values = torch.stack(prev_outputs, dim=0)          # (p, B, d)
            keys = self.rmsnorm(values)
            q = self.query.view(1, 1, -1).expand(1, values.size(1), -1)
            logits = torch.einsum('b d, p b d -> p b', q.squeeze(0), keys)
            weights = F.softmax(logits, dim=0).unsqueeze(-1)
            agg = (weights * values).sum(dim=0)
        else:
            agg = torch.zeros_like(out)
        return self.relu(out + agg)


class PilgrimAttnRes(nn.Module):
    """One-hot + BatchNorm MLP with plain or attention-residual blocks.

    Port of @ogurtsov's PilgrimAttnRes, re-parameterized with explicit kwargs and
    extended with `output_dim`, `features()`, and chunked eval-mode forwards so it can
    drive every stage of this pipeline. Math is unchanged.
    """

    def __init__(self, state_size=120, num_classes=120, hd1=1024, hd2=256, nrd=4,
                 n_attn_blocks=0, block_type='residual', dropout_rate=0.0,
                 output_dim=1, inference_chunk_size=4096):
        super().__init__()
        self.state_size = state_size
        self.num_classes = num_classes
        self.block_type = block_type
        self.output_dim = output_dim
        self.inference_chunk_size = inference_chunk_size

        self.input_layer = nn.Linear(state_size * num_classes, hd1)
        self.bn1 = nn.BatchNorm1d(hd1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        if hd2 > 0:
            self.hidden_layer = nn.Linear(hd1, hd2)
            self.bn2 = nn.BatchNorm1d(hd2)
            hidden_dim_for_output = hd2
        else:
            self.hidden_layer = None
            self.bn2 = None
            hidden_dim_for_output = hd1
        if block_type == 'residual':
            self.residual_blocks = (nn.ModuleList(
                [BNResidualBlock(hd2, dropout_rate) for _ in range(nrd)])
                if nrd > 0 and hd2 > 0 else None)
        elif block_type == 'attn_res':
            self.residual_blocks = (nn.ModuleList(
                [AttnResidualBlock(hd2, dropout_rate) for _ in range(n_attn_blocks)])
                if n_attn_blocks > 0 and hd2 > 0 else None)
        else:
            raise ValueError(f'unknown block_type {block_type!r}')
        self.output_layer = nn.Linear(hidden_dim_for_output, output_dim)
        self.feature_dim = hidden_dim_for_output

    def features(self, x):
        target_dtype = self.input_layer.weight.dtype
        h = F.one_hot(x.long(), num_classes=self.num_classes).view(x.size(0), -1).to(target_dtype)
        h = self.dropout(self.relu(self.bn1(self.input_layer(h))))
        if self.hidden_layer is not None:
            h = self.dropout(self.relu(self.bn2(self.hidden_layer(h))))
        if self.residual_blocks is not None:
            if self.block_type == 'residual':
                for block in self.residual_blocks:
                    h = block(h)
            else:
                prev_outputs = []
                for block in self.residual_blocks:
                    h = block(h, prev_outputs)
                    prev_outputs.append(h)
        return h

    def _forward_single(self, x):
        out = self.output_layer(self.features(x))
        return out.squeeze(-1) if self.output_dim == 1 else out

    def forward(self, x):
        if self.training or self.inference_chunk_size is None or x.shape[0] <= self.inference_chunk_size:
            return self._forward_single(x)
        outs = []
        for i in range(0, x.shape[0], self.inference_chunk_size):
            outs.append(self._forward_single(x[i:i + self.inference_chunk_size]))
        return torch.cat(outs, dim=0)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ---------------- registry + builders ----------------

MODEL_REGISTRY = {
    'ResMLPDistance': {'cls': ResMLPDistance, 'head_attr': 'head'},
    'PilgrimAttnRes': {'cls': PilgrimAttnRes, 'head_attr': 'output_layer'},
}


def model_params(model_type=None):
    t = model_type or CFG['model']['model_type']
    return dict(CFG['model'][t])


def build_v_model(model_type=None, params=None, output_dim=1):
    t = model_type or CFG['model']['model_type']
    p = dict(params) if params is not None else model_params(t)
    return MODEL_REGISTRY[t]['cls'](output_dim=output_dim, **p)


def v_model_config(model_type=None, params=None, output_dim=1):
    """Checkpoint model_config. For ResMLPDistance also writes the legacy flat keys so
    exported value heads stay drop-in compatible with our production solver stack."""
    t = model_type or CFG['model']['model_type']
    p = dict(params) if params is not None else model_params(t)
    mc = {'model_type': t, 'params': p, 'output_dim': output_dim}
    if t == 'ResMLPDistance':
        mc.update({'model_class': 'ResMLPDistance', 'state_size': p.get('state_size', 120),
                   'num_classes': p.get('num_classes', 120),
                   'hidden_dims': list(p['hidden_dims']),
                   'num_res_blocks': p['num_res_blocks'],
                   'encoding': 'embedding', 'embed_dim': p['embed_dim']})
    return mc


class DualHeadModel(nn.Module):
    """Any registered backbone + policy head (n_actions) + value head (1).

    The backbone's own native head is carried but unused; policy/value read
    backbone.features(x). This is what the 'az' stage trains.
    """

    def __init__(self, backbone, n_actions):
        super().__init__()
        self.backbone = backbone
        self.n_actions = n_actions
        self.policy_head = nn.Linear(backbone.feature_dim, n_actions)
        self.value_head = nn.Linear(backbone.feature_dim, 1)

    def features(self, x):
        return self.backbone.features(x)

    def forward(self, x):
        h = self.features(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ---------------- checkpoint helpers ----------------

def clean_state_dict(sd):
    return {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}


def _legacy_resmlp_kwargs(mc):
    keep = ('state_size', 'num_classes', 'num_res_blocks', 'encoding', 'embed_dim',
            'output_dim')
    kw = {k: mc[k] for k in keep if k in mc}
    if 'hidden_dims' in mc:
        kw['hidden_dims'] = tuple(mc['hidden_dims'])
    return kw


def load_v_model(path, device=DEVICE):
    """Load any V checkpoint: new-style (model_type + params) or legacy ResMLP."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = clean_state_dict(ck['state_dict'])
    mc = ck.get('model_config', {})
    if 'model_type' in mc and 'params' in mc:
        model = build_v_model(mc['model_type'], mc['params'],
                              output_dim=mc.get('output_dim', 1))
    else:
        model = ResMLPDistance(**_legacy_resmlp_kwargs(mc))
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def warmstart_v_into_dual(dual, ckpt_path, model_type):
    """Copy a V checkpoint into the dual-head backbone; init value_head from its head."""
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = clean_state_dict(ck.get('state_dict', ck))
    backbone_sd = dual.backbone.state_dict()
    compatible = {k: v for k, v in sd.items()
                  if k in backbone_sd and tuple(backbone_sd[k].shape) == tuple(v.shape)}
    skipped = len(sd) - len(compatible)
    dual.backbone.load_state_dict(compatible, strict=False)
    print(f'  warm-start: loaded {len(compatible)} backbone tensors '
          f'({skipped} incompatible skipped)', flush=True)
    if skipped:
        print('  WARNING: incompatible tensors start RANDOM - a different model_type or '
              'shape needs the earlier stages retrained, not a warm-start.', flush=True)
    head = MODEL_REGISTRY[model_type]['head_attr']
    hw, hb = f'{head}.weight', f'{head}.bias'
    if hw in sd and tuple(dual.value_head.weight.shape) == tuple(sd[hw].shape):
        with torch.no_grad():
            dual.value_head.weight.copy_(sd[hw])
            dual.value_head.bias.copy_(sd[hb])
        print('  initialized value_head = V distance head', flush=True)


def export_dual_head(az_ckpt_path, out_path, head='value'):
    """Extract one head of a dual-head checkpoint as a NATIVE V-model checkpoint of the
    same backbone type (value -> beam-search distance model; policy -> n_actions scorer)."""
    ck = torch.load(az_ckpt_path, map_location='cpu', weights_only=False)
    sd = clean_state_dict(ck['state_dict'])
    mc = ck['model_config']
    t, p = mc['model_type'], mc['params']
    out_dim = 1 if head == 'value' else int(mc.get('n_actions', N_GENERATORS))
    head_attr = MODEL_REGISTRY[t]['head_attr']
    new_sd = {k.removeprefix('backbone.'): v for k, v in sd.items()
              if k.startswith('backbone.')}
    src = 'value_head' if head == 'value' else 'policy_head'
    new_sd[f'{head_attr}.weight'] = sd[f'{src}.weight']
    new_sd[f'{head_attr}.bias'] = sd[f'{src}.bias']
    model = build_v_model(t, p, output_dim=out_dim)   # validates shapes on load
    model.load_state_dict(new_sd)
    torch.save({'epoch': ck.get('epoch', -1), 'state_dict': new_sd,
                'model_config': v_model_config(t, p, output_dim=out_dim),
                'source_az_checkpoint': str(az_ckpt_path)}, out_path)
    print(f'exported {head} head -> {out_path}', flush=True)
    return out_path


_probe = build_v_model()
print(f"model pool: {list(MODEL_REGISTRY)} | selected: {CFG['model']['model_type']} "
      f'({_probe.num_parameters():,} params)')
del _probe
