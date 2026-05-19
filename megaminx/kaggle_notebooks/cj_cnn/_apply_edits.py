"""Add four 1D-CNN architectures to the cj_mlp clone:
   CNN (Embedded1DCNN_Flexible), TCN (TCNModel), GatedTCN (GatedTCN1D), HierCNN (HierarchicalCNN1D).

Edits:
 - Cell 12: append CNN class defs and extend GetModel.
 - Cell 2:  append 4 commented cfg variants.
 - Insert 4 commented run cells after the existing MLP_m05 run cell.
"""
import json

p = "megaminx/kaggle_notebooks/cj_cnn/cayleypy-rw-cj-cnn.ipynb"
with open(p, encoding="utf-8") as f:
    nb = json.load(f)

# ---- Cell 12: insert CNN class defs + replace GetModel ----
cell12 = nb["cells"][12]
src12 = "".join(cell12["source"]) if isinstance(cell12["source"], list) else cell12["source"]

marker = "def GetModel(CFG: dict[str, Any]):"
idx = src12.find(marker)
assert idx != -1, "GetModel marker not found in cell 12"
src12_head = src12[:idx].rstrip() + "\n\n"

cnn_classes_and_dispatcher = '''# ---- 1D-CNN architectures (ported from notebooks 1dcnn_local.ipynb / Colab) ----
# Four families:
#   * Embedded1DCNN_Flexible: 2-conv-block CNN, length-agnostic via AdaptiveAvgPool1d.
#   * TCNModel              : dilated TCN with simple residual blocks, dilations 1,2,4,...
#   * GatedTCN1D            : WaveNet-style TCN with `tanh(filt) * sigmoid(gate)` activation.
#   * HierarchicalCNN1D     : 1D U-Net with skip connections and ConvTranspose1d upsampling.
#
# All four take (B, seq_len) integer-token inputs and return a scalar prediction.

class Embedded1DCNN_Flexible(nn.Module):
    """Length-agnostic 2-conv-block CNN with global pooling."""
    def __init__(self, vocab_size=96, embed_dim=32, out_dim=1, dropout_p=0.3):
        super().__init__()
        self.out_dim = out_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout_embed = nn.Dropout(p=dropout_p)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout_conv1 = nn.Dropout(p=dropout_p)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout_conv2 = nn.Dropout(p=dropout_p)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 512)
        self.dropout_fc1 = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.embedding(x.long())
        x = self.dropout_embed(x)
        x = x.permute(0, 2, 1)
        x = self.dropout_conv1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout_conv2(F.relu(self.bn2(self.conv2(x))))
        x = self.global_pool(x).view(x.size(0), -1)
        x = self.dropout_fc1(F.relu(self.fc1(x)))
        out = self.fc2(x)
        return out.squeeze(1) if self.out_dim == 1 else out


class TCNBlock(nn.Module):
    """Vanilla TCN block: two dilated convs + residual skip."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.3):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.dropout(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        return F.relu(out + residual)


class TCNModel(nn.Module):
    """Stacked TCNBlocks with dilations 1, 2, 4, 8, ..."""
    def __init__(self, vocab_size=100, embed_dim=32, hidden_dim=128,
                 num_levels=4, kernel_size=3, dropout=0.3, out_dim=1):
        super().__init__()
        self.out_dim = out_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1)
        blocks = []
        d = 1
        for _ in range(num_levels):
            blocks.append(TCNBlock(hidden_dim, hidden_dim, kernel_size, d, dropout))
            d *= 2
        self.tcn = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.embedding(x.long()).permute(0, 2, 1)
        x = self.tcn(self.proj(x))
        x = self.global_pool(x).squeeze(-1)
        out = self.fc(x)
        return out.squeeze(-1) if self.out_dim == 1 else out


class GatedResidualBlock1D(nn.Module):
    """WaveNet-style block: tanh(filt) * sigmoid(gate)."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.3):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv_filter = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation)
        self.conv_gate = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation)
        self.bn_filter = nn.BatchNorm1d(out_channels)
        self.bn_gate = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        f = torch.tanh(self.bn_filter(self.conv_filter(x)))
        g = torch.sigmoid(self.bn_gate(self.conv_gate(x)))
        out = self.bn2(self.conv2(self.dropout(f * g)))
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        return F.relu(out + residual)


class GatedTCN1D(nn.Module):
    """Stacked GatedResidualBlock1D, dilations 1, 2, 4, 8, ..."""
    def __init__(self, vocab_size=50, embed_dim=32, hidden_dim=128,
                 num_levels=4, kernel_size=3, dropout=0.3, out_dim=1):
        super().__init__()
        self.out_dim = out_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList()
        d = 1
        for _ in range(num_levels):
            self.blocks.append(GatedResidualBlock1D(hidden_dim, hidden_dim, kernel_size, d, dropout))
            d *= 2
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.embedding(x.long()).permute(0, 2, 1)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x).squeeze(-1)
        out = self.fc(x)
        return out.squeeze(-1) if self.out_dim == 1 else out


class _ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.0):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.conv1(x))))
        return F.relu(self.bn2(self.conv2(x)))


class _DownBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool1d(2, stride=2)
        self.conv_block = _ConvBlock1D(in_ch, out_ch, kernel_size, dropout)

    def forward(self, x):
        return self.conv_block(self.pool(x))


class _UpBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv_block = _ConvBlock1D(out_ch * 2, out_ch, kernel_size, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        diff = skip.size(2) - x.size(2)
        if diff > 0:
            x = F.pad(x, (0, diff))
        elif diff < 0:
            x = x[:, :, :skip.size(2)]
        x = torch.cat([skip, x], dim=1)
        return self.conv_block(x)


class HierarchicalCNN1D(nn.Module):
    """1D U-Net: encoder with MaxPool downsampling, decoder with ConvTranspose1d, skip connections."""
    def __init__(self, vocab_size=50, embed_dim=16, base_ch=32,
                 num_levels=3, kernel_size=3, dropout=0.2, out_dim=1):
        super().__init__()
        self.out_dim = out_dim
        self.num_levels = num_levels
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.in_conv = _ConvBlock1D(embed_dim, base_ch, kernel_size, dropout)

        self.down_blocks = nn.ModuleList()
        in_ch = base_ch
        for i in range(num_levels):
            out_ch = base_ch * (2 ** (i + 1))
            self.down_blocks.append(_DownBlock1D(in_ch, out_ch, kernel_size, dropout))
            in_ch = out_ch

        self.up_blocks = nn.ModuleList()
        for i in range(num_levels):
            in_ch = base_ch * (2 ** (num_levels - i))
            out_ch = base_ch * (2 ** (num_levels - i - 1))
            self.up_blocks.append(_UpBlock1D(in_ch, out_ch, kernel_size, dropout))

        self.out_conv = nn.Sequential(
            nn.Conv1d(base_ch, base_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv1d(base_ch, base_ch, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_ch, out_dim)

    def forward(self, x):
        x = self.embedding(x.long()).permute(0, 2, 1)
        x = self.in_conv(x)
        skips = [x]
        for db in self.down_blocks:
            x = db(x)
            skips.append(x)
        for i, ub in enumerate(self.up_blocks):
            skip_idx = len(skips) - (i + 2)
            x = ub(x, skips[skip_idx])
        x = self.out_conv(x)
        x = self.global_pool(x).squeeze(-1)
        out = self.fc(x)
        return out.squeeze(-1) if self.out_dim == 1 else out


def GetModel(CFG: dict[str, Any]):
    mt = CFG["model_type"]
    if mt == "MLPRes1":
        return Pilgrim(CFG)
    elif mt == "MLP":
        return SimpleMLP(CFG)
    elif mt == "MLP_m05":
        return ResMLPDistance(CFG)
    elif mt == "CNN":
        return Embedded1DCNN_Flexible(
            vocab_size=CFG["num_classes"],
            embed_dim=CFG.get("embed_dim", 32),
            out_dim=CFG.get("out_dim", 1),
            dropout_p=CFG.get("dropout_p", 0.3),
        )
    elif mt == "TCN":
        return TCNModel(
            vocab_size=CFG["num_classes"],
            embed_dim=CFG.get("embed_dim", 32),
            hidden_dim=CFG.get("hidden_dim", 128),
            num_levels=CFG.get("num_levels", 4),
            kernel_size=CFG.get("kernel_size", 3),
            dropout=CFG.get("dropout", 0.3),
            out_dim=CFG.get("out_dim", 1),
        )
    elif mt == "GatedTCN":
        return GatedTCN1D(
            vocab_size=CFG["num_classes"],
            embed_dim=CFG.get("embed_dim", 32),
            hidden_dim=CFG.get("hidden_dim", 128),
            num_levels=CFG.get("num_levels", 4),
            kernel_size=CFG.get("kernel_size", 3),
            dropout=CFG.get("dropout", 0.3),
            out_dim=CFG.get("out_dim", 1),
        )
    elif mt == "HierCNN":
        return HierarchicalCNN1D(
            vocab_size=CFG["num_classes"],
            embed_dim=CFG.get("embed_dim", 16),
            base_ch=CFG.get("base_ch", 32),
            num_levels=CFG.get("num_levels", 3),
            kernel_size=CFG.get("kernel_size", 3),
            dropout=CFG.get("dropout", 0.2),
            out_dim=CFG.get("out_dim", 1),
        )
    else:
        raise ValueError("Unknown model type: " + mt)
'''

cell12["source"] = (src12_head + cnn_classes_and_dispatcher).splitlines(keepends=True)

# ---- Cell 2: append 4 commented cfg variants for the new architectures ----
cell2 = nb["cells"][2]
src2 = "".join(cell2["source"]) if isinstance(cell2["source"], list) else cell2["source"]

cnn_cfg_blocks = """

# --- Alternative cfg: CNN (Embedded1DCNN_Flexible) ---
# cfg = {
#         'model_type':  "CNN",
#         'embed_dim':   32,
#         'dropout_p':   0.3,
#         'out_dim':     1,
#         "rw_width": 2500, "rw_length": 40, "rw_mode": "nbt",
#         "lr": 1e-3, "num_epochs": 100, "val_ratio": 1, "batch_size": 1_000,
#         'beam_width': 2**12, 'max_steps': 100, 'beam_mode': "iterated",
#         'history_depth': 0, 'hashed_neigbourhood': 0, 'memory_cleanup': False,
#         'return_path': True, 'path_device': 'auto',
#         'verbose': 0, 'frequency_to_print_log_in_beam_search': 'auto',
#         'list_states_to_solve': list(range(100, 150)),
#         }

# --- Alternative cfg: TCN (vanilla dilated TCN) ---
# cfg = {
#         'model_type':  "TCN",
#         'embed_dim':   64,
#         'hidden_dim':  128,
#         'num_levels':  4,         # dilations 1, 2, 4, 8
#         'kernel_size': 3,
#         'dropout':     0.3,
#         'out_dim':     1,
#         "rw_width": 2500, "rw_length": 40, "rw_mode": "nbt",
#         "lr": 1e-3, "num_epochs": 100, "val_ratio": 1, "batch_size": 1_000,
#         'beam_width': 2**12, 'max_steps': 100, 'beam_mode': "iterated",
#         'history_depth': 0, 'hashed_neigbourhood': 0, 'memory_cleanup': False,
#         'return_path': True, 'path_device': 'auto',
#         'verbose': 0, 'frequency_to_print_log_in_beam_search': 'auto',
#         'list_states_to_solve': list(range(100, 150)),
#         }

# --- Alternative cfg: GatedTCN (WaveNet-style gated TCN) ---
# cfg = {
#         'model_type':  "GatedTCN",
#         'embed_dim':   64,
#         'hidden_dim':  256,
#         'num_levels':  6,         # dilations 1..32
#         'kernel_size': 3,
#         'dropout':     0.3,
#         'out_dim':     1,
#         "rw_width": 2500, "rw_length": 40, "rw_mode": "nbt",
#         "lr": 1e-3, "num_epochs": 100, "val_ratio": 1, "batch_size": 1_000,
#         'beam_width': 2**12, 'max_steps': 100, 'beam_mode': "iterated",
#         'history_depth': 0, 'hashed_neigbourhood': 0, 'memory_cleanup': False,
#         'return_path': True, 'path_device': 'auto',
#         'verbose': 0, 'frequency_to_print_log_in_beam_search': 'auto',
#         'list_states_to_solve': list(range(100, 150)),
#         }

# --- Alternative cfg: HierCNN (1D U-Net with skip connections) ---
# cfg = {
#         'model_type':  "HierCNN",
#         'embed_dim':   64,
#         'base_ch':     32,
#         'num_levels':  4,         # depth of encoder/decoder cycles
#         'kernel_size': 3,
#         'dropout':     0.2,
#         'out_dim':     1,
#         "rw_width": 2500, "rw_length": 40, "rw_mode": "nbt",
#         "lr": 1e-3, "num_epochs": 100, "val_ratio": 1, "batch_size": 1_000,
#         'beam_width': 2**12, 'max_steps': 100, 'beam_mode': "iterated",
#         'history_depth': 0, 'hashed_neigbourhood': 0, 'memory_cleanup': False,
#         'return_path': True, 'path_device': 'auto',
#         'verbose': 0, 'frequency_to_print_log_in_beam_search': 'auto',
#         'list_states_to_solve': list(range(100, 150)),
#         }
"""
cell2["source"] = (src2 + cnn_cfg_blocks).splitlines(keepends=True)

# ---- Insert 4 new commented run cells after the existing MLP_m05 run cell (idx 20). ----
# After insertion they will sit at indices 21..24.

run_cell_templates = [
    ("CNN", "Embedded1DCNN_Flexible", """cfg_cnn = dict(cfg)
# cfg_cnn.update({
#     'model_type':  "CNN",
#     'embed_dim':   32,
#     'dropout_p':   0.3,
#     'out_dim':     1,
#     'num_classes': int(max(graph.central_state)) + 1,
#     'state_size':  graph.definition.state_size,
# })
# model_cnn = GetModel(cfg_cnn)
# print(model_cnn)
# print('params:', sum(p.numel() for p in model_cnn.parameters()))
# dict_train_stat_cnn = train_model(cfg_cnn, model_cnn)
# torch.save(model_cnn.state_dict(), "weights_cnn.pth")"""),

    ("TCN", "TCNModel (vanilla dilated TCN)", """cfg_tcn = dict(cfg)
# cfg_tcn.update({
#     'model_type':  "TCN",
#     'embed_dim':   64,
#     'hidden_dim':  128,
#     'num_levels':  4,
#     'kernel_size': 3,
#     'dropout':     0.3,
#     'out_dim':     1,
#     'num_classes': int(max(graph.central_state)) + 1,
#     'state_size':  graph.definition.state_size,
# })
# model_tcn = GetModel(cfg_tcn)
# print(model_tcn)
# print('params:', sum(p.numel() for p in model_tcn.parameters()))
# dict_train_stat_tcn = train_model(cfg_tcn, model_tcn)
# torch.save(model_tcn.state_dict(), "weights_tcn.pth")"""),

    ("GatedTCN", "GatedTCN1D (WaveNet-style)", """cfg_gtcn = dict(cfg)
# cfg_gtcn.update({
#     'model_type':  "GatedTCN",
#     'embed_dim':   64,
#     'hidden_dim':  256,
#     'num_levels':  6,
#     'kernel_size': 3,
#     'dropout':     0.3,
#     'out_dim':     1,
#     'num_classes': int(max(graph.central_state)) + 1,
#     'state_size':  graph.definition.state_size,
# })
# model_gtcn = GetModel(cfg_gtcn)
# print(model_gtcn)
# print('params:', sum(p.numel() for p in model_gtcn.parameters()))
# dict_train_stat_gtcn = train_model(cfg_gtcn, model_gtcn)
# torch.save(model_gtcn.state_dict(), "weights_gtcn.pth")"""),

    ("HierCNN", "HierarchicalCNN1D (1D U-Net)", """cfg_hier = dict(cfg)
# cfg_hier.update({
#     'model_type':  "HierCNN",
#     'embed_dim':   64,
#     'base_ch':     32,
#     'num_levels':  4,
#     'kernel_size': 3,
#     'dropout':     0.2,
#     'out_dim':     1,
#     'num_classes': int(max(graph.central_state)) + 1,
#     'state_size':  graph.definition.state_size,
# })
# model_hier = GetModel(cfg_hier)
# print(model_hier)
# print('params:', sum(p.numel() for p in model_hier.parameters()))
# dict_train_stat_hier = train_model(cfg_hier, model_hier)
# torch.save(model_hier.state_dict(), "weights_hier.pth")"""),
]

new_cells = []
for key, name, body in run_cell_templates:
    body_lines = body.splitlines(keepends=True)
    body_commented = []
    for line in body_lines:
        if line.startswith("cfg_"):
            body_commented.append("# " + line)
        else:
            body_commented.append(line)
    src = (
        f"# --- Optional: train with the {key} branch ({name}) (commented) ---\n"
        f"# Uncomment to run; uses GetModel(cfg_{key.lower()}) so the existing train_model works as-is.\n"
        f"#\n"
        + "".join(body_commented)
        + "\n"
    )
    new_cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    })

# Insert after current cell 20 (MLP_m05 run); they become cells 21..24.
for offset, c in enumerate(new_cells):
    nb["cells"].insert(21 + offset, c)

with open(p, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"OK - cell count: {len(nb['cells'])}")
