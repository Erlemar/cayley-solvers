"""One-shot edit applier for the cloned Christopher Jewel notebook.

Adds:
 - ResMLPDistance class + extended GetModel into cell 12
 - Commented MLP_m05 cfg variant in cell 2
 - New commented "run with MLP_m05" cell after the existing commented MLP run cell
"""
import json

p = "megaminx/kaggle_notebooks/cj_mlp/cayleypy-rw-cj-mlp.ipynb"
with open(p, encoding="utf-8") as f:
    nb = json.load(f)

# ---- Cell 12: append ResMLPDistance + extended GetModel ----
cell12 = nb["cells"][12]
src12 = "".join(cell12["source"]) if isinstance(cell12["source"], list) else cell12["source"]

marker = "def GetModel(CFG: dict[str, Any]):"
idx = src12.find(marker)
assert idx != -1, "GetModel marker not found in cell 12"
src12_head = src12[:idx].rstrip() + "\n\n"

addendum = '''# ---- m05-style ResMLPDistance (our megaminx best heuristic architecture) ---------
# Embedding-encoded ResMLP with LayerNorm + ResBlocks. Same architecture as our
# `cayley.model.ResMLPDistance` (megaminx m05 / m07): hidden_dims=(2048, 512),
# num_res_blocks=2, encoding="embedding", embed_dim=16. Trained from scratch via
# random-walk targets (m07) and Bellman-warmstarted (m05).

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
    """Distance predictor: int permutation -> scalar predicted distance to solved.

    Config keys:
        state_size       (int)         length of the permutation vector
        num_classes      (int)         number of distinct piece labels
        hidden_dims      (list[int])   widths of the input stack (default [2048, 512])
        num_res_blocks   (int)         residual blocks after the input stack (default 2)
        encoding         (str)         "embedding" (default) or "onehot"
        embed_dim        (int)         only used if encoding == "embedding" (default 16)
    """
    def __init__(self, config):
        super().__init__()
        self.dtype = torch.float32
        self.state_size = config["state_size"]
        self.num_classes = config["num_classes"]
        self.hidden_dims = list(config.get("hidden_dims", [2048, 512]))
        self.num_res_blocks = int(config.get("num_res_blocks", 2))
        self.encoding = config.get("encoding", "embedding")
        self.embed_dim = int(config.get("embed_dim", 16))
        self.output_dim = 1

        if self.encoding == "onehot":
            in_dim = self.state_size * self.num_classes
            self.embedding = None
        elif self.encoding == "embedding":
            in_dim = self.state_size * self.embed_dim
            self.embedding = nn.Embedding(self.num_classes, self.embed_dim)
        else:
            raise ValueError(f"encoding must be 'onehot' or 'embedding', got {self.encoding!r}")

        layers = []
        prev = in_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        self.input_stack = nn.Sequential(*layers)
        self.res_blocks = nn.ModuleList([ResBlock(prev) for _ in range(self.num_res_blocks)])
        self.head = nn.Linear(prev, self.output_dim)

    def encode(self, x):
        target_dtype = self.input_stack[0].weight.dtype
        if self.encoding == "onehot":
            one_hot = F.one_hot(x.long(), num_classes=self.num_classes)
            return one_hot.to(target_dtype).flatten(start_dim=-2)
        return self.embedding(x.long()).to(target_dtype).flatten(start_dim=-2)

    def forward(self, x):
        h = self.encode(x)
        h = self.input_stack(h)
        for block in self.res_blocks:
            h = block(h)
        out = self.head(h)
        return out.squeeze(-1)


def GetModel(CFG: dict[str, Any]):
    if CFG["model_type"] == "MLPRes1":
        return Pilgrim(CFG)
    elif CFG["model_type"] == "MLP":
        return SimpleMLP(CFG)
    elif CFG["model_type"] == "MLP_m05":
        return ResMLPDistance(CFG)
    else:
        raise ValueError("Unknown model type: " + CFG["model_type"])
'''

cell12["source"] = (src12_head + addendum).splitlines(keepends=True)

# ---- Cell 2: append commented MLP_m05 cfg block ----
cell2 = nb["cells"][2]
src2 = "".join(cell2["source"]) if isinstance(cell2["source"], list) else cell2["source"]

mlp_m05_cfg = """

# --- Alternative cfg using the MLP_m05 (ResMLPDistance) branch of GetModel (commented) ---
# Uncomment this block (and comment out the active cfg above) to train with our
# megaminx m05-style architecture: embedding-encoded ResMLP, hidden=[2048,512], 2 ResBlocks.
#
# cfg = {
#         # Neural network:
#         'model_type':     "MLP_m05",
#         'hidden_dims':    [2048, 512],
#         'num_res_blocks': 2,
#         'encoding':       "embedding",   # or "onehot"
#         'embed_dim':      16,            # only used if encoding == "embedding"
#
#         # Training:
#         "rw_width": 2500,
#         "rw_length": 40,
#         "rw_mode": "nbt",
#         "lr": 1e-3,
#         "num_epochs": 100,
#         "val_ratio": 1,
#         "batch_size": 1_000,
#
#         # Beam search:
#         'beam_width': 2**12,
#         'max_steps': 100,
#         'beam_mode': "iterated",
#         'history_depth': 0,
#         'hashed_neigbourhood': 0,
#         'memory_cleanup': False,
#         'return_path': True,
#         'path_device': 'auto',
#         'verbose': 0,
#         'frequency_to_print_log_in_beam_search': 'auto',
#
#         'list_states_to_solve': list(range(100, 150)),
#         }
"""
cell2["source"] = (src2 + mlp_m05_cfg).splitlines(keepends=True)

# ---- New commented "run with MLP_m05" cell, inserted after the existing commented MLP run cell (idx 19) ----
new_cell_source = """# --- Optional: train with the MLP_m05 (ResMLPDistance) branch (commented) ---
# Mirrors the canonical training cell above but switches model_type to "MLP_m05".
# This is our megaminx m05-style architecture; on the small Christopher Jewel
# state space it is heavily overparameterised, so train loss will saturate fast.
# Provided here for parity with the megaminx pipeline.
#
# cfg_m05 = dict(cfg)  # copy current cfg
# cfg_m05.update({
#     'model_type':     "MLP_m05",
#     'hidden_dims':    [2048, 512],
#     'num_res_blocks': 2,
#     'encoding':       "embedding",
#     'embed_dim':      16,
#     'num_classes':    int(max(graph.central_state)) + 1,
#     'state_size':     graph.definition.state_size,
# })
#
# model_m05 = GetModel(cfg_m05)
# print(model_m05)
# print('params:', sum(p.numel() for p in model_m05.parameters()))
#
# dict_train_stat_m05 = train_model(cfg_m05, model_m05)
# torch.save(model_m05.state_dict(), "weights_m05.pth")
#
# # To run beam search with this model, point the beam-search cell below at
# # `model_m05` and `cfg_m05` (or assign: cfg = cfg_m05; model = model_m05).
"""
new_cell = {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": new_cell_source.splitlines(keepends=True),
}
nb["cells"].insert(20, new_cell)

with open(p, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"OK - cell count: {len(nb['cells'])}")
