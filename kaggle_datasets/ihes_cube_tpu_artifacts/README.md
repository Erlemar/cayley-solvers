# CayleyPy IHES Cube TPU Artifacts

Assets for a JAX SPMD beam-search solver for the
[CayleyPy SuperCube](https://www.kaggle.com/competitions/cayleypy-ihes-cube)
competition (IHES picture cube: 72 facelets, 18 generators).

## Files

| File | Contents |
| --- | --- |
| `e6_epoch_0499.pt` | Distance ("V") model: ResMLP, embedding encoding (72 classes, dim 16), hidden dims (1024, 256), 1 residual block, ~1.6M params. Trained on random walks (k_max=26), then Bellman-refined for 500 epochs with self-bootstrapped targets. |
| `puzzle_info.json` | The 18 generators (`f/r/d` x layers `0/1/2` x CW/CCW) as length-72 facelet permutations, plus the solved state. |
| `cube_symmetries.npy` | `(48, 72)` int array: the 48 octahedral symmetry permutations (24 rotations + 24 mirrors) acting on facelets. |
| `cube_symmetries_inv.npy` | `(48, 72)` inverses of the above. |
| `cube_symmetry_group.npz` | The full 96-element set (48 symmetries x inverse antisymmetry): keys `perms`, `perms_inv`, `is_antisymmetry`. |
| `cube_symmetries_meta.json` | Group-order metadata. |

## Model usage (PyTorch state_dict layout)

The checkpoint is a dict with a `state_dict` key. Layer layout:
`embedding (72,16)` -> flatten to 1152 -> `Linear->LayerNorm->ReLU` (1024) ->
`Linear->LayerNorm->ReLU` (256) -> 1 ResBlock (`Linear->LN->ReLU->Linear->LN->residual->ReLU`) ->
`head Linear(256,1)`. Output is predicted distance-to-solved (beam search uses
relative ordering; the absolute scale has a small offset at the solved state).

## Symmetry usage

For a symmetry `P` (row of `cube_symmetries.npy`) and state `s` (length-72 int
array mapping position -> facelet id), the transformed state is
`P_inv[s[P]]`-style conjugation; move relabeling under conjugation can be
computed at runtime from the generators. See the companion notebook for
working code.
