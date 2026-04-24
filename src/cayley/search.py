"""Beam search driven by the trained distance predictor.

Wraps CayleyPy's `BeamSearchAlgorithm`. Our `ResMLPDistance` can be passed directly to
`Predictor(graph, model)` because CayleyPy accepts any `torch.nn.Module` whose forward
takes (B, state_size) int tensors and returns (B,) floats — which is exactly our model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from cayley.model import ResMLPDistance
from cayley.puzzle import PictureCube


@dataclass
class SearchConfig:
    beam_width: int = 2**12
    max_steps: int = 100
    beam_mode: str = "simple"      # "advanced" doesn't return paths (CayleyPy library limitation)
    history_depth: int = 0          # only used by advanced mode


@dataclass
class SearchResult:
    found: bool
    path: list[str]              # empty if not found
    path_length: int


class Solver:
    """Holds a cayleypy CayleyGraph + Predictor + BeamSearchAlgorithm once and reuses
    them across puzzles. Creating a fresh CayleyGraph per puzzle breaks search because
    the graph's random hash vectors differ between constructions, causing state-tracking
    inconsistencies. Also faster: graph build has one-time setup cost.

    Optional MITM (meet-in-the-middle): if `mitm_depth > 0`, build a BFS frontier from
    the solved state out to `mitm_depth`. Beam search terminates when any beam state hits
    that frontier; the path is extended through the BFS. Adds 6-7 moves of effective
    beam reach for the cost of a one-time BFS.
    """

    def __init__(
        self,
        puzzle: PictureCube,
        model: ResMLPDistance,
        device: str = "auto",
        graph_batch_size: int = 65536,
        memory_limit_gb: float = 10.0,
        random_seed: int = 0,
        mitm_depth: int = 0,
    ):
        from cayleypy import CayleyGraph, CayleyGraphDef, Predictor
        from cayleypy.algo.beam_search import BeamSearchAlgorithm
        from cayleypy.cayley_graph_def import GeneratorType

        self.puzzle = puzzle
        graph_def = CayleyGraphDef(
            generators_type=GeneratorType.PERMUTATION,
            generators_permutations=[list(puzzle.generators[n]) for n in puzzle.move_names],
            generators_matrices=[],
            generator_names=list(puzzle.move_names),
            central_state=list(puzzle.solved_state),
            name="picture_cube_333",
        )
        self.graph = CayleyGraph(
            graph_def,
            device=device,
            batch_size=graph_batch_size,
            memory_limit_gb=memory_limit_gb,
            random_seed=random_seed,
        )
        self.predictor = Predictor(self.graph, model)
        self.searcher = BeamSearchAlgorithm(self.graph)
        self.bfs_result = None
        if mitm_depth > 0:
            self.bfs_result = self.graph.bfs(
                max_diameter=mitm_depth,
                max_layer_size_to_explore=100_000_000,
                return_all_hashes=True,
            )

    def solve(self, initial_state, cfg: SearchConfig) -> SearchResult:
        kwargs = dict(
            start_state=list(initial_state),
            predictor=self.predictor,
            beam_width=cfg.beam_width,
            beam_mode=cfg.beam_mode,
            max_steps=cfg.max_steps,
            return_path=True,
        )
        if cfg.beam_mode == "advanced":
            kwargs["history_depth"] = cfg.history_depth
        if self.bfs_result is not None:
            kwargs["bfs_result_for_mitm"] = self.bfs_result
        try:
            result = self.searcher.search(**kwargs)
        except torch.AcceleratorError as e:
            # Per-puzzle OOM: free cache and give up on this puzzle — the submission
            # builder will fall back to the provided fallback CSV.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return SearchResult(found=False, path=[], path_length=0)
        if not result.path_found or result.path is None:
            return SearchResult(found=False, path=[], path_length=0)
        names = [self.puzzle.move_names[i] for i in result.path]
        return SearchResult(found=True, path=names, path_length=len(names))


def search_cayleypy(
    puzzle: PictureCube,
    model: ResMLPDistance,
    initial_state,
    cfg: SearchConfig,
    device: str = "auto",
) -> SearchResult:
    """One-shot convenience wrapper. Prefer `Solver(...).solve(...)` for many puzzles."""
    solver = Solver(puzzle, model, device=device)
    return solver.solve(initial_state, cfg)


def load_model_checkpoint(
    path: str | Path,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    compile_inference: bool = False,
) -> ResMLPDistance:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    mcfg = ckpt.get("model_config") or {}
    if not mcfg:
        raise ValueError(f"checkpoint {path} has no 'model_config'; was it saved by an older version?")
    model = ResMLPDistance(
        state_size=int(mcfg["state_size"]),
        num_classes=int(mcfg["num_classes"]),
        hidden_dims=tuple(mcfg["hidden_dims"]),
        num_res_blocks=int(mcfg["num_res_blocks"]),
        encoding=mcfg.get("encoding", "onehot"),
        embed_dim=int(mcfg.get("embed_dim", 16)),
        output_dim=int(mcfg.get("output_dim", 1)),
    )
    # torch.compile wraps the model and prefixes state_dict keys with "_orig_mod." —
    # strip that so the plain (uncompiled) model can load.
    state_dict = ckpt["state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device=device, dtype=dtype).eval()
    if compile_inference and device == "cuda":
        # dynamic=True so the compiled graph handles the variable batch sizes that beam
        # search and inference-chunking both produce.
        model = torch.compile(model, dynamic=True)
    return model
