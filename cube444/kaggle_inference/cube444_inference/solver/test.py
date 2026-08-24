import argparse
from collections import Counter
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import scipy.stats
import torch

try:
    from unified_training.cube4 import Cube4StrictReduction
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from unified_training.cube4 import Cube4StrictReduction

from pilgrim import (
    QCPUOffloadSearcher,
    QSearcher,
    QVRerankSearcher,
    Searcher,
    WeightedEnsembleModel,
    build_model_from_info,
    generate_inverse_moves,
    generate_random_walk_states,
    load_torch_file,
    parse_generator_spec,
)
from pilgrim.parallel import maybe_wrap_dataparallel, resolve_device


def resolve_model_info_path(log_dir, group_id, target_id, model_id):
    base = f"model_p{int(group_id):03d}-t{int(target_id):03d}"
    candidates = sorted(Path(log_dir).glob(f"{base}*_{int(model_id)}.json"))
    if not candidates:
        raise FileNotFoundError(f"metadata for model_id={model_id} not found under {log_dir}")
    if len(candidates) > 1:
        raise RuntimeError(
            f"multiple metadata files found for model_id={model_id}: {[str(path) for path in candidates]}"
        )
    return candidates[0]


def is_q_model(info):
    return (
        str(info.get("model_name", "")).endswith("-q")
        or str(info.get("name", "")).find("-q-") >= 0
        or str(info.get("training_mode", "")).startswith("q_")
        or int(info.get("num_actions", 0)) > 1
    )


def mean_confidence_interval(values):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), float(values[0])

    mean = statistics.fmean(values)
    standard_error = float(scipy.stats.sem(values, ddof=1))
    if standard_error == 0.0:
        return mean, mean
    low, high = scipy.stats.t.interval(
        confidence=0.95,
        df=len(values) - 1,
        loc=mean,
        scale=standard_error,
    )
    return float(low), float(high)


def wilson_confidence_interval(successes, total, confidence=0.95):
    if total == 0:
        return 0.0, 0.0
    probability = successes / total
    z = float(scipy.stats.norm.ppf(0.5 + confidence / 2))
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    radius = (
        z
        * ((probability * (1.0 - probability) / total + z * z / (4 * total * total)) ** 0.5)
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def reduction_outcome(status):
    if bool(status.strict[0]):
        return "strict"
    oll = bool(status.oll_parity[0])
    pll = bool(status.pll_parity[0])
    if oll and pll:
        return "oll+pll"
    if oll:
        return "oll"
    if pll:
        return "pll"
    return "invalid"


def load_model_for_info(
    *,
    info,
    model_id,
    epoch_label,
    num_classes,
    state_size,
    output_dim,
    V0,
    device,
    gpu_ids,
    compile_enabled,
    compile_mode,
    compile_skip_dynamic_cudagraphs,
    weights_path_override=None,
):
    model = build_model_from_info(
        info,
        num_classes=num_classes,
        state_size=state_size,
        output_dim=output_dim,
    )
    model_name = info.get("model_name")
    if weights_path_override is not None:
        weights_path = str(weights_path_override)
    else:
        weights_path = f"weights/{model_name}_{model_id}_best.pth" if epoch_label == "best" else f"weights/{model_name}_{model_id}_e{int(epoch_label):05d}.pth"
    state = load_torch_file(weights_path, weights_only=False, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    if device.type == "cuda":
        model.half()
        model.dtype = torch.float16
    else:
        model.dtype = torch.float32
    if V0.min() < 0:
        model.z_add = -V0.min().item()

    model.to(device)
    if len(gpu_ids) > 1 and compile_enabled:
        raise RuntimeError("torch.compile is only supported with single-GPU inference")
    if len(gpu_ids) > 1:
        model = maybe_wrap_dataparallel(model, gpu_ids)
    elif compile_enabled:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch build")
        triton_cfg = getattr(getattr(torch, "_inductor", None), "config", None)
        triton_cfg = getattr(triton_cfg, "triton", None)
        if triton_cfg is not None and hasattr(triton_cfg, "cudagraph_dynamic_shape_warn_limit"):
            triton_cfg.cudagraph_dynamic_shape_warn_limit = None
        if compile_skip_dynamic_cudagraphs:
            if triton_cfg is not None and hasattr(triton_cfg, "cudagraph_skip_dynamic_graphs"):
                triton_cfg.cudagraph_skip_dynamic_graphs = True
            print("Dynamic-shape CUDAGraph capture disabled for torch.compile")
        model = torch.compile(model, mode=compile_mode)
    return model, weights_path


def resolve_search_state_dtype(name, V0):
    if name == "auto":
        vmin = int(V0.min().item())
        vmax = int(V0.max().item())
        for dtype in (torch.uint8, torch.int16, torch.int32, torch.int64):
            info = torch.iinfo(dtype)
            if info.min <= vmin and vmax <= info.max:
                return dtype
        raise ValueError(f"V0 values [{vmin}, {vmax}] do not fit in supported integer dtypes")

    return {
        "uint8": torch.uint8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
    }[name]


def print_args(args):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] Evaluation config:")
    for key, value in sorted(vars(args).items()):
        print(f"  {key:<15} {value}")


def cuda_memory_summary(device):
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    gib = 1024**3
    return {
        "allocated_gib": allocated / gib,
        "reserved_gib": reserved / gib,
        "peak_allocated_gib": peak_allocated / gib,
        "peak_reserved_gib": peak_reserved / gib,
    }


def print_cuda_memory(prefix, device):
    stats = cuda_memory_summary(device)
    if stats is None:
        return
    print(
        f"{prefix} CUDA memory: "
        f"allocated={stats['allocated_gib']:.2f}GiB, "
        f"reserved={stats['reserved_gib']:.2f}GiB, "
        f"peak_allocated={stats['peak_allocated_gib']:.2f}GiB, "
        f"peak_reserved={stats['peak_reserved_gib']:.2f}GiB"
    )


def start_cuda_memory_snapshot(args, device):
    if not args.cuda_memory_snapshot or device.type != "cuda":
        return False
    memory = getattr(torch.cuda, "memory", None)
    recorder = getattr(memory, "_record_memory_history", None)
    if recorder is None:
        print("CUDA memory snapshot requested, but torch.cuda.memory._record_memory_history is unavailable")
        return False
    try:
        recorder(enabled="all", stacks="all", max_entries=args.cuda_memory_snapshot_entries)
    except TypeError:
        try:
            recorder(max_entries=args.cuda_memory_snapshot_entries)
        except TypeError:
            recorder()
    return True


def stop_cuda_memory_snapshot(enabled):
    if not enabled:
        return
    recorder = getattr(torch.cuda.memory, "_record_memory_history", None)
    if recorder is None:
        return
    try:
        recorder(enabled=None)
    except TypeError:
        try:
            recorder(False)
        except TypeError:
            pass


def dump_cuda_memory_snapshot(path, enabled):
    if not enabled:
        return
    dumper = getattr(torch.cuda.memory, "_dump_snapshot", None)
    if dumper is None:
        print("CUDA memory snapshot recording was enabled, but _dump_snapshot is unavailable")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dumper(path)
    print(f"CUDA memory snapshot saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Run beam-search evaluation for a Megaminx value or Q model.")
    parser.add_argument("--group_id", type=int, required=True, help="Puzzle group id.")
    parser.add_argument("--target_id", type=int, default=0, help="Target id.")
    parser.add_argument("--states_path", type=str, help="Load test state(s) from a .pt tensor file.")
    parser.add_argument("--rnd_depth", type=int, help="Generate fixed-depth random scrambles on the fly.")
    parser.add_argument("--rnd_seed", type=int, default=0, help="Seed for on-the-fly random scrambles.")
    parser.add_argument("--search_seed", type=int, default=0, help="Seed for deterministic search hashing.")
    parser.add_argument("--model_id", type=int, help="Legacy model id.")
    parser.add_argument(
        "--model_info_path",
        type=Path,
        help="Direct training-core model metadata JSON.",
    )
    parser.add_argument(
        "--weights_path",
        type=Path,
        help="Direct bare state-dict .pth; use with --model_info_path.",
    )
    parser.add_argument(
        "--ensemble_model_info_path",
        type=Path,
        help="Metadata JSON for a secondary Q model blended with the primary model.",
    )
    parser.add_argument(
        "--ensemble_weights_path",
        type=Path,
        help="Weights for --ensemble_model_info_path.",
    )
    parser.add_argument(
        "--ensemble_secondary_weight",
        type=float,
        default=0.3,
        help="Secondary Q-model weight; the primary model receives 1-weight.",
    )
    parser.add_argument(
        "--ensemble_tertiary_model_info_path",
        type=Path,
        help="Metadata JSON for an optional third Q model.",
    )
    parser.add_argument(
        "--ensemble_tertiary_weights_path",
        type=Path,
        help="Weights for --ensemble_tertiary_model_info_path.",
    )
    parser.add_argument(
        "--ensemble_tertiary_weight",
        type=float,
        default=0.0,
        help=(
            "Absolute third-model weight. With three models, the primary "
            "weight is 1-secondary-tertiary."
        ),
    )
    parser.add_argument("--rerank_model_id", type=int, help="Scalar V model id for Q-shortlist + V-rerank mode.")
    parser.add_argument("--qshort_alpha", type=float, default=2.0, help="Q shortlist size multiplier before V rerank.")
    parser.add_argument("--epoch", type=int, help="Epoch checkpoint to load unless --best is set.")
    parser.add_argument("--best", action="store_true", help="Load the validation-selected checkpoint.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for single-GPU inference.")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead", help="torch.compile mode.")
    parser.add_argument(
        "--compile_skip_dynamic_cudagraphs",
        action="store_true",
        help="Disable dynamic-shape CUDAGraph capture for torch.compile.",
    )
    parser.add_argument("--B", type=int, default=2**18, help="Beam width.")
    parser.add_argument("--num_attempts", type=int, default=2, help="Number of search restarts.")
    parser.add_argument("--num_steps", type=int, default=200, help="Maximum search steps.")
    parser.add_argument("--tail_bfs_depth", type=int, default=0, help="Check beam states against a solved-state BFS table up to this depth.")
    parser.add_argument("--tests_num", type=int, default=10, help="Number of scrambles to evaluate.")
    parser.add_argument("--gpu_ids", type=str, help="Comma-separated CUDA ids, e.g. '0,3,5,6'. Defaults to GPU 0.")
    parser.add_argument("--eval_batch_size", type=int, default=2**14, help="Batch size for expansion and scoring.")
    parser.add_argument(
        "--search_state_dtype",
        choices=("auto", "uint8", "int16", "int32", "int64"),
        default="auto",
        help="Integer dtype for search states. auto picks the smallest safe dtype.",
    )
    parser.add_argument("--cuda_memory_stats", action="store_true", help="Print PyTorch CUDA allocator memory stats.")
    parser.add_argument(
        "--cuda_memory_snapshot",
        action="store_true",
        help="Record and dump a PyTorch CUDA memory snapshot for memory_viz.",
    )
    parser.add_argument(
        "--cuda_memory_snapshot_entries",
        type=int,
        default=2000000,
        help="Maximum allocation history entries for --cuda_memory_snapshot.",
    )
    parser.add_argument(
        "--cpu_offload_beam",
        action="store_true",
        help="Experimental Q-search mode: keep beam and candidate buffers on CPU.",
    )
    parser.add_argument(
        "--cpu_candidate_buffer_factor",
        type=float,
        default=1.25,
        help="CPU candidate buffer size as a multiple of B for --cpu_offload_beam.",
    )
    parser.add_argument(
        "--cpu_batch_topk",
        type=int,
        help="Fixed per-GPU-batch top-k for --cpu_offload_beam. Defaults to B / num_batches * factor.",
    )
    parser.add_argument(
        "--cpu_batch_topk_factor",
        type=float,
        default=1.25,
        help="Auto per-batch top-k multiplier for --cpu_offload_beam.",
    )
    parser.add_argument(
        "--cpu_candidate_prune_margin",
        type=float,
        default=1.05,
        help="Prune CPU candidate buffer only after it exceeds buffer_size * this margin.",
    )
    parser.add_argument(
        "--cpu_no_dedup_candidates",
        action="store_true",
        help="Disable hash deduplication of CPU-offload candidates. Faster but much lower beam diversity.",
    )
    parser.add_argument(
        "--cpu_visited_mode",
        choices=("none", "recent", "all"),
        default="none",
        help="Visited-hash filtering mode for --cpu_offload_beam.",
    )
    parser.add_argument("--verbose", type=int, default=0, help="Use tqdm if verbose > 0.")
    parser.add_argument(
        "--goal",
        choices=("solved", "cube4_strict_reduction", "cube4_centers_edges"),
        default="solved",
        help="Exact terminal predicate used by beam search.",
    )
    args = parser.parse_args()
    print_args(args)

    if not args.best and args.epoch is None:
        parser.error("--epoch is required unless --best is set")

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    if (args.model_info_path is None) != (args.weights_path is None):
        parser.error("--model_info_path and --weights_path must be supplied together")
    if (args.ensemble_model_info_path is None) != (
        args.ensemble_weights_path is None
    ):
        parser.error(
            "--ensemble_model_info_path and --ensemble_weights_path "
            "must be supplied together"
        )
    if not 0.0 <= args.ensemble_secondary_weight <= 1.0:
        parser.error("--ensemble_secondary_weight must be between 0 and 1")
    if (args.ensemble_tertiary_model_info_path is None) != (
        args.ensemble_tertiary_weights_path is None
    ):
        parser.error(
            "--ensemble_tertiary_model_info_path and "
            "--ensemble_tertiary_weights_path must be supplied together"
        )
    if (
        args.ensemble_tertiary_model_info_path is not None
        and args.ensemble_model_info_path is None
    ):
        parser.error("the tertiary ensemble model requires a secondary model")
    if not 0.0 <= args.ensemble_tertiary_weight <= 1.0:
        parser.error("--ensemble_tertiary_weight must be between 0 and 1")
    if args.ensemble_secondary_weight + args.ensemble_tertiary_weight > 1.0:
        parser.error("secondary and tertiary ensemble weights must sum to at most 1")
    if args.model_info_path is not None:
        if args.model_id is not None:
            parser.error("use either --model_id or direct --model_info_path/--weights_path")
        info_path = args.model_info_path
        model_label = str(info_path.stem).removeprefix("model_")
    else:
        if args.model_id is None:
            parser.error("--model_id or direct --model_info_path/--weights_path is required")
        info_path = resolve_model_info_path(log_dir, args.group_id, args.target_id, args.model_id)
        model_label = str(args.model_id)
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    ensemble_info = None
    ensemble_label = None
    if args.ensemble_model_info_path is not None:
        with args.ensemble_model_info_path.open("r", encoding="utf-8") as handle:
            ensemble_info = json.load(handle)
        ensemble_label = str(args.ensemble_model_info_path.stem).removeprefix(
            "model_"
        )
        if not is_q_model(info) or not is_q_model(ensemble_info):
            raise ValueError("Q ensembling requires two Q models")
    tertiary_info = None
    tertiary_label = None
    if args.ensemble_tertiary_model_info_path is not None:
        with args.ensemble_tertiary_model_info_path.open("r", encoding="utf-8") as handle:
            tertiary_info = json.load(handle)
        tertiary_label = str(
            args.ensemble_tertiary_model_info_path.stem
        ).removeprefix("model_")
        if not is_q_model(tertiary_info):
            raise ValueError("Q ensembling requires all models to be Q models")
    rerank_info = None
    if args.rerank_model_id is not None:
        rerank_info_path = resolve_model_info_path(log_dir, args.group_id, args.target_id, args.rerank_model_id)
        with rerank_info_path.open("r", encoding="utf-8") as handle:
            rerank_info = json.load(handle)
        if is_q_model(rerank_info):
            raise ValueError("--rerank_model_id must point to a scalar value model, not a Q model")

    device, gpu_ids = resolve_device(args.gpu_ids)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] Start testing with device: {device}.")

    with open(f"generators/p{int(args.group_id):03d}.json", "r", encoding="utf-8") as handle:
        data = json.load(handle)
    moves, move_names = parse_generator_spec(data)
    all_moves = torch.tensor(moves, dtype=torch.int64, device=device)

    V0 = load_torch_file(
        f"targets/p{int(args.group_id):03d}-t{int(args.target_id):03d}.pt",
        weights_only=True,
        map_location=device,
    )

    n_gens = all_moves.size(0)
    state_size = all_moves.size(1)
    num_classes = torch.unique(V0).numel()
    search_state_dtype = resolve_search_state_dtype(args.search_state_dtype, V0)

    print("Group info:")
    print(f"  # generators   {n_gens}")
    print(f"  # classes      {num_classes}")
    print(f"  state size     {state_size}")
    print(f"  search dtype   {str(search_state_dtype).replace('torch.', '')}")
    print(f"  search seed    {args.search_seed}")
    if gpu_ids:
        print(f"  cuda devices   {gpu_ids}")
    if args.compile:
        print(f"  compile mode   {args.compile_mode}")
    if args.cpu_offload_beam:
        print(f"  beam storage   cpu")
        print(f"  cpu buffer x   {args.cpu_candidate_buffer_factor:g}")
        print(f"  cpu batch topk {args.cpu_batch_topk if args.cpu_batch_topk is not None else 'auto'}")
        print(f"  cpu batch x    {args.cpu_batch_topk_factor:g}")
        print(f"  cpu dedup      {not args.cpu_no_dedup_candidates}")
        print(f"  cpu visited    {args.cpu_visited_mode}")

    inverse_moves = torch.tensor(generate_inverse_moves(move_names), dtype=torch.int64, device=device)
    reduction = None
    goal_test = None
    if args.goal in {"cube4_strict_reduction", "cube4_centers_edges"}:
        if state_size != 96 or n_gens != 24:
            raise ValueError("cube4 reduction goals expect the p002 4x4 problem")
        reduction = Cube4StrictReduction(V0)
        goal_test = (
            reduction.is_strict
            if args.goal == "cube4_strict_reduction"
            else reduction.is_centers_edges
        )
        if args.cpu_offload_beam:
            raise ValueError("--goal cube4 reduction is not supported with --cpu_offload_beam")
        goal_label = (
            "strict parity-free 3x3 reduction"
            if args.goal == "cube4_strict_reduction"
            else "centers solved and all edges paired"
        )
        print(f"  exact goal     {goal_label}")
    q_model = is_q_model(info)
    epoch_label = "best" if args.best else str(args.epoch)
    model_name = info.get("model_name", f"p{int(args.group_id):03d}-t{int(args.target_id):03d}")
    model, weights_path = load_model_for_info(
        info=info,
        model_id=args.model_id,
        epoch_label=epoch_label,
        num_classes=num_classes,
        state_size=state_size,
        output_dim=n_gens if q_model else 1,
        V0=V0,
        device=device,
        gpu_ids=gpu_ids,
        compile_enabled=args.compile,
        compile_mode=args.compile_mode,
        compile_skip_dynamic_cudagraphs=args.compile_skip_dynamic_cudagraphs,
        weights_path_override=args.weights_path,
    )
    if ensemble_info is not None:
        ensemble_model, ensemble_weights_path = load_model_for_info(
            info=ensemble_info,
            model_id=None,
            epoch_label=epoch_label,
            num_classes=num_classes,
            state_size=state_size,
            output_dim=n_gens,
            V0=V0,
            device=device,
            gpu_ids=gpu_ids,
            compile_enabled=args.compile,
            compile_mode=args.compile_mode,
            compile_skip_dynamic_cudagraphs=(
                args.compile_skip_dynamic_cudagraphs
            ),
            weights_path_override=args.ensemble_weights_path,
        )
        secondary_weight = float(args.ensemble_secondary_weight)
        tertiary_weight = float(args.ensemble_tertiary_weight)
        ensemble_models = [model, ensemble_model]
        ensemble_weights = [
            1.0 - secondary_weight - tertiary_weight,
            secondary_weight,
        ]
        tertiary_weights_path = None
        if tertiary_info is not None:
            tertiary_model, tertiary_weights_path = load_model_for_info(
                info=tertiary_info,
                model_id=None,
                epoch_label=epoch_label,
                num_classes=num_classes,
                state_size=state_size,
                output_dim=n_gens,
                V0=V0,
                device=device,
                gpu_ids=gpu_ids,
                compile_enabled=args.compile,
                compile_mode=args.compile_mode,
                compile_skip_dynamic_cudagraphs=(
                    args.compile_skip_dynamic_cudagraphs
                ),
                weights_path_override=args.ensemble_tertiary_weights_path,
            )
            ensemble_models.append(tertiary_model)
            ensemble_weights.append(tertiary_weight)
        model = WeightedEnsembleModel(ensemble_models, ensemble_weights).to(device)
        print(
            "Q ensemble enabled: "
            f"primary={1.0 - secondary_weight - tertiary_weight:g}, "
            f"secondary={secondary_weight:g}, "
            f"secondary_weights={ensemble_weights_path}"
            + (
                ""
                if tertiary_info is None
                else (
                    f", tertiary={tertiary_weight:g}, "
                    f"tertiary_weights={tertiary_weights_path}"
                )
            )
        )
    rerank_model = None
    if rerank_info is not None:
        if not q_model:
            raise ValueError("--rerank_model_id is only valid when --model_id points to a Q model")
        rerank_model, rerank_weights_path = load_model_for_info(
            info=rerank_info,
            model_id=args.rerank_model_id,
            epoch_label="best",
            num_classes=num_classes,
            state_size=state_size,
            output_dim=1,
            V0=V0,
            device=device,
            gpu_ids=gpu_ids,
            compile_enabled=args.compile,
            compile_mode=args.compile_mode,
            compile_skip_dynamic_cudagraphs=args.compile_skip_dynamic_cudagraphs,
        )
        print(f"Q-shortlist+V-rerank enabled: q_model_id={args.model_id}, v_model_id={args.rerank_model_id}, alpha={args.qshort_alpha:g}")

    if args.states_path is not None:
        states_path = Path(args.states_path)
        tests = load_torch_file(
            states_path,
            weights_only=False,
            map_location=device,
        )
        if tests.ndim == 1:
            tests = tests.unsqueeze(0)
        elif tests.ndim != 2:
            raise ValueError(f"--states_path must contain a 1D or 2D tensor, got shape {tuple(tests.shape)}")
        tests = tests[: args.tests_num]
        dataset_label = states_path.stem
        print(f"Loaded states from {states_path}: {tuple(tests.shape)}")
    elif args.rnd_depth is not None:
        tests = generate_random_walk_states(
            V0=V0,
            all_moves=all_moves,
            inverse_moves=inverse_moves,
            num_states=args.tests_num,
            depth=args.rnd_depth,
            device=device,
            seed=args.rnd_seed,
        )
        dataset_label = f"rnd-k{args.rnd_depth}-s{args.rnd_seed}"
        print(f"Generated rnd dataset on the fly: depth={args.rnd_depth}, seed={args.rnd_seed}")
    else:
        tests = load_torch_file(
            f"datasets/p{int(args.group_id):03d}-t{int(args.target_id):03d}-rnd.pt",
            weights_only=False,
            map_location=device,
        )[: args.tests_num]
        dataset_label = "rnd"
    args.tests_num = tests.size(0)
    print(f"Test dataset size: {args.tests_num}")

    if args.cpu_offload_beam and not q_model:
        raise RuntimeError("--cpu_offload_beam is currently implemented only for Q models")
    if args.cpu_offload_beam and rerank_model is not None:
        raise RuntimeError("--cpu_offload_beam is not implemented for Q-shortlist+V-rerank")
    if args.cpu_offload_beam and args.tail_bfs_depth > 0:
        raise RuntimeError("--tail_bfs_depth is not implemented for --cpu_offload_beam")
    searcher_cls = QVRerankSearcher if rerank_model is not None else (QCPUOffloadSearcher if args.cpu_offload_beam else (QSearcher if q_model else Searcher))
    searcher_kwargs = {}
    if rerank_model is not None:
        searcher_kwargs.update(rerank_model=rerank_model, qshort_alpha=args.qshort_alpha)
    if args.cpu_offload_beam:
        searcher_kwargs.update(
            cpu_candidate_buffer_factor=args.cpu_candidate_buffer_factor,
            cpu_batch_topk=args.cpu_batch_topk,
            cpu_batch_topk_factor=args.cpu_batch_topk_factor,
            cpu_candidate_prune_margin=args.cpu_candidate_prune_margin,
            cpu_dedup_candidates=not args.cpu_no_dedup_candidates,
            cpu_visited_mode=args.cpu_visited_mode,
        )
    searcher = searcher_cls(
        model=model,
        all_moves=all_moves,
        V0=V0,
        device=device,
        verbose=args.verbose,
        move_names=move_names,
        inverse_moves=inverse_moves,
        normalize_path=True,
        batch_size=args.eval_batch_size,
        hash_seed=args.search_seed,
        state_dtype=search_state_dtype,
        tail_bfs_depth=args.tail_bfs_depth,
        goal_test=goal_test,
        **searcher_kwargs,
    )

    rerank_suffix = "" if args.rerank_model_id is None else f"_rerank{args.rerank_model_id}_a{args.qshort_alpha:g}".replace(".", "p")
    ensemble_suffix = ""
    if ensemble_info is not None:
        if tertiary_info is not None:
            ensemble_key = hashlib.sha256(
                f"{ensemble_label}|{tertiary_label}".encode("utf-8")
            ).hexdigest()[:10]
            ensemble_suffix = (
                f"_ens3-{ensemble_key}"
                f"-w{args.ensemble_secondary_weight:g}"
                f"-w{args.ensemble_tertiary_weight:g}"
            ).replace(".", "p")
        else:
            ensemble_suffix = (
                f"_ens-{ensemble_label}-w{args.ensemble_secondary_weight:g}"
            ).replace(".", "p")
    tail_suffix = "" if args.tail_bfs_depth <= 0 else f"_tailbfs{args.tail_bfs_depth}"
    goal_suffix = {
        "solved": "",
        "cube4_strict_reduction": "_strict3x",
        "cube4_centers_edges": "_centers_edges",
    }[args.goal]
    log_suffix = f"{ensemble_suffix}{rerank_suffix}{tail_suffix}{goal_suffix}{'_compile' if args.compile else ''}"
    log_file = f"{log_dir}/test_{model_name}-{dataset_label}_{model_label}_{epoch_label}_B{args.B}{log_suffix}.json"

    results = []
    total_length = 0
    started = time.time()

    for test_num, state in enumerate(tests):
        solve_started = time.time()
        snapshot_enabled = start_cuda_memory_snapshot(args, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if args.cuda_memory_stats:
            print_cuda_memory(f"[before solution {test_num}]", device)
        moves, attempts = searcher.get_solution(
            state,
            B=args.B,
            num_steps=args.num_steps,
            num_attempts=args.num_attempts,
        )
        if args.cuda_memory_stats:
            print_cuda_memory(f"[after solution {test_num}]", device)
        if args.cuda_memory_snapshot:
            snapshot_path = f"{log_dir}/cuda_memory_{model_name}-{dataset_label}_{args.model_id}_{epoch_label}_B{args.B}_test{test_num}.pickle"
            dump_cuda_memory_snapshot(snapshot_path, snapshot_enabled)
        stop_cuda_memory_snapshot(snapshot_enabled)
        solve_time = time.time() - solve_started
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        vertex_num = searcher.counter[:, 0] / searcher.counter[:, 1]
        searcher.counter = torch.zeros((3, 2), dtype=torch.int64)

        if moves is not None:
            terminal = state.to(device=device, dtype=search_state_dtype)
            for move in moves.tolist():
                terminal = terminal.index_select(0, all_moves[int(move)])
            terminal_ok = (
                bool(goal_test(terminal.unsqueeze(0))[0])
                if goal_test is not None
                else torch.equal(terminal, V0.to(dtype=search_state_dtype))
            )
            if not terminal_ok:
                raise RuntimeError(f"solution {test_num} failed exact terminal replay")
            solution_length = len(moves)
            total_length += solution_length
            entry = {
                "test_num": test_num,
                "solution_length": solution_length,
                "attempts": attempts + 1,
                "time": round(solve_time, 2),
                "moves": moves.tolist(),
                "vertex_num": f"[{vertex_num[0]:.2e}, {vertex_num[1]:.2e}, {vertex_num[2]:.2e}]",
            }
            if reduction is not None:
                entry["reduction_outcome"] = reduction_outcome(
                    reduction.classify(terminal.unsqueeze(0))
                )
            print(f"[{timestamp}] Solution {test_num}: Length = {solution_length}")
        else:
            entry = {
                "test_num": test_num,
                "solution_length": None,
                "attempts": None,
                "time": round(solve_time, 2),
                "moves": None,
                "vertex_num": f"[{vertex_num[0]:.2e}, {vertex_num[1]:.2e}, {vertex_num[2]:.2e}]",
            }
            print(f"[{timestamp}] Solution {test_num} not found")

        results.append(entry)
        with open(log_file, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=4)

    finished = time.time()
    solved_results = [entry for entry in results if entry["solution_length"] is not None]
    avg_length = total_length / len(solved_results) if solved_results else 0.0
    length_ci_low, length_ci_high = mean_confidence_interval(
        [entry["solution_length"] for entry in solved_results]
    )

    print(f"Test completed in {(finished - started):.2f}s.")
    print(f"Average solution length: {avg_length:.2f}.")
    print(f"95% Student t confidence interval: [{length_ci_low:.2f}, {length_ci_high:.2f}]")
    print(f"Solved {len(solved_results)}/{args.tests_num} scrambles.")
    if args.goal == "cube4_centers_edges":
        outcomes = Counter(entry["reduction_outcome"] for entry in solved_results)
        strict_count = outcomes["strict"]
        rate = strict_count / len(solved_results) if solved_results else 0.0
        rate_low, rate_high = wilson_confidence_interval(strict_count, len(solved_results))
        print(
            "Reduction outcomes: "
            f"strict={outcomes['strict']}, OLL={outcomes['oll']}, "
            f"PLL={outcomes['pll']}, OLL+PLL={outcomes['oll+pll']}, "
            f"invalid={outcomes['invalid']}."
        )
        print(
            f"Strict parity-free rate: {100 * rate:.2f}% "
            f"(95% Wilson CI: [{100 * rate_low:.2f}%, {100 * rate_high:.2f}%])."
        )
    print(f"Results saved to {log_file}.")


if __name__ == "__main__":
    main()
