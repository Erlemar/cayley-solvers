#!/usr/bin/env python3
"""Solve a 4x4x4 Kaggle dataset with a weighted Q-model ensemble.

The progress database is durable after every attempted cube.  Successful paths
are replayed against the exact target before they are stored or emitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

import torch

from pilgrim import (
    QSearcher,
    WeightedEnsembleModel,
    build_model_from_info,
    generate_inverse_moves,
    load_torch_file,
    parse_generator_spec,
)
from pilgrim.parallel import maybe_wrap_dataparallel, resolve_device


BASE_DIR = Path(__file__).resolve().parent


def parse_state(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]


def load_csv_states(path: Path) -> list[tuple[int, torch.Tensor]]:
    rows: list[tuple[int, torch.Tensor]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                (
                    int(row["initial_state_id"]),
                    torch.tensor(parse_state(row["initial_state"]), dtype=torch.int64),
                )
            )
    if not rows:
        raise ValueError(f"{path} contains no states")
    if len({state_id for state_id, _ in rows}) != len(rows):
        raise ValueError(f"{path} contains duplicate initial_state_id values")
    return rows


def load_paths(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    result: dict[int, str] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[int(row["initial_state_id"])] = row["path"]
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(state: torch.Tensor) -> str:
    return hashlib.sha256(state.contiguous().numpy().tobytes()).hexdigest()


def path_length(path: str) -> int:
    return 0 if not path else len(path.split("."))


def apply_path(
    state: torch.Tensor,
    path: str,
    move_to_idx: dict[str, int],
    all_moves_cpu: torch.Tensor,
) -> torch.Tensor:
    current = state.clone()
    if not path:
        return current
    for name in path.split("."):
        if name not in move_to_idx:
            raise ValueError(f"unknown move in path: {name!r}")
        current = current.index_select(0, all_moves_cpu[move_to_idx[name]])
    return current


def moves_to_path(moves: torch.Tensor | None, move_names: list[str]) -> str | None:
    if moves is None:
        return None
    return ".".join(move_names[int(move)] for move in moves.tolist())


def resolve_search_state_dtype(name: str, target: torch.Tensor) -> torch.dtype:
    if name == "auto":
        vmin = int(target.min().item())
        vmax = int(target.max().item())
        for dtype in (torch.uint8, torch.int16, torch.int32, torch.int64):
            info = torch.iinfo(dtype)
            if info.min <= vmin and vmax <= info.max:
                return dtype
        raise ValueError(f"target values [{vmin}, {vmax}] do not fit")
    return {
        "uint8": torch.uint8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
    }[name]


def load_q_model(
    *,
    info_path: Path,
    weights_path: Path,
    num_classes: int,
    state_size: int,
    num_actions: int,
    target: torch.Tensor,
    device: torch.device,
    gpu_ids: list[int],
    compile_enabled: bool,
    compile_mode: str,
    compile_skip_dynamic_cudagraphs: bool,
) -> torch.nn.Module:
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    if int(info.get("num_actions", num_actions)) != num_actions:
        raise ValueError(f"{info_path} is not a {num_actions}-action Q model")
    model = build_model_from_info(
        info,
        num_classes=num_classes,
        state_size=state_size,
        output_dim=num_actions,
    )
    weights = load_torch_file(weights_path, weights_only=False, map_location="cpu")
    model.load_state_dict(weights, strict=True)
    model.eval()
    if device.type == "cuda":
        model.half()
        model.dtype = torch.float16
    else:
        model.dtype = torch.float32
    if target.min() < 0:
        model.z_add = -target.min().item()
    model.to(device)

    if len(gpu_ids) > 1 and compile_enabled:
        raise RuntimeError("torch.compile is supported only with one GPU")
    if len(gpu_ids) > 1:
        model = maybe_wrap_dataparallel(model, gpu_ids)
    elif compile_enabled:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this PyTorch build")
        triton_cfg = getattr(getattr(torch, "_inductor", None), "config", None)
        triton_cfg = getattr(triton_cfg, "triton", None)
        if triton_cfg is not None and hasattr(
            triton_cfg, "cudagraph_dynamic_shape_warn_limit"
        ):
            triton_cfg.cudagraph_dynamic_shape_warn_limit = None
        if (
            compile_skip_dynamic_cudagraphs
            and triton_cfg is not None
            and hasattr(triton_cfg, "cudagraph_skip_dynamic_graphs")
        ):
            triton_cfg.cudagraph_skip_dynamic_graphs = True
        model = torch.compile(model, mode=compile_mode)
    return model


def open_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS solutions (
            initial_state_id INTEGER PRIMARY KEY,
            state_hash TEXT NOT NULL,
            path TEXT NOT NULL,
            path_length INTEGER NOT NULL,
            attempts INTEGER,
            seconds REAL NOT NULL,
            beam_size INTEGER NOT NULL,
            run_key TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trials (
            initial_state_id INTEGER NOT NULL,
            state_hash TEXT NOT NULL,
            run_key TEXT NOT NULL,
            solved INTEGER NOT NULL,
            seconds REAL NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (initial_state_id, state_hash, run_key)
        );
        CREATE TABLE IF NOT EXISTS runs (
            run_key TEXT PRIMARY KEY,
            config_json TEXT NOT NULL,
            started_at TEXT NOT NULL
        );
        """
    )
    connection.commit()
    return connection


def load_database_paths(
    connection: sqlite3.Connection,
    states_by_id: dict[int, torch.Tensor],
) -> dict[int, str]:
    result: dict[int, str] = {}
    query = "SELECT initial_state_id, state_hash, path FROM solutions"
    for initial_state_id, stored_hash, path in connection.execute(query):
        state = states_by_id.get(int(initial_state_id))
        if state is not None and state_hash(state) == stored_hash:
            result[int(initial_state_id)] = str(path)
    return result


def choose_best_paths(
    ordered_states: list[tuple[int, torch.Tensor]],
    database_paths: dict[int, str],
    baseline_paths: dict[int, str],
) -> tuple[dict[int, str], dict[int, str]]:
    best: dict[int, str] = {}
    sources: dict[int, str] = {}
    for initial_state_id, _ in ordered_states:
        candidates: list[tuple[int, str, str]] = []
        if initial_state_id in baseline_paths:
            path = baseline_paths[initial_state_id]
            candidates.append((path_length(path), path, "baseline"))
        if initial_state_id in database_paths:
            path = database_paths[initial_state_id]
            candidates.append((path_length(path), path, "ensemble"))
        if candidates:
            _, path, source = min(candidates, key=lambda item: item[0])
            best[initial_state_id] = path
            sources[initial_state_id] = source
    return best, sources


def write_outputs(
    *,
    output: Path,
    ordered_states: list[tuple[int, torch.Tensor]],
    database_paths: dict[int, str],
    baseline_paths: dict[int, str],
    run_config: dict,
) -> dict:
    best, sources = choose_best_paths(ordered_states, database_paths, baseline_paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["initial_state_id", "path"])
        writer.writeheader()
        for initial_state_id, _ in ordered_states:
            writer.writerow(
                {
                    "initial_state_id": initial_state_id,
                    "path": best.get(initial_state_id, ""),
                }
            )
    os.replace(temporary, output)

    baseline_total = (
        sum(path_length(baseline_paths[state_id]) for state_id, _ in ordered_states)
        if all(state_id in baseline_paths for state_id, _ in ordered_states)
        else None
    )
    best_total = (
        sum(path_length(best[state_id]) for state_id, _ in ordered_states)
        if len(best) == len(ordered_states)
        else None
    )
    report = {
        "output": str(output),
        "dataset_count": len(ordered_states),
        "ensemble_solved_count": len(database_paths),
        "ensemble_selected_count": sum(
            source == "ensemble" for source in sources.values()
        ),
        "complete_count": len(best),
        "baseline_total_length": baseline_total,
        "best_total_length": best_total,
        "moves_saved_vs_baseline": (
            None
            if baseline_total is None or best_total is None
            else baseline_total - best_total
        ),
        "run": run_config,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    report_path = output.with_suffix(".report.json")
    report_tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    with report_tmp.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    os.replace(report_tmp, report_path)
    return report


def validate_external_paths(
    *,
    label: str,
    paths: dict[int, str],
    states_by_id: dict[int, torch.Tensor],
    target_cpu: torch.Tensor,
    move_to_idx: dict[str, int],
    all_moves_cpu: torch.Tensor,
) -> dict[int, str]:
    verified: dict[int, str] = {}
    for initial_state_id, path in paths.items():
        state = states_by_id.get(initial_state_id)
        if state is None:
            continue
        try:
            final = apply_path(state, path, move_to_idx, all_moves_cpu)
        except (KeyError, ValueError):
            continue
        if torch.equal(final, target_cpu):
            verified[initial_state_id] = path
    print(f"Verified {len(verified)}/{len(paths)} paths from {label}")
    return verified


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve and incrementally improve a 4x4x4 Kaggle submission."
    )
    parser.add_argument("--test-csv", type=Path, default=BASE_DIR / "test.csv")
    parser.add_argument(
        "--baseline-submission", type=Path, default=BASE_DIR / "baseline_submission.csv"
    )
    parser.add_argument(
        "--transformer-info",
        type=Path,
        default=BASE_DIR / "models/transformer/model.json",
    )
    parser.add_argument(
        "--transformer-weights",
        type=Path,
        default=BASE_DIR / "models/transformer/model.pth",
    )
    parser.add_argument(
        "--mlp-info", type=Path, default=BASE_DIR / "models/mlp_x16/model.json"
    )
    parser.add_argument(
        "--mlp-weights", type=Path, default=BASE_DIR / "models/mlp_x16/model.pth"
    )
    parser.add_argument(
        "--generator-file", type=Path, default=BASE_DIR / "generators/p002.json"
    )
    parser.add_argument(
        "--target-file", type=Path, default=BASE_DIR / "targets/p002-t000.pt"
    )
    parser.add_argument("--mlp-weight", type=float, default=0.4)
    parser.add_argument(
        "--tail-bfs-depth",
        type=int,
        default=0,
        help="exact endgame: terminate as soon as any beam state is within this many moves "
        "of solved, and finish with the exact BFS path. Implemented in Searcher and "
        "inherited by QSearcher; this driver simply never passed it. 0 = off.",
    )
    parser.add_argument(
        "--history-depth",
        type=int,
        default=0,
        help="cap cross-layer dedup to the last N layers. NOTE the inverted semantics vs "
        "KhoruzhiiSolver: this searcher already dedups against EVERY previous layer, so "
        "N>0 is a RELAXATION, not an addition. 0 = keep unbounded dedup.",
    )
    parser.add_argument("--B", type=int, default=2**21)
    parser.add_argument("--num-steps", type=int, default=150)
    parser.add_argument("--num-attempts", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2**14)
    parser.add_argument("--gpu-ids", type=str, default="0")
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument(
        "--search-state-dtype",
        choices=("auto", "uint8", "int16", "int32", "int64"),
        default="auto",
    )
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument(
        "--compile-skip-dynamic-cudagraphs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output", type=Path, default=BASE_DIR / "results/submission_ensemble_w040.csv"
    )
    parser.add_argument(
        "--progress-db", type=Path, default=BASE_DIR / "results/progress.sqlite3"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--pids",
        type=str,
        default=None,
        help="comma-separated initial_state_id list; overrides sharding",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--rerun-solved", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.mlp_weight <= 1.0:
        raise ValueError("--mlp-weight must be in [0, 1]")
    if args.B <= 0 or args.num_steps <= 0 or args.num_attempts <= 0:
        raise ValueError("B, num-steps, and num-attempts must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num-shards > 0 and 0 <= shard-index < num-shards")

    ordered_states = load_csv_states(args.test_csv)
    states_by_id = dict(ordered_states)
    if args.pids:
        # Fixed pid set, for A/B benchmarking on a matched (and non-trivial) sample.
        # Sharding by index gives pids 0,1,... which are 1- and 2-move solves.
        wanted = [p.strip() for p in args.pids.split(",") if p.strip()]
        by_id = {str(pid): (pid, state) for pid, state in ordered_states}
        missing = [p for p in wanted if p not in by_id]
        if missing:
            raise ValueError(f"--pids not present in test csv: {missing}")
        selected = [by_id[p] for p in wanted]
    else:
        selected = [
            row
            for index, row in enumerate(ordered_states)
            if index % args.num_shards == args.shard_index
        ]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        selected = selected[: args.limit]

    with args.generator_file.open("r", encoding="utf-8") as handle:
        generator_spec = json.load(handle)
    moves, move_names = parse_generator_spec(generator_spec)
    move_to_idx = {name: index for index, name in enumerate(move_names)}
    all_moves_cpu = torch.tensor(moves, dtype=torch.int64)
    target_cpu = load_torch_file(
        args.target_file, weights_only=True, map_location="cpu"
    ).long()

    baseline_paths = validate_external_paths(
        label=str(args.baseline_submission),
        paths=load_paths(args.baseline_submission),
        states_by_id=states_by_id,
        target_cpu=target_cpu,
        move_to_idx=move_to_idx,
        all_moves_cpu=all_moves_cpu,
    )

    print("Hashing model weights for reproducible resume metadata...")
    transformer_sha = sha256_file(args.transformer_weights)
    mlp_sha = sha256_file(args.mlp_weights)
    run_config = {
        "transformer_sha256": transformer_sha,
        "mlp_sha256": mlp_sha,
        "transformer_weight": 1.0 - args.mlp_weight,
        "mlp_weight": args.mlp_weight,
        "B": args.B,
        "num_steps": args.num_steps,
        "num_attempts": args.num_attempts,
        "search_seed": args.search_seed,
    }
    run_key = hashlib.sha256(
        json.dumps(run_config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    connection = open_database(args.progress_db)
    connection.execute(
        "INSERT OR IGNORE INTO runs VALUES (?, ?, ?)",
        (
            run_key,
            json.dumps(run_config, sort_keys=True),
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ),
    )
    connection.commit()

    database_paths = load_database_paths(connection, states_by_id)
    database_paths = validate_external_paths(
        label=str(args.progress_db),
        paths=database_paths,
        states_by_id=states_by_id,
        target_cpu=target_cpu,
        move_to_idx=move_to_idx,
        all_moves_cpu=all_moves_cpu,
    )
    initial_report = write_outputs(
        output=args.output,
        ordered_states=ordered_states,
        database_paths=database_paths,
        baseline_paths=baseline_paths,
        run_config=run_config,
    )

    print(
        f"Dataset={len(ordered_states)}, selected={len(selected)}, "
        f"already solved={len(database_paths)}, B={args.B}, "
        f"ensemble={1.0 - args.mlp_weight:g}*TR+{args.mlp_weight:g}*MLP"
    )
    if args.dry_run:
        print(f"Dry run complete. Current output: {args.output}")
        return 0

    device, gpu_ids = resolve_device(args.gpu_ids)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    target = target_cpu.to(device)
    all_moves = all_moves_cpu.to(device)
    num_actions = len(move_names)
    num_classes = int(torch.unique(target).numel())
    state_size = int(target.numel())

    print(f"Loading transformer on {device}...")
    transformer = load_q_model(
        info_path=args.transformer_info,
        weights_path=args.transformer_weights,
        num_classes=num_classes,
        state_size=state_size,
        num_actions=num_actions,
        target=target,
        device=device,
        gpu_ids=gpu_ids,
        compile_enabled=args.compile,
        compile_mode=args.compile_mode,
        compile_skip_dynamic_cudagraphs=args.compile_skip_dynamic_cudagraphs,
    )
    if args.mlp_weight == 0.0:
        # Standalone transformer. WeightedEnsembleModel would still run the MLP forward for
        # a zero-weighted contribution, roughly doubling the per-step cost for nothing.
        print("mlp_weight=0 -- running the transformer STANDALONE, MLP not loaded")
        model = transformer
    else:
        print(f"Loading x16 MLP on {device}...")
        mlp = load_q_model(
            info_path=args.mlp_info,
            weights_path=args.mlp_weights,
            num_classes=num_classes,
            state_size=state_size,
            num_actions=num_actions,
            target=target,
            device=device,
            gpu_ids=gpu_ids,
            compile_enabled=args.compile,
            compile_mode=args.compile_mode,
            compile_skip_dynamic_cudagraphs=args.compile_skip_dynamic_cudagraphs,
        )
        model = WeightedEnsembleModel(
            [transformer, mlp], [1.0 - args.mlp_weight, args.mlp_weight]
        ).to(device)
    inverse_moves = torch.tensor(
        generate_inverse_moves(move_names), dtype=torch.int64, device=device
    )
    searcher = QSearcher(
        model=model,
        all_moves=all_moves,
        V0=target,
        device=device,
        verbose=args.verbose,
        move_names=move_names,
        inverse_moves=inverse_moves,
        normalize_path=True,
        batch_size=args.eval_batch_size,
        hash_seed=args.search_seed,
        state_dtype=resolve_search_state_dtype(args.search_state_dtype, target),
        move_order=4,
        tail_bfs_depth=args.tail_bfs_depth,
        history_depth=args.history_depth,
    )
    if args.tail_bfs_depth > 0:
        n_tail = (
            0
            if searcher.tail_bfs_hashes is None
            else int(searcher.tail_bfs_hashes.numel())
        )
        print(
            f"endgame table: depth {args.tail_bfs_depth}, {n_tail:,} states",
            flush=True,
        )
        if n_tail == 0:
            raise SystemExit("endgame table is empty -- the flag is not doing anything")

    processed = 0
    solved_now = 0
    run_started = time.time()
    for position, (initial_state_id, state_cpu) in enumerate(selected, start=1):
        current_hash = state_hash(state_cpu)
        if not args.rerun_solved and initial_state_id in database_paths:
            continue
        prior_trial = connection.execute(
            "SELECT solved FROM trials WHERE initial_state_id=? AND state_hash=? AND run_key=?",
            (initial_state_id, current_hash, run_key),
        ).fetchone()
        if prior_trial is not None and not args.retry_failed:
            continue

        solve_started = time.time()
        moves_idx, attempts = searcher.get_solution(
            state_cpu.to(device),
            B=args.B,
            num_steps=args.num_steps,
            num_attempts=args.num_attempts,
        )
        seconds = time.time() - solve_started
        candidate = moves_to_path(moves_idx, move_names)
        verified = candidate is not None and torch.equal(
            apply_path(state_cpu, candidate, move_to_idx, all_moves_cpu), target_cpu
        )
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        connection.execute(
            "INSERT OR REPLACE INTO trials VALUES (?, ?, ?, ?, ?, ?)",
            (initial_state_id, current_hash, run_key, int(verified), seconds, now),
        )
        if verified and candidate is not None:
            connection.execute(
                """
                INSERT INTO solutions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(initial_state_id) DO UPDATE SET
                    state_hash=excluded.state_hash,
                    path=excluded.path,
                    path_length=excluded.path_length,
                    attempts=excluded.attempts,
                    seconds=excluded.seconds,
                    beam_size=excluded.beam_size,
                    run_key=excluded.run_key,
                    updated_at=excluded.updated_at
                WHERE excluded.state_hash != solutions.state_hash
                   OR excluded.path_length < solutions.path_length
                """,
                (
                    initial_state_id,
                    current_hash,
                    candidate,
                    path_length(candidate),
                    int(attempts) + 1,
                    seconds,
                    args.B,
                    run_key,
                    now,
                ),
            )
            solved_now += 1
        connection.commit()
        processed += 1

        database_paths = load_database_paths(connection, states_by_id)
        report = write_outputs(
            output=args.output,
            ordered_states=ordered_states,
            database_paths=database_paths,
            baseline_paths=baseline_paths,
            run_config=run_config,
        )
        status = "solved" if verified else "not-found"
        length = path_length(candidate or "")
        print(
            f"[{position}/{len(selected)}] id={initial_state_id} {status} "
            f"len={length} time={seconds:.1f}s model_n={len(database_paths)} "
            f"best_total={report['best_total_length']} "
            f"saved={report['moves_saved_vs_baseline']}",
            flush=True,
        )

    final_paths = load_database_paths(connection, states_by_id)
    final_report = write_outputs(
        output=args.output,
        ordered_states=ordered_states,
        database_paths=final_paths,
        baseline_paths=baseline_paths,
        run_config=run_config,
    )
    connection.close()
    print(
        f"Finished: attempted={processed}, solved_now={solved_now}, "
        f"elapsed={time.time() - run_started:.1f}s"
    )
    print(
        f"Submission: {args.output}; total={final_report['best_total_length']}; "
        f"saved={final_report['moves_saved_vs_baseline']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
