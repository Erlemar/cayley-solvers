"""Query the trained Jewel policy for one next move.

The public interface accepts the competition's 48-integer sticker state and
returns official competition move names.  ``JewelQueryEngine`` keeps the model
resident, so applications should create it once and issue many queries.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .ball import ExactBall
from .interactive import interactive_solve, predict_batch
from .model import JewelTransformer, load_transformer
from .official import OfficialPuzzle, parse_official_state
from .pdb import EdgePatternDatabase
from .puzzle import JewelState, apply_path


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PACKAGE_ROOT / "models" / "transformer_v4_public" / "best.pt"
DEFAULT_PUZZLE_INFO = PACKAGE_ROOT / "data" / "puzzle_info.json"
DEFAULT_TEST = PACKAGE_ROOT / "data" / "test.csv"
DEFAULT_BALL = PACKAGE_ROOT / "artifacts" / "ball_d8_official"
DEFAULT_PDBS = (
    PACKAGE_ROOT / "artifacts" / "pdb_edges_0_4_official.npy",
    PACKAGE_ROOT / "artifacts" / "pdb_edges_5_9_official.npy",
)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot see a CUDA device")
    return resolved


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float64) - float(np.max(values))
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


class JewelQueryEngine:
    """Persistent state -> next-move query engine.

    By default an exact-ball lookup guards the model choice.  Outside that
    ball, ``next_move`` is exactly the model's top policy action.  Set
    ``use_exact=False`` on a query to obtain the raw policy choice everywhere.
    """

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        *,
        puzzle_info: str | Path = DEFAULT_PUZZLE_INFO,
        ball: str | Path | None = DEFAULT_BALL,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = _resolve_device(device)
        self.model = load_transformer(str(checkpoint), self.device)
        self.official = OfficialPuzzle.load(puzzle_info)
        self.ball = ExactBall.load(ball) if ball is not None else None

    @classmethod
    def from_components(
        cls,
        model: JewelTransformer,
        official: OfficialPuzzle,
        *,
        ball: ExactBall | None = None,
        device: str | torch.device | None = None,
    ) -> "JewelQueryEngine":
        """Construct an engine from already-loaded parts (also useful in tests)."""
        engine = cls.__new__(cls)
        engine.device = _resolve_device(device or next(model.parameters()).device)
        engine.model = model.to(engine.device).eval()
        engine.official = official
        engine.ball = ball
        return engine

    def query(
        self,
        official_state: str | Sequence[int] | np.ndarray,
        *,
        top_k: int = 3,
        use_exact: bool = True,
    ) -> dict:
        """Return the next official move and model diagnostics."""
        if not 1 <= top_k <= 12:
            raise ValueError("top_k must be between 1 and 12")
        stickers = (
            parse_official_state(official_state)
            if isinstance(official_state, str)
            else np.asarray(official_state, dtype=np.uint8)
        )
        state = self.official.to_structured(stickers)
        rank = int(state.rank())
        exact_distance = self.ball.distance(state) if self.ball is not None else None

        if rank == 0:
            return {
                "status": "solved",
                "next_move": None,
                "model_move": None,
                "source": "solved",
                "state_rank": 0,
                "exact_distance": 0 if exact_distance is not None else None,
                "model_predicted_distance": 0.0,
                "alternatives": [],
            }

        predictions = predict_batch(self.model, [state], self.device)
        policy_probabilities = _softmax(predictions["policy_logits"][0])
        geodesic_probabilities = _sigmoid(predictions["geodesic_logits"][0])
        regrets = predictions["regret"][0]
        order = np.argsort(-policy_probabilities)
        model_action = int(order[0])

        exact_actions: list[int] = []
        if use_exact and self.ball is not None and exact_distance is not None:
            mask = self.ball.optimal_action_mask(state, exact_distance)
            exact_actions = [action for action in range(12) if mask & (1 << action)]
            # All these actions are provably optimal; use the model to break ties.
            selected_action = max(exact_actions, key=lambda action: policy_probabilities[action])
            source = "exact_ball"
        else:
            selected_action = model_action
            source = "model"

        def describe(action: int) -> dict:
            return {
                "move": self.official.format_path([action]),
                "internal_action": action,
                "policy_probability": float(policy_probabilities[action]),
                "geodesic_probability": float(geodesic_probabilities[action]),
                "predicted_regret": float(regrets[action]),
                "exact_descending": action in exact_actions if exact_actions else None,
            }

        selected = describe(selected_action)
        return {
            "status": "move",
            "next_move": selected["move"],
            "model_move": self.official.format_path([model_action]),
            "source": source,
            "state_rank": rank,
            "exact_distance": exact_distance,
            "model_predicted_distance": float(predictions["distance"][0]),
            "policy_probability": selected["policy_probability"],
            "geodesic_probability": selected["geodesic_probability"],
            "predicted_regret": selected["predicted_regret"],
            "exact_optimal_moves": [self.official.format_path([action]) for action in exact_actions],
            "alternatives": [describe(int(action)) for action in order[:top_k]],
        }

    def solve_query(
        self,
        official_state: str | Sequence[int] | np.ndarray,
        pdbs: Sequence[EdgePatternDatabase],
        *,
        top_k: int = 3,
        widths: Sequence[int] = (1, 4, 16, 64),
        max_learned_steps: int = 18,
    ) -> dict:
        """Return the first move of a complete, independently verified solution."""
        if not 1 <= top_k <= 12:
            raise ValueError("top_k must be between 1 and 12")
        if self.ball is None:
            raise ValueError("solve_query requires an exact ball")
        stickers = (
            parse_official_state(official_state)
            if isinstance(official_state, str)
            else np.asarray(official_state, dtype=np.uint8)
        )
        state = self.official.to_structured(stickers)
        # Keep diagnostics for every action so the search-selected move can be
        # reported even when it was not the raw policy argmax.
        response = self.query(stickers, top_k=12, use_exact=True)
        all_alternatives = response["alternatives"]
        result = interactive_solve(
            state,
            self.model,
            self.ball,
            pdbs,
            widths=widths,
            max_learned_steps=max_learned_steps,
            device=self.device,
            try_all_widths=True,
            incumbent_length=None,
        )
        response["alternatives"] = response["alternatives"][:top_k]
        response.update(
            {
                "search_reason": result.reason,
                "search_width": result.width,
                "expanded_states": result.expanded_states,
                "generated_states": result.generated_states,
                "search_seconds": result.seconds,
            }
        )
        if result.path is None:
            response.update(
                {
                    "status": "search_failed",
                    "next_move": None,
                    "source": "interactive_search_failed",
                    "solution_path": None,
                    "solution_length": None,
                    "verified": False,
                }
            )
            return response

        internal_valid = apply_path(state, result.path).rank() == 0
        official_valid = np.array_equal(
            self.official.apply_path(stickers, result.path), self.official.central_state
        )
        if not internal_valid or not official_valid:
            raise RuntimeError("interactive search returned a path that failed replay verification")

        solution_path = self.official.format_path(result.path)
        if not result.path:
            response.update(
                {
                    "status": "solved",
                    "next_move": None,
                    "source": "solved",
                    "solution_path": "",
                    "solution_length": 0,
                    "verified": True,
                }
            )
            return response

        selected_action = int(result.path[0])
        selected = next(
            item for item in all_alternatives if item["internal_action"] == selected_action
        )
        response.update(
            {
                "status": "move",
                "next_move": selected["move"],
                "source": "exact_ball" if result.reason == "exact_ball" else "interactive_search",
                "policy_probability": selected["policy_probability"],
                "geodesic_probability": selected["geodesic_probability"],
                "predicted_regret": selected["predicted_regret"],
                "solution_path": solution_path,
                "solution_length": len(result.path),
                "verified": True,
            }
        )
        return response


def _state_from_file(path: str | Path) -> str | Sequence[int]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError("JSON state files must contain an array of 48 integers")
        return value
    if text.startswith("{"):
        value = json.loads(text)
        if not isinstance(value, dict) or "initial_state" not in value:
            raise ValueError("JSON state objects must contain an 'initial_state' field")
        return value["initial_state"]
    return text


def _state_from_test(test_path: str | Path, initial_state_id: str) -> str:
    with Path(test_path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["initial_state_id"] == initial_state_id:
                return row["initial_state"]
    raise ValueError(f"initial_state_id {initial_state_id!r} was not found in {test_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Give the trained Christopher's Jewel model a state and receive its next move."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--state", help="official state as 48 comma-separated integers")
    source.add_argument("--state-file", help="text or JSON file containing an official state")
    source.add_argument("--initial-state-id", help="load this puzzle from --test")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--puzzle-info", default=str(DEFAULT_PUZZLE_INFO))
    parser.add_argument("--test", default=str(DEFAULT_TEST))
    parser.add_argument("--ball", default=str(DEFAULT_BALL))
    parser.add_argument("--pdb", action="append", help="edge PDB path; may be repeated")
    parser.add_argument("--widths", default="1,4,16,64")
    parser.add_argument("--max-learned-steps", type=int, default=18)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="skip interactive search and return the raw one-step model choice",
    )
    parser.add_argument(
        "--move-only",
        action="store_true",
        help="print only the official move name (or SOLVED)",
    )
    args = parser.parse_args()

    if args.state is not None:
        state = args.state
    elif args.state_file is not None:
        state = _state_from_file(args.state_file)
    else:
        state = _state_from_test(args.test, args.initial_state_id)

    engine = JewelQueryEngine(
        args.checkpoint,
        puzzle_info=args.puzzle_info,
        ball=None if args.model_only else args.ball,
        device=args.device,
    )
    if args.model_only:
        result = engine.query(state, top_k=args.top_k, use_exact=False)
    else:
        pdb_paths = args.pdb or [str(path) for path in DEFAULT_PDBS]
        pdbs = [EdgePatternDatabase.load(path) for path in pdb_paths]
        result = engine.solve_query(
            state,
            pdbs,
            top_k=args.top_k,
            widths=tuple(int(width) for width in args.widths.split(",")),
            max_learned_steps=args.max_learned_steps,
        )
    if args.move_only:
        print(result["next_move"] or "SOLVED")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
