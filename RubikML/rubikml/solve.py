"""
python -m rubikml.solve --checkpoint checkpoints/best.pt --depth 5 --mode greedy
python -m rubikml.solve --checkpoint checkpoints/best.pt --depth 7 --mode beam
"""
import argparse
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .cube import Cube
from .encoding import encode_one_hot
from .moves import MOVES, inverse_move, index_to_move
from .model import build_model


def _encode_for_torch(cube: Cube, device: torch.device) -> torch.Tensor:
    x = encode_one_hot(cube.as_stickers())
    return torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)


def _get_policy_logits(model, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


def _get_value(model, x: torch.Tensor) -> float:
    out = model(x)
    return out[1].item() if isinstance(out, tuple) else 0.0


# --- greedy ---

def solve_greedy(model, cube: Cube, device: torch.device, max_steps: int = 100):
    model.eval()
    previous_move = None
    seen_states: set = set()
    moves_used: list[str] = []

    with torch.no_grad():
        for _ in range(max_steps):
            if cube.is_solved():
                return True, moves_used

            state_key = tuple(cube.as_stickers().tolist())
            if state_key in seen_states:
                return False, moves_used
            seen_states.add(state_key)

            logits = _get_policy_logits(model, _encode_for_torch(cube, device)).squeeze(0)
            ranked = logits.argsort(descending=True).tolist()

            move = None
            for idx in ranked:
                candidate = index_to_move(idx)
                if previous_move is not None and candidate == inverse_move(previous_move):
                    continue
                move = candidate
                break
            if move is None:
                move = index_to_move(ranked[0])

            cube.apply(move)
            moves_used.append(move)
            previous_move = move

    return cube.is_solved(), moves_used


# --- top-k value guided ---

def solve_topk_value(model, cube: Cube, device: torch.device, k: int = 5, max_steps: int = 100):
    model.eval()
    previous_move = None
    seen_states: set = set()
    moves_used: list[str] = []

    with torch.no_grad():
        for _ in range(max_steps):
            if cube.is_solved():
                return True, moves_used

            state_key = tuple(cube.as_stickers().tolist())
            if state_key in seen_states:
                return False, moves_used
            seen_states.add(state_key)

            logits = _get_policy_logits(model, _encode_for_torch(cube, device)).squeeze(0)
            top_indices = logits.argsort(descending=True).tolist()

            best_move = None
            best_value = float("inf")
            for idx in top_indices[:k]:
                candidate = index_to_move(idx)
                if previous_move is not None and candidate == inverse_move(previous_move):
                    continue
                test_cube = cube.clone()
                test_cube.apply(candidate)
                if test_cube.is_solved():
                    best_move = candidate
                    break
                v = _get_value(model, _encode_for_torch(test_cube, device))
                if v < best_value:
                    best_value = v
                    best_move = candidate

            if best_move is None:
                best_move = index_to_move(top_indices[0])

            cube.apply(best_move)
            moves_used.append(best_move)
            previous_move = best_move

    return cube.is_solved(), moves_used


# --- beam search ---

@dataclass(order=True)
class _BeamNode:
    score: float
    path: list[str] = field(compare=False)
    # actual sticker array — avoids replaying from solved + scramble each step
    stickers: tuple = field(compare=False)
    last_move: str | None = field(compare=False)


def solve_beam(
    model,
    cube: Cube,
    device: torch.device,
    beam_size: int = 16,
    top_k_moves: int = 5,
    max_search_depth: int = 30,
    path_length_weight: float = 0.1,
    value_weight: float = 1.0,
    policy_weight: float = 0.1,
    avoid_inverse: bool = True,
    avoid_seen_states: bool = True,
):
    """
    Beam search guided by policy + value outputs.
    score = path_length_weight * len(path) + value_weight * predicted_distance
            - policy_weight * log_policy_probability
    Lower score is better.
    """
    model.eval()
    seen_states: set = set()

    initial_stickers = tuple(cube.as_stickers().tolist())
    if avoid_seen_states:
        seen_states.add(initial_stickers)

    beam: list[_BeamNode] = [_BeamNode(score=0.0, path=[], stickers=initial_stickers, last_move=None)]

    with torch.no_grad():
        for _ in range(max_search_depth):
            candidates: list[_BeamNode] = []

            for node in beam:
                import numpy as np
                c = Cube(np.array(node.stickers, dtype=np.int8))

                x = _encode_for_torch(c, device)
                out = model(x)
                if isinstance(out, tuple):
                    logits, _ = out
                else:
                    logits = out

                log_probs = F.log_softmax(logits.squeeze(0), dim=0)
                top_indices = logits.squeeze(0).argsort(descending=True).tolist()

                for idx in top_indices[:top_k_moves]:
                    move = index_to_move(idx)
                    if avoid_inverse and node.last_move is not None and move == inverse_move(node.last_move):
                        continue

                    next_cube = c.clone()
                    next_cube.apply(move)

                    if next_cube.is_solved():
                        return True, node.path + [move]

                    state_key = tuple(next_cube.as_stickers().tolist())
                    if avoid_seen_states and state_key in seen_states:
                        continue

                    nx = _encode_for_torch(next_cube, device)
                    nout = model(nx)
                    next_value = nout[1].item() if isinstance(nout, tuple) else 0.0

                    new_path = node.path + [move]
                    score = (
                        path_length_weight * len(new_path)
                        + value_weight * next_value
                        - policy_weight * log_probs[idx].item()
                    )
                    candidates.append(_BeamNode(
                        score=score,
                        path=new_path,
                        stickers=state_key,
                        last_move=move,
                    ))
                    if avoid_seen_states:
                        seen_states.add(state_key)

            if not candidates:
                break

            candidates.sort()
            beam = candidates[:beam_size]

    best = beam[0] if beam else _BeamNode(0.0, [], initial_stickers, None)
    return False, best.path


# --- model loading ---

def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model_type = cfg.get("model_type", "policy")
    model = build_model(model_type).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description="Solve a scrambled cube with the neural model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--mode", choices=["greedy", "topk_value", "beam"], default="greedy")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--beam-size", type=int, default=16)
    parser.add_argument("--max-search-depth", type=int, default=30)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    from .moves import random_scramble

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    model = load_model(args.checkpoint, device)

    successes = total_moves = 0
    for _ in range(args.trials):
        scramble = random_scramble(args.depth)
        cube = Cube.solved()
        cube.apply_many(scramble)
        if args.mode == "greedy":
            solved, moves = solve_greedy(model, cube, device, args.max_steps)
        elif args.mode == "topk_value":
            solved, moves = solve_topk_value(model, cube, device, args.topk, args.max_steps)
        else:
            solved, moves = solve_beam(model, cube, device, args.beam_size, args.topk, args.max_search_depth)
        if solved:
            successes += 1
            total_moves += len(moves)

    success_rate = successes / args.trials * 100
    avg_moves = total_moves / successes if successes else float("nan")
    print(f"depth={args.depth}  mode={args.mode}  success={success_rate:.1f}%  avg_moves={avg_moves:.1f}")


if __name__ == "__main__":
    main()
