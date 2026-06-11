"""
python -m rubikml.evaluate --checkpoint checkpoints/best.pt
"""
import argparse
import json
import os
import time

import torch

from .cube import Cube
from .moves import random_scramble
from .solve import load_model, solve_greedy, solve_topk_value, solve_beam


def evaluate(checkpoint: str, depths: list[int], trials: int, mode: str,
             max_steps: int, topk: int, device: torch.device, output_dir: str = "runs",
             beam_size: int = 16, max_search_depth: int = 30):
    model = load_model(checkpoint, device)
    results = []

    print(f"\ncheckpoint: {checkpoint}")
    print(f"mode: {mode}  trials per depth: {trials}")
    print(f"{'Depth':>6} | {'Success':>8} | {'Avg Moves':>10} | {'Loop/Fail':>10}")
    print("-" * 44)

    import random
    random.seed(0)

    for depth in depths:
        successes = loops = total_moves = 0
        for _ in range(trials):
            scramble = random_scramble(depth)
            cube = Cube.solved()
            cube.apply_many(scramble)
            if mode == "greedy":
                solved, moves = solve_greedy(model, cube, device, max_steps)
            elif mode == "topk_value":
                solved, moves = solve_topk_value(model, cube, device, topk, max_steps)
            else:
                solved, moves = solve_beam(model, cube, device, beam_size, topk, max_search_depth)

            if solved:
                successes += 1
                total_moves += len(moves)
            else:
                loops += 1

        success_rate = successes / trials * 100
        fail_rate = loops / trials * 100
        avg_moves = total_moves / successes if successes else float("nan")
        print(f"{depth:>6} | {success_rate:>7.1f}% | {avg_moves:>10.1f} | {fail_rate:>9.1f}%")
        results.append(dict(depth=depth, success_rate=success_rate,
                            avg_moves=avg_moves, fail_rate=fail_rate,
                            trials=trials))

    os.makedirs(output_dir, exist_ok=True)
    fname = f"eval_{mode}_d{'_'.join(str(d) for d in depths)}_{int(time.time())}.json"
    out_path = os.path.join(output_dir, fname)
    with open(out_path, "w") as f:
        json.dump({"checkpoint": checkpoint, "mode": mode, "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RubikML solve success by depth")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3, 5, 7, 10])
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--mode", choices=["greedy", "topk_value", "beam"], default="greedy")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--beam-size", type=int, default=16)
    parser.add_argument("--max-search-depth", type=int, default=30)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)

    evaluate(args.checkpoint, args.depths, args.trials, args.mode,
             args.max_steps, args.topk, device, beam_size=args.beam_size,
             max_search_depth=args.max_search_depth)


if __name__ == "__main__":
    main()
