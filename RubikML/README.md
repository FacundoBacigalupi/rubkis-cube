# RubikML

Neural network solver for the Unity Rubik's Cube project.

Trains a residual policy+value network to solve the Rubik's Cube using beam search.
No classical solver, no Kociemba, no move history — the model learns purely from
randomly generated scrambles of a Python cube simulator.

## Results

| Scramble depth | Success rate | Avg moves |
|---|---|---|
| 15 | 100.0 % | 27.2 |
| 20 | 100.0 % | 29.9 |
| 25 | 100.0 % | 32.6 |

*(beam-size=128, max-search-depth=75)*

## Model: `RubikPolicyValueNetV2`

- **Input**: 54 stickers × 6 one-hot colors = 324 floats
- **Architecture**: embed → 4 × ResBlock (Linear + LayerNorm + ReLU + residual) → policy head + value head
- **Hidden size**: 1024, ~9.5 M parameters
- **Policy head**: 12 move logits (`R R' L L' U U' D D' F F' B B'`)
- **Value head**: scalar ∈ [0, 1], target = `scramble_depth / 26`

## Setup

```powershell
cd RubikML
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Training curriculum

The model is trained in phases of increasing scramble depth with a fixed `value_norm=26`
so the value head scale stays stable across phases.

```powershell
# Phase 1 — depth 1-15
py -m rubikml.train --model-type policy_value_v2 `
  --min-depth 1 --max-depth 15 --epochs 80 --lr 1e-3 `
  --batch-size 512 --samples-per-epoch 200000 --value-norm 26 --device auto

# Phase 2 — depth 1-20
py -m rubikml.train --resume checkpoints/best.pt --model-type policy_value_v2 `
  --min-depth 1 --max-depth 20 --epochs 60 --lr 2e-4 `
  --batch-size 512 --samples-per-epoch 200000 --value-norm 26 --device auto

# Phase 3 — depth 5-25
py -m rubikml.train --resume checkpoints/best.pt --model-type policy_value_v2 `
  --min-depth 5 --max-depth 25 --epochs 60 --lr 5e-5 `
  --batch-size 512 --samples-per-epoch 200000 --value-norm 26 --device auto
```

## Benchmark

```powershell
py -m rubikml.solve --checkpoint checkpoints/best.pt `
  --mode beam --depth 25 --trials 200 --beam-size 128 --max-search-depth 75
```

## Export to Unity

```powershell
py -m rubikml.export_onnx --checkpoint checkpoints/best.pt `
  --output exports/rubik_pv.onnx --include-value
copy exports\rubik_pv.onnx ..\Assets\ML\rubik_pv.onnx
```

## Tests

```powershell
python -m pytest
```

## Shuffler constraint

Consecutive moves must be on different rotation axes (X/Y/Z), matching Unity's
`RubikShuffler`. This is enforced in `moves.random_scramble`.

## Move order (fixed — required for Unity ONNX integration)

```
["R", "R'", "L", "L'", "U", "U'", "D", "D'", "F", "F'", "B", "B'"]
```

## Sticker encoding

54 stickers, each one-hot over 6 colors → 324-float input vector.

| Sticker range | Face | Color id |
|---|---|---|
| [0..8]   | U (top)   | 0 |
| [9..17]  | D (bottom)| 1 |
| [18..26] | R (right) | 2 |
| [27..35] | L (left)  | 3 |
| [36..44] | F (front) | 4 |
| [45..53] | B (back)  | 5 |
