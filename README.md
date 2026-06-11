# Rubik's Cube 3D

A 3D Rubik's Cube game built in Unity 6 with a neural network AI solver.

## Download

Go to [Releases](../../releases) and download the latest `.zip` → unzip → run `RubiksCube.exe`.  
Windows only.

## Features

- 3×3×3 cube with 27 individual cubies, physically-based sticker materials
- Instant shuffle followed by a live timer
- **Neural AI solver** — beam-search guided by a residual policy+value network, solves up to 25-move scrambles with near-100% success rate
- Rewind solver replays the inverse of the shuffle history with animation
- Camera orbit via middle-mouse drag and camera flip

## Controls

| Input | Action |
|---|---|
| Middle mouse drag | Orbit camera |
| Left click + drag → release on a sticker | Rotate the layer under that sticker |
| **Shuffle** (top-left) | Reset cube to solved, scramble 25 moves, start timer |
| **Solve** (top-right) | Neural AI solves the cube with animated moves |

## How it works

### Cube representation
Each cubie holds a logical coordinate `(x, y, z) ∈ {-1, 0, 1}³`. Layer turns are identified by axis + layer index. After each animated move the logical coords are updated.

### Input
A ray is cast against the clicked sticker's collider. Only front-facing stickers are accepted (dot product of face normal and ray direction must be negative, preventing accidental clicks on back faces). The sticker's current world-space normal and the mouse drag direction are projected onto screen space to determine which layer and direction to turn. The move fires on mouse-up rather than at a drag threshold, making it less sensitive.

### Shuffle
Pressing **Shuffle** resets the cube to the fully-solved state (destroys and rebuilds all cubies), then applies 25 random outer-layer moves instantly. No two consecutive moves share the same rotation axis. History is cleared and the timer starts.

### Neural AI Solver
Pressing **Neural Solve** runs a beam search guided by a trained neural network:

- **Model**: `RubikPolicyValueNetV2` — residual network with LayerNorm, 4 blocks, 1024 hidden units (~9.5 M parameters)
- **Inputs**: one-hot encoded cube state (54 stickers × 6 colors = 324 floats)
- **Outputs**: policy head (move logits over 12 moves) + value head (estimated distance to solved, normalised by 26)
- **Search**: beam search with `beamSize=128`, `maxBeamDepth=75`, scoring each candidate as `0.1 × pathLen + value − 0.1 × log_policy`
- **Runtime**: Unity InferenceEngine (Sentis) running the exported ONNX model on GPU

### Manual solve detection
After every player move, each sticker's current world-space normal is compared against its original `LocalNormal`. If all stickers match, the cube is solved and the timer stops.

## ML project (`RubikML/`)

Self-contained Python package for training and exporting the solver.
