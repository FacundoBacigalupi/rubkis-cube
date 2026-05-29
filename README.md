# Rubik's Cube 3D

A 3D Rubik's Cube game built in Unity 6.

## Download

Go to [Releases](../../releases) and download the latest `.zip` → unzip → run `RubiksCube.exe`.  
Windows only.

## Features

- 3×3×3 cube with 27 individual cubies
- Instant shuffle followed by a live timer
- Solve button replays the inverse of the shuffle history with animation
- Manual solve detection: timer stops automatically when the cube is solved by hand
- Camera orbit via middle-mouse drag

## Controls

| Input | Action |
|---|---|
| Middle mouse drag | Orbit camera |
| Left click + drag on a sticker | Rotate the layer under that sticker |
| **Shuffle** (top-left) | Reset cube to solved, scramble 25 moves, start timer |
| **Solve** (top-right) | Animate the inverse move sequence back to solved |

## How it works

### Cube representation
Each cubie holds a logical coordinate `(x, y, z) ∈ {-1, 0, 1}³`. Layer turns are identified by axis + layer index. After each animated move the logical coords are updated.

### Input
A ray is cast against the clicked sticker's collider. The sticker's current world-space normal and the mouse drag direction are projected onto screen space to determine which layer and direction to turn.

### Shuffle
Pressing **Shuffle** resets the cube to the fully-solved state (destroys and rebuilds all cubies), then applies 25 random outer-layer moves instantly, never letting two consecutive moves cancel each other. History is cleared and the timer starts.

### Solve
Pressing **Solve** replays each history move in inverse order with full animation. Because the cube is always reset before shuffling, the solve always returns to the genuine solved state regardless of how many times Shuffle was pressed.

### Manual solve detection
After every player move, each sticker's current world-space normal is compared against its original `LocalNormal`. If all stickers match their original face, the cube is solved and the timer stops.
