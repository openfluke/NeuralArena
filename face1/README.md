# Face Diffusion Model (Version 1)

Trains a masked diffusion model to generate 5x5 ASCII emoji faces.

## What It Does

- **Pixel-level generation** — Generates 25-pixel (5×5) face patterns
- **Discrete tokens** — 4 tokens: background(0), eyes(1), mouth(2), features(3), plus [MASK]
- **Training on patterns** — Learns from happy, sad, angry, surprised faces
- **Masked diffusion** — Progressive unmasking during generation

## Face Patterns

```
Happy:    Sad:      Angry:    Surprised:
. . . .   . . . .   . . . .   . . . .
. o . o   . o . o   . o ^ o   . o . o
. . . .   . . . .   . ^ . ^   . ^ ^ ^
. - - -   . . - .   . - - -   . - . -
. . . .   . - - -   . . . .   . . . .
```

## Network Architecture

- **DModel**: 32
- **Heads**: 2
- **Layers**: 2
- **Vocab**: 5 tokens

## Training

- **Epochs**: 200
- **Learning Rate**: 0.001 with linear decay
- **Batch Size**: 3

## Running

```bash
go run .
```

Shows sample generations every 20 epochs.
