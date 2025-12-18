# Small-Scale Masked Diffusion Test

Minimal masked diffusion transformer for debugging and rapid iteration.

## What It Does

- **Simple dataset** — Only 4 sentences: "hello world", "goodbye world", "hello there", "goodbye there"
- **Small network** — DModel=32, 2 heads, 1 layer
- **Quick training** — 100 epochs for fast iteration
- **Debugging focus** — Simplified setup to verify masked diffusion works

## Network Architecture

| Parameter | Value |
|-----------|-------|
| DModel | 32 |
| Heads | 2 |
| Layers | 1 |
| FeedForward | 32 |
| Max Length | 5 |
| Vocab Size | ~10 |

## Training Config

- **Epochs**: 100
- **Learning Rate**: 0.0001
- **Batch Size**: 4
- **Timesteps**: 10

## Running

```bash
go run .
```

Generates 3 text samples after training.
