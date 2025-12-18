# Digit Pattern Diffusion Model

Trains a masked diffusion model to generate binary digit patterns (0-4) represented as 2x3 grids.

## What It Does

- **Digit representation** — 5 digits (0-4) as 6-element binary patterns (2×3 grid flattened)
- **Masked diffusion training** — Progressive masking/denoising approach
- **Transformer-based** — Uses a transformer encoder for denoising
- **Generation** — Iteratively unmasks positions to generate valid digit patterns

## Digit Patterns

```
0: 110   1: 010   2: 101   3: 101   4: 011
   101      011      110      011      010
```

## Network Architecture

- **DModel**: 128
- **Heads**: 4
- **Layers**: 1
- **FeedForward**: 256
- **Vocab**: 3 tokens (0, 1, [MASK])

## Training Config

- **Timesteps**: 20 diffusion steps
- **Epochs**: 100
- **Learning Rate**: 0.001
- **Mask schedule**: 0% → 80% noise

## Evaluation

- Generates 5 patterns
- Matches to closest digit by Hamming distance
- Success = within 1 Hamming distance of valid digit

## Running

```bash
go run .
```
