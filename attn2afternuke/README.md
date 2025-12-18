# Attention GPU vs CPU Comparison (After Weight Nuking)

This experiment tests GPU vs CPU forward/backward pass equivalence after resetting ("nuking") network weights.

## What It Tests

- **Dense vs Attention models** — Compares pure dense networks with mixed dense+attention layers
- **GPU vs CPU training** — Trains identical models on both CPU and GPU
- **Output equivalence** — Compares outputs within tolerance (1e-5 for exact match, 1e-3 for acceptable)
- **Training speed** — Measures CPU vs GPU speedup

## Network Architecture

```
8x8 input → 4x4 hidden → 4x4 mixed (dense/attn/dense/attn) → 1x4 output (softmax)
```

## Configurations Tested

| Model | Attention Sharing | Normalization | Heads |
|-------|-------------------|---------------|-------|
| `dense` | N/A | N/A | N/A |
| `attnLayer_norm_H2` | layer | Yes | 2 |
| `attnPerSlice_norm_H2` | per-slice | Yes | 2 |

## Key Features

- Cosine learning rate scheduling
- Gradient clipping (±1.0)
- Model save/reload between CPU and GPU tests
- 4-class classification task (same row, same column, same diagonal, other)

## Running

```bash
go run .
```

Outputs GPU comparison files to `gpu_comparison/` directory.
