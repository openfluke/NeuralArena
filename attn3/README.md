# Attention CPU vs GPU Timing Test

Benchmarks forward pass timing and output deltas between CPU and GPU execution for attention models.

## What It Tests

- **Load timing** — Time to deserialize model from JSON
- **GPU initialization** — Time to set up optimized GPU resources
- **Forward pass timing** — CPU vs GPU per-input latency
- **Output deltas** — Maximum absolute difference between CPU and GPU results

## Network Architecture

```
8x8 input → 4x4 hidden → 4x4 mixed (dense/attn/dense/attn) → 1x4 output (softmax)
```

## Configurations

16 model variants: dense baseline + 5 attention configs × 3 head counts (1, 2, 3)

## Test Inputs

1. **Zero** — All zeros
2. **Two-Query** — Fixed two-point classification
3. **Random** — Seeded random (12345)

## Output Format

```
[model_name]
 load(cpu)=Xms  load(gpu)=Xms
 init(gpu)=Xms
 CPU forward: zero=Xµs  twoQ=Xµs  rand=Xµs
 GPU forward: zero=Xµs  twoQ=Xµs  rand=Xµs
 max|Δ|: zero=X  twoQ=X  rand=X
```

## Running

```bash
go run .
```
