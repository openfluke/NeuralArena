# CPU/GPU Forward Pass Exact Match Test (Fixed Seed)

Tests CPU and GPU forward pass equivalence using fixed seeds and saved models for reproducibility.

## What It Tests

- **Deterministic initialization** — Uses fixed seed (12345) for reproducible results
- **Model persistence** — Saves base models to JSON, loads separate copies for CPU/GPU
- **Dense model** — Pure dense network test
- **Attention model** — Mixed dense+attention layers  
- **Multiple passes** — 10 forward passes with different inputs

## Network Architecture

```
8x8 input → 4x4 hidden → 4x4 (dense or mixed) → 1x4 output (softmax)
```

## Pass/Fail Criteria

| Max Difference | Result |
|----------------|--------|
| < 1e-6 | ✓ PASSED (exact match) |
| < 1e-3 | ⚠ PASSED (acceptable) |
| ≥ 1e-3 | ✗ FAILED |

## Key Features

- Fixed random seed for reproducible results
- Model save/load from JSON for identical weights
- Neuron-by-neuron comparison with detailed output
- Summary statistics (max diff, percentage of differing neurons)

## Running

```bash
go run .
```

Creates `base_dense_model.json` and `base_attn_model.json`.
