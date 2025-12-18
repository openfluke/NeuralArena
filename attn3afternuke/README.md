# CPU/GPU Forward Pass Exact Match Test

Validates that CPU and GPU implementations produce identical forward pass results, down to floating point precision.

## What It Tests

- **Neuron-by-neuron comparison** — Compares every neuron value between CPU and GPU networks
- **Dense model** — Tests pure dense network
- **Attention model** — Tests mixed dense+attention layers
- **Consistency over multiple passes** — Runs 10 forward passes with different inputs

## Network Architecture

```
8x8 input → 4x4 hidden → 4x4 (dense or mixed) → 1x4 output (softmax)
```

## Pass/Fail Criteria

| Max Difference | Result |
|----------------|--------|
| < 1e-6 | ✓ PASS — Within FP32 precision |
| < 1e-3 | ⚠ ACCEPTABLE — OK for training |
| ≥ 1e-3 | ✗ FAIL — Implementation issue |

## Key Features

- Model save/reload to ensure identical starting weights
- Automatic GPU fallback detection
- Layer-by-layer neuron comparison output
- Cleanup of test files after completion

## Running

```bash
go run .
```

Creates temporary `test_dense.json` and `test_attn.json` files (auto-deleted).
