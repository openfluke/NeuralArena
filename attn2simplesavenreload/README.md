# Attention Model Save/Reload Test

Tests the serialization and deserialization of attention-based networks using the bundle JSON format.

## What It Tests

- **Model persistence** — Save network to JSON, reload, verify identical outputs
- **Attention variants** — Tests dense baseline + 15 attention configurations
- **Deterministic verification** — Uses SHA-12 hashes to verify output consistency

## Network Architecture

```
8x8 input → 4x4 hidden → 4x4 mixed (dense/attn/dense/attn) → 1x4 output (softmax)
```

## Configurations Tested

| Base Config | Heads | Description |
|-------------|-------|-------------|
| `dense` | - | Pure dense baseline |
| `attnLayer_noNorm` | 1, 2, 3 | Layer-shared attention, no normalization |
| `attnLayer_norm` | 1, 2, 3 | Layer-shared attention + LayerNorm |
| `attnPerSlice_noNorm` | 1, 2, 3 | Per-slice attention, no normalization |
| `attnPerSlice_norm` | 1, 2, 3 | Per-slice attention + LayerNorm |
| `attnLayer_norm_replay` | 1, 2, 3 | Layer-shared + replay mechanism |

## Test Inputs

1. **Zero input** — All zeros (stability check)
2. **Two-query** — Two 1s at fixed positions (classification check)
3. **Random** — Seeded random input (general verification)

## Running

```bash
go run .
```

Outputs saved models to `bundles_v3/` directory.
