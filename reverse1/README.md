# Network Inversion / Reverse Pass

Tests various methods to invert (reverse) a neural network's forward pass.

## What It Does

Given a network output, attempts to reconstruct the original input using:
1. **ReverseExact** — Analytical inverse for linear networks with equal layer sizes
2. **ReverseLayerByLayer** — Layer-by-layer inversion for uniform widths
3. **SimulatedAnnealing** — Stochastic optimization for non-linear networks
4. **Adam optimizer** — Gradient-based reconstruction

## Test Configurations

| Name | Architecture | Activation |
|------|--------------|------------|
| Shallow-Small-4 | 4→4→4 | Linear |
| Shallow-Wide-8 | 4→8→4 | Linear |
| Shallow-NonLinear | 4→4→4 | LeakyReLU |
| Deep-Small-4x3 | 4→4→4→4→4 | Linear |
| Deep-Wide-8x4 | 4→8→8→8→8→4 | Linear |
| Deep-NonLinear | 4→4→4→4→4 | LeakyReLU |
| Unequal-Sizes | 4→8→8→6 | Linear |

## Evaluation

- Mean Squared Error between original and reconstructed inputs
- Multiple input samples for robustness
- Summary table with pass/fail status

## Success Criteria

| Error | Status |
|-------|--------|
| < 1e-3 | ✓ Approximate reconstruction |
| ≥ 1e-3 | ⚠ Error too high |

## Running

```bash
go run .
```
