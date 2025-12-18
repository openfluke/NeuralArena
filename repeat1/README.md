# Replay Mechanism with GPU/CPU Comparison

Tests the replay mechanism combined with GPU vs CPU forward pass equivalence.

## What It Tests

- **Network cloning** — Marshal/unmarshal networks via JSON to create identical copies
- **GPU vs CPU output** — Compares forward pass outputs within tolerance (1e-3)
- **Static replay** — Tests replay with "before" and "after" phases
- **Training with replay** — Compares GPU/CPU after backward pass with replay enabled

## Network Architecture

```
28x28 input → 128 hidden (x3) → 10 output
```

All ReLU activations except linear input/output.

## Tests Performed

1. **Clone verification** — Net1, Net2, Net3 produce identical outputs
2. **GPU vs CPU forward** — Timing and output comparison
3. **Replay "before"** — Replay before layer computation
4. **Replay "after"** — Replay after layer computation
5. **Training with replay** — Backward pass on both GPU/CPU + replay

## Key Features

- WebGPU initialization with fallback to CPU
- Tolerance-based comparison (1e-3)
- Timing measurements for GPU vs CPU
- Configurable replay offset, phase, and max iterations

## Running

```bash
go run testingcode.go
```
