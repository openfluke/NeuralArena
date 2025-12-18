# WebGPU Training Test (Version 1)

Tests WebGPU-accelerated training on MNIST dataset.

## What It Does

- **MNIST dataset** — Downloads and loads handwritten digit data
- **WebGPU training** — Forward and backward pass on GPU
- **CPU fallback** — Falls back to CPU if WebGPU unavailable

## Files

| File | Description |
|------|-------------|
| `testingcode.go` | Main experiment code |
| `mnist.go` | MNIST data loading utilities |

## Running

```bash
go run .
```

Requires WebGPU support.
