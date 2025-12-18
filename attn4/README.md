# Attention Training: CPU vs GPU Comparison

Tests that training (forward + backward pass) produces equivalent results on CPU and GPU.

## What It Tests

- **Training equivalence** — Same model trained on CPU and GPU should converge similarly
- **Backward pass on GPU** — Verifies GPU gradient computation
- **Speedup measurement** — GPU vs CPU per-epoch performance

## Network Architecture

```
8x8 input → 4x4 hidden → 4x4 (dense or mixed attn) → 1x4 output (softmax)
```

## Configurations

| Model | Heads | DK | Normalization |
|-------|-------|----|----|
| `dense` | - | - | - |
| `attn_H1_noNorm` | 1 | 64 | No |
| `attn_H2_norm` | 2 | 32 | Yes |

## Training Config

- **Epochs**: 50
- **Learning Rate**: 0.01
- **Gradient Clipping**: ±1.0
- **Loss**: MSE (0.5 × Σ(pred - target)²)

## Output Format

```
[model_name]
  CPU: total=Xms  avg_epoch=Xµs  final_loss=X.XXXXXX
       final_output: <sha12>
  GPU: init=Xms  total=Xms  avg_epoch=Xµs  final_loss=X.XXXXXX
       final_output: <sha12>
  Output delta: X.XXXXe-XX
  Speedup: X.XXx (GPU vs CPU per epoch)
```

## Running

```bash
go run .
```

Saves models to `bundles_training/` directory.
