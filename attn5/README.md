# Attention Parallel GPU Training

Comprehensive attention mechanism test with parallel training of multiple model variants on GPU.

## What It Tests

- **Parallel training** — Trains all model variants concurrently using goroutines
- **GPU initialization** — Tests forward and backward GPU initialization
- **Multiple attention configs** — Dense baseline + attention variants with 1/2/3 heads
- **ADHD performance metrics** — Bucket comparison and scoring
- **Model persistence** — Save/reload verification

## Network Architecture

```
8x8 input → 4x4 hidden → 4x4 mixed (dense/attn/dense/attn) → 1x4 output (softmax)
```

## Task: Global Two-Query Classification

4-class problem on 8x8 grid with two marked positions:
- Class 0: Same row
- Class 1: Same column  
- Class 2: Same main diagonal
- Class 3: Otherwise

## Training Config

- **Epochs**: 28
- **Learning Rate**: Cosine schedule from 0.01 → 0.003
- **Training samples**: 4096
- **Test samples**: 1024
- **Label smoothing**: 0.05

## Features

- Parallel model training with `sync.WaitGroup`
- Pre/post-training output inspection
- Confusion matrix per model
- ADHD bucket comparison table
- Post-reload accuracy verification

## Running

```bash
go run .
```

Outputs saved to `bundles_v3/` directory.
