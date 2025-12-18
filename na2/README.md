# Masked Diffusion Transformer - Text Generation

Enhanced diffusion model using masked tokens (BERT-style) instead of noise for denoising.

## What It Does

- **Masked diffusion** — Uses [MASK] tokens instead of noise at each timestep
- **Continuous timestep** — Samples t ~ U[0,1] instead of discrete steps
- **Cross-entropy on masked positions** — Only computes loss where tokens are masked
- **Multi-threaded** — 80% CPU utilization with goroutines

## Difference from na1

| Feature | na1 | na2 |
|---------|-----|-----|
| Noise method | AddNoise | AddNoiseMasked |
| Timestep | Discrete | Continuous U[0,1] |
| Loss positions | All | Only [MASK] tokens |
| Timesteps | 5 | 20 |

## Training Data

500 sentences (23 hand-crafted + 477 generated)

## Running

```bash
go run .
```

Generates text every epoch with loss progress.
