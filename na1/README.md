# Diffusion Transformer (V5) - Text Generation

Trains a transformer-based diffusion model to generate natural language sentences.

## What It Does

- **Sentence generation** — Trains on conditional "if...then" sentences and generates new ones
- **Transformer encoder** — Custom tokenizer + transformer for text modeling
- **Multi-threaded training** — Uses 80% of CPU cores with goroutines
- **Cosine LR schedule** — Learning rate decay over epochs

## Training Data

- 23 hand-crafted sentences + 477 randomly generated "if the X Y then it Z" patterns
- Total: 500 sentences

## Network Architecture

- **DModel**: 128
- **Heads**: 4
- **Layers**: 2
- **FeedForward**: 512
- **Max Sequence Length**: 10

## Training Config

- **Epochs**: 2000
- **Learning Rate**: 0.002 (with cosine decay)
- **Batch Size**: 10
- **Diffusion Timesteps**: 5
- **Gradient Clipping**: ±5.0

## Generation

Uses temperature-controlled sampling with TopK=3.

## Running

```bash
go run .
```

Prints generated text every 10 epochs and final output.
