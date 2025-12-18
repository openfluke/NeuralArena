# Small Diffusion PoC - Text Generation

Proof-of-concept for text generation using a simple diffusion model.

## What It Does

- **Simple text corpus** — Uses famous quotes/sentences for training
- **Masked diffusion** — Progressively unmasks tokens to generate text
- **Vocabulary building** — Automatic tokenizer from training data

## Training Data

```
"the quick brown fox jumps over the lazy dog"
"a journey of a thousand miles begins with a single step"
"to be or not to be that is the question"
```

## Network Architecture

Simple 3-layer network:
```
input (seq_len × vocab_size) → 128 hidden → output (seq_len × vocab_size)
```

## Running

```bash
go run .
```

Generates text after training.
