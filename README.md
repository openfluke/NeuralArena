# 🧠 NeuralArena

> ⚠️ **Note**: Development has moved on to **[Loom](https://github.com/openfluke/loom)** — the successor framework with 3D grid architectures, parallel layer linking, and advanced telemetry.

---

## What is NeuralArena?

**NeuralArena** is a research sandbox — a collection of 56 standalone experiments testing different approaches to neural network training. Think of it as a scientist's workbench where ideas get prototyped before being promoted to production code.

This repository sits on top of the **[paragon](https://github.com/openfluke/paragon)** neural network library, also written in Go. Each folder here is a self-contained experiment exploring something different — from trying gradient-free training to messing with GPU acceleration.

If you've stumbled into this repo, you're looking at someone's R&D lab, not a polished library. Some things work, some don't, some are half-finished. That's the point.

---

## Getting Started

### Prerequisites

- Go 1.21+
- Git

### Clone Setup

NeuralArena must be cloned **inside** the paragon directory:

```bash
# Step 1: Clone paragon
git clone https://github.com/openfluke/paragon.git
cd paragon

# Step 2: Clone NeuralArena inside paragon
git clone https://github.com/openfluke/NeuralArena.git

# Step 3: Run an experiment
cd NeuralArena/replay2
go run .
```

**Why inside paragon?** Each experiment's `go.mod` uses `replace paragon => ../` so Go can find the library.

---

## What's Being Tested Here

### 🔄 Replay Mechanism
**Directories**: `replay2`-`replay7`, `nlpReplay1`-`3`, `replayEyeState`, `repeat1`

Trying something a bit different — re-running a layer's computation multiple times during training. The idea is that maybe repeating the forward pass through a layer could help it learn better patterns.

**What we found**: Seems to help shallow networks (1-2 hidden layers) by ~0.5% accuracy, but actually hurts deeper networks. Interesting tradeoff. Entropy-gated replay (where the network decides when to replay based on uncertainty) showed some promise.

---

### 🎭 Behavioral Mimicry (Gradient-Free Training)
**Directories**: `invert1`-`invert4`

This is an attempt at training without backpropagation. Instead of computing gradients, we try to directly adjust weights based on how wrong the output is. Kind of like how you might teach by showing examples rather than explaining the math.

**Why try this?** Backprop is great but maybe there are simpler approaches for certain problems. Also potentially interesting for edge devices where you can't do full gradient computation. Still very experimental.

---

### 🧬 Neural Surgery & Reattachment
**Directories**: `reattachment1`, `reattachment2`

Trying to extract a piece of a trained network, train it separately, then plug it back in. Like removing an organ, doing some work on it, and transplanting it back.

**Use cases we're exploring**:
- Hot-swapping parts of a deployed model
- Training different "modules" separately then combining
- Transfer learning with more control

---

### 👁️ Attention Mechanisms
**Directories**: `attn1`-`attn5`, `attn*afternuke`, `attn*simplesavenreload`

Adding attention layers to paragon and testing different configurations — single-head, multi-head, with/without layer normalization, with replay, etc.

Also spent a lot of time making sure CPU and GPU produce the same outputs for these (the `afternuke` experiments are about testing this after resetting weights).

---

### ⚡ GPU/WebGPU Acceleration
**Directories**: `gpuNativeHandover`, `gpuTesting`, `gpu1`, `gpu2Backdebug`, `trainWebgpu1`-`2`, `device1`, `typeGpu1`-`2`

Trying to get GPU acceleration working through Dawn (Google's WebGPU implementation). This involves:
- CGO bindings to call Dawn from Go
- Writing WGSL compute shaders
- Lots of debugging to make sure GPU outputs match CPU

`device1` just lists available GPUs. `gpuNativeHandover` has the actual shader code for vector operations.

---

### 🎨 Generative / Diffusion Models
**Directories**: `face1`-`4`, `time1`-`4`, `smalldiffpoc1`-`2`, `na1`-`3`, `exp1`

Playing with masked diffusion — the idea of starting with all [MASK] tokens and gradually "unmasking" to generate content. Applied to:

- **Emoji faces** (`face*`) — Generating 5x5 pixel faces
- **Stock prices** (`time*`) — Predicting price movements (up/down/flat)
- **Text** (`na*`, `smalldiff*`) — Generating sentences

These use transformer encoders with cross-entropy training on masked positions. Quality varies.

---

### 📊 Benchmarking & Type Testing
**Directories**: `benchmark1`, `type1`-`2`, `typeDyn1`

Testing performance across different numeric types (int8, float32, float64) and activation functions.

**Key finding**: Pre-converting types before operations is 20-30% faster than inline conversion. Sounds obvious but good to verify.

---

### 🔃 Network Inversion
**Directories**: `reverse1`

Trying to run a network "backwards" — given an output, recover the input. Tests different approaches:
- Exact analytical inverse (only works for linear networks with square weights)
- Layer-by-layer inversion
- Simulated annealing (random search)
- Adam optimizer

Linear networks can be inverted quite well. Non-linear ones... not so much.

---

### 💰 Real-World Data
**Directories**: `fin1`

Testing on actual data — Bank Marketing dataset for predicting customer subscriptions. Downloads data, trains, saves model in JSON and GOB formats, reloads and verifies accuracy is preserved.

---

## Codebase Structure

Every experiment follows roughly the same pattern:

```
experiment_name/
├── engine.go       # Main code
├── go.mod          # Module definition
├── go.sum          # Deps
├── README.md       # What this experiment does
└── *.json/*.csv    # Saved models/data (gitignored)
```

---

## The Paragon Framework

Everything here depends on `paragon`, which provides:

| Feature | Description |
|---------|-------------|
| Generic types | `Network[float32]`, `Network[float64]`, etc. |
| Layer types | Dense, Conv2D, RNN, LSTM, Attention, Transformer |
| Activations | ReLU, Sigmoid, Tanh, LeakyReLU, ELU, Softmax |
| Serialization | JSON and GOB formats |
| GPU support | WebGPU via Dawn (experimental) |
| ADHD metrics | Custom performance evaluation |

---

## Rough Assessment

| What | Status |
|------|--------|
| **Research exploration** | ✅ 56 experiments, lots of ground covered |
| **Production ready** | ❌ Nope, this is a playground |
| **Documentation** | ⚠️ Some experiments have good READMEs, some don't |
| **Testing** | ⚠️ Ad-hoc, experiment-by-experiment |
| **Reproducibility** | ⚠️ Mixed — some use seeds, some don't |

---

## Stuff That Might Be Worth Looking At

If you're poking around and want to see something interesting:

1. **`invert1`** — Gradient-free training. Unusual approach, might spark ideas.
2. **`reattachment1`** — Network surgery. The workflow for extracting and reattaching subnetworks is kind of neat.
3. **`nlpReplay1`** — Entropy-gated replay. The network decides when to repeat computations.
4. **`gpuNativeHandover`** — If you're curious about Dawn/WebGPU from Go.
5. **`face1`** — Simple masked diffusion. Easy to understand the concept.

---

## Related Projects

| Project | What It Is |
|---------|------------|
| **[paragon](https://github.com/openfluke/paragon)** | The neural network library this builds on |
| **[Loom](https://github.com/openfluke/loom)** | Where development has moved — 3D grid architectures |

---

## License

Same as paragon. See that repo for details.
