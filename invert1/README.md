# 🧠 Behavioral Mimicry with Paragon Networks

## Overview

This project explores an **alternative to backpropagation**: using **behavioral mimicry** to train simple neural networks in the `paragon` framework. The core idea is to compare a network's output to a target (either a known function or a trained "teacher" model) and manually adjust weights and biases in response to the error — without relying on traditional gradients.

Rather than training networks with abstract loss functions and full backprop pipelines, this approach focuses on **directly shaping behavior** by incrementally modifying the output layer and, optionally, upstream layers. The goal is interpretable, targeted learning through controlled corrections.

## Goals

1. **Demonstrate mimicry-based adaptation** of a student network toward a simple mathematical function (`y = 2x + 3`).
2. **Explore teacher-student behavioral alignment**, where a trained network (`y = 3x - 2`) serves as a dynamic oracle for a randomly initialized student.
3. Develop a **lightweight, modular routine** for experimenting with behavioral deltas, gradient-free learning, and upstream propagation of feedback.

## Components

### `test1()`

- A baseline mimicry loop.
- Uses a **handwritten oracle** (`y = 2x + 3`) as the ground truth.
- A student network with 3 linear layers (1x1 neurons) is adjusted at the **output layer only**, mimicking the behavior based on raw prediction error.
- Updates are **damped and clipped** for stability.

### `behavioralMimicFromTrainedModel()`

- Introduces a **teacher-student framework**.
- The teacher is a **trained Paragon model** that learns `y = 3x - 2` using built-in gradient training.
- The student mimics the teacher’s outputs using **manual behavioral updates** across all layers, not just the output.
- Error propagation is performed using a crude **proxy signal** through the network hierarchy (a kind of hand-crafted signal decay), simulating upstream influence.

### `adjustOutputLayer(...)` and `adjustNetworkUpstream(...)`

- These functions manually tweak weights and biases in response to prediction error.
- They apply behavioral updates **without gradient computation**, based purely on feedback from observed discrepancies.
- This is a **human-readable, deterministic mechanism** for low-level correction and inspection.

## Why This Matters

- Provides a **transparent alternative to gradient descent**, especially useful for small, interpretable networks.
- Supports experimentation with **hybrid learning strategies**, where only part of a network learns behaviorally.
- Aligns with broader goals in the **NMES project**, including layerwise tweaking, ADHD metric evaluation, and low-fidelity generalization.
- Opens pathways for **real-time, low-overhead model correction** — ideal for physical or embedded systems with limited compute.

## Future Directions

- Integrate ADHD performance metrics to visualize and track mimicry quality.
- Extend to **multi-dimensional inputs and outputs**.
- Apply this approach to **construct control** or **adaptive motor behavior** in modular systems.
- Refine upstream adjustment logic with **local learning rules**, e.g. Oja’s rule or feedback alignment.

This project is a foundation for more complex **emergent behavioral learning** in the Biofoundry and NMES ecosystems, where interpretability and modularity are as important as performance.

## OUTPUT

🎯 Starting Behavioral Mimicry from Trained Paragon Model
⚠️ Negative loss (-28.2146) detected at sample 0, epoch 0. Stopping training early.
✅ Teacher Model Trained:
x = 1.00 → y = 1.7523
x = 2.00 → y = 3.5046
x = 3.00 → y = 5.2568
x = 4.00 → y = 7.0091
x = 5.00 → y = 8.7614

🌀 Iteration 0
x = 1.00 | target = 1.7523 | pred = 2.3275 | Δ = -0.5753
x = 2.00 | target = 3.5046 | pred = 4.4122 | Δ = -0.9077
x = 3.00 | target = 5.2568 | pred = 5.7880 | Δ = -0.5311
x = 4.00 | target = 7.0091 | pred = 6.8874 | Δ = 0.1217
x = 5.00 | target = 8.7614 | pred = 8.9980 | Δ = -0.2366

🌀 Iteration 1
x = 1.00 | target = 1.7523 | pred = 1.4333 | Δ = 0.3190
x = 2.00 | target = 3.5046 | pred = 3.2490 | Δ = 0.2555
x = 3.00 | target = 5.2568 | pred = 5.2108 | Δ = 0.0460
x = 4.00 | target = 7.0091 | pred = 7.0893 | Δ = -0.0801
x = 5.00 | target = 8.7614 | pred = 8.6959 | Δ = 0.0655

🌀 Iteration 2
x = 1.00 | target = 1.7523 | pred = 1.6279 | Δ = 0.1243
x = 2.00 | target = 3.5046 | pred = 3.4963 | Δ = 0.0083
x = 3.00 | target = 5.2568 | pred = 5.3407 | Δ = -0.0838
x = 4.00 | target = 7.0091 | pred = 7.0408 | Δ = -0.0317
x = 5.00 | target = 8.7614 | pred = 8.7632 | Δ = -0.0018

🌀 Iteration 3
x = 1.00 | target = 1.7523 | pred = 1.5993 | Δ = 0.1529
x = 2.00 | target = 3.5046 | pred = 3.4467 | Δ = 0.0579
x = 3.00 | target = 5.2568 | pred = 5.3072 | Δ = -0.0503
x = 4.00 | target = 7.0091 | pred = 7.0478 | Δ = -0.0386
x = 5.00 | target = 8.7614 | pred = 8.7494 | Δ = 0.0120

🌀 Iteration 4
x = 1.00 | target = 1.7523 | pred = 1.6191 | Δ = 0.1332
x = 2.00 | target = 3.5046 | pred = 3.4619 | Δ = 0.0426
x = 3.00 | target = 5.2568 | pred = 5.3093 | Δ = -0.0525
x = 4.00 | target = 7.0091 | pred = 7.0429 | Δ = -0.0338
x = 5.00 | target = 8.7614 | pred = 8.7532 | Δ = 0.0082
