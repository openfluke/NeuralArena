# NeuralArena: Black-Box Behavioral Distillation Experiments

NeuralArena is a focused research sandbox for exploring **black-box neural distillation** — where a student network learns to mimic a fully trained teacher solely from behavior, without access to internal weights or gradients.

The central objective is simple but profound:

> Can a randomly initialized student network learn to match a teacher's outputs by only observing its predictions?

---

## 🧠 Project Goal

To discover and benchmark a range of _non-gradient_ distillation techniques that push the student's accuracy (measured by ADHD Score) as close as possible to that of the teacher.

- **Teacher**: Fully trained on MNIST with conventional backpropagation.
- **Student**: Starts from scratch; receives no gradients or architecture knowledge.
- **Target**: Reach an ADHD Score of **≥ 50** with only behavioral mimicry.

---

## 🧪 Strategies Attempted

Across dozens of prototypes, methods were developed around five core ideas:

---

### 🔁 **Upstream Feedback Methods**

These directly propagate scalar error signals through the student’s layers, backward from output to input:

- **`adjustNetworkUpstream`**  
  _Simple mean-pixel proxy and scalar error feedback (our strongest performer)._  
  ✅ _Peak Score: ~50_

- `adjustNetworkUpstreamSmart`  
  _Normalized input proxy with layer scaling._

- `adjustNetworkUpstreamDepthScaled`  
  _RMS-based proxy with per-weight clipping._

- `adjustNetworkWaveProp`  
  _Wave-like attenuation through spatial and momentum fields._

- `adjustNetworkPulseFlow`  
  _Randomized sparsity with logarithmic error modulation._

---

### 🧬 **Biological / Synaptic Learning**

Inspired by STDP, Hebbian rules, and synaptic memory:

- `adjustNetworkSTDPDirect`  
  _Plasticity based on timing and directional correlation._

- `adjustNetworkHebbError`  
  _Hebbian coincidence scaled by signed teacher error._

- `adjustNetworkMomentum`  
  _First-order momentum memory per weight and bias._

---

### 🧭 **Structural / Feature-Oriented**

Focus on spatial or quadrant-based rewiring using feature salience:

- `adjustNetworkFeatureEcho`  
  _Per-quadrant activation shaping and echo feedback._

- `adjustNetworkSparseEcho`  
  _Max-intensity pixel proxy with selective routing._

---

### 🛰️ **Teacher Behavior Distillation Methods**

Designed to emulate the teacher via observed outputs only (true black-box mimicry):

- `projectiveDistillationUpstream`  
  _Behavioral deltas broadcasted backward with decay._

- `echoDistillationPulse`  
  _Temporal echo using teacher’s confidence-weighted error._

- `reverseCausalTraceAlign`  
  _Causal tracing from output error to early layers._

- `errorSculptPropagation`  
  _Tanh-sculpted feedback scaled by activation magnitude._

- `eventTraceAlignment`  
  _Trace-based reinforcement tied to teacher’s predicted class._

- `eventTraceAlignTopK`  
  _Top-K activation-based reinforcement strategy._

---

### 🧠 **Correlation & Memory-Based Conditioning**

Persistent reinforcement guided by correlation strength over time:

- `correlationTraceAdjustment`  
  _Three-way causal correlation trace across neurons._

- `correlationTraceReinforceV2`  
  _Trust-based memory per neuron and synapse (best among these)._

---

### 🧪 **Exploratory Experimental Variants**

New conceptual territory, including latent and counterfactual ideas:

- `latentSpacePulseInjection`  
  _Latent modulation via inferred attractor pulses._

- `multiverseInversionAblation`  
  _Node salience identified via lesion tests and ablation._

- `eventTraceAlignmentTopK`  
  _Sparse reinforcement over top-activated paths._

- `correlationTraceReinforceV2`  
  _Trust memory for reinforcing consistent causal channels._

---

## 📊 Evaluation Metric: ADHD Score

**ADHD Score** is a custom metric indicating student fidelity.  
It's derived from the percent deviation of predictions vs. teacher output, penalizing large mismatches.

| Range    | Meaning                          |
| -------- | -------------------------------- |
| `90–100` | Nearly perfect mimicry (teacher) |
| `70–90`  | Very close output shapes         |
| `50–70`  | Partial structural alignment     |
| `30–50`  | General behavior shaping         |
| `< 30`   | Poor mimicry / noise             |

---

## 🏆 Final Results Summary

| Strategy                         | Peak Student ADHD | Notes                         |
| -------------------------------- | ----------------- | ----------------------------- |
| `adjustNetworkUpstream`          | **~50**           | Fast, robust, most effective  |
| `correlationTraceReinforceV2`    | ~42.9             | Best memory-based strategy    |
| `projectiveDistillationUpstream` | ~46.9             | Best pure behavior projection |
| `latentSpacePulseInjection`      | ~43.1             | Novel latent-pulse idea       |
| Others                           | 30–40             | Innovative but plateaued      |

All strategies plateaued well below the teacher’s ADHD score (~96–97), confirming the difficulty of behavioral mimicry in the absence of internal supervision.

---

## ✅ Conclusion

> **`adjustNetworkUpstream` is the most consistently effective method** across all tested variants, managing to reach ADHD ~50 without needing gradient information or teacher internals.

This validates the power of:

- Scalar error broadcast,
- Layer-wise decay,
- Proxy-driven feedback.

Most other methods — while creatively structured — failed to surpass this simple yet potent mechanism.

## Couldn't beat adjustNetworkUpstream output:

🧠 No pre-trained model found. Starting training...
Epoch 0, Loss: 0.4799
Epoch 1, Loss: 0.2952
Epoch 2, Loss: 0.2546
Epoch 3, Loss: 0.2329
Epoch 4, Loss: 0.2185
Epoch 5, Loss: 0.2094
Epoch 6, Loss: 0.2010
Epoch 7, Loss: 0.1948
Epoch 8, Loss: 0.1888
Epoch 9, Loss: 0.1826
✅ Training complete.

---------SimplePRINT----------
🧠 ADHD Score: 96.41
📊 Deviation Buckets:

- 0-10% → 9424 samples
- 10-20% → 74 samples
- 20-30% → 27 samples
- 30-40% → 86 samples
- 40-50% → 72 samples
- 50-100% → 189 samples
- 100%+ → 128 samples

---------PrintFullDiagnostics----------
🧠 Full Composite Performance Report
===================================
📦 Samples Evaluated: 10000
✅ Exact Matches: 9424 (94.24%)
📉 Mean Absolute Error: 0.1989
📐 Mean % Deviation: 4.26%
📊 Std Dev of Abs Error: 0.9635
🧮 ADHD Score: 96.41
🧮 Composite Score: 95.32
📊 Deviation Buckets:

- 0-10% → 9424 samples
- 10-20% → 74 samples
- 20-30% → 27 samples
- 30-40% → 86 samples
- 40-50% → 72 samples
- 50-100% → 189 samples
- 100%+ → 128 samples
  🚨 Worst 5 Samples:
  [9051] Expected=1.000, Actual=8.000 | Abs=7.000 | %=700.00%
  [9071] Expected=1.000, Actual=8.000 | Abs=7.000 | %=700.00%
  [5457] Expected=1.000, Actual=8.000 | Abs=7.000 | %=700.00%
  [9540] Expected=1.000, Actual=8.000 | Abs=7.000 | %=700.00%
  [2266] Expected=1.000, Actual=6.000 | Abs=5.000 | %=500.00%

---------PrintSAMPLEDiagnostics----------
🧠 Sample-Level Evaluation (per vector)
======================================
🧪 Total Samples: 10000
✅ Exact Matches (ε=0.0100): 6008 (60.08%)
📉 Mean Absolute Error (per sample): 0.0170
📐 Mean % Deviation (per sample): 0.85%
📊 Std Dev of Abs Error: 0.0448
🧮 ADHD Score (sample-level view): 99.15
🧮 Composite Score (ADHD + Exact): 79.62
📊 Deviation Buckets:

- 0-10% → 10000 samples
- 10-20% → 0 samples
- 20-30% → 0 samples
- 30-40% → 0 samples
- 40-50% → 0 samples
- 50-100% → 0 samples
- 100%+ → 0 samples
  🚨 Worst 5 Samples (by % deviation):
  [4176] MAE=0.2000 | %=10.00%
  [1790] MAE=0.2000 | %=10.00%
  [3817] MAE=0.2000 | %=10.00%
  [5888] MAE=0.2000 | %=10.00%
  [6081] MAE=0.2000 | %=10.00%

---------Student Network Iterative Distillation on Training Data----------

⚙️ maxUpdate=0.50 damping=0.30
Iteration Teacher ADHD Student ADHD Δ
0 96.75 36.34 ⬆
1 96.75 36.33 ⬇
2 96.75 36.32 ⬇

⚙️ maxUpdate=0.50 damping=0.70
Iteration Teacher ADHD Student ADHD Δ
0 96.75 43.08 ⬆
1 96.75 43.08 ⬇
2 96.75 43.07 ⬇

⚙️ maxUpdate=0.50 damping=0.20
Iteration Teacher ADHD Student ADHD Δ
0 96.75 32.05 ⬆
1 96.75 32.06 ⬆
2 96.75 32.06 ⬆

⚙️ maxUpdate=0.50 damping=0.10
Iteration Teacher ADHD Student ADHD Δ
0 96.75 36.13 ⬆
1 96.75 36.13 =
2 96.75 36.13 =

⚙️ maxUpdate=0.50 damping=0.01
Iteration Teacher ADHD Student ADHD Δ
0 96.75 38.98 ⬆
1 96.75 39.25 ⬆
2 96.75 39.32 ⬆

⚙️ maxUpdate=0.10 damping=0.01
Iteration Teacher ADHD Student ADHD Δ
0 96.75 47.10 ⬆
1 96.75 47.10 =
2 96.75 47.10 =

⚙️ maxUpdate=5.00 damping=0.01
Iteration Teacher ADHD Student ADHD Δ
0 96.75 43.26 ⬆
1 96.75 43.38 ⬆
2 96.75 43.40 ⬆

---------Student Network Permuted Error/LR Sweep (🧪 Experimental Upstream Divergence)----------
Error LR Student ADHD  
-0.30 0.001 43.99  
-0.30 0.010 37.89  
-0.30 0.050 42.25  
-0.30 0.100 45.59  
-0.30 0.200 20.63  
-0.10 0.001 39.70  
-0.10 0.010 31.67  
-0.10 0.050 30.76  
-0.10 0.100 11.81  
-0.10 0.200 38.86  
-0.05 0.001 34.94  
-0.05 0.010 37.71  
-0.05 0.050 38.33  
-0.05 0.100 32.66  
-0.05 0.200 48.62  
0.05 0.001 39.09  
0.05 0.010 40.58  
0.05 0.050 40.25  
0.05 0.100 35.21  
0.05 0.200 43.33  
0.10 0.001 37.32  
0.10 0.010 43.75  
0.10 0.050 44.82  
0.10 0.100 42.64  
0.10 0.200 38.17  
0.30 0.001 38.20  
0.30 0.010 27.50  
0.30 0.050 42.76  
0.30 0.100 45.03  
0.30 0.200 49.15

---------Student Network Permuted Error/LR Sweep (🧪 Extreme Range Exploration)----------
Error LR Student ADHD  
-1.00 0.0001 43.77  
-1.00 0.0010 40.30  
-1.00 0.0100 36.24  
-1.00 0.0500 45.41  
-1.00 0.1000 42.61  
-1.00 0.2000 34.82  
-1.00 0.5000 50.62  
-1.00 1.0000 32.80  
-0.50 0.0001 25.19  
-0.50 0.0010 42.27  
-0.50 0.0100 44.51  
-0.50 0.0500 32.62  
-0.50 0.1000 47.50  
-0.50 0.2000 35.85  
-0.50 0.5000 19.34  
-0.50 1.0000 46.71  
-0.30 0.0001 35.20  
-0.30 0.0010 42.49  
-0.30 0.0100 31.35  
-0.30 0.0500 38.51  
-0.30 0.1000 43.02  
-0.30 0.2000 44.85  
-0.30 0.5000 40.34  
-0.30 1.0000 9.87  
-0.10 0.0001 41.22  
-0.10 0.0010 34.77  
-0.10 0.0100 35.36  
-0.10 0.0500 32.31  
-0.10 0.1000 46.61  
-0.10 0.2000 41.50  
-0.10 0.5000 36.23  
-0.10 1.0000 14.27  
-0.05 0.0001 33.90  
-0.05 0.0010 45.90  
-0.05 0.0100 18.72  
-0.05 0.0500 43.17  
-0.05 0.1000 38.23  
-0.05 0.2000 29.43  
-0.05 0.5000 47.98  
-0.05 1.0000 43.96  
0.00 0.0001 36.13  
0.00 0.0010 19.19  
0.00 0.0100 28.94  
0.00 0.0500 34.43  
0.00 0.1000 12.38  
0.00 0.2000 41.23  
0.00 0.5000 41.33  
0.00 1.0000 37.39  
0.05 0.0001 33.73  
0.05 0.0010 32.02  
0.05 0.0100 48.90  
0.05 0.0500 41.32  
0.05 0.1000 36.12  
0.05 0.2000 46.41  
0.05 0.5000 19.33  
0.05 1.0000 45.53  
0.10 0.0001 30.51  
0.10 0.0010 29.01  
0.10 0.0100 38.74  
0.10 0.0500 45.14  
0.10 0.1000 41.37  
0.10 0.2000 36.00  
0.10 0.5000 33.22  
0.10 1.0000 19.81  
0.30 0.0001 42.34  
0.30 0.0010 39.50  
0.30 0.0100 27.40  
0.30 0.0500 46.84  
0.30 0.1000 38.74  
0.30 0.2000 43.60  
0.30 0.5000 35.32  
0.30 1.0000 9.86  
0.50 0.0001 46.26  
0.50 0.0010 19.19  
0.50 0.0100 34.94  
0.50 0.0500 42.05  
0.50 0.1000 43.88  
0.50 0.2000 44.58  
0.50 0.5000 44.35  
0.50 1.0000 40.56  
1.00 0.0001 27.70  
1.00 0.0010 41.97  
1.00 0.0100 33.28  
1.00 0.0500 37.90  
1.00 0.1000 36.25  
1.00 0.2000 31.08  
1.00 0.5000 41.51  
1.00 1.0000 42.84
