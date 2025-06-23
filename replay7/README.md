# Replay Mechanism Experiments in Paragon

This repository contains experiments evaluating the "replay" mechanism in the Paragon neural network framework, where a hidden layer's computation is re-run during the forward or backward pass to enhance learning or performance. Experiments were conducted on the MNIST dataset with various network configurations.

## Experiments Overview

We tested the replay mechanism across multiple benchmarks:

1. **Single Compare**: Compared a baseline network (no replay) with a replay-enabled network on the first hidden layer.
2. **Benchmark Replay vs Baseline**: Ran 10 and 100 instances of both baseline and replay networks for statistical analysis.
3. **Benchmark Replay Depths**: Varied the number of hidden layers (2–4) and the depth of replay application.
4. **Benchmark Max Replay**: Tested different maximum replay cycles (0–3) in a 4-hidden-layer network.
5. **Benchmark Replay Before After**: Compared replay applied before vs. after normal layer computation in a single-hidden-layer network.

## Key Findings

- **Shallow Networks (1–2 Hidden Layers)**:

  - Replay consistently improved performance.
  - **Single Compare**: Replay network scored ADHD 96.17 vs. 95.79 (baseline) and accuracy 93.93% vs. 93.34%.
  - **100 Runs**: Replay averaged ADHD 96.03 vs. 95.68 (baseline) and accuracy 93.71% vs. 93.18%.
  - **Before vs. After**: Replay "after" (ADHD 96.09, accuracy 93.80%) slightly outperformed "before" (ADHD 96.07, accuracy 93.73%) and baseline (ADHD 95.71, accuracy 93.22%).

- **Deeper Networks (3–4 Hidden Layers)**:

  - Replay benefits diminished or reversed.
  - **Replay Depths**: For 2 layers, replay on both (ReplayD=2) yielded ADHD 83.94 and accuracy 75.06%, better than no replay (ADHD 83.28, accuracy 74.12%). For 3–4 layers, no replay often performed best (e.g., 4 layers: ADHD 51.97, accuracy 33.79% with no replay vs. ADHD 49.58, accuracy 28.46% with ReplayD=4).
  - **Max Replay**: In a 4-layer network, increasing replay cycles hurt performance (e.g., MaxReplay=0: ADHD 52.00, accuracy 33.99%; MaxReplay=3: ADHD 47.58, accuracy 27.72%).

- **Optimal Configuration**:
  - Replay is most effective in shallow networks with a single cycle, applied after computation.
  - Deeper or capacity-constrained networks see performance degradation with replay, likely due to amplified gradient noise.

## Experimental Setup

- **Dataset**: MNIST (28x28 handwritten digits).
- **Architecture**: Varied from 1 hidden layer (16x16) to 4 hidden layers (8x8), with input 28x28 and output 10x1 (softmax).
- **Training**: 20 epochs, learning rate 0.001, 80% CPU utilization for parallel runs.
- **Metrics**:
  - **ADHD Score**: Custom metric rewarding predictions closer to true labels (higher is better).
  - **Accuracy**: Percentage of correct classifications.

## Results Summary

| Experiment           | Configuration            | ADHD (avg) | Accuracy (avg) |
| -------------------- | ------------------------ | ---------- | -------------- |
| Single Compare       | Baseline (1 hidden)      | 95.79      | 93.34%         |
|                      | Replay (1 hidden, after) | 96.17      | 93.93%         |
| 100 Runs             | Baseline                 | 95.68      | 93.18%         |
|                      | Replay (after, 1 cycle)  | 96.03      | 93.71%         |
| Replay Depths (2 HL) | No Replay                | 83.28      | 74.12%         |
|                      | ReplayD=2                | 83.94      | 75.06%         |
| Replay Depths (4 HL) | No Replay                | 51.97      | 33.79%         |
|                      | ReplayD=4                | 49.58      | 28.46%         |
| Max Replay (4 HL)    | MaxReplay=0              | 52.00      | 33.99%         |
|                      | MaxReplay=3              | 47.58      | 27.72%         |
| Before vs After      | Baseline                 | 95.71      | 93.22%         |
|                      | Replay Before            | 96.07      | 93.73%         |
|                      | Replay After             | 96.09      | 93.80%         |

## Limitations and Future Work

- **Dataset**: Limited to MNIST; results may not generalize to complex datasets (e.g., CIFAR-10).
- **Configurations**: Not all replay variants (e.g., layer-specific offsets) were exhaustively tested.
- **Future Directions**: Explore adaptive learning rates, warm-up phases, or selective replay for deeper networks.

For detailed results and implementation, see the benchmark scripts and Paragon source code in this repository.

# Benchmark Results

## 10× Benchmark

| Run | Kind     | ADHD  | Acc%  |
| --- | -------- | ----- | ----- |
| 5   | baseline | 95.79 | 93.34 |
| 6   | replay   | 96.17 | 93.93 |
| 1   | baseline | 95.78 | 93.35 |
| 8   | replay   | 96.20 | 93.93 |
| 4   | baseline | 95.68 | 93.19 |
| 1   | replay   | 96.05 | 93.79 |
| 3   | baseline | 95.72 | 93.27 |
| 3   | replay   | 96.13 | 93.81 |
| 8   | baseline | 95.86 | 93.29 |
| 0   | replay   | 96.17 | 93.91 |
| 2   | baseline | 95.76 | 93.29 |
| 7   | replay   | 96.05 | 93.46 |
| 9   | baseline | 95.36 | 92.80 |
| 2   | replay   | 95.87 | 93.37 |
| 0   | baseline | 95.71 | 93.40 |
| 4   | replay   | 95.58 | 93.23 |
| 7   | baseline | 95.59 | 93.08 |
| 9   | replay   | 95.98 | 93.51 |
| 6   | baseline | 95.75 | 93.19 |
| 5   | replay   | 95.94 | 93.58 |

**Average:**

- Baseline: ADHD = 95.70, Acc% = 93.22
- Replay: ADHD = 96.01, Acc% = 93.65

## 100× Benchmark (80% CPUs)

| Run | ADHD_bas | ADHD_rep | ΔADHD | Acc%\_bas | Acc%\_rep | ΔAcc% |
| --- | -------- | -------- | ----- | --------- | --------- | ----- |
| 0   | 95.78    | 96.13    | +0.35 | 93.43     | 93.78     | +0.35 |
| 1   | 95.24    | 96.20    | +0.95 | 92.60     | 93.92     | +1.32 |
| 2   | 95.62    | 96.13    | +0.51 | 93.18     | 93.95     | +0.77 |
| 3   | 95.68    | 96.14    | +0.45 | 93.10     | 93.91     | +0.81 |
| 4   | 95.74    | 95.90    | +0.16 | 93.27     | 93.63     | +0.36 |
| 5   | 95.77    | 96.18    | +0.41 | 93.14     | 94.00     | +0.86 |
| 6   | 95.69    | 96.15    | +0.46 | 93.19     | 93.89     | +0.70 |
| 7   | 95.64    | 96.27    | +0.63 | 93.07     | 93.93     | +0.86 |
| 8   | 95.41    | 96.16    | +0.75 | 92.85     | 93.94     | +1.09 |
| 9   | 95.48    | 95.92    | +0.44 | 92.88     | 93.60     | +0.72 |
| 10  | 95.67    | 95.97    | +0.30 | 93.30     | 93.72     | +0.42 |
| 11  | 95.44    | 96.20    | +0.76 | 92.80     | 93.94     | +1.14 |
| 12  | 95.53    | 96.08    | +0.55 | 92.95     | 93.81     | +0.86 |
| 13  | 95.92    | 95.88    | -0.03 | 93.39     | 93.65     | +0.26 |
| 14  | 95.40    | 95.91    | +0.51 | 92.57     | 93.61     | +1.04 |
| 15  | 95.60    | 96.03    | +0.43 | 93.09     | 93.64     | +0.55 |
| 16  | 95.40    | 96.10    | +0.70 | 92.80     | 93.66     | +0.86 |
| 17  | 95.74    | 96.03    | +0.29 | 93.30     | 93.81     | +0.51 |
| 18  | 95.39    | 95.95    | +0.56 | 92.78     | 93.61     | +0.83 |
| 19  | 95.78    | 96.18    | +0.41 | 93.22     | 93.97     | +0.75 |
| 20  | 95.66    | 95.88    | +0.22 | 93.16     | 93.48     | +0.32 |
| 21  | 95.66    | 96.02    | +0.37 | 93.24     | 93.72     | +0.48 |
| 22  | 95.88    | 96.36    | +0.49 | 93.44     | 94.13     | +0.69 |
| 23  | 95.76    | 96.39    | +0.64 | 93.29     | 94.12     | +0.83 |
| 24  | 95.65    | 95.90    | +0.25 | 93.22     | 93.53     | +0.31 |
| 25  | 95.93    | 96.06    | +0.13 | 93.47     | 93.72     | +0.25 |
| 26  | 95.69    | 96.00    | +0.31 | 93.21     | 93.46     | +0.25 |
| 27  | 95.63    | 96.10    | +0.47 | 92.99     | 93.78     | +0.79 |
| 28  | 95.64    | 95.89    | +0.25 | 93.12     | 93.55     | +0.43 |
| 29  | 95.34    | 96.01    | +0.67 | 92.82     | 93.53     | +0.71 |
| 30  | 95.72    | 96.21    | +0.49 | 93.26     | 93.94     | +0.68 |
| 31  | 95.58    | 96.22    | +0.64 | 92.85     | 93.76     | +0.91 |
| 32  | 95.81    | 96.06    | +0.25 | 93.35     | 93.71     | +0.36 |
| 33  | 95.58    | 96.07    | +0.49 | 93.04     | 93.82     | +0.78 |
| 34  | 95.49    | 96.09    | +0.60 | 93.03     | 93.83     | +0.80 |
| 35  | 95.77    | 96.14    | +0.37 | 93.35     | 93.82     | +0.47 |
| 36  | 95.68    | 95.96    | +0.27 | 93.28     | 93.56     | +0.28 |
| 37  | 95.71    | 96.00    | +0.29 | 93.32     | 93.67     | +0.35 |
| 38  | 95.61    | 95.96    | +0.34 | 93.24     | 93.70     | +0.46 |
| 39  | 95.72    | 95.95    | +0.22 | 93.19     | 93.59     | +0.40 |
| 40  | 95.60    | 96.07    | +0.47 | 93.03     | 93.83     | +0.80 |
| 41  | 95.65    | 95.91    | +0.26 | 93.11     | 93.48     | +0.37 |
| 42  | 95.59    | 95.89    | +0.30 | 93.03     | 93.54     | +0.51 |
| 43  | 95.59    | 95.99    | +0.40 | 93.21     | 93.70     | +0.49 |
| 44  | 95.58    | 96.20    | +0.62 | 93.23     | 93.91     | +0.68 |
| 45  | 95.61    | 95.97    | +0.36 | 92.93     | 93.61     | +0.68 |
| 46  | 95.69    | 96.11    | +0.42 | 93.17     | 93.94     | +0.77 |
| 47  | 95.74    | 95.78    | +0.03 | 93.23     | 93.37     | +0.14 |
| 48  | 95.50    | 96.23    | +0.73 | 93.03     | 94.06     | +1.03 |
| 49  | 95.55    | 95.99    | +0.43 | 93.02     | 93.42     | +0.40 |
| 50  | 95.69    | 96.35    | +0.66 | 93.22     | 94.25     | +1.03 |
| 51  | 95.84    | 95.84    | -0.00 | 93.46     | 93.31     | -0.15 |
| 52  | 95.61    | 96.00    | +0.39 | 93.09     | 93.68     | +0.59 |
| 53  | 95.63    | 96.17    | +0.54 | 92.90     | 94.02     | +1.12 |
| 54  | 95.69    | 95.83    | +0.14 | 93.13     | 93.54     | +0.41 |
| 55  | 95.75    | 96.11    | +0.37 | 93.21     | 93.68     | +0.47 |
| 56  | 95.75    | 95.94    | +0.19 | 93.31     | 93.63     | +0.32 |
| 57  | 95.48    | 95.93    | +0.45 | 92.98     | 93.42     | +0.44 |
| 58  | 95.75    | 95.90    | +0.15 | 93.34     | 93.54     | +0.20 |
| 59  | 95.81    | 96.13    | +0.32 | 93.19     | 94.02     | +0.83 |
| 60  | 95.76    | 96.07    | +0.31 | 93.24     | 93.70     | +0.46 |
| 61  | 95.88    | 96.02    | +0.14 | 93.53     | 93.79     | +0.26 |
| 62  | 95.96    | 95.96    | -0.01 | 93.57     | 93.72     | +0.15 |
| 63  | 95.57    | 95.91    | +0.34 | 92.98     | 93.40     | +0.42 |
| 64  | 95.56    | 95.96    | +0.40 | 93.20     | 93.76     | +0.56 |
| 65  | 95.75    | 95.77    | +0.02 | 93.16     | 93.39     | +0.23 |
| 66  | 95.78    | 96.05    | +0.27 | 93.25     | 93.63     | +0.38 |
| 67  | 95.90    | 96.17    | +0.28 | 93.40     | 93.78     | +0.38 |
| 68  | 95.65    | 96.02    | +0.37 | 93.05     | 93.69     | +0.64 |
| 69  | 95.82    | 95.93    | +0.10 | 93.52     | 93.35     | -0.17 |
| 70  | 95.91    | 96.18    | +0.27 | 93.56     | 93.98     | +0.42 |
| 71  | 95.62    | 96.05    | +0.43 | 93.15     | 93.76     | +0.61 |
| 72  | 95.74    | 96.32    | +0.58 | 93.28     | 94.05     | +0.77 |
| 73  | 95.79    | 96.14    | +0.36 | 93.38     | 93.92     | +0.54 |
| 74  | 95.55    | 96.17    | +0.62 | 92.96     | 93.86     | +0.90 |
| 75  | 95.64    | 95.67    | +0.03 | 93.13     | 93.31     | +0.18 |
| 76  | 95.77    | 95.89    | +0.12 | 93.31     | 93.48     | +0.17 |
| 77  | 95.61    | 96.17    | +0.56 | 92.99     | 93.80     | +0.81 |
| 78  | 95.78    | 96.01    | +0.23 | 93.50     | 93.71     | +0.21 |
| 79  | 95.92    | 96.04    | +0.12 | 93.52     | 93.59     | +0.07 |
| 80  | 95.73    | 95.82    | +0.09 | 93.08     | 93.39     | +0.31 |
| 81  | 95.77    | 96.02    | +0.26 | 93.44     | 93.91     | +0.47 |
| 82  | 95.65    | 96.07    | +0.42 | 93.19     | 93.64     | +0.45 |
| 83  | 95.67    | 95.92    | +0.25 | 93.24     | 93.76     | +0.52 |
| 84  | 95.82    | 96.11    | +0.29 | 93.23     | 93.78     | +0.55 |
| 85  | 95.58    | 95.77    | +0.19 | 93.02     | 93.55     | +0.53 |
| 86  | 95.71    | 95.79    | +0.08 | 93.16     | 93.41     | +0.25 |
| 87  | 95.50    | 95.88    | +0.38 | 93.00     | 93.54     | +0.54 |
| 88  | 95.59    | 95.94    | +0.35 | 93.19     | 93.66     | +0.47 |
| 89  | 95.65    | 96.04    | +0.39 | 93.13     | 93.87     | +0.74 |
| 90  | 95.82    | 95.88    | +0.06 | 93.42     | 93.48     | +0.06 |
| 91  | 95.71    | 95.88    | +0.16 | 93.20     | 93.44     | +0.24 |
| 92  | 95.66    | 95.84    | +0.18 | 93.03     | 93.38     | +0.35 |
| 93  | 95.70    | 96.14    | +0.44 | 93.05     | 93.72     | +0.67 |
| 94  | 95.50    | 96.13    | +0.63 | 92.93     | 93.97     | +1.04 |
| 95  | 95.72    | 96.00    | +0.27 | 93.13     | 93.57     | +0.44 |
| 96  | 95.92    | 95.88    | -0.04 | 93.49     | 93.37     | -0.12 |
| 97  | 95.80    | 96.01    | +0.20 | 93.53     | 93.68     | +0.15 |
| 98  | 95.84    | 96.20    | +0.36 | 93.33     | 93.91     | +0.58 |
| 99  | 96.00    | 96.21    | +0.22 | 93.61     | 93.95     | +0.34 |
| AVG | 95.68    | 96.03    | +0.35 | 93.18     | 93.71     | +0.53 |

## Multi-Hidden Layer Replay Benchmark (10 runs each)

| Hidden | ReplayD | ADHD(avg) | ±sd  | Acc%(avg) | ±sd  |
| ------ | ------- | --------- | ---- | --------- | ---- |
| 2      | 0       | 83.28     | 0.38 | 74.12     | 0.40 |
| 2      | 1       | 83.42     | 0.36 | 74.41     | 0.27 |
| 2      | 2       | 83.94     | 0.23 | 75.06     | 0.24 |
| 3      | 0       | 71.11     | 0.95 | 55.99     | 0.70 |
| 3      | 1       | 71.54     | 0.83 | 56.54     | 0.69 |
| 3      | 2       | 71.25     | 0.65 | 56.35     | 0.70 |
| 3      | 3       | 71.27     | 1.10 | 56.11     | 1.21 |
| 4      | 0       | 51.97     | 0.71 | 33.79     | 0.29 |
| 4      | 1       | 51.88     | 0.73 | 33.18     | 1.58 |
| 4      | 2       | 51.95     | 1.23 | 34.00     | 0.80 |
| 4      | 3       | 52.35     | 0.53 | 34.25     | 0.36 |
| 4      | 4       | 49.58     | 2.64 | 28.46     | 2.93 |

## Max Replay Benchmark (4 Hidden Layers, 10 runs each)

| MaxReplay | ADHD(avg) | ±sd  | Acc%(avg) | ±sd  |
| --------- | --------- | ---- | --------- | ---- |
| 0         | 52.00     | 1.08 | 33.99     | 0.57 |
| 1         | 46.66     | 4.51 | 24.29     | 7.01 |
| 2         | 48.33     | 4.30 | 28.83     | 3.46 |
| 3         | 47.58     | 2.82 | 27.72     | 3.19 |

## Replay Before vs After (1 hidden layer, 10 runs)

| Variant       | ADHD(avg) | ±sd  | Acc%(avg) | ±sd  |
| ------------- | --------- | ---- | --------- | ---- |
| baseline      | 95.71     | 0.16 | 93.22     | 0.25 |
| replay-before | 96.07     | 0.10 | 93.73     | 0.18 |
| replay-after  | 96.09     | 0.18 | 93.80     | 0.18 |
