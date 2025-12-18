# Bank Marketing Dataset Classification

Binary classification on the UCI Bank Marketing dataset to predict subscription outcomes.

## What It Does

- **Downloads dataset** — Fetches UCI Bank Marketing dataset automatically
- **Feature extraction** — Uses age, balance, and duration as input features
- **Binary classification** — Predicts "yes" or "no" bank subscription
- **Model persistence** — Tests JSON and GOB serialization

## Dataset

- **Source**: UCI Machine Learning Repository
- **Features**: Age, account balance, call duration
- **Target**: Term deposit subscription (yes/no)
- **Samples**: 1000 (limited for demo speed)
- **Split**: 80% train, 20% test

## Network Architecture

```
3 inputs (age, balance, duration) → 8 hidden (ReLU) → 2 outputs (softmax)
```

## Training Config

- **Epochs**: 50
- **Learning Rate**: 0.01

## Evaluation

- Training and testing accuracy
- Sample predictions with actual values
- Comparison between original, JSON-loaded, and GOB-loaded models

## Output

```
Training on Bank Marketing dataset...
---------- Trained Model ----------
Training Accuracy: XX.XX%
Testing Accuracy: XX.XX%
Sample 1: Age=XX, Balance=XX, Duration=XX
  Predicted: yes/no
  Actual: yes/no
```

## Running

```bash
go run .
```

Creates `model.json` and `model.gob`.
