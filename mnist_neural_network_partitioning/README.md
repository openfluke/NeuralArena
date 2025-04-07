# Neural Network Partitioning

A Go-based implementation of tag-based neural network partitioning — a minimalist and functional approach to conditional computation within a single neural network structure.

This project demonstrates how to statically route different training samples through specific partitions of a network using tagged forward and backward passes. It simulates a hard-coded **mixture-of-experts (MoE)** model without a learned gating mechanism, enabling multiple expert subnetworks to coexist within a unified architecture.

## Overview

The network is split into independent regions based on _tags_. In the primary example:

- **Even-digit classification** is handled by one partition.
- **Odd-digit classification** is handled by another.

Each tag activates only its designated subset of the network during forward and backward propagation. This means:

- Neurons outside of the tag’s partition remain **inactive and untrained**.
- Accuracy and performance are evaluated on the combined expert performance of the network as a whole.

## Key Concepts

- **Tag-aware Forward/Backward Passes**: The network processes inputs conditionally based on an explicit tag (e.g., label parity). Only neurons with the associated tag are active and updated.
- **Partitioned Layers**: Hidden layers are partially connected. Subsets of neurons belong exclusively to a specific tag, enabling specialization.
- **Shared Output Layer**: Despite having separate pathways, both partitions share the same output layer, creating a unified output space for classification.
- **Sparse Execution**: The network activates only the relevant sub-network per training sample, mimicking the behavior of a sparse MoE.

## Why This Matters

This implementation enables:

- Efficient experimentation with multi-task learning and conditional models.
- Isolation of learning pathways within a single network instance.
- Greater modularity and potential for extending to learned gating or dynamic routing.

## Results

Using the MNIST dataset:

- Each tag (even/odd digits) trains its own partition.
- After 10 epochs of alternating tag-specific passes, the network achieves ~83% test accuracy.

## Future Work

- Add dynamic tagging or learned gating mechanisms.
- Explore tag overlap (shared neurons between partitions).
- Extend to multi-task settings (e.g., digit classification + rotation prediction).
- Build recursive self-generating models leveraging partitioned specialization.
- Enable tags to dynamically activate sub-networks during inference.
