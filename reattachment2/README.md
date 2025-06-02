# ADHD-Based Neural Network Growth Demonstration

## Overview

This project demonstrates an innovative approach to neural network optimization using **ADHD-based growth** (Adaptive Dynamic Hierarchical Development). The system identifies underperforming layers in a neural network and strategically grows the architecture to improve performance, guided by fine-grained ADHD metrics. The demonstration showcases how a small initial network can evolve to handle complex decision boundaries, achieving significant performance improvements.

## Key Features

- **ADHD Evaluation**: Measures network performance with detailed deviation metrics, identifying poorly performing samples and layers.
- **Dynamic Growth**: Adds new layers to the network where ADHD scores indicate deficiencies, using micro-network candidates to test improvements.
- **Efficient Processing**: Processes batches of 64 checkpoint samples and evaluates 160 micro-networks per growth attempt, ensuring computational efficiency.
- **Configurable Parameters**: Allows customization of growth parameters such as batch size, micro-network count, ADHD score thresholds, and learning rates.
- **Comprehensive Analysis**: Provides detailed breakdowns of ADHD scores, network architecture, performance benchmarks, and growth efficiency.

## How It Works

The demonstration follows a structured process to evolve a neural network:

1. **Initialize Network**: Starts with a small network (3 → 4 → 3 → 2) designed for growth.
2. **Generate Training Data**: Creates 1000 samples with a complex, non-linear decision boundary to challenge the network.
3. **Baseline ADHD Evaluation**: Assesses initial performance, producing a baseline ADHD score (e.g., 0.1625).
4. **Configure Growth**: Sets parameters like batch size (64), micro-network count (160), ADHD score threshold (85.0), and improvement threshold (0.1).
5. **ADHD-Based Growth**:
   - Identifies the layer with the lowest ADHD performance (e.g., layer 1 with 135 poor samples).
   - Tests 160 micro-network candidates, each trained for 1 epoch.
   - Selects the best candidate based on ADHD improvement (e.g., 0.8057 improvement).
   - Integrates the improved micro-network into the full model, adding a new layer (e.g., 10 neurons with ReLU activation).
6. **Final Evaluation**: Verifies the improved ADHD score (e.g., 0.3300, a 0.1675 improvement).
7. **Performance Analysis**: Benchmarks inference speed (e.g., 2324284.1 inferences/second) and analyzes growth efficiency (e.g., 10.2704 score improvement/second).

## Results

The demonstration achieved a **successful ADHD-based growth**:

- **ADHD Score Improvement**: Increased from 0.1625 to 0.3300 (0.1675 improvement, 103.08% relative gain).
- **Architecture Growth**: Added 1 new hidden layer with 10 neurons, increasing total parameters to 89.
- **Deviation Distribution Improvement**:
  - Initial: 67.5% of samples in the 50-100% deviation bucket (poor performance).
  - Final: 66.0% of samples in the 0-10% deviation bucket (excellent performance).
- **Efficiency**: Processed 160 micro-networks in 16.31ms, averaging 0.10ms per micro-network.
- **Key Insight**: ADHD-guided growth added 1 layer and improved fine-grained prediction quality by 0.17 points, demonstrating the system's ability to target and enhance weak areas.

## How We Found the Improvement

The improvement was discovered through the following steps:

1. **Baseline Assessment**: The initial ADHD evaluation revealed a low score (0.1625), with 135 samples showing high deviation (50-100%), indicating the network struggled with the complex decision boundary.
2. **Layer Selection**: ADHD analysis pinpointed layer 1 as the weakest, with the highest number of poor-performing samples.
3. **Micro-Network Testing**: 160 micro-networks were trained on a 64-sample checkpoint batch, each adding a new layer. The best candidate achieved a significant ADHD improvement (0.8057).
4. **Full Model Integration**: The best micro-network was reattached to the original network, adding a 10-neuron layer. A full ADHD evaluation confirmed the score increase to 0.3300.
5. **Verification**: The final deviation distribution showed a shift from high-deviation (50-100%) to low-deviation (0-10%) samples, validating the growth's effectiveness.

## Getting Started

### Prerequisites

- Go programming language (version 1.18 or later)
- Basic understanding of neural networks and Go programming

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Ensure the `paragon` package is correctly imported in the test code.

### Running the Demonstration

1. Navigate to the project directory.
2. Run the test code:
   ```bash
   go run main.go
   ```
3. Observe the step-by-step output, including ADHD evaluations, growth process, and final results.

### Configuration

Modify the `GrowConfig` in `main.go` to experiment with different parameters:

- `BatchSize`: Number of checkpoint samples (default: 64).
- `MicroNetCount`: Number of micro-networks per attempt (default: 160).
- `MinADHDScore`: Threshold for triggering growth (default: 85.0).
- `ImprovementThreshold`: Minimum ADHD improvement required (default: 0.1).
- `NewLayerWidth`: Size of new layers (default: 10).

## Code Structure

- **paragon package** (`paragon/`):
  - Defines the `GrowConfig`, `GrowthResult`, `CheckpointBatch`, and `MicroNetCandidate` structs.
  - Implements the `Grow` method for ADHD-based network growth.
  - Includes helper functions for ADHD evaluation, checkpoint creation, and micro-network processing.
- **Test Code** (`main.go`):
  - Orchestrates the demonstration process.
  - Provides functions for network creation, data generation, training, evaluation, and analysis.
  - Outputs detailed logs and metrics for each step.

## Future Improvements

- **Dynamic Layer Sizing**: Adaptively determine the size of new layers based on problem complexity.
- **Multi-Layer Growth**: Allow simultaneous growth at multiple layers for larger networks.
- **Advanced ADHD Metrics**: Incorporate additional performance indicators to refine growth decisions.
- **Parallel Processing**: Optimize micro-network training with concurrent execution for faster growth.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Inspired by research in adaptive neural network architectures and performance optimization.
- Built with the Go programming language for efficiency and type safety.
