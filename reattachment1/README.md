# NeuralArena: Neural Network Surgery Verification

## Overview

NeuralArena is a Go-based project that demonstrates and verifies neural network surgery, including network creation, micro-network extraction, three-way verification, and network modification. The program performs a series of tests to ensure functional equivalence between the main network, its checkpoint, and an extracted micro-network, while also showcasing performance metrics for each operation.

## Features

- **Network Creation/Loading**: Creates a new neural network (3 → 8 → 6 → 2) or loads existing networks from JSON files (`original_network.json`, `modified_network.json`).
- **Micro-Network Extraction**: Extracts a micro-network from a specified checkpoint layer for independent verification.
- **Three-Way Verification**: Compares outputs from:
  - Main network full forward pass
  - Main network from a checkpoint layer
  - Micro-network from the same checkpoint
- **Network Surgery**: Modifies the network structure and verifies the changes, ensuring the modified network maintains compatibility with the original structure.
- **Performance Timing**: Measures and reports execution time for key operations (network creation, verification, surgery, and saving).
- **File I/O**: Saves original, modified, and micro-networks to JSON files for persistence and subsequent testing.

## Prerequisites

- **Go**: Version 1.16 or higher.
- **Paragon Library**: A custom neural network library (replace `"paragon"` with the actual import path in the code).
- **Dependencies**: Standard Go libraries (`fmt`, `log`, `math/rand`, `os`, `time`).

## Usage

1. **Clone the Repository**:

   ```bash
   git clone <repository-url>
   cd NeuralArena/reattachment1
   ```

2. **Run the Program**:

   ```bash
   go run .
   ```

3. **Output**:
   The program produces detailed console output, including:

   - Network setup details (creation or loading).
   - Test input and checkpoint layer.
   - Micro-network extraction results.
   - Three-way verification results with output comparisons.
   - Surgery results (if performed) with pre- and post-surgery outputs.
   - Performance metrics for each step.
   - Confirmation of saved network files.

4. **Starting Fresh**:
   To perform new surgery, delete the saved JSON files (`original_network.json`, `modified_network.json`, `micro_network.json`) and rerun the program.

## Example Output

The program executes in two main scenarios:

1. **Fresh Run (No Saved Files)**:

   - Creates a new network.
   - Extracts a micro-network.
   - Performs three-way verification.
   - Conducts network surgery.
   - Saves all networks to JSON files.

   Key output:

   ```
   === Complete Neural Network Surgery Verification with Timing ===
   🏗  Step 1: Setting up networks...
   ✅ Created network: 3 → 8 → 6 → 2 in 53.516µs
   🔬 Step 2: Setting up micro network...
   ✅ Micro network extracted: 3 layers in 3.369µs
   🧪 Step 3: Running 3-way verification...
   🎉 ALL THREE OUTPUTS MATCH PERFECTLY!
   🚀 Step 5: Demonstrating complete surgery...
   🏆 Surgery complete! Micro network has 3 layers
   💾 Step 6: Saving networks after surgery...
   ✅ Complete verification test finished!
   ```

2. **Run with Saved Files**:

   - Loads modified and original networks from JSON.
   - Extracts a micro-network from the original network.
   - Verifies compatibility between the modified and original structures.
   - Skips surgery to maintain verification integrity.

   Key output:

   ```
   === Complete Neural Network Surgery Verification with Timing ===
   🏗  Step 1: Setting up networks...
   📁 Loading modified network from modified_network.json...
   📁 Loading original network from original_network.json...
   🔬 Step 2: Setting up micro network...
   ✅ Micro network extracted: 3 layers in 36.359µs
   🧪 Step 3: Running 3-way verification...
   🎉 ALL THREE OUTPUTS MATCH PERFECTLY!
   🚀 Step 5: Surgery skipped on loaded modified network
   ✅ Complete verification test finished!
   ```

## Code Structure

- **main.go**:
  - **Main Function**: Orchestrates the entire workflow, including network setup, verification, surgery, and saving.
  - **Helper Functions**:
    - `setupNetworks`: Creates or loads networks based on file existence.
    - `loadNetworkFromFile`: Loads a network from a JSON file.
    - `createNewNetwork`: Creates a new network with specified layer sizes and activations.
    - `extractNewMicroNetwork`: Extracts a micro-network from a checkpoint layer.
    - `demonstrateCompleteSurgery`: Performs and verifies network surgery.
    - `saveAllNetworks`: Saves all networks to JSON files.
    - `fileExists`, `abs`, `getCheckMark`: Utility functions for file checking, absolute value, and status icons.

## Notes

- **Verification Tolerance**: Uses a tolerance of `1e-10` for output comparisons and `1e-6` for surgery and difference testing.
- **Checkpoint Layer**: Set to layer 2 for all tests.
- **Test Input**: Fixed at `[0.1, 0.5, 0.9]` for consistency.
- **Surgery Limitation**: Surgery is only performed on fresh networks to avoid corrupting verification results.
- **Random Seed**: Set to `42` for reproducibility.

## Future Improvements

- Add support for configurable test inputs and checkpoint layers.
- Enhance the Paragon library to support more activation functions and network architectures.
- Implement additional verification methods for robustness.
- Optimize performance for larger networks.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
