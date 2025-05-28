// gpu_debug_test.go
package main

import (
	"fmt"
	"math"
	"math/rand"
	"paragon"
)

func main() {
	// Set seed for reproducible results
	rand.Seed(42)

	fmt.Println("=== GPU Backward Pass Debug Test ===")

	// Create a simple network for testing
	layerSizes := []struct{ Width, Height int }{
		{28, 28}, // Input layer (784 neurons)
		{10, 1},  // Hidden layer (10 neurons)
		{10, 1},  // Output layer (10 neurons)
	}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{false, true, true}

	network := paragon.NewNetwork[float32](layerSizes, activations, fullyConnected)
	network.Debug = true
	network.WebGPUNative = true

	fmt.Printf("Created network with %d layers\n", len(network.Layers))

	// Initialize GPU resources
	fmt.Println("Initializing GPU resources...")
	if err := network.InitializeGPUComplete(); err != nil {
		fmt.Printf("Failed to initialize GPU: %v\n", err)
		return
	}

	fmt.Println("GPU initialization successful!")
	RunSafeBufferWriteTest(network)

	// In your main debug function, add this after GPU initialization:
	fmt.Println("=== Testing Simple Shader Capabilities ===")
	network.TestWithSimpleShader(1)

	// In your main debug function, after the simple shader test:
	fmt.Println("=== Testing Working Backward Shader ===")
	network.TestWorkingBackwardShader(1)

	// In your main debug function, after the working shader test:
	fmt.Println("=== Testing Weight Application ===")
	network.TestWeightApplication()

	fmt.Println("=== Testing Saturation Issue ===")
	network.TestSaturationIssue()

	// In your main debug function, after the working shader test:
	fmt.Println("=== Quick Weight Application Test ===")
	network.QuickWeightApplicationTest()

	fmt.Println("=== Debug Weight Application Process ===")
	network.DebugWeightApplication(1)

	// In your main debug function, after GPU initialization:
	fmt.Println("=== Testing Weight Extraction Cycle ===")
	for l := 1; l <= network.OutputLayer; l++ {
		network.TestWeightCycle(l)
	}

	// Create simple test data
	input := make([][]float64, 28)
	for i := range input {
		input[i] = make([]float64, 28)
		for j := range input[i] {
			input[i][j] = rand.Float64()
		}
	}

	// Create target (one-hot encoding for class 5)
	target := make([][]float64, 1)
	target[0] = make([]float64, 10)
	target[0][5] = 1.0 // Target class

	fmt.Println("=== Testing Forward Pass ===")

	// Test forward pass
	network.Forward(input)

	// Print initial outputs
	outputs := network.GetOutput()
	fmt.Printf("Initial outputs: ")
	for i, out := range outputs {
		fmt.Printf("%.4f ", out)
		if i >= 9 {
			break
		}
	}
	fmt.Println()

	// Compute initial loss
	initialLoss := network.ComputeLoss(target)
	fmt.Printf("Initial loss: %.6f\n", initialLoss)

	fmt.Println("=== Testing GPU Backward Pass ===")

	// Test the fixed backward pass
	learningRate := 0.01
	var clipUpper, clipLower float32 = 1000, -1000

	// Validate GPU backward pass
	err := network.ValidateGPUBackward(input, target, learningRate)
	if err != nil {
		fmt.Printf("GPU backward validation failed: %v\n", err)
		return
	}

	fmt.Println("=== Training Loop Test ===")

	// Test multiple iterations
	for epoch := 0; epoch < 5; epoch++ {
		// In your debug test, before the main training loop:
		fmt.Println("=== Testing GPU Write Capability ===")
		network.TestShaderWeightUpdate()
		network.Forward(input)
		loss := network.ComputeLoss(target)

		fmt.Printf("Epoch %d: Loss=%.6f", epoch, loss)

		// Check for NaN
		if math.IsNaN(loss) {
			fmt.Printf(" - NaN detected!\n")
			break
		}

		// Run backward pass
		network.Backward(target, learningRate, clipUpper, clipLower)

		// Check outputs after backward pass
		network.Forward(input)
		newLoss := network.ComputeLoss(target)

		fmt.Printf(" -> %.6f (change: %+.6f)\n", newLoss, newLoss-loss)

		// Print some output values
		outputs := network.GetOutput()
		fmt.Printf("  Outputs: ")
		for i := 0; i < min(5, len(outputs)); i++ {
			fmt.Printf("%.4f ", outputs[i])
		}
		fmt.Println("...")
	}

	fmt.Println("=== Comparison Test: CPU vs GPU ===")

	// Create identical networks for comparison
	networkCPU := paragon.NewNetwork[float32](layerSizes, activations, fullyConnected)
	networkGPU := paragon.NewNetwork[float32](layerSizes, activations, fullyConnected)

	networkCPU.Debug = false
	networkGPU.Debug = false
	networkGPU.WebGPUNative = true

	// Copy weights to ensure identical starting conditions
	copyNetworkWeights(networkCPU, networkGPU)

	// Initialize GPU for second network
	if err := networkGPU.InitializeGPUComplete(); err != nil {
		fmt.Printf("Failed to initialize second GPU network: %v\n", err)
		return
	}

	// Test both networks on same data
	fmt.Println("Running CPU training...")
	networkCPU.Forward(input)
	lossCPU1 := networkCPU.ComputeLoss(target)
	networkCPU.Backward(target, learningRate, clipUpper, clipLower)
	networkCPU.Forward(input)
	lossCPU2 := networkCPU.ComputeLoss(target)

	fmt.Println("Running GPU training...")
	networkGPU.Forward(input)
	lossGPU1 := networkGPU.ComputeLoss(target)
	networkGPU.Backward(target, learningRate, clipUpper, clipLower)
	networkGPU.Forward(input)
	lossGPU2 := networkGPU.ComputeLoss(target)

	fmt.Printf("CPU: %.6f -> %.6f (change: %+.6f)\n", lossCPU1, lossCPU2, lossCPU2-lossCPU1)
	fmt.Printf("GPU: %.6f -> %.6f (change: %+.6f)\n", lossGPU1, lossGPU2, lossGPU2-lossGPU1)

	// Compare outputs
	outputsCPU := networkCPU.GetOutput()
	outputsGPU := networkGPU.GetOutput()

	fmt.Println("Output comparison:")
	maxDiff := 0.0
	for i := 0; i < min(len(outputsCPU), len(outputsGPU)); i++ {
		diff := math.Abs(outputsCPU[i] - outputsGPU[i])
		if diff > maxDiff {
			maxDiff = diff
		}
		if i < 5 {
			fmt.Printf("  [%d] CPU=%.6f, GPU=%.6f, diff=%.6f\n", i, outputsCPU[i], outputsGPU[i], diff)
		}
	}
	fmt.Printf("Maximum output difference: %.6f\n", maxDiff)

	if maxDiff < 0.001 {
		fmt.Println("✓ GPU and CPU results are very close!")
	} else if maxDiff < 0.01 {
		fmt.Println("⚠ GPU and CPU results have small differences")
	} else {
		fmt.Println("✗ GPU and CPU results differ significantly")
	}

	fmt.Println("=== Test Complete ===")
}

// Helper function to copy weights between networks
func copyNetworkWeights(src, dst *paragon.Network[float32]) {
	for l := 1; l < len(src.Layers); l++ {
		srcLayer := src.Layers[l]
		dstLayer := dst.Layers[l]

		for y := 0; y < srcLayer.Height; y++ {
			for x := 0; x < srcLayer.Width; x++ {
				srcNeuron := srcLayer.Neurons[y][x]
				dstNeuron := dstLayer.Neurons[y][x]

				// Copy bias
				dstNeuron.Bias = srcNeuron.Bias

				// Copy weights
				for i, conn := range srcNeuron.Inputs {
					if i < len(dstNeuron.Inputs) {
						dstNeuron.Inputs[i].Weight = conn.Weight
					}
				}
			}
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Add this to your debug test to check buffer write capability:
// Modified debug test that prevents contamination
func RunSafeBufferWriteTest(network *paragon.Network[float32]) {
	fmt.Println("=== Testing Direct Buffer Write (Safe) ===")

	// Save network state BEFORE test
	savedWeights := network.SaveNetworkWeights()

	// Run isolated test
	err := network.TestBufferWriteIsolated()
	if err != nil {
		fmt.Printf("Buffer write test failed: %v\n", err)
	}

	// Restore network state AFTER test
	network.RestoreNetworkWeights(savedWeights)

	fmt.Println("Network weights restored after test")
}
