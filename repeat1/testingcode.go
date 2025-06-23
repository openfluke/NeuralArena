package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"paragon" // Replace with actual import path
)

// Global tolerance constant
const tolerance = 1e-3

// getOutput performs a forward pass and returns the network's output
func getOutput(net *paragon.Network[float32], input [][]float64) []float64 {
	net.Forward(input)
	outputLayer := net.Layers[net.OutputLayer]
	output := make([]float64, outputLayer.Width)
	for x := 0; x < outputLayer.Width; x++ {
		output[x] = float64(outputLayer.Neurons[0][x].Value)
	}
	return output
}

// approxEqual compares two float slices using the global tolerance
func approxEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > tolerance {
			return false
		}
	}
	return true
}

// compareNetworks compares outputs and speed of GPU vs CPU networks
func compareNetworks(netGPU, netCPU *paragon.Network[float32], input [][]float64) {
	// Enable WebGPU on netGPU
	netGPU.WebGPUNative = true
	if err := netGPU.InitializeOptimizedGPU(); err != nil {
		fmt.Printf("⚠️ Failed to initialize WebGPU: %v\n", err)
		fmt.Println("   Continuing with CPU-only processing...")
		netGPU.WebGPUNative = false
	} else {
		fmt.Println("✅ WebGPU initialized successfully")
	}

	// Ensure netCPU remains on CPU
	netCPU.WebGPUNative = false

	// Warm-up phase
	netGPU.Forward(input)
	netCPU.Forward(input)

	// Time GPU forward pass
	startGPU := time.Now()
	outputGPU := getOutput(netGPU, input)
	durationGPU := time.Since(startGPU)

	// Time CPU forward pass
	startCPU := time.Now()
	outputCPU := getOutput(netCPU, input)
	durationCPU := time.Since(startCPU)

	// Compare outputs using global tolerance
	if approxEqual(outputGPU, outputCPU) {
		fmt.Println("GPU and CPU outputs match within tolerance")
	} else {
		fmt.Println("GPU and CPU outputs do not match")
		fmt.Println("GPU Output:", outputGPU)
		fmt.Println("CPU Output:", outputCPU)
	}

	// Print timing results
	fmt.Printf("GPU forward pass time: %v\n", durationGPU)
	fmt.Printf("CPU forward pass time: %v\n", durationCPU)
}

func main() {
	// Set seed for reproducibility
	rand.Seed(42)

	// Define network architecture: 28x28 input, 128 hidden, 10 output
	layerSizes := []struct{ Width, Height int }{{28, 28}, {128, 1}, {128, 1}, {128, 1}, {10, 1}}
	activations := []string{"linear", "relu", "relu", "relu", "linear"}
	fullyConnected := []bool{true, true, true, true, true}
	seed := int64(42)

	// Generate random input (28x28)
	input := make([][]float64, 28)
	for i := range input {
		input[i] = make([]float64, 28)
		for j := range input[i] {
			input[i][j] = rand.Float64()
		}
	}

	// Step 1: Create Neural Network 1 (net1)
	net1 := paragon.NewNetwork[float32](layerSizes, activations, fullyConnected, seed)
	net1.WebGPUNative = false

	// Step 2: Create Neural Networks 2 and 3 (net2 and net3)
	net2 := paragon.NewNetwork[float32](layerSizes, activations, fullyConnected, seed)
	net3 := paragon.NewNetwork[float32](layerSizes, activations, fullyConnected, seed)
	net2.WebGPUNative = false
	net3.WebGPUNative = false

	// Step 3: Copy net1 to net2 and net3
	jsonBytes, err := net1.MarshalJSONModel()
	if err != nil {
		fmt.Println("Failed to marshal net1 to JSON:", err)
		return
	}

	err = net2.UnmarshalJSONModel(jsonBytes)
	if err != nil {
		fmt.Println("Failed to unmarshal net2:", err)
		return
	}

	err = net3.UnmarshalJSONModel(jsonBytes)
	if err != nil {
		fmt.Println("Failed to unmarshal net3:", err)
		return
	}

	// Step 4: Compare outputs of net1, net2, and net3
	output1 := getOutput(net1, input)
	output2 := getOutput(net2, input)
	output3 := getOutput(net3, input)

	fmt.Println("Net1 Output:", output1)
	fmt.Println("Net2 Output:", output2)
	fmt.Println("Net3 Output:", output3)

	if !approxEqual(output1, output2) {
		fmt.Println("Mismatch between net1 and net2")
	} else {
		fmt.Println("Net1 and net2 match")
	}

	if !approxEqual(output1, output3) {
		fmt.Println("Mismatch between net1 and net3")
	} else {
		fmt.Println("Net1 and net3 match")
	}

	// Step 5: Compare net2 (with GPU) vs net3 (without GPU)
	compareNetworks(net2, net3, input)

	// Test replay with "before" on middle layer (index 2)
	compareStaticReplayOutputs(net2, net3, input, 2, -1, "before", 2)

	// Test replay with "after" on middle layer (index 2)
	compareStaticReplayOutputs(net2, net3, input, 2, -1, "after", 2)

	// Generate random target (1, 10)
	target := make([][]float64, 1)
	target[0] = make([]float64, 10)
	for i := range target[0] {
		target[0][i] = rand.Float64()
	}

	// Test with replay "before" on layer 1
	compareStaticReplayOutputsWithTraining(net2, net3, input, target, 1, -1, "before", 1, 0.01, float32(1.0), float32(-1.0))

	// Test with replay "after" on layer 1
	compareStaticReplayOutputsWithTraining(net2, net3, input, target, 1, -1, "after", 1, 0.01, float32(1.0), float32(-1.0))
}

// compareStaticReplayOutputs tests static replay with "before" and "after" on CPU and GPU
func compareStaticReplayOutputs(netGPU, netCPU *paragon.Network[float32], input [][]float64, layerIdx int, replayOffset int, replayPhase string, maxReplay int) {

	// Configure replay settings for both networks
	for _, net := range []*paragon.Network[float32]{netGPU, netCPU} {
		if layerIdx >= len(net.Layers) {
			fmt.Println("Invalid layer index for replay configuration")
			return
		}
		net.Layers[layerIdx].ReplayOffset = replayOffset
		net.Layers[layerIdx].ReplayPhase = replayPhase
		net.Layers[layerIdx].MaxReplay = maxReplay
	}

	// Enable GPU on netGPU
	netGPU.WebGPUNative = true
	if err := netGPU.InitializeOptimizedGPU(); err != nil {
		fmt.Printf("⚠️ Failed to initialize WebGPU: %v\n", err)
		fmt.Println("   Continuing with CPU-only processing...")
		netGPU.WebGPUNative = false
	} else {
		fmt.Println("✅ WebGPU initialized successfully")
	}

	// Ensure netCPU remains on CPU
	netCPU.WebGPUNative = false

	// Perform forward pass on GPU network
	outputGPU := getOutput(netGPU, input)

	// Perform forward pass on CPU network
	outputCPU := getOutput(netCPU, input)

	// Compare outputs using approxEqual with global tolerance
	if approxEqual(outputGPU, outputCPU) {
		fmt.Println("GPU and CPU outputs match within tolerance with static replay")
	} else {
		fmt.Println("GPU and CPU outputs do not match with static replay")
		fmt.Println("GPU Output:", outputGPU)
		fmt.Println("CPU Output:", outputCPU)
	}
}

func compareStaticReplayOutputsWithTraining(netGPU, netCPU *paragon.Network[float32], input [][]float64, targets [][]float64, layerIdx int, replayOffset int, replayPhase string, maxReplay int, lr float64, clipUpper float32, clipLower float32) {
	// Configure replay settings for both networks
	for _, net := range []*paragon.Network[float32]{netGPU, netCPU} {
		if layerIdx >= len(net.Layers) {
			fmt.Println("Invalid layer index for replay configuration")
			return
		}
		net.Layers[layerIdx].ReplayOffset = replayOffset
		net.Layers[layerIdx].ReplayPhase = replayPhase
		net.Layers[layerIdx].MaxReplay = maxReplay
	}

	// Enable GPU on netGPU
	netGPU.WebGPUNative = true
	if err := netGPU.InitializeOptimizedGPU(); err != nil {
		fmt.Printf("⚠️ Failed to initialize WebGPU: %v\n", err)
		fmt.Println("   Continuing with CPU-only processing...")
		netGPU.WebGPUNative = false
	} else {
		fmt.Println("✅ WebGPU initialized successfully")
	}

	// Ensure netCPU remains on CPU
	netCPU.WebGPUNative = false

	// Perform training step on both networks
	for _, net := range []*paragon.Network[float32]{netGPU, netCPU} {
		net.Forward(input)
		net.Backward(targets, lr, clipUpper, clipLower)
	}

	// Perform forward pass after training to get updated outputs
	outputGPU := getOutput(netGPU, input)
	outputCPU := getOutput(netCPU, input)

	// Compare outputs using approxEqual with global tolerance
	if approxEqual(outputGPU, outputCPU) {
		fmt.Println("GPU and CPU outputs match within tolerance after training with static replay")
	} else {
		fmt.Println("GPU and CPU outputs do not match after training with static replay")
		fmt.Println("GPU Output:", outputGPU)
		fmt.Println("CPU Output:", outputCPU)
	}
}
