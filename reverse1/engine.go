package main

import (
	"fmt"
	"math"
	"paragon"
)

// TestConfig defines a network configuration for testing
type TestConfig struct {
	inputDim       int
	hiddenLayers   []int
	outputDim      int
	fullyConnected bool
	linearOnly     bool
	name           string
}

// Engine wraps a paragon network
type Engine struct {
	network *paragon.Network[float32]
}

// NewEngine creates a new engine with an invertible network
func NewEngine(inputDim, outputDim, numHiddenLayers int, hiddenWidth int, fullyConnected, linearOnly bool) *Engine {
	network := paragon.CreateInvertibleNetwork[float32](inputDim, hiddenWidth, outputDim, numHiddenLayers, linearOnly)
	network.Debug = true // Enable debug logging
	return &Engine{network: network}
}

// Run performs the forward pass
func (e *Engine) Run(input [][]float64) []float64 {
	e.network.Forward(input)
	return e.network.GetOutput()
}

// Reverse performs the reverse pass
func (e *Engine) Reverse(output [][]float64, config TestConfig) ([][]float64, error) {
	// Use ReverseExact for linear networks with equal layer sizes
	if config.linearOnly && config.inputDim == config.outputDim && (len(config.hiddenLayers) == 0 || config.hiddenLayers[0] == config.inputDim) {
		return e.network.ReverseExact(output)
	}
	// Use ReverseLayerByLayer for networks with uniform layer widths
	allEqual := true
	width := config.inputDim
	for _, h := range config.hiddenLayers {
		if h != width {
			allEqual = false
			break
		}
	}
	if allEqual && config.outputDim == width {
		return e.network.ReverseLayerByLayer(output)
	}
	// Enhanced Simulated Annealing for non-linear networks as fallback
	if !config.linearOnly {
		bestReconstruction := [][]float64{}
		bestError := math.Inf(1)
		for i := 0; i < 5; i++ { // Run 5 times and pick the best
			reconstructed := e.network.ReverseUsingSimulatedAnnealing(output, 2000, 10.0)
			e.network.Forward(reconstructed)
			currentOutput := e.network.GetOutput()
			currentError := 0.0
			for j := range currentOutput {
				diff := currentOutput[j] - output[0][j]
				currentError += diff * diff
			}
			currentError = math.Sqrt(currentError)
			if currentError < bestError {
				bestError = currentError
				bestReconstruction = reconstructed
			}
		}
		return bestReconstruction, nil
	}
	// Fallback to Adam for other cases
	reconstructed := e.network.ReverseUsingAdam(output, 2000)
	return reconstructed, nil
}

// calculateInputError computes Mean Squared Error between original and reconstructed inputs
func calculateInputError(original, reconstructed []float64) float64 {
	totalError := 0.0
	for i := range original {
		diff := original[i] - reconstructed[i]
		totalError += diff * diff
	}
	return totalError / float64(len(original))
}

// runTest evaluates a single network configuration with multiple inputs
func runTest(config TestConfig) (float64, error) {
	engine := NewEngine(
		config.inputDim,
		config.outputDim,
		len(config.hiddenLayers),
		config.hiddenLayers[0],
		config.fullyConnected,
		config.linearOnly,
	)

	// Multiple input samples for robustness
	inputs := [][][]float64{
		{{0.5, -0.3, 0.8, 0.2}},
		{{0.1, 0.2, 0.3, 0.4}},
		{{-0.5, 0.5, -0.2, 0.8}},
	}
	if config.inputDim != 4 {
		inputs = [][][]float64{
			{make([]float64, config.inputDim)},
			{make([]float64, config.inputDim)},
			{make([]float64, config.inputDim)},
		}
		for j := range inputs {
			for i := range inputs[j][0] {
				inputs[j][0][i] = 0.5 - float64(i)*0.1
			}
		}
	}

	// Average error across inputs
	totalError := 0.0
	count := 0
	for _, input := range inputs {
		output := engine.Run(input)
		reconstructed, err := engine.Reverse([][]float64{output}, config)
		if err != nil {
			return 0, fmt.Errorf("reverse failed: %v", err)
		}
		errVal := calculateInputError(input[0], reconstructed[0])
		if !math.IsNaN(errVal) {
			totalError += errVal
			count++
		}
	}

	if count == 0 {
		return 0, fmt.Errorf("no valid reconstructions")
	}
	return totalError / float64(count), nil
}

// summarizeResults prints a formatted summary of test results
func summarizeResults(results map[string]float64, errors map[string]error) {
	fmt.Println("\n=== Test Results Summary ===")
	fmt.Println("Name\t\t\tDepth\tHidden Widths\tActivation\tError\t\tStatus")
	fmt.Println("------------------------------------------------------------------------")
	for _, config := range testConfigs {
		err, hasErr := errors[config.name]
		errorVal := results[config.name]
		status := "✓ Approximate"
		if hasErr {
			status = fmt.Sprintf("✗ Failed: %v", err)
		} else if errorVal > 1e-3 {
			status = "⚠ Error Too High"
		}

		activation := "Linear"
		if !config.linearOnly {
			activation = "LeakyReLU"
		}
		fmt.Printf("%-20s\t%d\t%-12s\t%-10s\t%.6f\t%s\n",
			config.name,
			len(config.hiddenLayers)+2,
			fmt.Sprintf("%v", config.hiddenLayers),
			activation,
			errorVal,
			status,
		)
	}

	var validErrors []float64
	for _, err := range results {
		if !math.IsNaN(err) && err != 0 {
			validErrors = append(validErrors, err)
		}
	}
	if len(validErrors) > 0 {
		var mean, std float64
		for _, err := range validErrors {
			mean += err
		}
		mean /= float64(len(validErrors))
		for _, err := range validErrors {
			std += (err - mean) * (err - mean)
		}
		std = math.Sqrt(std / float64(len(validErrors)))
		fmt.Printf("\nAverage Error: %.6f\n", mean)
		fmt.Printf("Std Dev Error: %.6f\n", std)
	}
}

// Test configurations
var testConfigs = []TestConfig{
	{name: "Shallow-Small-4", inputDim: 4, hiddenLayers: []int{4}, outputDim: 4, fullyConnected: true, linearOnly: true},
	{name: "Shallow-Wide-8", inputDim: 4, hiddenLayers: []int{8}, outputDim: 4, fullyConnected: true, linearOnly: true},
	{name: "Shallow-Med-4x4", inputDim: 4, hiddenLayers: []int{4, 4}, outputDim: 4, fullyConnected: true, linearOnly: true},
	{name: "Shallow-NonLinear", inputDim: 4, hiddenLayers: []int{4}, outputDim: 4, fullyConnected: true, linearOnly: false},
	{name: "Deep-Small-4x3", inputDim: 4, hiddenLayers: []int{4, 4, 4}, outputDim: 4, fullyConnected: true, linearOnly: true},
	{name: "Deep-Wide-8x4", inputDim: 4, hiddenLayers: []int{8, 8, 8, 8}, outputDim: 4, fullyConnected: true, linearOnly: true},
	{name: "Deep-Large-16x5", inputDim: 4, hiddenLayers: []int{16, 16, 16, 16, 16}, outputDim: 4, fullyConnected: true, linearOnly: true},
	{name: "Deep-NonLinear", inputDim: 4, hiddenLayers: []int{4, 4, 4}, outputDim: 4, fullyConnected: true, linearOnly: false},
	{name: "Unequal-Sizes", inputDim: 4, hiddenLayers: []int{8, 8}, outputDim: 6, fullyConnected: true, linearOnly: true},
}

func main() {
	fmt.Println("=== Neural Network Forward and Reverse Pass Demo ===")

	results := make(map[string]float64)
	errors := make(map[string]error)

	for _, config := range testConfigs {
		fmt.Printf("\nTesting %s...\n", config.name)
		fmt.Printf("Input: %d, Hidden: %v, Output: %d, Linear: %v\n",
			config.inputDim, config.hiddenLayers, config.outputDim, config.linearOnly)

		errorVal, err := runTest(config)
		results[config.name] = errorVal
		if err != nil {
			errors[config.name] = err
			fmt.Printf("Error: %v\n", err)
		} else {
			fmt.Printf("Reconstruction Error: %.6f\n", errorVal)
			if errorVal < 1e-3 {
				fmt.Println("✓ Approximate reconstruction achieved!")
			} else {
				fmt.Println("⚠ Reconstruction error too high.")
			}
		}
	}

	summarizeResults(results, errors)
}
