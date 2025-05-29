package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"paragon"
	"time"
)

const iterations = 1000
const tolerance = 1e-5

// Simplified layer structure: same for all tests
var layers = []struct{ Width, Height int }{
	{28, 28}, // Input layer (MNIST)
	{32, 32}, // First hidden layer
	{32, 32}, // Second hidden layer
	{10, 1},  // Output layer
}

// Activation functions to test
var actsToTest = []string{
	"linear",     // Linear (identity)
	"relu",       // ReLU
	"leaky_relu", // Leaky ReLU
	"elu",        // ELU
	"swish",      // Swish/SiLU
	"gelu",       // GELU
	"tanh",       // Hyperbolic tangent
	"softmax",    // Softmax
}

var fc = []bool{true, true, true, true}

const mnistDir = "./mnist_data"

type ActivationTestResult struct {
	Type       string
	Activation string
	CPUTime    time.Duration
	GPUTime    time.Duration
	Speedup    float64
	Mismatch   bool
}

func main() {
	rand.Seed(42)

	// Load MNIST data
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatalf("MNIST download error: %v", err)
	}

	testInputs, _, err := loadMNISTData(mnistDir, false)
	if err != nil {
		log.Fatalf("Failed to load test data: %v", err)
	}

	fmt.Printf("Loaded MNIST: %d test samples\n", len(testInputs))

	// Collect results for all types and activations
	results := []ActivationTestResult{}
	types := []string{"float32", "int32", "uint32"}

	for _, typeName := range types {
		fmt.Printf("\n=== Testing %s ===\n", typeName)

		var actResults []ActivationTestResult
		switch typeName {
		case "float32":
			actResults = benchmarkTypeWithActivations[float32](typeName, testInputs[0])
		case "int32":
			actResults = benchmarkTypeWithActivations[int32](typeName, testInputs[0])
		case "uint32":
			actResults = benchmarkTypeWithActivations[uint32](typeName, testInputs[0])
		}
		results = append(results, actResults...)
	}

	// Print comprehensive summary
	fmt.Printf("\n==== Activation Function Performance (%d iterations) ====\n", iterations)
	fmt.Printf("%-10s | %-12s | %-12s | %-12s | %-10s | Match?\n", "Type", "Activation", "CPU Time", "GPU Time", "Speedup")
	fmt.Println("-----------------------------------------------------------------------")

	for _, r := range results {
		match := "✅"
		if r.Mismatch {
			match = "❌"
		}
		fmt.Printf("%-10s | %-12s | %-12v | %-12v | %-9.2fx | %s\n",
			r.Type, r.Activation, r.CPUTime, r.GPUTime, r.Speedup, match)
	}

	// Print activation compatibility summary
	fmt.Printf("\n==== Activation Function Compatibility ====\n")
	fmt.Printf("%-12s | %-8s | %-8s | %-8s\n", "Activation", "float32", "int32", "uint32")
	fmt.Println("-------------------------------------------")

	for _, activation := range actsToTest {
		fmt.Printf("%-12s | ", activation)
		for _, typeName := range types {
			// Find the result for this type and activation
			found := false
			for _, r := range results {
				if r.Type == typeName && r.Activation == activation {
					status := "✅"
					if r.Mismatch {
						status = "❌"
					}
					fmt.Printf("%-8s | ", status)
					found = true
					break
				}
			}
			if !found {
				fmt.Printf("%-8s | ", "N/A")
			}
		}
		fmt.Println()
	}
}

func benchmarkTypeWithActivations[T paragon.Numeric](typeName string, input [][]float64) []ActivationTestResult {
	activationResults := []ActivationTestResult{}

	fmt.Printf("Testing activation functions for %s...\n", typeName)

	for _, activation := range actsToTest {
		// Define activation functions: use the test activation for hidden layers
		acts := []string{"linear", activation, activation, "softmax"}

		nn := paragon.NewNetwork[T](layers, acts, fc)

		// CPU test
		nn.WebGPUNative = false
		cpuStart := time.Now()
		for j := 0; j < iterations; j++ {
			nn.Forward(input)
		}
		cpuDuration := time.Since(cpuStart)
		cpuOut := nn.GetOutput()

		// GPU test
		nn.WebGPUNative = true
		err := nn.InitializeOptimizedGPU()
		if err != nil {
			fmt.Printf("[%s/%s] GPU unsupported: %v\n", typeName, activation, err)
			activationResults = append(activationResults, ActivationTestResult{
				Type:       typeName,
				Activation: activation,
				CPUTime:    cpuDuration,
				GPUTime:    0,
				Speedup:    0,
				Mismatch:   true,
			})
			continue
		}

		// Warm-up
		nn.Forward(input)
		gpuStart := time.Now()
		for j := 0; j < iterations; j++ {
			nn.Forward(input)
		}
		gpuDuration := time.Since(gpuStart)
		gpuOut := nn.GetOutput()

		nn.CleanupOptimizedGPU()

		// Compare outputs
		mismatch := false
		for k := range cpuOut {
			if math.Abs(cpuOut[k]-gpuOut[k]) > tolerance {
				fmt.Printf("[%s/%s] Mismatch at index %d: CPU=%.6f GPU=%.6f\n",
					typeName, activation, k, cpuOut[k], gpuOut[k])
				mismatch = true
				break
			}
		}

		speedup := 0.0
		if gpuDuration > 0 {
			speedup = float64(cpuDuration) / float64(gpuDuration)
		}

		activationResults = append(activationResults, ActivationTestResult{
			Type:       typeName,
			Activation: activation,
			CPUTime:    cpuDuration,
			GPUTime:    gpuDuration,
			Speedup:    speedup,
			Mismatch:   mismatch,
		})

		fmt.Printf("  %s: CPU=%v GPU=%v Speedup=%.2fx Match=%t\n",
			activation, cpuDuration, gpuDuration, speedup, !mismatch)
	}

	return activationResults
}
