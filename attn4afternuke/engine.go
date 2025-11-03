// test_forward_exact_match.go
package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"

	"paragon"
)

// saveModel saves a network to JSON
func saveModel[T paragon.Numeric](net *paragon.Network[T], path string) error {
	js, err := paragon.ExportBundleJSON[T]("test", net, nil)
	if err != nil {
		return fmt.Errorf("export failed: %w", err)
	}
	if err := os.WriteFile(path, []byte(js), 0o644); err != nil {
		return fmt.Errorf("write failed: %w", err)
	}
	fmt.Printf("✓ Saved model to %s\n", path)
	return nil
}

// loadModel loads a network from JSON
func loadModel[T paragon.Numeric](path string) (*paragon.Network[T], error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read failed: %w", err)
	}
	anyNet, err := paragon.ImportBundleJSON(string(data))
	if err != nil {
		return nil, fmt.Errorf("import failed: %w", err)
	}
	loaded, ok := anyNet.(*paragon.Network[T])
	if !ok {
		return nil, fmt.Errorf("type mismatch")
	}
	fmt.Printf("✓ Loaded model from %s\n", path)
	return loaded, nil
}

// createBaseModel creates and saves a base model
func createBaseModel[T paragon.Numeric](sizes []paragon.GridSpec, acts []string, fully []bool, seed int64, path string) (*paragon.Network[T], error) {
	// Set seed for reproducibility
	rand.Seed(seed)

	fmt.Println("\n=== Creating Base Model ===")
	fmt.Printf("Seed: %d\n", seed)
	fmt.Printf("Architecture: %v\n", sizes)
	fmt.Printf("Activations: %v\n", acts)

	n, err := paragon.BuildGridNet[T](paragon.BuildOpts[T]{
		Sizes:       sizes,
		Activations: acts,
		FullyConn:   fully,
	})
	if err != nil {
		return nil, err
	}

	// Save the model
	if err := saveModel(n, path); err != nil {
		return nil, err
	}

	return n, nil
}

// createBaseModelWithAttention creates and saves a model with attention
func createBaseModelWithAttention[T paragon.Numeric](seed int64, path string) (*paragon.Network[T], error) {
	rand.Seed(seed)

	fmt.Println("\n=== Creating Base Model with Attention ===")
	fmt.Printf("Seed: %d\n", seed)

	sizes := []paragon.GridSpec{
		{Width: 8, Height: 8}, // input
		{Width: 4, Height: 4}, // hidden1
		{Width: 4, Height: 4}, // hidden2 - mixed dense/attn
		{Width: 4, Height: 1}, // output
	}
	acts := []string{"relu", "relu", "relu", "softmax"}
	fully := []bool{true, true, true, true}

	slices := make([][]string, len(sizes))
	slices[2] = []string{"dense", "attn", "dense", "attn"}

	attnCfg := make([]*paragon.AttnConfig[T], len(sizes))
	attnCfg[2] = &paragon.AttnConfig[T]{
		Heads:     2,
		DK:        32,
		UseWo:     true,
		Share:     "layer",
		Dropout:   0.0,
		PosEnc2D:  true,
		UseNorm:   true,
		PosEncAmp: 0.02,
		NormEps:   1e-6,
	}

	n, err := paragon.BuildGridNet[T](paragon.BuildOpts[T]{
		Sizes:       sizes,
		Activations: acts,
		FullyConn:   fully,
		SliceTypes:  slices,
		Attn:        attnCfg,
	})
	if err != nil {
		return nil, err
	}

	// Save the model
	if err := saveModel(n, path); err != nil {
		return nil, err
	}

	return n, nil
}

// generateTestInput creates deterministic test input
func generateTestInput(seed int64) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	input := make([][]float64, 8)
	for i := range input {
		input[i] = make([]float64, 8)
		for j := range input[i] {
			input[i][j] = rng.Float64()
		}
	}
	return input
}

// compareOutputs compares CPU and GPU outputs
func compareOutputs(cpuOutput, gpuOutput []float64, testName string) (maxDiff float64, passed bool) {
	fmt.Printf("\n=== Output Comparison: %s ===\n", testName)

	if len(cpuOutput) != len(gpuOutput) {
		fmt.Printf("✗ Length mismatch: CPU=%d, GPU=%d\n", len(cpuOutput), len(gpuOutput))
		return math.MaxFloat64, false
	}

	maxDiff = 0.0
	for i := range cpuOutput {
		diff := math.Abs(cpuOutput[i] - gpuOutput[i])
		if diff > maxDiff {
			maxDiff = diff
		}
		if i < 4 || diff > 1e-6 {
			fmt.Printf("  [%d] CPU=%.15e GPU=%.15e diff=%.15e\n",
				i, cpuOutput[i], gpuOutput[i], diff)
		}
	}

	fmt.Printf("Max difference: %.15e\n", maxDiff)

	if maxDiff < 1e-6 {
		fmt.Println("✓ PASS: Outputs match within FP32 precision")
		passed = true
	} else if maxDiff < 1e-3 {
		fmt.Println("⚠ ACCEPTABLE: Small differences (acceptable for training)")
		passed = true
	} else {
		fmt.Println("✗ FAIL: Outputs differ significantly")
		passed = false
	}

	return maxDiff, passed
}

// compareNeurons performs detailed neuron-by-neuron comparison
func compareNeurons[T paragon.Numeric](cpu, gpu *paragon.Network[T]) (maxDiff float64) {
	fmt.Println("\n=== Neuron-by-Neuron Comparison ===")

	totalDiff := 0.0
	maxDiff = 0.0
	diffCount := 0
	totalNeurons := 0

	for l := 0; l < len(cpu.Layers); l++ {
		cpuLayer := &cpu.Layers[l]
		gpuLayer := &gpu.Layers[l]

		fmt.Printf("\nLayer %d (%dx%d):\n", l, cpuLayer.Width, cpuLayer.Height)

		layerMaxDiff := 0.0
		layerDiffs := 0

		for y := 0; y < cpuLayer.Height; y++ {
			for x := 0; x < cpuLayer.Width; x++ {
				cpuVal := float64(any(cpuLayer.Neurons[y][x].Value).(T))
				gpuVal := float64(any(gpuLayer.Neurons[y][x].Value).(T))

				diff := math.Abs(cpuVal - gpuVal)
				totalDiff += diff
				totalNeurons++

				if diff > 1e-10 {
					diffCount++
					layerDiffs++
					if layerDiffs <= 5 {
						fmt.Printf("  Neuron[%d][%d]: CPU=%.15e, GPU=%.15e, diff=%.15e\n",
							y, x, cpuVal, gpuVal, diff)
					}
				}

				if diff > layerMaxDiff {
					layerMaxDiff = diff
				}
				if diff > maxDiff {
					maxDiff = diff
				}
			}
		}

		fmt.Printf("  Layer max diff: %.15e\n", layerMaxDiff)
		if layerDiffs > 5 {
			fmt.Printf("  (...%d more differences)\n", layerDiffs-5)
		}
	}

	avgDiff := totalDiff / float64(totalNeurons)

	fmt.Printf("\n=== Summary ===\n")
	fmt.Printf("Total neurons: %d\n", totalNeurons)
	fmt.Printf("Neurons with differences: %d (%.2f%%)\n", diffCount, 100.0*float64(diffCount)/float64(totalNeurons))
	fmt.Printf("Max difference: %.15e\n", maxDiff)
	fmt.Printf("Average difference: %.15e\n", avgDiff)

	return maxDiff
}

func main() {
	fmt.Println("=== Forward Pass CPU/GPU Exact Match Test ===")
	fmt.Println("=== Using Fixed Seed and Saved Models ===\n")

	// Suppress GPU warnings
	os.Setenv("WGPU_LOG_LEVEL", "error")

	const seed = int64(12345)

	// Test 1: Simple Dense Model
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("TEST 1: Simple Dense Model")
	fmt.Println(strings.Repeat("=", 60))

	denseModelPath := "base_dense_model.json"

	// Create or load base model
	if _, err := os.Stat(denseModelPath); os.IsNotExist(err) {
		sizes := []paragon.GridSpec{
			{Width: 8, Height: 8}, // input
			{Width: 4, Height: 4}, // hidden1
			{Width: 4, Height: 4}, // hidden2
			{Width: 4, Height: 1}, // output
		}
		acts := []string{"relu", "relu", "relu", "softmax"}
		fully := []bool{true, true, true, true}

		_, err = createBaseModel[float32](sizes, acts, fully, seed, denseModelPath)
		if err != nil {
			panic(err)
		}
	} else {
		fmt.Println("\n=== Loading Existing Base Model ===")
		fmt.Printf("✓ Loaded model from %s\n", denseModelPath)
	}

	// Generate deterministic test input
	testInput := generateTestInput(seed + 1)

	// Load two independent copies for CPU and GPU
	cpuDense, err := loadModel[float32](denseModelPath)
	if err != nil {
		panic(err)
	}

	gpuDense, err := loadModel[float32](denseModelPath)
	if err != nil {
		panic(err)
	}

	// Run on CPU
	fmt.Println("\n--- Running CPU Forward Pass ---")
	cpuDense.Forward(testInput)
	cpuOutput := cpuDense.GetOutput()
	fmt.Printf("CPU Output (first 4): ")
	for i := 0; i < len(cpuOutput) && i < 4; i++ {
		fmt.Printf("%.15e ", cpuOutput[i])
	}
	fmt.Println()

	// Enable GPU
	fmt.Println("\n--- Enabling GPU ---")
	if err := gpuDense.EnableGPU(); err != nil {
		fmt.Printf("GPU not available: %v\n", err)
		fmt.Println("Skipping GPU tests")
		return
	}
	fmt.Println("✓ GPU enabled")

	// Run on GPU
	fmt.Println("\n--- Running GPU Forward Pass ---")
	gpuDense.Forward(testInput)
	gpuOutput := gpuDense.GetOutput()
	fmt.Printf("GPU Output (first 4): ")
	for i := 0; i < len(gpuOutput) && i < 4; i++ {
		fmt.Printf("%.15e ", gpuOutput[i])
	}
	fmt.Println()

	// Compare
	maxDiffDense, passedDense := compareOutputs(cpuOutput, gpuOutput, "Dense Model")
	maxDiffNeuronDense := compareNeurons(cpuDense, gpuDense)

	// Test 2: Model with Attention
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("TEST 2: Model with Attention")
	fmt.Println(strings.Repeat("=", 60))

	attnModelPath := "base_attn_model.json"

	// Create or load base model with attention
	if _, err := os.Stat(attnModelPath); os.IsNotExist(err) {
		_, err = createBaseModelWithAttention[float32](seed, attnModelPath)
		if err != nil {
			panic(err)
		}
	} else {
		fmt.Println("\n=== Loading Existing Base Model with Attention ===")
		fmt.Printf("✓ Loaded model from %s\n", attnModelPath)
	}

	// Load two independent copies
	cpuAttn, err := loadModel[float32](attnModelPath)
	if err != nil {
		panic(err)
	}

	gpuAttn, err := loadModel[float32](attnModelPath)
	if err != nil {
		panic(err)
	}

	// Run on CPU
	fmt.Println("\n--- Running CPU Forward Pass ---")
	cpuAttn.Forward(testInput)
	cpuOutputAttn := cpuAttn.GetOutput()
	fmt.Printf("CPU Output (first 4): ")
	for i := 0; i < len(cpuOutputAttn) && i < 4; i++ {
		fmt.Printf("%.15e ", cpuOutputAttn[i])
	}
	fmt.Println()

	// Enable GPU
	fmt.Println("\n--- Enabling GPU ---")
	if err := gpuAttn.EnableGPU(); err != nil {
		fmt.Printf("GPU not available: %v\n", err)
		return
	}
	fmt.Println("✓ GPU enabled")

	// Run on GPU
	fmt.Println("\n--- Running GPU Forward Pass ---")
	gpuAttn.Forward(testInput)
	gpuOutputAttn := gpuAttn.GetOutput()
	fmt.Printf("GPU Output (first 4): ")
	for i := 0; i < len(gpuOutputAttn) && i < 4; i++ {
		fmt.Printf("%.15e ", gpuOutputAttn[i])
	}
	fmt.Println()

	// Compare
	maxDiffAttn, passedAttn := compareOutputs(cpuOutputAttn, gpuOutputAttn, "Attention Model")
	maxDiffNeuronAttn := compareNeurons(cpuAttn, gpuAttn)

	// Test 3: Multiple Forward Passes
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("TEST 3: Multiple Forward Passes (Consistency)")
	fmt.Println(strings.Repeat("=", 60))

	fmt.Println("\nTesting consistency over 10 forward passes...")
	maxOverallDiff := 0.0

	for i := 0; i < 10; i++ {
		// Generate different input with different seed
		input := generateTestInput(seed + int64(100+i))

		// Run on both
		cpuDense.Forward(input)
		gpuDense.Forward(input)

		cpuOut := cpuDense.GetOutput()
		gpuOut := gpuDense.GetOutput()

		// Compare
		maxDiff := 0.0
		for j := range cpuOut {
			diff := math.Abs(cpuOut[j] - gpuOut[j])
			if diff > maxDiff {
				maxDiff = diff
			}
		}

		if maxDiff > maxOverallDiff {
			maxOverallDiff = maxDiff
		}

		status := "✓"
		if maxDiff > 1e-6 {
			status = "⚠"
		}
		if maxDiff > 1e-3 {
			status = "✗"
		}

		fmt.Printf("  Pass %2d: max diff = %.15e %s\n", i+1, maxDiff, status)
	}

	fmt.Printf("\nMax difference across all passes: %.15e\n", maxOverallDiff)

	// Final Verdict
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("FINAL VERDICT")
	fmt.Println(strings.Repeat("=", 60))

	allPassed := true

	fmt.Printf("\n1. Dense Model:\n")
	fmt.Printf("   Output max diff:  %.15e\n", maxDiffDense)
	fmt.Printf("   Neuron max diff:  %.15e\n", maxDiffNeuronDense)
	if !passedDense {
		allPassed = false
		fmt.Println("   Status: ✗ FAILED")
	} else if maxDiffDense < 1e-6 {
		fmt.Println("   Status: ✓ PASSED (exact match)")
	} else {
		fmt.Println("   Status: ⚠ PASSED (acceptable)")
	}

	fmt.Printf("\n2. Attention Model:\n")
	fmt.Printf("   Output max diff:  %.15e\n", maxDiffAttn)
	fmt.Printf("   Neuron max diff:  %.15e\n", maxDiffNeuronAttn)
	if !passedAttn {
		allPassed = false
		fmt.Println("   Status: ✗ FAILED")
	} else if maxDiffAttn < 1e-6 {
		fmt.Println("   Status: ✓ PASSED (exact match)")
	} else {
		fmt.Println("   Status: ⚠ PASSED (acceptable)")
	}

	fmt.Printf("\n3. Multiple Passes:\n")
	fmt.Printf("   Max diff overall: %.15e\n", maxOverallDiff)
	if maxOverallDiff > 1e-3 {
		allPassed = false
		fmt.Println("   Status: ✗ FAILED")
	} else if maxOverallDiff < 1e-6 {
		fmt.Println("   Status: ✓ PASSED (exact match)")
	} else {
		fmt.Println("   Status: ⚠ PASSED (acceptable)")
	}

	fmt.Println()
	if allPassed {
		if maxOverallDiff < 1e-6 && maxDiffDense < 1e-6 && maxDiffAttn < 1e-6 {
			fmt.Println("✓✓✓ ALL TESTS PASSED - CPU AND GPU PRODUCE IDENTICAL RESULTS ✓✓✓")
		} else {
			fmt.Println("✓ ALL TESTS PASSED - DIFFERENCES WITHIN ACCEPTABLE RANGE")
		}
	} else {
		fmt.Println("✗ SOME TESTS FAILED - INVESTIGATION NEEDED")
	}

	fmt.Println("\n✓ Test complete!")
}
