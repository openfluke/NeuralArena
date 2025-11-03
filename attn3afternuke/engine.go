// test_forward_match.go
package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"

	"paragon"
)

func compareNeurons[T paragon.Numeric](cpu, gpu *paragon.Network[T]) {
	fmt.Println("\n=== Neuron-by-Neuron Comparison ===")

	totalDiff := 0.0
	maxDiff := 0.0
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
					if diffCount <= 5 { // Show first 5 differences per layer
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
	fmt.Printf("Total neurons compared: %d\n", totalNeurons)
	fmt.Printf("Neurons with differences: %d\n", diffCount)
	fmt.Printf("Max difference: %.15e\n", maxDiff)
	fmt.Printf("Average difference: %.15e\n", avgDiff)

	if maxDiff < 1e-6 {
		fmt.Println("✓ CPU and GPU outputs match exactly (within FP32 precision)")
	} else if maxDiff < 1e-3 {
		fmt.Println("⚠ CPU and GPU outputs differ slightly (acceptable for training)")
	} else {
		fmt.Println("✗ CPU and GPU outputs differ significantly!")
	}
}

func buildTestModel[T paragon.Numeric]() *paragon.Network[T] {
	sizes := []paragon.GridSpec{
		{Width: 8, Height: 8}, // input
		{Width: 4, Height: 4}, // hidden1
		{Width: 4, Height: 4}, // hidden2
		{Width: 4, Height: 1}, // output
	}
	acts := []string{"relu", "relu", "relu", "softmax"}
	fully := []bool{true, true, true, true}

	n, err := paragon.BuildGridNet[T](paragon.BuildOpts[T]{
		Sizes:       sizes,
		Activations: acts,
		FullyConn:   fully,
	})
	if err != nil {
		panic(err)
	}
	return n
}

func buildTestModelWithAttention[T paragon.Numeric]() *paragon.Network[T] {
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
		panic(err)
	}
	return n
}

func generateTestInput() [][]float64 {
	rng := rand.New(rand.NewSource(42))
	input := make([][]float64, 8)
	for i := range input {
		input[i] = make([]float64, 8)
		for j := range input[i] {
			input[i][j] = rng.Float64()
		}
	}
	return input
}

func saveAndLoad[T paragon.Numeric](net *paragon.Network[T], path string) *paragon.Network[T] {
	// Save
	js, err := paragon.ExportBundleJSON[T]("test", net, nil)
	if err != nil {
		panic(err)
	}
	if err := os.WriteFile(path, []byte(js), 0o644); err != nil {
		panic(err)
	}

	// Load
	data, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	anyNet, err := paragon.ImportBundleJSON(string(data))
	if err != nil {
		panic(err)
	}

	loaded, ok := anyNet.(*paragon.Network[T])
	if !ok {
		panic("type mismatch")
	}
	return loaded
}

func main() {
	fmt.Println("=== Forward Pass CPU/GPU Exact Match Test ===\n")

	// Suppress GPU warnings by setting environment variable
	os.Setenv("WGPU_LOG_LEVEL", "error")

	// Test 1: Dense model
	fmt.Println("────────────────────────────────────────")
	fmt.Println("TEST 1: Dense Model")
	fmt.Println("────────────────────────────────────────")

	cpuDense := buildTestModel[float32]()
	testInput := generateTestInput()

	// Run on CPU
	fmt.Println("\nRunning forward pass on CPU...")
	cpuDense.Forward(testInput)
	cpuOutput := cpuDense.GetOutput()
	fmt.Printf("CPU Output: ")
	for i := 0; i < len(cpuOutput) && i < 4; i++ {
		fmt.Printf("%.15e ", cpuOutput[i])
	}
	fmt.Println()

	// Save and reload to create identical copy
	fmt.Println("\nSaving and reloading model for GPU test...")
	gpuDense := saveAndLoad(cpuDense, "test_dense.json")

	// Enable GPU
	fmt.Println("Enabling GPU...")
	if err := gpuDense.EnableGPU(); err != nil {
		fmt.Printf("GPU not available: %v\n", err)
		fmt.Println("Skipping GPU tests")
		return
	}
	fmt.Println("GPU enabled ✓")

	// Run on GPU
	fmt.Println("\nRunning forward pass on GPU...")
	gpuDense.Forward(testInput)
	gpuOutput := gpuDense.GetOutput()
	fmt.Printf("GPU Output: ")
	for i := 0; i < len(gpuOutput) && i < 4; i++ {
		fmt.Printf("%.15e ", gpuOutput[i])
	}
	fmt.Println()

	// Compare outputs
	fmt.Println("\nOutput comparison:")
	maxOutputDiff := 0.0
	for i := range cpuOutput {
		diff := math.Abs(cpuOutput[i] - gpuOutput[i])
		if diff > maxOutputDiff {
			maxOutputDiff = diff
		}
		if i < 4 {
			fmt.Printf("  [%d] CPU=%.15e GPU=%.15e diff=%.15e\n",
				i, cpuOutput[i], gpuOutput[i], diff)
		}
	}
	fmt.Printf("Max output difference: %.15e\n", maxOutputDiff)

	// Compare all neurons
	compareNeurons(cpuDense, gpuDense)

	// Test 2: Model with attention
	fmt.Println("\n────────────────────────────────────────")
	fmt.Println("TEST 2: Model with Attention (Mixed)")
	fmt.Println("────────────────────────────────────────")

	cpuAttn := buildTestModelWithAttention[float32]()

	// Run on CPU
	fmt.Println("\nRunning forward pass on CPU...")
	cpuAttn.Forward(testInput)
	cpuOutputAttn := cpuAttn.GetOutput()
	fmt.Printf("CPU Output: ")
	for i := 0; i < len(cpuOutputAttn) && i < 4; i++ {
		fmt.Printf("%.15e ", cpuOutputAttn[i])
	}
	fmt.Println()

	// Save and reload
	fmt.Println("\nSaving and reloading model for GPU test...")
	gpuAttn := saveAndLoad(cpuAttn, "test_attn.json")

	// Enable GPU
	fmt.Println("Enabling GPU...")
	if err := gpuAttn.EnableGPU(); err != nil {
		fmt.Printf("GPU not available: %v\n", err)
		return
	}
	fmt.Println("GPU enabled ✓")

	// Run on GPU
	fmt.Println("\nRunning forward pass on GPU...")
	gpuAttn.Forward(testInput)
	gpuOutputAttn := gpuAttn.GetOutput()
	fmt.Printf("GPU Output: ")
	for i := 0; i < len(gpuOutputAttn) && i < 4; i++ {
		fmt.Printf("%.15e ", gpuOutputAttn[i])
	}
	fmt.Println()

	// Compare outputs
	fmt.Println("\nOutput comparison:")
	maxOutputDiffAttn := 0.0
	for i := range cpuOutputAttn {
		diff := math.Abs(cpuOutputAttn[i] - gpuOutputAttn[i])
		if diff > maxOutputDiffAttn {
			maxOutputDiffAttn = diff
		}
		if i < 4 {
			fmt.Printf("  [%d] CPU=%.15e GPU=%.15e diff=%.15e\n",
				i, cpuOutputAttn[i], gpuOutputAttn[i], diff)
		}
	}
	fmt.Printf("Max output difference: %.15e\n", maxOutputDiffAttn)

	// Compare all neurons
	compareNeurons(cpuAttn, gpuAttn)

	// Test 3: Multiple forward passes
	fmt.Println("\n────────────────────────────────────────")
	fmt.Println("TEST 3: Multiple Forward Passes")
	fmt.Println("────────────────────────────────────────")

	fmt.Println("\nTesting consistency over 10 forward passes...")
	maxOverallDiff := 0.0

	for i := 0; i < 10; i++ {
		// Generate different input
		rng := rand.New(rand.NewSource(int64(100 + i)))
		input := make([][]float64, 8)
		for r := range input {
			input[r] = make([]float64, 8)
			for c := range input[r] {
				input[r][c] = rng.Float64()
			}
		}

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

		fmt.Printf("  Pass %2d: max diff = %.15e\n", i+1, maxDiff)
	}

	fmt.Printf("\nMax difference across all passes: %.15e\n", maxOverallDiff)

	// Final verdict
	fmt.Println("\n────────────────────────────────────────")
	fmt.Println("FINAL VERDICT")
	fmt.Println("────────────────────────────────────────")

	if maxOverallDiff < 1e-6 {
		fmt.Println("✓ PASS: CPU and GPU produce identical results")
		fmt.Println("  (differences within floating point precision)")
	} else if maxOverallDiff < 1e-3 {
		fmt.Println("⚠ ACCEPTABLE: CPU and GPU differ slightly")
		fmt.Println("  (differences acceptable for neural network training)")
	} else {
		fmt.Println("✗ FAIL: CPU and GPU produce different results")
		fmt.Println("  (differences too large, check implementation)")
	}

	// Cleanup
	os.Remove("test_dense.json")
	os.Remove("test_attn.json")

	fmt.Println("\n✓ Test complete!")
}
