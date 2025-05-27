package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"paragon"
)

const mnistDir = "./mnist_data"

func main() {
	rand.Seed(42)

	// --- Download and Load MNIST ---
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatalf("MNIST download error: %v", err)
	}
	testInputs, _, err := loadMNISTData(mnistDir, false)
	if err != nil {
		log.Fatalf("Failed to load test data: %v", err)
	}

	fmt.Printf("Loaded MNIST: %d test samples\n", len(testInputs))

	// --- Model Definition (exact as CompareCPUVsGPU) ---
	layers := []struct{ Width, Height int }{
		{28, 28},
		{32, 32},
		{32, 32},
		{32, 32},
		{32, 32},
		{32, 32},
		{10, 1},
	}
	acts := []string{"leaky_relu", "leaky_relu", "leaky_relu", "leaky_relu", "leaky_relu", "leaky_relu", "softmax"}
	fc := []bool{true, true, true, true, true, true, true}

	const iterations = 1000

	// ======= Build model =======
	nn := paragon.NewNetwork[float32](layers, acts, fc)

	// ======= CPU forward =======
	nn.WebGPUNative = false
	fmt.Println("\nRunning CPU forward pass benchmark...")
	cpuStart := time.Now()
	for i := 0; i < iterations; i++ {
		nn.Forward(testInputs[0])
	}
	cpuDuration := time.Since(cpuStart)
	cpuOut := nn.GetOutput()

	// ======= GPU forward =======
	nn.WebGPUNative = true
	if err := nn.InitializeOptimizedGPU(); err != nil {
		log.Fatalf("GPU init failed: %v", err)
	}
	defer nn.CleanupOptimizedGPU()

	// Warm-up
	nn.Forward(testInputs[0])

	fmt.Println("Running GPU forward pass benchmark...")
	gpuStart := time.Now()
	for i := 0; i < iterations; i++ {
		nn.Forward(testInputs[0])
	}
	gpuDuration := time.Since(gpuStart)
	gpuOut := nn.GetOutput()

	// ======= Output =======
	fmt.Printf("\n==== Timing Summary (%d iterations) ====\n", iterations)
	fmt.Printf("CPU total time: %v\n", cpuDuration)
	fmt.Printf("GPU total time: %v\n", gpuDuration)
	fmt.Printf("Average CPU Forward Pass: %v\n", cpuDuration/time.Duration(iterations))
	fmt.Printf("Average GPU Forward Pass: %v\n", gpuDuration/time.Duration(iterations))
	if gpuDuration > 0 {
		fmt.Printf("GPU is %.2fx faster than CPU (per forward)\n", float64(cpuDuration)/float64(gpuDuration))
	}

	// Compare outputs
	mismatch := false
	for i := range cpuOut {
		if math.Abs(cpuOut[i]-gpuOut[i]) > 1e-5 {
			fmt.Printf("Mismatch at index %d: CPU=%f, GPU=%f\n", i, cpuOut[i], gpuOut[i])
			mismatch = true
		}
	}
	if !mismatch {
		fmt.Println("CPU and GPU outputs match within tolerance (1e-5)")
	}
}
