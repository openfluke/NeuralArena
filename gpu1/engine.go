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
	testInputs, testTargets, err := loadMNISTData(mnistDir, false)
	if err != nil {
		log.Fatalf("Failed to load test data: %v", err)
	}

	fmt.Printf("Loaded MNIST: %d test samples\n", len(testInputs))

	// --- Model Definition (exact as CompareCPUVsGPU) ---
	/*layers := []struct{ Width, Height int }{
		{28, 28},
		{32, 32},
		{32, 32},
		{32, 32},
		{32, 32},
		{32, 32},
		{10, 1},
	}
	acts := []string{"leaky_relu", "leaky_relu", "leaky_relu", "leaky_relu", "leaky_relu", "leaky_relu", "softmax"}
	fc := []bool{true, true, true, true, true, true, true}*/

	layers := []struct{ Width, Height int }{
		{28, 28}, {16, 16}, {10, 1},
	}
	acts := []string{"leaky_relu", "leaky_relu", "softmax"}
	fc := []bool{true, false, true}

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

	// ======= GPU Training Benchmark =======
	fmt.Println("\n==== GPU Training Benchmark ====")

	// Initialize complete GPU (forward + backward)
	if err := nn.InitializeGPUComplete(); err != nil {
		log.Fatalf("Failed to initialize complete GPU: %v", err)
	}

	// Prepare training data (first 100 samples)
	const trainingSamples = 100
	trainingInputs := make([][][]float64, trainingSamples)
	trainingTargets := make([][][]float64, trainingSamples)

	for i := 0; i < trainingSamples; i++ {
		trainingInputs[i] = testInputs[i]
		trainingTargets[i] = testTargets[i]
	}

	fmt.Printf("Training on %d samples...\n", trainingSamples)

	// Calculate initial loss
	totalInitialLoss := 0.0
	for i := 0; i < trainingSamples; i++ {
		nn.Forward(trainingInputs[i])
		loss := nn.ComputeLoss(trainingTargets[i])
		totalInitialLoss += loss
	}
	avgInitialLoss := totalInitialLoss / float64(trainingSamples)
	fmt.Printf("Initial average loss: %.6f\n", avgInitialLoss)

	// Training parameters
	const (
		epochs       = 5
		learningRate = 0.05
		clipUpper    = float32(1.0)
		clipLower    = float32(-1.0)
	)

	// Time the training
	trainingStart := time.Now()

	for epoch := 0; epoch < epochs; epoch++ {
		epochStart := time.Now()
		totalLoss := 0.0

		// Shuffle training data
		perm := rand.Perm(trainingSamples)

		for _, i := range perm {
			// Forward pass
			nn.Forward(trainingInputs[i])

			// Compute loss
			loss := nn.ComputeLoss(trainingTargets[i])
			if math.IsNaN(loss) {
				fmt.Printf("Warning: NaN loss at sample %d, epoch %d\n", i, epoch)
				continue
			}
			totalLoss += loss

			// Backward pass (GPU)
			err := nn.BackwardGPU(trainingTargets[i], learningRate, clipUpper, clipLower)
			if err != nil {
				log.Printf("GPU backward pass failed at sample %d, epoch %d: %v", i, epoch, err)
				// Fall back to CPU for this sample
				nn.Backward(trainingTargets[i], learningRate, clipUpper, clipLower)
			}

			nn.SyncAllGPUWeights()
		}

		epochDuration := time.Since(epochStart)
		avgLoss := totalLoss / float64(trainingSamples)
		fmt.Printf("Epoch %d/%d: Loss=%.6f, Time=%v\n", epoch+1, epochs, avgLoss, epochDuration)
	}

	trainingDuration := time.Since(trainingStart)

	// Calculate final loss
	totalFinalLoss := 0.0
	correct := 0

	for i := 0; i < trainingSamples; i++ {
		nn.Forward(trainingInputs[i])
		loss := nn.ComputeLoss(trainingTargets[i])
		totalFinalLoss += loss

		// Check accuracy
		output := nn.GetOutput()
		predicted := argmax(output)
		actual := argmax(flatten2D(trainingTargets[i]))
		if predicted == actual {
			correct++
		}
	}

	avgFinalLoss := totalFinalLoss / float64(trainingSamples)
	accuracy := float64(correct) / float64(trainingSamples) * 100.0

	fmt.Printf("\n==== Training Results ====\n")
	fmt.Printf("Training time: %v\n", trainingDuration)
	fmt.Printf("Time per epoch: %v\n", trainingDuration/epochs)
	fmt.Printf("Time per sample: %v\n", trainingDuration/time.Duration(trainingSamples*epochs))
	fmt.Printf("Initial loss: %.6f\n", avgInitialLoss)
	fmt.Printf("Final loss: %.6f\n", avgFinalLoss)
	fmt.Printf("Loss improvement: %.6f (%.2f%%)\n", avgInitialLoss-avgFinalLoss, (avgInitialLoss-avgFinalLoss)/avgInitialLoss*100)
	fmt.Printf("Final accuracy: %.2f%% (%d/%d)\n", accuracy, correct, trainingSamples)

	// Test a few individual predictions
	fmt.Printf("\n==== Sample Predictions ====\n")
	for i := 0; i < min(5, trainingSamples); i++ {
		nn.Forward(trainingInputs[i])
		output := nn.GetOutput()
		predicted := argmax(output)
		actual := argmax(flatten2D(trainingTargets[i]))
		confidence := output[predicted]

		fmt.Printf("Sample %d: Predicted=%d, Actual=%d, Confidence=%.4f %s\n",
			i, predicted, actual, confidence,
			func() string {
				if predicted == actual {
					return "✓"
				} else {
					return "✗"
				}
			}())
	}
}

// Helper functions
func argmax(slice []float64) int {
	maxIdx := 0
	maxVal := slice[0]
	for i, val := range slice {
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}
	return maxIdx
}

func flatten2D(data [][]float64) []float64 {
	result := make([]float64, 0)
	for _, row := range data {
		result = append(result, row...)
	}
	return result
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
