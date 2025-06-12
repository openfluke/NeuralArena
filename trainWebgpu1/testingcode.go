package main

import (
	"fmt"
	"math/rand"
	"os"
	"time"

	"paragon"

	"github.com/openfluke/pilot"
	"github.com/openfluke/pilot/experiments"
)

const (
	epochs       = 2
	learningRate = 0.05
	modelsDir    = "./models"
)

func main() {
	// Create models directory if it doesn't exist
	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		fmt.Printf("❌ Failed to create models directory: %v\n", err)
		return
	}

	// Load MNIST data
	mnist := experiments.NewMNISTDatasetStage("./data/mnist")
	exp := pilot.NewExperiment("MNIST", mnist)
	if err := exp.RunAll(); err != nil {
		fmt.Println("❌ Experiment failed:", err)
		os.Exit(1)
	}

	allInputs, allTargets, err := loadMNISTData("./data/mnist")
	if err != nil {
		fmt.Println("❌ Failed to load MNIST:", err)
		return
	}

	// Split into 80% training and 20% testing
	trainInputs, trainTargets, testInputs, testTargets := paragon.SplitDataset(allInputs, allTargets, 0.8)
	fmt.Printf("📊 Dataset sizes: Train=%d, Test=%d\n", len(trainInputs), len(testInputs))

	// Select 1000 training samples for faster testing
	sampleInputs := trainInputs[:1000]
	sampleTargets := trainTargets[:1000]

	// Set random seed for reproducibility
	rand.Seed(42)

	// Create CPU-only network
	nnCPU := paragon.NewNetwork[float32](
		[]struct{ Width, Height int }{{28, 28}, {32, 32}, {10, 1}},
		[]string{"linear", "relu", "softmax"},
		[]bool{true, true, true},
	)
	nnCPU.WebGPUNative = false // Explicitly disable GPU for CPU comparison
	//nnCPU.Debug = true

	// Reset seed to ensure same initial weights for GPU network
	rand.Seed(42)

	// Create GPU network - NewNetwork will automatically try to initialize GPU
	nnGPU := paragon.NewNetwork[float32](
		[]struct{ Width, Height int }{{28, 28}, {32, 32}, {10, 1}},
		[]string{"linear", "relu", "softmax"},
		[]bool{true, true, true},
	)
	//nnGPU.Debug = true

	// Check if GPU was successfully initialized
	gpuAvailable := false
	if nnGPU.IsGPUAvailable() {
		if err := nnGPU.VerifyGPUSetup(); err != nil {
			fmt.Printf("⚠️ GPU setup verification failed: %v\n", err)
			fmt.Println("🧠 Training on CPU...")
			nnGPU.WebGPUNative = false
		} else {
			fmt.Printf("✅ GPU setup verified successfully\n")
			gpuAvailable = true
			defer nnGPU.CleanupOptimizedGPU()
		}
	} else {
		fmt.Printf("⚠️ GPU not available, status: %s\n", nnGPU.GetGPUStatus())
		fmt.Println("🧠 Training on CPU...")
	}

	// Print GPU training stats
	if gpuAvailable {
		stats := nnGPU.GetGPUTrainingStats()
		fmt.Printf("🚀 GPU Training Stats: %+v\n", stats)
	}

	// Train CPU network
	fmt.Println("🧠 Training CPU network...")
	startCPU := time.Now()
	nnCPU.Train(
		sampleInputs,
		sampleTargets,
		epochs,
		learningRate,
		float32(2.0),  // clipUpper
		float32(-2.0), // clipLower
	)
	cpuTime := time.Since(startCPU)
	fmt.Printf("CPU training time: %v\n", cpuTime)

	// Train GPU network if available
	var gpuTime time.Duration
	if gpuAvailable {
		fmt.Println("🚀 Training GPU network...")
		startGPU := time.Now()
		nnGPU.Train(
			sampleInputs,
			sampleTargets,
			epochs,
			learningRate,
			float32(2.0),  // clipUpper
			float32(-2.0), // clipLower
		)
		gpuTime = time.Since(startGPU)
		fmt.Printf("GPU training time: %v\n", gpuTime)

		// Calculate speedup
		if cpuTime > 0 && gpuTime > 0 {
			speedup := float64(cpuTime) / float64(gpuTime)
			fmt.Printf("🏃 GPU speedup: %.2fx\n", speedup)
		}
	}

	// Evaluate both networks on test set
	fmt.Println("📊 Evaluating models...")
	cpuAccuracy := evaluateNetwork(nnCPU, testInputs, testTargets)
	fmt.Printf("CPU model test accuracy: %.4f\n", cpuAccuracy*100)

	if gpuAvailable {
		gpuAccuracy := evaluateNetwork(nnGPU, testInputs, testTargets)
		fmt.Printf("GPU model test accuracy: %.4f\n", gpuAccuracy*100)

		// Compare accuracy
		accuracyDiff := gpuAccuracy - cpuAccuracy
		fmt.Printf("📈 Accuracy difference (GPU - CPU): %.4f\n", accuracyDiff*100)
	}

	// Test individual GPU operations if available
	if gpuAvailable {
		fmt.Println("🔍 Testing GPU operations...")
		testSample := sampleInputs[0]
		nnGPU.TestGPUDirect(testSample)
	}

	// Save models if SaveJSON method exists
	/*
		modelPathCPU := filepath.Join(modelsDir, "mnist_model_cpu.json")
		if err := nnCPU.SaveJSON(modelPathCPU); err != nil {
			fmt.Printf("❌ Failed to save CPU model: %v\n", err)
		} else {
			fmt.Printf("💾 Saved CPU model to %s\n", modelPathCPU)
		}

		if gpuAvailable {
			modelPathGPU := filepath.Join(modelsDir, "mnist_model_gpu.json")
			if err := nnGPU.SaveJSON(modelPathGPU); err != nil {
				fmt.Printf("❌ Failed to save GPU model: %v\n", err)
			} else {
				fmt.Printf("💾 Saved GPU model to %s\n", modelPathGPU)
			}
		}
	*/
}

// Updated evaluation function with proper accuracy calculation
func evaluateNetwork[T paragon.Numeric](nn *paragon.Network[T], inputs, targets [][][]float64) float64 {
	correct := 0
	total := len(inputs)

	for i := range inputs {
		// Forward pass
		nn.Forward(inputs[i])

		// Get network output
		output := nn.GetOutput()

		// Find predicted class (argmax of output)
		predictedClass := argMax(output)

		// Find true class (argmax of target)
		trueClass := argMax(targets[i][0])

		if predictedClass == trueClass {
			correct++
		}
	}

	accuracy := float64(correct) / float64(total)

	// Update network performance stats if available
	if nn.Performance != nil {
		nn.Performance.Score = accuracy
	}

	return accuracy
}

// Helper function to find argmax
func argMax(slice []float64) int {
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

// Optional: Add performance monitoring
func monitorTraining[T paragon.Numeric](nn *paragon.Network[T], sampleInput [][]float64) {
	if nn.IsGPUAvailable() {
		// Test a single forward pass
		start := time.Now()
		nn.Forward(sampleInput)
		duration := time.Since(start)
		fmt.Printf("⏱️  Single forward pass time: %v\n", duration)

		// Check GPU memory usage or other metrics if available
		stats := nn.GetGPUTrainingStats()
		fmt.Printf("📊 Current GPU stats: %+v\n", stats)
	}
}
