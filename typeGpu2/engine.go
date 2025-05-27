package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"sync"
	"time"

	"paragon"
)

const (
	mnistDir  = "mnist_data"
	numEpochs = 5
	learnRate = 0.01
)

type BenchmarkResult struct {
	ModelName string
	TypeName  string
	Duration  time.Duration // total train+eval time
	Score     float64
}

type LoadedBenchmarkResult struct {
	ModelName string
	TypeName  string
	Duration  time.Duration // just eval (inference) time
	Score     float64
}

var (
	results       []BenchmarkResult
	loadedResults []LoadedBenchmarkResult
	wg            sync.WaitGroup
	mu            sync.Mutex
	gpuOn         bool
)

func CompareCPUVsGPU() {
	rand.Seed(42)
	nn := paragon.NewNetwork[float32]([]struct{ Width, Height int }{
		{28, 28}, // Input layer (MNIST-like)
		{16, 16}, // Hidden layer
		{10, 1},  // Output layer
	}, []string{"leaky_relu", "leaky_relu", "softmax"}, []bool{true, true, true})

	input := make([][]float64, 28)
	for i := range input {
		input[i] = make([]float64, 28)
		for j := range input[i] {
			input[i][j] = rand.Float64()
		}
	}

	const iterations = 100
	var cpuDuration, gpuDuration time.Duration

	nn.WebGPUNative = false
	for i := 0; i < iterations; i++ {
		start := time.Now()
		nn.Forward(input)
		cpuDuration += time.Since(start)
	}
	cpuOut := nn.GetOutput()

	nn.WebGPUNative = true
	nn.BuildGPUKernels()
	for i := 0; i < iterations; i++ {
		start := time.Now()
		nn.Forward(input)
		gpuDuration += time.Since(start)
	}
	gpuOut := nn.GetOutput()

	fmt.Printf("Average CPU Forward Pass (%d iterations): %v\n", iterations, cpuDuration/time.Duration(iterations))
	fmt.Printf("Average GPU Forward Pass (%d iterations): %v\n", iterations, gpuDuration/time.Duration(iterations))

	if len(cpuOut) != len(gpuOut) {
		fmt.Printf("Output length mismatch: CPU=%d, GPU=%d\n", len(cpuOut), len(gpuOut))
		return
	}
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
	fmt.Printf("CPU Output: %v\n", cpuOut)
	fmt.Printf("GPU Output: %v\n", gpuOut)
}

// Add this function to your test file to verify GPU is working
func TestGPUPerformance() {
	fmt.Println("\n=== GPU Performance Test ===")

	// Create a network
	nn := paragon.NewNetwork[float32]([]struct{ Width, Height int }{
		{28, 28}, // Input
		{16, 16}, // Hidden
		{10, 1},  // Output
	}, []string{"leaky_relu", "leaky_relu", "softmax"}, []bool{true, true, true})

	// Load the model
	if err := nn.LoadJSON("model_Standard_float32.json"); err != nil {
		fmt.Printf("Failed to load model: %v\n", err)
		return
	}

	// Create test input
	input := make([][]float64, 28)
	for i := range input {
		input[i] = make([]float64, 28)
		for j := range input[i] {
			input[i][j] = rand.Float64()
		}
	}

	// Test CPU performance
	nn.WebGPUNative = false
	nn.Debug = true // Enable debug output

	start := time.Now()
	for i := 0; i < 100; i++ {
		nn.Forward(input)
	}
	cpuTime := time.Since(start)
	cpuOutput := nn.GetOutput()

	fmt.Printf("\nCPU: 100 iterations in %v (%.2f ms/iter)\n",
		cpuTime, float64(cpuTime.Milliseconds())/100)

	// Test GPU performance
	nn.WebGPUNative = true
	nn.BuildGPUKernels()

	// Verify GPU setup
	if err := nn.VerifyGPUSetup(); err != nil {
		fmt.Printf("GPU setup verification failed: %v\n", err)
		return
	}

	nn.Debug = false // Disable debug for timing

	start = time.Now()
	for i := 0; i < 100; i++ {
		nn.Forward(input)
	}
	gpuTime := time.Since(start)
	gpuOutput := nn.GetOutput()

	fmt.Printf("GPU: 100 iterations in %v (%.2f ms/iter)\n",
		gpuTime, float64(gpuTime.Milliseconds())/100)

	// Compare outputs
	maxDiff := 0.0
	for i := range cpuOutput {
		diff := math.Abs(cpuOutput[i] - gpuOutput[i])
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	fmt.Printf("\nMax difference between CPU and GPU: %e\n", maxDiff)
	fmt.Printf("Speedup: %.2fx\n", float64(cpuTime)/float64(gpuTime))

	// Test with batch
	fmt.Println("\n--- Batch Test (1000 samples) ---")

	// CPU batch
	nn.WebGPUNative = false
	start = time.Now()
	for i := 0; i < 1000; i++ {
		nn.Forward(input)
	}
	cpuBatchTime := time.Since(start)

	// GPU batch
	nn.WebGPUNative = true
	start = time.Now()
	for i := 0; i < 1000; i++ {
		nn.Forward(input)
	}
	gpuBatchTime := time.Since(start)

	fmt.Printf("CPU Batch: %v (%.2f ms/sample)\n",
		cpuBatchTime, float64(cpuBatchTime.Milliseconds())/1000)
	fmt.Printf("GPU Batch: %v (%.2f ms/sample)\n",
		gpuBatchTime, float64(gpuBatchTime.Milliseconds())/1000)
	fmt.Printf("Batch Speedup: %.2fx\n", float64(cpuBatchTime)/float64(gpuBatchTime))
}

func TestGPUScaling() {
	fmt.Println("\n=== GPU Scaling Test ===")

	sizes := []struct {
		name   string
		layers []struct{ Width, Height int }
	}{
		{"Tiny (784→256→10)", []struct{ Width, Height int }{{28, 28}, {16, 16}, {10, 1}}},
		{"Small (784→512→10)", []struct{ Width, Height int }{{28, 28}, {32, 16}, {10, 1}}},
		{"Medium (784→1024→10)", []struct{ Width, Height int }{{28, 28}, {32, 32}, {10, 1}}},
		{"Large (784→2048→10)", []struct{ Width, Height int }{{28, 28}, {64, 32}, {10, 1}}},
	}

	input := make([][]float64, 28)
	for i := range input {
		input[i] = make([]float64, 28)
		for j := range input[i] {
			input[i][j] = rand.Float64()
		}
	}

	for _, size := range sizes {
		fmt.Printf("\n%s:\n", size.name)

		nn := paragon.NewNetwork[float32](
			size.layers,
			[]string{"leaky_relu", "leaky_relu", "softmax"},
			[]bool{true, true, true},
		)

		// CPU timing
		start := time.Now()
		for i := 0; i < 100; i++ {
			nn.Forward(input)
		}
		cpuTime := time.Since(start)

		// GPU timing
		nn.WebGPUNative = true
		nn.BuildGPUKernels()

		// Warm up
		nn.Forward(input)

		start = time.Now()
		for i := 0; i < 100; i++ {
			nn.Forward(input)
		}
		gpuTime := time.Since(start)

		speedup := float64(cpuTime) / float64(gpuTime)
		fmt.Printf("  CPU: %.2f ms/iter\n", float64(cpuTime.Microseconds())/100/1000)
		fmt.Printf("  GPU: %.2f ms/iter\n", float64(gpuTime.Microseconds())/100/1000)
		fmt.Printf("  Speedup: %.2fx %s\n", speedup,
			map[bool]string{true: "✓", false: "✗"}[speedup > 1.0])
	}
}

// Test batch processing (process multiple samples in one GPU call)
func TestBatchProcessing() {
	fmt.Println("\n=== Batch Processing Test ===")

	nn := paragon.NewNetwork[float32]([]struct{ Width, Height int }{
		{28, 28}, {32, 32}, {10, 1},
	}, []string{"leaky_relu", "leaky_relu", "softmax"}, []bool{true, true, true})

	// Create batch of inputs
	batchSizes := []int{1, 10, 50, 100}

	for _, batchSize := range batchSizes {
		fmt.Printf("\nBatch size %d:\n", batchSize)

		// Create batch
		inputs := make([][][]float64, batchSize)
		for b := 0; b < batchSize; b++ {
			inputs[b] = make([][]float64, 28)
			for i := 0; i < 28; i++ {
				inputs[b][i] = make([]float64, 28)
				for j := 0; j < 28; j++ {
					inputs[b][i][j] = rand.Float64()
				}
			}
		}

		// CPU timing
		nn.WebGPUNative = false
		start := time.Now()
		for _, input := range inputs {
			nn.Forward(input)
		}
		cpuTime := time.Since(start)

		// GPU timing
		nn.WebGPUNative = true
		nn.BuildGPUKernels()
		start = time.Now()
		for _, input := range inputs {
			nn.Forward(input)
		}
		gpuTime := time.Since(start)

		fmt.Printf("  CPU total: %v (%.3f ms/sample)\n",
			cpuTime, float64(cpuTime.Microseconds())/float64(batchSize)/1000)
		fmt.Printf("  GPU total: %v (%.3f ms/sample)\n",
			gpuTime, float64(gpuTime.Microseconds())/float64(batchSize)/1000)
		fmt.Printf("  Speedup: %.2fx\n", float64(cpuTime)/float64(gpuTime))
	}
}

func main() {

	CompareCPUVsGPU()
	TestGPUPerformance()

	return
	// --- Load MNIST ---
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatalf("MNIST download error: %v", err)
	}
	//trainInputs, trainTargets, _ := loadMNISTData(mnistDir, true)
	testInputs, testTargets, _ := loadMNISTData(mnistDir, false)
	//trainSetInputs, trainSetTargets, _, _ := paragon.SplitDataset(trainInputs, trainTargets, 0.8)

	// --- Evaluate all models by loading only (no retrain) ---
	runAllLoadedOnly(testInputs, testTargets)

	fmt.Println("\n📦 Reloaded Benchmark Results (Loaded from JSON):")
	fmt.Printf("%-14s %-10s %-12s %s\n", "Model", "Type", "Time", "ADHD Score")
	for _, r := range loadedResults {
		fmt.Printf("%-14s %-10s %-12s %.2f\n", r.ModelName, r.TypeName, r.Duration.Truncate(time.Millisecond), r.Score)
	}

	gpuOn = true

	runAllLoadedOnly(testInputs, testTargets)

	fmt.Println("\n📦 Reloaded Benchmark Results (Loaded from JSON):")
	fmt.Printf("%-14s %-10s %-12s %s\n", "Model", "Type", "Time", "ADHD Score")
	for _, r := range loadedResults {
		fmt.Printf("%-14s %-10s %-12s %.2f\n", r.ModelName, r.TypeName, r.Duration.Truncate(time.Millisecond), r.Score)
	}
}

// 🔁 Launch jobs for every T and model combination
func runAll(
	trainInputs, trainTargets, testInputs, testTargets [][][]float64,
) {
	models := []string{"Standard", "Replay", "DynamicReplay"}

	for _, model := range models {
		launch(model, "float32", trainAndEvaluate[float32], trainInputs, trainTargets, testInputs, testTargets, 5, -5)
		launch(model, "float64", trainAndEvaluate[float64], trainInputs, trainTargets, testInputs, testTargets, 5, -5)

		launch(model, "int", trainAndEvaluate[int], trainInputs, trainTargets, testInputs, testTargets, 5, -5)
		launch(model, "int8", trainAndEvaluate[int8], trainInputs, trainTargets, testInputs, testTargets, 5, -5)
		launch(model, "int16", trainAndEvaluate[int16], trainInputs, trainTargets, testInputs, testTargets, 5, -5)
		launch(model, "int32", trainAndEvaluate[int32], trainInputs, trainTargets, testInputs, testTargets, 5, -5)
		launch(model, "int64", trainAndEvaluate[int64], trainInputs, trainTargets, testInputs, testTargets, 5, -5)

		launch(model, "uint", trainAndEvaluate[uint], trainInputs, trainTargets, testInputs, testTargets, 5, 0)
		launch(model, "uint8", trainAndEvaluate[uint8], trainInputs, trainTargets, testInputs, testTargets, 5, 0)
		launch(model, "uint16", trainAndEvaluate[uint16], trainInputs, trainTargets, testInputs, testTargets, 5, 0)
		launch(model, "uint32", trainAndEvaluate[uint32], trainInputs, trainTargets, testInputs, testTargets, 5, 0)
		launch(model, "uint64", trainAndEvaluate[uint64], trainInputs, trainTargets, testInputs, testTargets, 5, 0)
	}
}

// 👇 Launch a single benchmark in a goroutine
func launch[T paragon.Numeric](
	model, typeName string,
	evalFn func(string, [][][]float64, [][][]float64, [][][]float64, [][][]float64, T, T) (float64, time.Duration),
	trainInputs, trainTargets, testInputs, testTargets [][][]float64,
	clipUpper, clipLower T,
) {
	wg.Add(1)
	go func() {
		defer wg.Done()
		start := time.Now()
		score, _ := evalFn(model, trainInputs, trainTargets, testInputs, testTargets, clipUpper, clipLower)

		totalElapsed := time.Since(start) // total (train + eval + save) time

		mu.Lock()
		results = append(results, BenchmarkResult{
			ModelName: model,
			TypeName:  typeName,
			Duration:  totalElapsed,
			Score:     score,
		})
		mu.Unlock()
	}()
}

// 🧠 Main per-model benchmark and save model
func trainAndEvaluate[T paragon.Numeric](
	modelType string,
	trainInputs, trainTargets [][][]float64,
	testInputs, testTargets [][][]float64,
	clipUpper, clipLower T, // still passed but ignored here
) (float64, time.Duration) {
	layers := []struct{ Width, Height int }{
		{28, 28}, {16, 16}, {10, 1},
	}
	acts := []string{"leaky_relu", "leaky_relu", "softmax"}
	full := []bool{true, false, true}
	nn := paragon.NewNetwork[T](layers, acts, full)

	if modelType == "Replay" {
		layer := &nn.Layers[1]
		layer.ReplayEnabled = true
		layer.ReplayPhase = "after"
		layer.ReplayOffset = -1
		layer.MaxReplay = 1
	} else if modelType == "DynamicReplay" {
		layer := &nn.Layers[1]
		layer.ReplayEnabled = true
		layer.ReplayBudget = 3
		layer.ReplayGateFunc = func(_ [][]T) float64 { return 0.6 }
		layer.ReplayGateToReps = func(score float64) int {
			if score > 0.8 {
				return 3
			} else if score > 0.6 {
				return 2
			}
			return 1
		}
	}
	nn.Train(trainInputs, trainTargets, numEpochs, learnRate, true, clipUpper, clipLower)

	// Evaluate trained model (inference timing)
	var expected, predicted []float64
	startEval := time.Now()
	for i := range testInputs {
		nn.Forward(testInputs[i])
		nn.ApplySoftmax()
		out := nn.ExtractOutput()
		pred := paragon.ArgMax(out)
		label := paragon.ArgMax(testTargets[i][0])
		expected = append(expected, float64(label))
		predicted = append(predicted, float64(pred))
	}
	nn.EvaluateModel(expected, predicted)
	origScore := nn.Performance.Score
	evalElapsed := time.Since(startEval)

	// SAVE MODEL
	filename := fmt.Sprintf("model_%s_%s.json", modelType, nn.TypeName)
	if err := nn.SaveJSON(filename); err != nil {
		fmt.Printf("❌ Could not save model %s: %v\n", filename, err)
	}

	return origScore, evalElapsed
}

// --- New: Evaluate by loading only ---
func runAllLoadedOnly(testInputs, testTargets [][][]float64) {
	models := []string{"Standard", "Replay", "DynamicReplay"}
	types := []string{"float32" /*"int32", "uint32"*/}

	for _, model := range models {
		for _, typeName := range types {
			filename := fmt.Sprintf("model_%s_%s.json", model, typeName)
			score, elapsed := evaluateLoadedModel(model, typeName, filename, testInputs, testTargets)
			loadedResults = append(loadedResults, LoadedBenchmarkResult{model, typeName, elapsed, score})
		}
	}
}

func evaluateLoadedModel(
	model, typeName, filename string,
	testInputs, testTargets [][][]float64,
) (float64, time.Duration) {
	layers := []struct{ Width, Height int }{{28, 28}, {16, 16}, {10, 1}}
	acts := []string{"leaky_relu", "leaky_relu", "softmax"}
	full := []bool{true, false, true}

	switch typeName {
	case "float32":
		return evalLoadedHelper[float32](filename, layers, acts, full, testInputs, testTargets)
	case "float64":
		return evalLoadedHelper[float64](filename, layers, acts, full, testInputs, testTargets)
	case "int":
		return evalLoadedHelper[int](filename, layers, acts, full, testInputs, testTargets)
	case "int8":
		return evalLoadedHelper[int8](filename, layers, acts, full, testInputs, testTargets)
	case "int16":
		return evalLoadedHelper[int16](filename, layers, acts, full, testInputs, testTargets)
	case "int32":
		return evalLoadedHelper[int32](filename, layers, acts, full, testInputs, testTargets)
	case "int64":
		return evalLoadedHelper[int64](filename, layers, acts, full, testInputs, testTargets)
	case "uint":
		return evalLoadedHelper[uint](filename, layers, acts, full, testInputs, testTargets)
	case "uint8":
		return evalLoadedHelper[uint8](filename, layers, acts, full, testInputs, testTargets)
	case "uint16":
		return evalLoadedHelper[uint16](filename, layers, acts, full, testInputs, testTargets)
	case "uint32":
		return evalLoadedHelper[uint32](filename, layers, acts, full, testInputs, testTargets)
	case "uint64":
		return evalLoadedHelper[uint64](filename, layers, acts, full, testInputs, testTargets)
	default:
		fmt.Printf("❌ Unknown type %s for %s\n", typeName, filename)
		return -1, 0
	}
}

func evalLoadedHelper[T paragon.Numeric](
	filename string,
	layers []struct{ Width, Height int },
	acts []string,
	full []bool,
	testInputs, testTargets [][][]float64,
) (float64, time.Duration) {
	nn := paragon.NewNetwork[T](layers, acts, full)
	if err := nn.LoadJSON(filename); err != nil {
		fmt.Printf("❌ Failed to load %s: %v\n", filename, err)
		return -1, 0
	}
	if gpuOn {
		nn.WebGPUNative = true
		nn.BuildGPUKernels()
	}
	var expected, predicted []float64
	start := time.Now()
	for i := range testInputs {
		nn.Forward(testInputs[i])
		nn.ApplySoftmax()
		out := nn.ExtractOutput()
		pred := paragon.ArgMax(out)
		label := paragon.ArgMax(testTargets[i][0])
		expected = append(expected, float64(label))
		predicted = append(predicted, float64(pred))
	}
	nn.EvaluateModel(expected, predicted)
	elapsed := time.Since(start)
	fmt.Println(elapsed)
	return nn.Performance.Score, elapsed
}
