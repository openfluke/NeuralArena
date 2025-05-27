package main

import (
	"fmt"
	"log"
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

func main() {
	// --- Load MNIST ---
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatalf("MNIST download error: %v", err)
	}
	//trainInputs, trainTargets, _ := loadMNISTData(mnistDir, true)
	testInputs, testTargets, _ := loadMNISTData(mnistDir, false)
	//trainSetInputs, trainSetTargets, _, _ := paragon.SplitDataset(trainInputs, trainTargets, 0.8)

	//trainSetInputs, trainSetTargets, _, _ := paragon.SplitDataset(trainInputs, trainTargets, 0.8)

	trainInputs, trainTargets, _ := loadMNISTData(mnistDir, true)

	runAll(trainInputs, trainTargets, testInputs, testTargets)

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
		/*	launch(model, "float64", trainAndEvaluate[float64], trainInputs, trainTargets, testInputs, testTargets, 5, -5)

			launch(model, "int", trainAndEvaluate[int], trainInputs, trainTargets, testInputs, testTargets, 5, -5)
			launch(model, "int8", trainAndEvaluate[int8], trainInputs, trainTargets, testInputs, testTargets, 5, -5)
			launch(model, "int16", trainAndEvaluate[int16], trainInputs, trainTargets, testInputs, testTargets, 5, -5)
			launch(model, "int32", trainAndEvaluate[int32], trainInputs, trainTargets, testInputs, testTargets, 5, -5)
			launch(model, "int64", trainAndEvaluate[int64], trainInputs, trainTargets, testInputs, testTargets, 5, -5)

			launch(model, "uint", trainAndEvaluate[uint], trainInputs, trainTargets, testInputs, testTargets, 5, 0)
			launch(model, "uint8", trainAndEvaluate[uint8], trainInputs, trainTargets, testInputs, testTargets, 5, 0)
			launch(model, "uint16", trainAndEvaluate[uint16], trainInputs, trainTargets, testInputs, testTargets, 5, 0)
			launch(model, "uint32", trainAndEvaluate[uint32], trainInputs, trainTargets, testInputs, testTargets, 5, 0)
			launch(model, "uint64", trainAndEvaluate[uint64], trainInputs, trainTargets, testInputs, testTargets, 5, 0)*/
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
	types := []string{"float32"}

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
		//nn.BuildGPUKernels()
		err := nn.InitializeOptimizedGPU()
		if err != nil {
			log.Fatalf("Failed to initialize GPU: %v", err)
		}

		defer nn.CleanupOptimizedGPU()
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
