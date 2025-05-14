package main

import (
	"fmt"
	"log"
	"sync"
	"time"

	"paragon"
)

const (
	baseURL   = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	mnistDir  = "mnist_data"
	numEpochs = 5
	learnRate = 0.01
)

type BenchmarkResult struct {
	ModelName string
	TypeName  string
	Duration  time.Duration
	Score     float64
}

var (
	results []BenchmarkResult
	wg      sync.WaitGroup
	mu      sync.Mutex
)

func main() {
	// --- Load MNIST ---
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatalf("MNIST download error: %v", err)
	}
	trainInputs, trainTargets, _ := loadMNISTData(mnistDir, true)
	testInputs, testTargets, _ := loadMNISTData(mnistDir, false)
	trainSetInputs, trainSetTargets, _, _ := paragon.SplitDataset(trainInputs, trainTargets, 0.8)

	// Launch benchmarks
	runAll(trainSetInputs, trainSetTargets, testInputs, testTargets)

	wg.Wait()

	// --- Summary Output ---
	fmt.Println("\n📊 Final Benchmark Results:")
	fmt.Printf("%-14s %-10s %-12s %s\n", "Model", "Type", "Time", "ADHD Score")
	for _, r := range results {
		fmt.Printf("%-14s %-10s %-12s %.2f\n",
			r.ModelName, r.TypeName, r.Duration.Truncate(time.Millisecond), r.Score)
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

		launch(model, "int64", trainAndEvaluate[int64], trainInputs, trainTargets, testInputs, testTargets, 5, -5)

		launch(model, "uint", trainAndEvaluate[uint], trainInputs, trainTargets, testInputs, testTargets, 5, 0)
		launch(model, "uint8", trainAndEvaluate[uint8], trainInputs, trainTargets, testInputs, testTargets, 5, 0)
		launch(model, "uint16", trainAndEvaluate[uint16], trainInputs, trainTargets, testInputs, testTargets, 5, 0)
		launch(model, "uint32", trainAndEvaluate[uint32], trainInputs, trainTargets, testInputs, testTargets, 5, 0)
		launch(model, "uint64", trainAndEvaluate[uint64], trainInputs, trainTargets, testInputs, testTargets, 5, 0)

		launch(model, "int32", trainAndEvaluate[int32], trainInputs, trainTargets, testInputs, testTargets, 5, -5)
	}
}

// 👇 Launch a single benchmark in a goroutine
func launch[T paragon.Numeric](
	model, typeName string,
	evalFn func(string, [][][]float64, [][][]float64, [][][]float64, [][][]float64, T, T) float64,
	trainInputs, trainTargets, testInputs, testTargets [][][]float64,
	clipUpper, clipLower T,
) {
	wg.Add(1)
	go func() {
		defer wg.Done()
		start := time.Now()
		score := evalFn(model, trainInputs, trainTargets, testInputs, testTargets, clipUpper, clipLower)
		elapsed := time.Since(start)

		mu.Lock()
		results = append(results, BenchmarkResult{
			ModelName: model,
			TypeName:  typeName,
			Duration:  elapsed,
			Score:     score,
		})
		mu.Unlock()
	}()
}

// 🧠 Main per-model benchmark
func trainAndEvaluate[T paragon.Numeric](
	modelType string,
	trainInputs, trainTargets [][][]float64,
	testInputs, testTargets [][][]float64,
	clipUpper, clipLower T, // still passed but ignored here
) float64 {
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
	//nn.Debug = true
	nn.Train(trainInputs, trainTargets, numEpochs, learnRate, true, clipUpper, clipLower)
	//nn.Debug = false
	var expected, predicted []float64
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
	return nn.Performance.Score
}

// scale.go
func scaleDataset(in [][][]float64, factor float64) [][][]float64 {
	out := make([][][]float64, len(in))
	for i, sample := range in {
		H, W := len(sample), len(sample[0])
		scaled := make([][]float64, H)
		for y := 0; y < H; y++ {
			scaled[y] = make([]float64, W)
			for x := 0; x < W; x++ {
				scaled[y][x] = sample[y][x] * factor
			}
		}
		out[i] = scaled
	}
	return out
}
