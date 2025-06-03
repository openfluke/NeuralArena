package main

import (
	"fmt"
	"math/rand"
	"runtime"
	"time"

	"paragon"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("=== Grow() Function Test with ADHD Evaluation ===")

	// Example MNIST-like dummy data (replace with real samples)
	inputs := [][][]float64{
		{{0.0, 0.1, 0.2, 0.0}},
		{{0.9, 0.8, 0.7, 0.6}},
		{{0.3, 0.2, 0.5, 0.1}},
	}

	labels := []float64{1, 0, 3} // dummy labels matching input

	// Define network structure
	layerSizes := []struct{ Width, Height int }{
		{4, 1}, {6, 1}, {4, 1}, // input → hidden → output
	}
	activations := []string{"linear", "relu", "softmax"}
	connected := []bool{false, true, true}

	// Create new network
	net := paragon.NewNetwork[float32](layerSizes, activations, connected)
	net.Debug = true

	// Show structure before
	fmt.Println("\n🧠 Initial Network Structure:")
	printNetworkShape(net)

	// Pre-evaluate base performance
	net.EvaluateModel(labels, extractPredictedLabels(net, inputs))
	fmt.Printf("📊 Initial ADHD Score: %.2f\n", net.Performance.Score)

	totalCores := runtime.NumCPU()
	maxThreads := int(0.8 * float64(totalCores)) // Use 80%
	if maxThreads < 1 {
		maxThreads = 1
	}

	fmt.Printf("🚀 Using %d of %d cores for Grow() exploration\n", maxThreads, totalCores)

	// Run grow experiment
	improved := net.Grow(
		1,         // checkpointLayer
		inputs,    // testInputs
		labels,    // expectedOutputs
		50,        // numCandidates
		5,         // epochs
		0.05,      // learningRate
		1e-6,      // tolerance
		1.0, -1.0, // clipUpper / clipLower
		2, 8, // minWidth, maxWidth
		1, 4, // ✅ minHeight, maxHeight
		[]string{"relu", "sigmoid", "tanh"}, // activation pool
		maxThreads,
	)

	// Show structure after
	fmt.Println("\n🧠 Final Network Structure:")
	printNetworkShape(net)

	if improved {
		fmt.Println("🚀 Network successfully improved by Grow()!")
	} else {
		fmt.Println("⚡️  No improvement found during Grow().")
	}
}

// Helper: extract predicted labels
func extractPredictedLabels(net *paragon.Network[float32], inputs [][][]float64) []float64 {
	labels := make([]float64, len(inputs))
	for i, in := range inputs {
		net.Forward(in)
		raw := net.GetOutput()
		labels[i] = float64(paragon.ArgMax(raw))
	}
	return labels
}

// Helper: print layer sizes and count
// Helper: print layer sizes and count
func printNetworkShape[T paragon.Numeric](net *paragon.Network[T]) {
	for i, layer := range net.Layers {
		activation := "unknown"
		if len(layer.Neurons) > 0 && len(layer.Neurons[0]) > 0 {
			activation = layer.Neurons[0][0].Activation
		}
		fmt.Printf("   Layer %d: %dx%d (%s)\n", i, layer.Width, layer.Height, activation)
	}
	fmt.Printf("   Total Layers: %d\n", len(net.Layers))
}
