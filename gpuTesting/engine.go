package main

import (
	"fmt"
	"os"
	"time"

	"paragon"
)

func main() {
	// 🧪 Dummy MNIST-style input: 28x28 zeros
	input := make([][]float64, 28)
	for i := range input {
		input[i] = make([]float64, 28)
	}

	// ⚙️ Model layout
	layers := []struct{ Width, Height int }{
		{28, 28},  // Input
		{2048, 1}, // Hidden 1
		{2048, 1}, // Hidden 2
		{1024, 1}, // Hidden 3
		{1024, 1}, // Hidden 4
		{512, 1},  // Hidden 5
		{512, 1},  // Hidden 6
		{256, 1},  // Hidden 7
		{128, 1},  // Hidden 8
		{10, 1},   // Output
	}
	acts := []string{
		"leaky_relu", // Input layer
		"relu",       // Hidden 1
		"relu",       // Hidden 2
		"relu",       // Hidden 3
		"relu",       // Hidden 4
		"relu",       // Hidden 5
		"relu",       // Hidden 6
		"relu",       // Hidden 7
		"relu",       // Hidden 8
		"softmax",    // Output
	}
	full := make([]bool, len(layers))
	for i := range full {
		full[i] = true
	}

	// 🚫 Standard model (no replay)
	standard := paragon.NewNetwork[float32](layers, acts, full)
	standard.TypeName = "float32"

	start1 := time.Now()
	standard.Forward(input)
	elapsed1 := time.Since(start1)

	// 🔁 Manual replay model (after, 10x, from previous layer)
	replay := paragon.NewNetwork[float32](layers, acts, full)
	replay.TypeName = "float32"

	for i := range replay.Layers {
		if i > 0 && i < len(replay.Layers)-1 {
			replay.Layers[i].ReplayEnabled = true
			replay.Layers[i].ReplayPhase = "after"
			replay.Layers[i].ReplayOffset = -1
			replay.Layers[i].MaxReplay = 10
		}
	}

	start2 := time.Now()
	replay.Forward(input)
	elapsed2 := time.Since(start2)

	// 🧠 Report
	fmt.Println("🧠 Inference Timing Comparison")
	fmt.Println("----------------------------------")
	fmt.Printf("Standard (no replay):  %v\n", elapsed1)
	fmt.Printf("Manual Replay (x10):   %v\n", elapsed2)

	// 💾 Save models
	if err := os.MkdirAll("models", 0755); err != nil {
		panic(err)
	}
	if err := standard.SaveJSON("models/mnist_standard.json"); err != nil {
		panic(err)
	}
	if err := replay.SaveJSON("models/mnist_manual_replay.json"); err != nil {
		panic(err)
	}
	fmt.Println("✅ Models saved to: mnist_standard.json & mnist_manual_replay.json")
}
