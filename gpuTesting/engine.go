package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
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

	// 💾 Save models to absolute paths
	modelDir := filepath.Join(os.Getenv("HOME"), "git", "paradawn", "models")
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		panic(err)
	}
	stdModel := filepath.Join(modelDir, "mnist_standard.json")
	repModel := filepath.Join(modelDir, "mnist_manual_replay.json")

	if err := standard.SaveJSON(stdModel); err != nil {
		panic(err)
	}
	if err := replay.SaveJSON(repModel); err != nil {
		panic(err)
	}
	fmt.Println("✅ Models saved to:")
	fmt.Println(" -", stdModel)
	fmt.Println(" -", repModel)

	// 🚀 Run compiled C++ backend
	execPath := filepath.Join(os.Getenv("HOME"), "git", "paradawn", "build", "paradawn")

	//fmt.Println("🔁 Running compiled executable with standard model (CPU)...")
	//RunParadawnExecutable(execPath, stdModel, 0)

	fmt.Println("🔁 Running compiled executable with replay model (GPU x8)...")
	RunParadawnExecutable(execPath, stdModel, 8)
}

// RunParadawnExecutable runs the C++ executable with stdin input and model
func RunParadawnExecutable(exePath, modelPath string, gpuLayers int) {
	fmt.Printf("🚀 Running: %s %s [--gpu=%d]\n", exePath, modelPath, gpuLayers)

	// Build 28x28 input stream
	var input strings.Builder
	for y := 0; y < 28; y++ {
		for x := 0; x < 28; x++ {
			input.WriteString("0 ")
		}
		input.WriteString("\n")
	}
	inputStr := input.String()

	// Show the input that will be streamed
	fmt.Println("📤 Sending input to paradawn (28x28 zeros):")
	fmt.Println(strings.TrimSpace(inputStr))

	// Prepare command args
	args := []string{modelPath}
	if gpuLayers > 0 {
		args = append(args, fmt.Sprintf("--gpu=%d", gpuLayers))
	}
	fmt.Printf("🛠️ Executing command: %s %s\n", exePath, strings.Join(args, " "))

	// Set up the command
	cmd := exec.Command(exePath, args...)
	cmd.Stdin = strings.NewReader(inputStr)

	// Create pipes for stdout/stderr
	stdout, _ := cmd.StdoutPipe()
	stderr, _ := cmd.StderrPipe()

	// Start the process
	start := time.Now()
	if err := cmd.Start(); err != nil {
		fmt.Println("❌ Failed to start command:", err)
		return
	}

	// Stream STDOUT
	go func() {
		fmt.Println("🔧 [paradawn STDOUT]")
		buf := make([]byte, 1024)
		for {
			n, err := stdout.Read(buf)
			if n > 0 {
				fmt.Print(string(buf[:n]))
			}
			if err != nil {
				break
			}
		}
	}()

	// Stream STDERR
	go func() {
		fmt.Println("⚠️ [paradawn STDERR]")
		buf := make([]byte, 1024)
		for {
			n, err := stderr.Read(buf)
			if n > 0 {
				fmt.Print(string(buf[:n]))
			}
			if err != nil {
				break
			}
		}
	}()

	// Wait for the process to finish
	if err := cmd.Wait(); err != nil {
		fmt.Println("❌ Process error:", err)
	}
	elapsed := time.Since(start)

	// Timing summary
	fmt.Println()
	fmt.Println("⏱️  Total Duration:", elapsed)
	fmt.Println("---------------------------------")
}
