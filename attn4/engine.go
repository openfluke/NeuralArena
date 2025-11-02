package main

import (
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"paragon"
)

/* ===================== helpers ===================== */

func sha12(v []float64) string {
	h := sha1.New()
	for _, x := range v {
		fmt.Fprintf(h, "%.9f,", x)
	}
	return hex.EncodeToString(h.Sum(nil))[:12]
}

func maxAbsDiff(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	m := 0.0
	for i := range a {
		d := math.Abs(a[i] - b[i])
		if d > m {
			m = d
		}
	}
	return m
}

func toFloat64(vals []float32) []float64 {
	out := make([]float64, len(vals))
	for i, v := range vals {
		out[i] = float64(v)
	}
	return out
}

/* ===================== inputs ===================== */

func zeroInput(h, w int) [][]float64 {
	X := make([][]float64, h)
	for r := 0; r < h; r++ {
		X[r] = make([]float64, w)
	}
	return X
}

func randInput(h, w int, seed int64) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	X := make([][]float64, h)
	for r := 0; r < h; r++ {
		row := make([]float64, w)
		for c := 0; c < w; c++ {
			row[c] = rng.Float64()
		}
		X[r] = row
	}
	return X
}

func randTarget(size int, seed int64) []float64 {
	rng := rand.New(rand.NewSource(seed + 999))
	target := make([]float64, size)
	sum := 0.0
	for i := range target {
		target[i] = rng.Float64()
		sum += target[i]
	}
	// Normalize to sum to 1
	for i := range target {
		target[i] /= sum
	}
	return target
}

/* ===================== model builders ===================== */

func buildDense() *paragon.Network[float32] {
	sizes := []paragon.GridSpec{
		{Width: 8, Height: 8},
		{Width: 4, Height: 4},
		{Width: 4, Height: 4},
		{Width: 4, Height: 1},
	}
	acts := []string{"relu", "relu", "relu", "softmax"}
	fully := []bool{false, true, true, true}

	n, err := paragon.BuildGridNet[float32](paragon.BuildOpts[float32]{
		Sizes: sizes, Activations: acts, FullyConn: fully,
	})
	if err != nil {
		panic(err)
	}
	return n
}

func buildAttnMix(heads, dk int, useNorm bool) *paragon.Network[float32] {
	sizes := []paragon.GridSpec{
		{Width: 8, Height: 8},
		{Width: 4, Height: 4},
		{Width: 4, Height: 4}, // mixed layer
		{Width: 4, Height: 1},
	}
	acts := []string{"relu", "relu", "relu", "softmax"}
	fully := []bool{false, true, true, true}

	slices := make([][]string, len(sizes))
	slices[2] = []string{"dense", "attn", "dense", "attn"}

	attn := make([]*paragon.AttnConfig[float32], len(sizes))
	attn[2] = &paragon.AttnConfig[float32]{
		Heads:       heads,
		DK:          dk,
		UseWo:       true,
		Share:       "layer",
		Dropout:     0,
		PosEnc2D:    true,
		UseNorm:     useNorm,
		PosEncAmp:   0.02,
		NormEps:     1e-6,
		UseReplay:   false,
		ForceReplay: false,
	}

	n, err := paragon.BuildGridNet[float32](paragon.BuildOpts[float32]{
		Sizes: sizes, Activations: acts, FullyConn: fully,
		SliceTypes: slices, Attn: attn,
	})
	if err != nil {
		panic(err)
	}
	return n
}

/* ===================== training helpers ===================== */

type trainResult struct {
	finalLoss    float64
	totalTime    time.Duration
	initTime     time.Duration
	avgEpochTime time.Duration
	finalOutput  []float64
}

func trainCPU(n *paragon.Network[float32], inputs [][]float64, target []float64, epochs int, lr float64) trainResult {
	n.WebGPUNative = false

	// Convert 1D target to 2D format for Backward
	targets2D := make([][]float64, 1)
	targets2D[0] = target

	start := time.Now()
	var totalEpochTime time.Duration
	var finalLoss float64

	for e := 0; e < epochs; e++ {
		epochStart := time.Now()

		n.Forward(inputs)
		output := n.GetOutput()

		// Calculate loss
		finalLoss = 0.0
		for i := range output {
			diff := output[i] - target[i]
			finalLoss += diff * diff
		}
		finalLoss *= 0.5

		n.Backward(targets2D, lr, 1.0, -1.0)

		totalEpochTime += time.Since(epochStart)
	}

	// Final forward pass
	n.Forward(inputs)
	finalOutput := n.GetOutput()

	return trainResult{
		finalLoss:    finalLoss,
		totalTime:    time.Since(start),
		initTime:     0,
		avgEpochTime: totalEpochTime / time.Duration(epochs),
		finalOutput:  finalOutput,
	}
}

func trainGPU(n *paragon.Network[float32], inputs [][]float64, target []float64, epochs int, lr float64) trainResult {
	n.WebGPUNative = true

	// Convert 1D target to 2D format for Backward
	targets2D := make([][]float64, 1)
	targets2D[0] = target

	// Initialize GPU
	initStart := time.Now()
	if err := n.InitializeOptimizedGPU(); err != nil {
		return trainResult{initTime: -1}
	}
	if err := n.InitializeBackwardGPU(); err != nil {
		return trainResult{initTime: -1}
	}
	initTime := time.Since(initStart)

	start := time.Now()
	var totalEpochTime time.Duration
	var finalLoss float64

	for e := 0; e < epochs; e++ {
		epochStart := time.Now()

		n.Forward(inputs)
		output := n.GetOutput()

		// Calculate loss
		finalLoss = 0.0
		for i := range output {
			diff := output[i] - target[i]
			finalLoss += diff * diff
		}
		finalLoss *= 0.5

		n.Backward(targets2D, lr, 1.0, -1.0)

		totalEpochTime += time.Since(epochStart)
	}

	// Final forward pass
	n.Forward(inputs)
	finalOutput := n.GetOutput()

	return trainResult{
		finalLoss:    finalLoss,
		totalTime:    time.Since(start),
		initTime:     initTime,
		avgEpochTime: totalEpochTime / time.Duration(epochs),
		finalOutput:  finalOutput,
	}
}

/* ===================== save/load ===================== */

func saveBundle(path, id string, n *paragon.Network[float32]) {
	js, err := paragon.ExportBundleJSON[float32](id, n, nil)
	if err != nil {
		panic(err)
	}
	if err := os.WriteFile(path, []byte(js), 0o644); err != nil {
		panic(err)
	}
}

func loadBundle(path string) (*paragon.Network[float32], error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	anyNet, err := paragon.ImportBundleJSON(string(b))
	if err != nil {
		return nil, err
	}
	rt, ok := anyNet.(*paragon.Network[float32])
	if !ok {
		return nil, fmt.Errorf("bundle type mismatch")
	}
	return rt, nil
}

/* ===================== main ===================== */

func main() {
	outDir := "bundles_training"
	_ = os.MkdirAll(outDir, 0o755)

	type spec struct {
		name   string
		make   func() *paragon.Network[float32]
		epochs int
		lr     float64
	}

	variants := []spec{
		{
			name:   "dense",
			make:   func() *paragon.Network[float32] { return buildDense() },
			epochs: 50,
			lr:     0.01,
		},
		{
			name:   "attn_H1_noNorm",
			make:   func() *paragon.Network[float32] { return buildAttnMix(1, 64, false) },
			epochs: 50,
			lr:     0.01,
		},
		{
			name:   "attn_H2_norm",
			make:   func() *paragon.Network[float32] { return buildAttnMix(2, 32, true) },
			epochs: 50,
			lr:     0.01,
		},
	}

	// Inputs and targets
	X := randInput(8, 8, 12345)
	target := randTarget(4, 67890)

	fmt.Println("=== Training Test: CPU vs GPU ===\n")
	fmt.Printf("Training config: %d epochs, lr=%.4f\n", variants[0].epochs, variants[0].lr)
	fmt.Printf("Input: 8x8 random, Target: [%.3f %.3f %.3f %.3f]\n\n",
		target[0], target[1], target[2], target[3])

	for _, sp := range variants {
		path := filepath.Join(outDir, sp.name+".bundle.json")

		// Build and save if missing
		if _, err := os.Stat(path); err != nil {
			net := sp.make()
			saveBundle(path, sp.name, net)
		}

		// Load for CPU
		cpuNet, err := loadBundle(path)
		if err != nil {
			fmt.Printf("[%s] CPU load error: %v\n", sp.name, err)
			continue
		}

		// Load for GPU
		gpuNet, err := loadBundle(path)
		if err != nil {
			fmt.Printf("[%s] GPU load error: %v\n", sp.name, err)
			continue
		}

		fmt.Printf("[%s]\n", sp.name)

		// Train CPU
		cpuRes := trainCPU(cpuNet, X, target, sp.epochs, sp.lr)
		fmt.Printf("  CPU: total=%v  avg_epoch=%v  final_loss=%.6f\n",
			cpuRes.totalTime, cpuRes.avgEpochTime, cpuRes.finalLoss)
		fmt.Printf("       final_output: %s\n", sha12(cpuRes.finalOutput))

		// Train GPU
		gpuRes := trainGPU(gpuNet, X, target, sp.epochs, sp.lr)
		if gpuRes.initTime < 0 {
			fmt.Printf("  GPU: init FAILED (skipping)\n\n")
			continue
		}

		fmt.Printf("  GPU: init=%v  total=%v  avg_epoch=%v  final_loss=%.6f\n",
			gpuRes.initTime, gpuRes.totalTime, gpuRes.avgEpochTime, gpuRes.finalLoss)
		fmt.Printf("       final_output: %s\n", sha12(gpuRes.finalOutput))

		// Compare outputs
		delta := maxAbsDiff(cpuRes.finalOutput, gpuRes.finalOutput)
		fmt.Printf("  Output delta: %.9g\n", delta)

		// Speedup
		if cpuRes.avgEpochTime > 0 && gpuRes.avgEpochTime > 0 {
			speedup := float64(cpuRes.avgEpochTime) / float64(gpuRes.avgEpochTime)
			fmt.Printf("  Speedup: %.2fx (GPU vs CPU per epoch)\n", speedup)
		}

		fmt.Println()
	}

	fmt.Println("Done.")
}
