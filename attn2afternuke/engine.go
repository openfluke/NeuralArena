// engine.go
package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"paragon"
)

/* =========================
   DATA: global two-query (8x8)
   ========================= */

type sample struct {
	X [][]float64 // 8x8 input
	Y [][]float64 // 1x4 one-hot (with label smoothing)
}

func oneHot4(k int, smooth float64) [][]float64 {
	y := make([][]float64, 1)
	y[0] = []float64{smooth / 3, smooth / 3, smooth / 3, smooth / 3}
	if k >= 0 && k < 4 {
		y[0][k] = 1.0 - smooth
	}
	return y
}

func mkFixedRng(seed int64) *rand.Rand { return rand.New(rand.NewSource(seed)) }

func buildGlobalTwoQuery(n int, rng *rand.Rand, smooth float64) []sample {
	const H, W = 8, 8
	out := make([]sample, n)
	for i := 0; i < n; i++ {
		X := make([][]float64, H)
		for r := 0; r < H; r++ {
			X[r] = make([]float64, W)
		}
		r1, c1 := rng.Intn(H), rng.Intn(W)
		r2, c2 := rng.Intn(H), rng.Intn(W)
		for r1 == r2 && c1 == c2 {
			r2, c2 = rng.Intn(H), rng.Intn(W)
		}
		X[r1][c1] = 1
		X[r2][c2] = 1

		var cls int
		switch {
		case r1 == r2:
			cls = 0
		case c1 == c2:
			cls = 1
		case (r1 - c1) == (r2 - c2):
			cls = 2
		default:
			cls = 3
		}
		out[i] = sample{X: X, Y: oneHot4(cls, smooth)}
	}
	return out
}

/* =========================
   PRINT HELPERS
   ========================= */

func argmax(probs []float64) int {
	mi := 0
	mx := probs[0]
	for i := 1; i < len(probs); i++ {
		if probs[i] > mx {
			mx, mi = probs[i], i
		}
	}
	return mi
}

func printFirst(out []float64, n int, label string) {
	fmt.Println(label)
	for i := 0; i < len(out) && i < n; i++ {
		fmt.Printf(" %.15e", out[i])
	}
	fmt.Println()
}

func printConfusion(name string, cm [4][4]int) {
	fmt.Printf("\n%s confusion matrix (rows=true, cols=pred):\n", name)
	for i := 0; i < 4; i++ {
		fmt.Printf("[%d %d %d %d]\n", cm[i][0], cm[i][1], cm[i][2], cm[i][3])
	}
}

/* =========================
   MODEL BUILDERS
   ========================= */

func buildShapes() ([]paragon.GridSpec, []string, []bool) {
	// 8x8 -> 4x4 -> 4x4 -> 1x4
	sizes := []paragon.GridSpec{
		{Width: 8, Height: 8}, // input
		{Width: 4, Height: 4}, // hidden1
		{Width: 4, Height: 4}, // hidden2 (mix: dense/attn in columns)
		{Width: 4, Height: 1}, // logits
	}
	acts := []string{"relu", "relu", "relu", "softmax"}
	fully := []bool{true, true, true, true}
	return sizes, acts, fully
}

type attnKnobs[T paragon.Numeric] struct {
	share     string
	useNorm   bool
	posEncAmp float64
	normEps   float64
	dk        int
	heads     int
}

func buildDense[T paragon.Numeric](sizes []paragon.GridSpec, acts []string, fully []bool) *paragon.Network[T] {
	n, err := paragon.BuildGridNet[T](paragon.BuildOpts[T]{
		Sizes:       sizes,
		Activations: acts,
		FullyConn:   fully,
	})
	if err != nil {
		panic(err)
	}
	return n
}

func buildAttnMix[T paragon.Numeric](sizes []paragon.GridSpec, acts []string, fully []bool, knobs attnKnobs[T]) *paragon.Network[T] {
	slices := make([][]string, len(sizes))
	slices[2] = []string{"dense", "attn", "dense", "attn"}

	attnCfg := make([]*paragon.AttnConfig[T], len(sizes))
	attnCfg[2] = &paragon.AttnConfig[T]{
		Heads:     knobs.heads,
		DK:        knobs.dk,
		UseWo:     true,
		Share:     knobs.share,
		Dropout:   0.0,
		PosEnc2D:  true,
		UseNorm:   knobs.useNorm,
		PosEncAmp: knobs.posEncAmp,
		NormEps:   knobs.normEps,
	}

	n, err := paragon.BuildGridNet[T](paragon.BuildOpts[T]{
		Sizes:       sizes,
		Activations: acts,
		FullyConn:   fully,
		SliceTypes:  slices,
		Attn:        attnCfg,
	})
	if err != nil {
		panic(err)
	}
	return n
}

/* =========================
   TRAIN / EVAL
   ========================= */

type trainCfg struct {
	epochs int
	lr0    float64
	lr1    float64
	clipHi float32
	clipLo float32
	seed   int64
	name   string
	useGPU bool
}

func cosineLR(t, T int, lr0, lr1 float64) float64 {
	ct := (1 - math.Cos(math.Pi*float64(t)/float64(T))) / 2
	return lr0*(1-ct) + lr1*ct
}

func trainNet[T paragon.Numeric](net *paragon.Network[T], data []sample, cfg trainCfg) time.Duration {
	trainStart := time.Now()
	permRng := rand.New(rand.NewSource(cfg.seed))

	for e := 0; e < cfg.epochs; e++ {
		epochStart := time.Now()

		lr := cosineLR(e, cfg.epochs-1, cfg.lr0, cfg.lr1)
		total := 0.0
		perm := permRng.Perm(len(data))
		for _, i := range perm {
			net.Forward(data[i].X)
			total += net.ComputeLoss(data[i].Y)
			net.Backward(data[i].Y, lr, T(cfg.clipHi), T(cfg.clipLo))
		}

		epochDur := time.Since(epochStart)
		fmt.Printf("  Epoch %02d: loss=%.4f lr=%.4g took=%s\n",
			e, total/float64(len(data)), lr, epochDur)
	}

	totalDur := time.Since(trainStart)
	avgPerEpoch := totalDur / time.Duration(cfg.epochs)
	fmt.Printf("[%s] Total: %s | Avg/epoch: %s\n", cfg.name, totalDur, avgPerEpoch)
	return totalDur
}

type evalOut struct {
	name string
	acc  float64
	cm   [4][4]int
}

func evaluate[T paragon.Numeric](name string, net *paragon.Network[T], test []sample) evalOut {
	correct := 0
	var cm [4][4]int

	for _, s := range test {
		exp := 0
		for j := 0; j < 4; j++ {
			if s.Y[0][j] > 0.5 {
				exp = j
				break
			}
		}

		net.Forward(s.X)
		out := net.GetOutput()
		pred := argmax(out)
		if pred == exp {
			correct++
		}

		cm[exp][pred]++
	}

	return evalOut{
		name: name,
		acc:  float64(correct) / float64(len(test)),
		cm:   cm,
	}
}

/* =========================
   SAVE / LOAD
   ========================= */

func saveModel(path, id string, n *paragon.Network[float32]) {
	js, err := paragon.ExportBundleJSON[float32](id, n, nil)
	if err != nil {
		panic(err)
	}
	if err := os.WriteFile(path, []byte(js), 0o644); err != nil {
		panic(err)
	}
}

func loadModel(path string) *paragon.Network[float32] {
	data, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	anyNet, err := paragon.ImportBundleJSON(string(data))
	if err != nil {
		panic(err)
	}
	rt, ok := anyNet.(*paragon.Network[float32])
	if !ok {
		panic("type mismatch after reload")
	}
	return rt
}

/* =========================
   OUTPUT COMPARISON
   ========================= */

func compareOutputs(cpu, gpu []float64) float64 {
	if len(cpu) != len(gpu) {
		panic("output length mismatch")
	}
	maxDiff := 0.0
	for i := range cpu {
		diff := math.Abs(cpu[i] - gpu[i])
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	return maxDiff
}

/* =========================
   MAIN
   ========================= */

func main() {
	const smooth = 0.05
	dataRng := mkFixedRng(42)
	train := buildGlobalTwoQuery(4096, dataRng, smooth)
	test := buildGlobalTwoQuery(1024, dataRng, smooth)

	sizes, acts, fully := buildShapes()

	// Create output directory
	outDir := "gpu_comparison"
	_ = os.MkdirAll(outDir, 0o755)

	// Model configurations to test
	type modelConfig struct {
		name    string
		isDense bool
		knobs   attnKnobs[float32]
	}

	configs := []modelConfig{
		{"dense", true, attnKnobs[float32]{}},
		{"attnLayer_norm_H2", false, attnKnobs[float32]{
			share: "layer", useNorm: true, posEncAmp: 0.02, normEps: 1e-6, dk: 32, heads: 2,
		}},
		{"attnPerSlice_norm_H2", false, attnKnobs[float32]{
			share: "per-slice", useNorm: true, posEncAmp: 0.02, normEps: 1e-6, dk: 32, heads: 2,
		}},
	}

	trainCfg := trainCfg{
		epochs: 28,
		lr0:    1e-2,
		lr1:    3e-3,
		clipHi: 1.0,
		clipLo: -1.0,
		seed:   42,
	}

	fmt.Println("=== GPU vs CPU Comparison ===\n")

	// Process each model type
	for _, config := range configs {
		fmt.Printf("\n========================================\n")
		fmt.Printf("MODEL: %s\n", config.name)
		fmt.Printf("========================================\n\n")

		// Create CPU version
		fmt.Println("Creating CPU model...")
		var cpuNet *paragon.Network[float32]
		if config.isDense {
			cpuNet = buildDense[float32](sizes, acts, fully)
		} else {
			cpuNet = buildAttnMix[float32](sizes, acts, fully, config.knobs)
		}

		// Test forward pass before training
		fmt.Println("\n--- Pre-training forward pass check ---")
		testInput := test[0].X
		cpuNet.Forward(testInput)
		cpuOut := cpuNet.GetOutput()
		printFirst(cpuOut, 4, "CPU output (first 4):")

		// Train on CPU
		fmt.Printf("\n--- Training on CPU ---\n")
		trainCfg.name = config.name + " [CPU]"
		trainCfg.useGPU = false
		cpuTime := trainNet[float32](cpuNet, train, trainCfg)

		// Save CPU model
		cpuPath := filepath.Join(outDir, config.name+"_cpu.bundle.json")
		saveModel(cpuPath, config.name+"_cpu", cpuNet)
		fmt.Printf("Saved CPU model to: %s\n", cpuPath)

		// Post-training CPU output
		fmt.Println("\n--- Post-training CPU output ---")
		cpuNet.Forward(testInput)
		cpuOutTrained := cpuNet.GetOutput()
		printFirst(cpuOutTrained, 4, "CPU trained output (first 4):")

		// Evaluate CPU
		cpuEval := evaluate(config.name+"_cpu", cpuNet, test)
		fmt.Printf("\nCPU Accuracy: %.2f%%\n", 100*cpuEval.acc)

		// Create GPU version by reloading the same saved model
		fmt.Println("\n--- Creating GPU version (reload from CPU model) ---")
		gpuNet := loadModel(cpuPath)

		// Enable GPU
		fmt.Println("Enabling GPU...")
		if err := gpuNet.EnableGPU(); err != nil {
			fmt.Printf("GPU not available: %v\nSkipping GPU comparison for this model.\n", err)
			continue
		}
		fmt.Println("GPU enabled successfully!")

		// Test GPU forward pass before training (should match CPU exactly)
		fmt.Println("\n--- Pre-training GPU vs CPU forward pass ---")
		gpuNet.Forward(testInput)
		gpuOut := gpuNet.GetOutput()
		printFirst(gpuOut, 4, "GPU output (first 4):")
		diff := compareOutputs(cpuOut, gpuOut)
		fmt.Printf("Max difference (CPU vs GPU): %.15e\n", diff)
		if diff < 1e-5 {
			fmt.Println("✓ GPU matches CPU (within tolerance)")
		} else {
			fmt.Println("✗ GPU differs from CPU!")
		}

		// Train on GPU
		fmt.Printf("\n--- Training on GPU ---\n")
		trainCfg.name = config.name + " [GPU]"
		trainCfg.useGPU = true
		trainCfg.seed = 42 // Same seed for fair comparison
		gpuTime := trainNet[float32](gpuNet, train, trainCfg)

		// Compare training times
		fmt.Printf("\n--- Training Time Comparison ---\n")
		fmt.Printf("CPU Time: %s\n", cpuTime)
		fmt.Printf("GPU Time: %s\n", gpuTime)
		speedup := float64(cpuTime) / float64(gpuTime)
		fmt.Printf("Speedup: %.2fx\n", speedup)

		// Save GPU model
		gpuPath := filepath.Join(outDir, config.name+"_gpu.bundle.json")
		saveModel(gpuPath, config.name+"_gpu", gpuNet)
		fmt.Printf("Saved GPU model to: %s\n", gpuPath)

		// Post-training GPU output
		fmt.Println("\n--- Post-training GPU output ---")
		gpuNet.Forward(testInput)
		gpuOutTrained := gpuNet.GetOutput()
		printFirst(gpuOutTrained, 4, "GPU trained output (first 4):")

		// Compare trained outputs
		diff = compareOutputs(cpuOutTrained, gpuOutTrained)
		fmt.Printf("\nMax difference in trained outputs: %.15e\n", diff)
		if diff < 1e-3 {
			fmt.Println("✓ GPU training matches CPU training (within tolerance)")
		} else {
			fmt.Println("✗ GPU training differs from CPU training")
		}

		// Evaluate GPU
		gpuEval := evaluate(config.name+"_gpu", gpuNet, test)
		fmt.Printf("\nGPU Accuracy: %.2f%%\n", 100*gpuEval.acc)

		// Compare accuracies
		fmt.Printf("\n--- Accuracy Comparison ---\n")
		fmt.Printf("CPU: %.2f%%\n", 100*cpuEval.acc)
		fmt.Printf("GPU: %.2f%%\n", 100*gpuEval.acc)
		fmt.Printf("Difference: %.2f%%\n", 100*(gpuEval.acc-cpuEval.acc))

		// Show confusion matrices
		printConfusion(config.name+"_cpu", cpuEval.cm)
		printConfusion(config.name+"_gpu", gpuEval.cm)
	}

	// Final summary
	fmt.Printf("\n========================================\n")
	fmt.Printf("FINAL COMPARISON - ALL MODELS\n")
	fmt.Printf("========================================\n\n")

	// Reload all models and evaluate
	fmt.Println("Loading all saved models for final evaluation...\n")

	for _, config := range configs {
		cpuPath := filepath.Join(outDir, config.name+"_cpu.bundle.json")
		gpuPath := filepath.Join(outDir, config.name+"_gpu.bundle.json")

		fmt.Printf("\n--- %s ---\n", config.name)

		// Load and evaluate CPU
		cpuNet := loadModel(cpuPath)
		cpuEval := evaluate(config.name+"_cpu", cpuNet, test)

		// Load and evaluate GPU
		gpuNet := loadModel(gpuPath)
		gpuEval := evaluate(config.name+"_gpu", gpuNet, test)

		// Display
		fmt.Printf("CPU Accuracy: %.2f%%\n", 100*cpuEval.acc)
		fmt.Printf("GPU Accuracy: %.2f%%\n", 100*gpuEval.acc)
		fmt.Printf("Match: %v\n", math.Abs(cpuEval.acc-gpuEval.acc) < 0.001)

		// Test forward pass
		testInput := test[0].X
		cpuNet.Forward(testInput)
		cpuOut := cpuNet.GetOutput()

		gpuNet.Forward(testInput) // Note: GPU disabled after reload
		gpuOut := gpuNet.GetOutput()

		diff := compareOutputs(cpuOut, gpuOut)
		fmt.Printf("Output max diff: %.15e\n", diff)
	}

	fmt.Println("\n✓ Comparison complete!")
}
