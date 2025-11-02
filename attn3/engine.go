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

func printFull(out []float64, label string) {
	fmt.Println(label)
	for i := range out {
		if i%8 == 0 && i != 0 {
			fmt.Println()
		}
		fmt.Printf(" %.15e", out[i])
	}
	fmt.Println()
}

func forwardReturn[T paragon.Numeric](n *paragon.Network[T], X [][]float64) []float64 {
	n.Forward(X)
	return n.GetOutput()
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

/* ===================== deterministic inputs ===================== */

func zeroInput(h, w int) [][]float64 {
	X := make([][]float64, h)
	for r := 0; r < h; r++ {
		X[r] = make([]float64, w)
	}
	return X
}

func fixedTwoQuery(h, w int) [][]float64 {
	X := zeroInput(h, w)
	X[1][6] = 1
	X[4][2] = 1
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

/* ===================== model builders (same layout) ===================== */

func buildShapes() ([]paragon.GridSpec, []string, []bool) {
	// 8x8 -> 4x4 -> 4x4 -> 1x4
	return []paragon.GridSpec{
			{Width: 8, Height: 8},
			{Width: 4, Height: 4},
			{Width: 4, Height: 4}, // attn/dense mix
			{Width: 4, Height: 1},
		},
		[]string{"relu", "relu", "relu", "softmax"},
		[]bool{false, true, true, true}
}

type attnKnobs[T paragon.Numeric] struct {
	share       string // "layer" | "per-slice"
	useNorm     bool
	useReplay   bool
	forceReplay bool
	posEncAmp   float64
	normEps     float64
	dk          int
	heads       int
}

func buildDense[T paragon.Numeric]() *paragon.Network[T] {
	sizes, acts, fully := buildShapes()
	n, err := paragon.BuildGridNet[T](paragon.BuildOpts[T]{
		Sizes: sizes, Activations: acts, FullyConn: fully,
	})
	if err != nil {
		panic(err)
	}
	return n
}

func buildAttnMix[T paragon.Numeric](k attnKnobs[T]) *paragon.Network[T] {
	sizes, acts, fully := buildShapes()
	// layer 2 mix: dense/attn/dense/attn (4 columns)
	slices := make([][]string, len(sizes))
	slices[2] = []string{"dense", "attn", "dense", "attn"}

	attn := make([]*paragon.AttnConfig[T], len(sizes))
	attn[2] = &paragon.AttnConfig[T]{
		Heads:       k.heads,
		DK:          k.dk,
		UseWo:       true,
		Share:       k.share,
		Dropout:     0,
		PosEnc2D:    true,
		UseNorm:     k.useNorm,
		PosEncAmp:   k.posEncAmp,
		NormEps:     k.normEps,
		UseReplay:   k.useReplay,
		ForceReplay: k.forceReplay,
		ReplayGain:  1.1,
	}

	n, err := paragon.BuildGridNet[T](paragon.BuildOpts[T]{
		Sizes: sizes, Activations: acts, FullyConn: fully,
		SliceTypes: slices, Attn: attn,
	})
	if err != nil {
		panic(err)
	}
	return n
}

/* ===================== save / load ===================== */

func saveBundle(path, id string, n *paragon.Network[float32]) {
	js, err := paragon.ExportBundleJSON[float32](id, n, nil)
	if err != nil {
		panic(err)
	}
	if err := os.WriteFile(path, []byte(js), 0o644); err != nil {
		panic(err)
	}
}

func loadBundle(path string) (*paragon.Network[float32], time.Duration, error) {
	start := time.Now()
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	anyNet, err := paragon.ImportBundleJSON(string(b))
	if err != nil {
		return nil, 0, err
	}
	rt, ok := anyNet.(*paragon.Network[float32])
	if !ok {
		return nil, 0, fmt.Errorf("bundle type mismatch")
	}
	return rt, time.Since(start), nil
}

/* ===================== CPU vs GPU test ===================== */

type cpuGpuOut struct {
	outZero []float64
	out2Q   []float64
	outRand []float64
	tZero   time.Duration
	t2Q     time.Duration
	tRand   time.Duration
}

func runCPU(n *paragon.Network[float32], X0, X2Q, XR [][]float64) cpuGpuOut {
	n.WebGPUNative = false
	t0 := time.Now()
	o0 := forwardReturn(n, X0)
	t1 := time.Now()
	o2 := forwardReturn(n, X2Q)
	t2 := time.Now()
	or := forwardReturn(n, XR)
	t3 := time.Now()
	return cpuGpuOut{
		outZero: o0, out2Q: o2, outRand: or,
		tZero: t1.Sub(t0), t2Q: t2.Sub(t1), tRand: t3.Sub(t2),
	}
}

type gpuResult struct {
	cpuGpuOut
	initTime time.Duration
}

func runGPU(n *paragon.Network[float32], X0, X2Q, XR [][]float64) gpuResult {
	n.WebGPUNative = true

	// Initialize optimized GPU explicitly to time just the init cost
	startInit := time.Now()
	if err := n.InitializeOptimizedGPU(); err != nil {
		// fall back noted in output
		return gpuResult{
			cpuGpuOut: cpuGpuOut{},
			initTime:  -1,
		}
	}
	initDur := time.Since(startInit)

	// First run (X0)
	t0 := time.Now()
	o0 := forwardReturn(n, X0)
	t1 := time.Now()
	o2 := forwardReturn(n, X2Q)
	t2 := time.Now()
	or := forwardReturn(n, XR)
	t3 := time.Now()

	return gpuResult{
		cpuGpuOut: cpuGpuOut{
			outZero: o0, out2Q: o2, outRand: or,
			tZero: t1.Sub(t0), t2Q: t2.Sub(t1), tRand: t3.Sub(t2),
		},
		initTime: initDur,
	}
}

/* ===================== main ===================== */

func main() {
	outDir := "bundles_v3"
	_ = os.MkdirAll(outDir, 0o755)

	type spec struct {
		name string
		make func() *paragon.Network[float32]
	}

	perHeadDK := map[int]int{1: 64, 2: 32, 3: 24}
	var variants []spec

	// dense baseline
	variants = append(variants, spec{
		name: "dense",
		make: func() *paragon.Network[float32] { return buildDense[float32]() },
	})

	// attention variants across heads 1,2,3
	type vcfg struct {
		base, share         string
		norm, replay, force bool
	}
	bases := []vcfg{
		{"attnLayer_noNorm", "layer", false, false, false},
		{"attnLayer_norm", "layer", true, false, false},
		{"attnPerSlice_noNorm", "per-slice", false, false, false},
		{"attnPerSlice_norm", "per-slice", true, false, false},
		{"attnLayer_norm_replay", "layer", true, true, true},
	}
	for _, h := range []int{1, 2, 3} {
		for _, b := range bases {
			name := fmt.Sprintf("%s-H%d", b.base, h)
			hh := h
			v := b
			variants = append(variants, spec{
				name: name,
				make: func() *paragon.Network[float32] {
					return buildAttnMix[float32](attnKnobs[float32]{
						share: v.share, useNorm: v.norm,
						useReplay: v.replay, forceReplay: v.force,
						posEncAmp: 0.02, normEps: 1e-6,
						dk: perHeadDK[hh], heads: hh,
					})
				},
			})
		}
	}

	// Inputs
	X0 := zeroInput(8, 8)
	X2Q := fixedTwoQuery(8, 8)
	XR := randInput(8, 8, 12345)

	fmt.Println("=== CPU vs GPU timings + output deltas (load if exists, else build+save) ===")
	for _, sp := range variants {
		path := filepath.Join(outDir, sp.name+".bundle.json")

		// Build & save once if missing
		if _, err := os.Stat(path); err != nil {
			net := sp.make()
			_ = forwardReturn(net, X0)
			saveBundle(path, sp.name, net)
		}

		// Load CPU copy
		cpuNet, loadDurCPU, err := loadBundle(path)
		if err != nil {
			fmt.Printf("[%s] load error: %v\n", sp.name, err)
			continue
		}

		// Load GPU copy
		gpuNet, loadDurGPU, err := loadBundle(path)
		if err != nil {
			fmt.Printf("[%s] GPU load error: %v\n", sp.name, err)
			continue
		}

		// Run CPU
		cpuRes := runCPU(cpuNet, X0, X2Q, XR)

		// Run GPU
		gpuRes := runGPU(gpuNet, X0, X2Q, XR)

		fmt.Printf("\n[%s]\n", sp.name)
		fmt.Printf(" load(cpu)=%v  load(gpu)=%v\n", loadDurCPU, loadDurGPU)
		if gpuRes.initTime >= 0 {
			fmt.Printf(" init(gpu)=%v\n", gpuRes.initTime)
		} else {
			fmt.Printf(" init(gpu)=FAILED (falling back would be handled inside Forward, but test skips GPU compare)\n")
		}

		// CPU output + hashes
		fmt.Printf(" CPU forward: zero=%v  twoQ=%v  rand=%v\n",
			cpuRes.tZero, cpuRes.t2Q, cpuRes.tRand)
		fmt.Printf("   hashes: zero=%s twoQ=%s rand=%s\n",
			sha12(cpuRes.outZero), sha12(cpuRes.out2Q), sha12(cpuRes.outRand))

		// 🔊 PRINT FULL CPU VECTORS
		printFull(cpuRes.outZero, " CPU zero:")
		printFull(cpuRes.out2Q, " CPU twoQ:")
		printFull(cpuRes.outRand, " CPU rand:")

		// GPU timing + hashes (if ok)
		if gpuRes.initTime >= 0 {
			fmt.Printf(" GPU forward: zero=%v  twoQ=%v  rand=%v\n",
				gpuRes.tZero, gpuRes.t2Q, gpuRes.tRand)
			fmt.Printf("   hashes: zero=%s twoQ=%s rand=%s\n",
				sha12(gpuRes.outZero), sha12(gpuRes.out2Q), sha12(gpuRes.outRand))

			// 🔊 PRINT FULL GPU VECTORS
			printFull(gpuRes.outZero, " GPU zero:")
			printFull(gpuRes.out2Q, " GPU twoQ:")
			printFull(gpuRes.outRand, " GPU rand:")

			// Deltas
			d0 := maxAbsDiff(cpuRes.outZero, gpuRes.outZero)
			d2 := maxAbsDiff(cpuRes.out2Q, gpuRes.out2Q)
			dr := maxAbsDiff(cpuRes.outRand, gpuRes.outRand)
			fmt.Printf(" max|Δ|   : zero=%.9g  twoQ=%.9g  rand=%.9g\n", d0, d2, dr)
		} else {
			fmt.Println(" (GPU forward skipped; no deltas)")
		}
	}

	fmt.Println("\nDone.")
}
