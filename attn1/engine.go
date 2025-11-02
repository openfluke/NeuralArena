// engine.go
package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"

	"paragon"
)

/* =========================
   DATA: global two-query (8x8)
   =========================
   Classes:
     0: same row
     1: same column
     2: same main-diagonal (r-c == r2-c2)
     3: otherwise
*/

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

func bucketMap(p *paragon.ADHDPerformance) map[string]int {
	m := map[string]int{}
	for k, b := range p.Buckets {
		m[k] = b.Count
	}
	return m
}

func printBucketCompare(title string, names []string, perfs []*paragon.ADHDPerformance) {
	keys := []string{"0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%", "100%+"}
	fmt.Println("\n" + title)

	// header
	fmt.Printf("Bucket     |")
	for _, nm := range names {
		fmt.Printf(" %14s |", nm)
	}
	fmt.Println()
	fmt.Print("-----------+")
	for range names {
		fmt.Print("---------------+")
	}
	fmt.Println()

	// rows
	maps := make([]map[string]int, len(perfs))
	for i, p := range perfs {
		maps[i] = bucketMap(p)
	}
	for _, k := range keys {
		fmt.Printf("%-10s |", k)
		for i := range perfs {
			fmt.Printf(" %14d |", maps[i][k])
		}
		fmt.Println()
	}

	// footer totals
	fmt.Print("-----------+")
	for range names {
		fmt.Print("---------------+")
	}
	fmt.Println()

	fmt.Printf("Total      |")
	for _, p := range perfs {
		fmt.Printf(" %14d |", p.Total)
	}
	fmt.Println()
	fmt.Printf("Failures   |")
	for _, p := range perfs {
		fmt.Printf(" %14d |", p.Failures)
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
		{Width: 8, Height: 8},
		{Width: 4, Height: 4},
		{Width: 4, Height: 4},
		{Width: 4, Height: 1},
	}
	acts := []string{"relu", "relu", "relu", "softmax"}
	fully := []bool{false, true, true, true}
	return sizes, acts, fully
}

type attnKnobs[T paragon.Numeric] struct {
	share       string // "layer" | "per-slice"
	useNorm     bool
	useReplay   bool
	forceReplay bool
	posEncAmp   float64
	normEps     float64
	dk          int
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
	// Per-slice pattern on the attention layer (index 2)
	// If share=="layer", this still gives dense columns + attn columns in the same layer.
	slices := make([][]string, len(sizes))
	slices[2] = []string{"dense", "attn", "dense", "attn"}

	attnCfg := make([]*paragon.AttnConfig[T], len(sizes))
	attnCfg[2] = &paragon.AttnConfig[T]{
		DK:          knobs.dk,
		UseWo:       true,
		Share:       knobs.share, // "layer" or "per-slice"
		Dropout:     0.0,
		PosEnc2D:    true,
		UseNorm:     knobs.useNorm,
		PosEncAmp:   knobs.posEncAmp,
		NormEps:     knobs.normEps,
		UseReplay:   knobs.useReplay,
		ForceReplay: knobs.forceReplay,
		ReplayGain:  1.1,
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
}

func cosineLR(t, T int, lr0, lr1 float64) float64 {
	ct := (1 - math.Cos(math.Pi*float64(t)/float64(T))) / 2
	return lr0*(1-ct) + lr1*ct
}

func trainNet[T paragon.Numeric](net *paragon.Network[T], data []sample, cfg trainCfg, wg *sync.WaitGroup) {
	defer wg.Done()
	permRng := rand.New(rand.NewSource(cfg.seed))
	for e := 0; e < cfg.epochs; e++ {
		lr := cosineLR(e, cfg.epochs-1, cfg.lr0, cfg.lr1)
		total := 0.0
		perm := permRng.Perm(len(data))
		for _, i := range perm {
			net.Forward(data[i].X)
			total += net.ComputeLoss(data[i].Y)
			// Backward expects clip bounds as T; cast explicitly.
			net.Backward(data[i].Y, lr, T(cfg.clipHi), T(cfg.clipLo))
		}
		fmt.Printf("Epoch %02d (%s): loss=%.4f lr=%.4g\n", e, cfg.name, total/float64(len(data)), lr)
	}
}

type evalOut struct {
	name string
	acc  float64
	perf *paragon.ADHDPerformance
	cm   [4][4]int
}

func evaluate[T paragon.Numeric](name string, net *paragon.Network[T], test []sample) evalOut {
	// Ensure the internal perf struct exists (important after reloads)
	net.Performance = paragon.NewADHDPerformance()

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

		res := net.EvaluatePrediction(float64(exp), float64(pred))
		// This calls into net.Performance (now guaranteed non-nil)
		net.UpdateADHDPerformance(res)

		cm[exp][pred]++
	}

	// Finalise score inside the net (the print helper reads from perf)
	net.Performance.Score = net.ComputeFinalScore()

	return evalOut{
		name: name,
		acc:  float64(correct) / float64(len(test)),
		perf: net.Performance, // return the same one the net updated
		cm:   cm,
	}
}

/* =========================
   SAVE / LOAD (bundle v2)
   ========================= */

func saveReload(path, id string, n *paragon.Network[float32]) *paragon.Network[float32] {
	js, err := paragon.ExportBundleJSON[float32](id, n, nil)
	if err != nil {
		panic(err)
	}
	if err := os.WriteFile(path, []byte(js), 0o644); err != nil {
		panic(err)
	}
	anyNet, err := paragon.ImportBundleJSON(string(js))
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
   MAIN
   ========================= */

func main() {
	const smooth = 0.05
	dataRng := mkFixedRng(42)
	train := buildGlobalTwoQuery(4096, dataRng, smooth)
	test := buildGlobalTwoQuery(1024, dataRng, smooth)

	// Shapes
	sizes, acts, fully := buildShapes()

	// Build 6 models
	dense := buildDense[float32](sizes, acts, fully)

	attnLayer_noNorm := buildAttnMix[float32](sizes, acts, fully, attnKnobs[float32]{
		share: "layer", useNorm: false, useReplay: false, forceReplay: false, posEncAmp: 0.02, normEps: 1e-6, dk: 64,
	})
	attnLayer_norm := buildAttnMix[float32](sizes, acts, fully, attnKnobs[float32]{
		share: "layer", useNorm: true, useReplay: false, forceReplay: false, posEncAmp: 0.02, normEps: 1e-6, dk: 64,
	})
	attnPerSlice_noNorm := buildAttnMix[float32](sizes, acts, fully, attnKnobs[float32]{
		share: "per-slice", useNorm: false, useReplay: false, forceReplay: false, posEncAmp: 0.02, normEps: 1e-6, dk: 64,
	})
	attnPerSlice_norm := buildAttnMix[float32](sizes, acts, fully, attnKnobs[float32]{
		share: "per-slice", useNorm: true, useReplay: false, forceReplay: false, posEncAmp: 0.02, normEps: 1e-6, dk: 64,
	})
	attnLayer_norm_replay := buildAttnMix[float32](sizes, acts, fully, attnKnobs[float32]{
		share: "layer", useNorm: true, useReplay: true, forceReplay: true, posEncAmp: 0.02, normEps: 1e-6, dk: 64,
	})

	// Pre-training peek
	inPeek := test[0].X
	dense.Forward(inPeek)
	attnLayer_noNorm.Forward(inPeek)
	attnLayer_norm.Forward(inPeek)
	attnPerSlice_noNorm.Forward(inPeek)
	attnPerSlice_norm.Forward(inPeek)
	attnLayer_norm_replay.Forward(inPeek)

	printFirst(dense.GetOutput(), 8, "Dense output (first 8):")
	printFirst(attnLayer_noNorm.GetOutput(), 8, "Attn layer (no norm) output (first 8):")
	printFirst(attnLayer_norm.GetOutput(), 8, "Attn layer (with norm) output (first 8):")
	fmt.Println("Layer 2 slice types (attn layer):", attnLayer_norm.Layers[2].SliceTypes)

	// Train all (parallel)
	cfg := trainCfg{
		epochs: 28,
		lr0:    1e-2,
		lr1:    3e-3,
		clipHi: 1.0,
		clipLo: -1.0,
		seed:   2025,
	}
	var wg sync.WaitGroup
	run := func(n *paragon.Network[float32], name string, seed int64) {
		wg.Add(1)
		go trainNet[float32](n, train, trainCfg{
			epochs: cfg.epochs, lr0: cfg.lr0, lr1: cfg.lr1,
			clipHi: cfg.clipHi, clipLo: cfg.clipLo, seed: seed, name: name,
		}, &wg)
	}
	fmt.Println("\n=== Training (parallel) ===")
	run(dense, "dense", 11)
	run(attnLayer_noNorm, "attnLayer_noNorm", 22)
	run(attnLayer_norm, "attnLayer_norm", 33)
	run(attnPerSlice_noNorm, "attnPerSlice_noNorm", 44)
	run(attnPerSlice_norm, "attnPerSlice_norm", 55)
	run(attnLayer_norm_replay, "attnLayer_norm_replay", 66)
	wg.Wait()

	// Post-training peek
	dense.Forward(inPeek)
	attnLayer_noNorm.Forward(inPeek)
	attnLayer_norm.Forward(inPeek)
	attnPerSlice_noNorm.Forward(inPeek)
	attnPerSlice_norm.Forward(inPeek)
	attnLayer_norm_replay.Forward(inPeek)
	fmt.Println("\n=== Post-training ===")
	printFirst(dense.GetOutput(), 8, "Dense output (first 8):")
	printFirst(attnLayer_noNorm.GetOutput(), 8, "Attn layer (no norm) output (first 8):")
	printFirst(attnLayer_norm.GetOutput(), 8, "Attn layer (with norm) output (first 8):")

	// Evaluate
	eDense := evaluate("dense", dense, test)
	eLay0 := evaluate("attnLayer_noNorm", attnLayer_noNorm, test)
	eLayN := evaluate("attnLayer_norm", attnLayer_norm, test)
	eSlice0 := evaluate("attnPerSlice_noNorm", attnPerSlice_noNorm, test)
	eSliceN := evaluate("attnPerSlice_norm", attnPerSlice_norm, test)
	eLayNR := evaluate("attnLayer_norm_replay", attnLayer_norm_replay, test)

	fmt.Printf("\n=== Accuracy (global two-query) ===\n")
	fmt.Printf("dense                 : %.2f%%\n", 100*eDense.acc)
	fmt.Printf("attnLayer_noNorm      : %.2f%%\n", 100*eLay0.acc)
	fmt.Printf("attnLayer_norm        : %.2f%%\n", 100*eLayN.acc)
	fmt.Printf("attnPerSlice_noNorm   : %.2f%%\n", 100*eSlice0.acc)
	fmt.Printf("attnPerSlice_norm     : %.2f%%\n", 100*eSliceN.acc)
	fmt.Printf("attnLayer_norm_replay : %.2f%%\n", 100*eLayNR.acc)

	printBucketCompare(
		"ADHD bucket comparison",
		[]string{
			"dense",
			"attnLayer_noNorm",
			"attnLayer_norm",
			"attnPerSlice_noNorm",
			"attnPerSlice_norm",
			"attnLayer_norm_replay",
		},
		[]*paragon.ADHDPerformance{
			eDense.perf, eLay0.perf, eLayN.perf, eSlice0.perf, eSliceN.perf, eLayNR.perf,
		},
	)
	printConfusion(eDense.name, eDense.cm)
	printConfusion(eLay0.name, eLay0.cm)
	printConfusion(eLayN.name, eLayN.cm)
	printConfusion(eSlice0.name, eSlice0.cm)
	printConfusion(eSliceN.name, eSliceN.cm)
	printConfusion(eLayNR.name, eLayNR.cm)

	// Save & reload all; re-eval to prove persistence
	denseRT := saveReload("dense_v2.bundle.json", "dense_v2", dense)
	lay0RT := saveReload("attn_layer_noNorm_v2.bundle.json", "attn_layer_noNorm_v2", attnLayer_noNorm)
	layNRT := saveReload("attn_layer_norm_v2.bundle.json", "attn_layer_norm_v2", attnLayer_norm)
	slice0RT := saveReload("attn_perSlice_noNorm_v2.bundle.json", "attn_perSlice_noNorm_v2", attnPerSlice_noNorm)
	sliceNRT := saveReload("attn_perSlice_norm_v2.bundle.json", "attn_perSlice_norm_v2", attnPerSlice_norm)
	layNRRT := saveReload("attn_layer_norm_replay_v2.bundle.json", "attn_layer_norm_replay_v2", attnLayer_norm_replay)

	reDense := evaluate("dense-RT", denseRT, test)
	reLay0 := evaluate("attnLayer_noNorm-RT", lay0RT, test)
	reLayN := evaluate("attnLayer_norm-RT", layNRT, test)
	reSlice0 := evaluate("attnPerSlice_noNorm-RT", slice0RT, test)
	reSliceN := evaluate("attnPerSlice_norm-RT", sliceNRT, test)
	reLayNR := evaluate("attnLayer_norm_replay-RT", layNRRT, test)

	fmt.Printf("\n=== Accuracy after reload ===\n")
	fmt.Printf("dense-RT              : %.2f%%\n", 100*reDense.acc)
	fmt.Printf("attnLayer_noNorm-RT   : %.2f%%\n", 100*reLay0.acc)
	fmt.Printf("attnLayer_norm-RT     : %.2f%%\n", 100*reLayN.acc)
	fmt.Printf("attnPerSlice_noNorm-RT: %.2f%%\n", 100*reSlice0.acc)
	fmt.Printf("attnPerSlice_norm-RT  : %.2f%%\n", 100*reSliceN.acc)
	fmt.Printf("attnLayer_norm_replay-RT: %.2f%%\n", 100*reLayNR.acc)

	printBucketCompare(
		"ADHD bucket comparison (reloaded)",
		[]string{
			"dense-RT",
			"attnLayer_noNorm-RT",
			"attnLayer_norm-RT",
			"attnPerSlice_noNorm-RT",
			"attnPerSlice_norm-RT",
			"attnLayer_norm_replay-RT",
		},
		[]*paragon.ADHDPerformance{
			reDense.perf, reLay0.perf, reLayN.perf, reSlice0.perf, reSliceN.perf, reLayNR.perf,
		},
	)

	fmt.Printf("\nScores / Failures (reloaded):\n")
	fmt.Printf("dense-RT                    -> score=%.3f  failures=%d\n", reDense.perf.Score, reDense.perf.Failures)
	fmt.Printf("attnLayer_noNorm-RT         -> score=%.3f  failures=%d\n", reLay0.perf.Score, reLay0.perf.Failures)
	fmt.Printf("attnLayer_norm-RT           -> score=%.3f  failures=%d\n", reLayN.perf.Score, reLayN.perf.Failures)
	fmt.Printf("attnPerSlice_noNorm-RT      -> score=%.3f  failures=%d\n", reSlice0.perf.Score, reSlice0.perf.Failures)
	fmt.Printf("attnPerSlice_norm-RT        -> score=%.3f  failures=%d\n", reSliceN.perf.Score, reSliceN.perf.Failures)
	fmt.Printf("attnLayer_norm_replay-RT    -> score=%.3f  failures=%d\n", reLayNR.perf.Score, reLayNR.perf.Failures)
}
