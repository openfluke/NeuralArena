// engine.go
package main

import (
	"fmt"
	"math/rand"

	"paragon"
)

/* ===== data: global two-query relation =====
   Pick two active cells in an 8×8 grid.
   Classes (4-way):
     0: same row
     1: same column
     2: same main-diagonal (r-c == r2-c2)
     3: otherwise
   This forces long-range pair interactions; attention helps.
*/

type sample struct {
	X [][]float64 // 8x8 input
	Y [][]float64 // 1x4 one-hot target
}

func oneHot4(k int) [][]float64 {
	y := make([][]float64, 1)
	y[0] = []float64{0, 0, 0, 0}
	if k >= 0 && k < 4 {
		y[0][k] = 1
	}
	return y
}

func buildGlobalTwoQuery(n int, rng *rand.Rand) []sample {
	const H, W = 8, 8
	out := make([]sample, n)
	for i := 0; i < n; i++ {
		X := make([][]float64, H)
		for r := 0; r < H; r++ {
			X[r] = make([]float64, W)
		}
		// choose two distinct positions
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
		out[i] = sample{X: X, Y: oneHot4(cls)}
	}
	return out
}

/* ===== helpers ===== */

func mkFixedRng(seed int64) *rand.Rand { return rand.New(rand.NewSource(seed)) }

// tiny printer
func printFirst(out []float64, n int, label string) {
	fmt.Println(label)
	for i := 0; i < len(out) && i < n; i++ {
		fmt.Printf(" %.15e", out[i])
	}
	fmt.Println()
}

func bucketCounts(p *paragon.ADHDPerformance) map[string]int {
	m := map[string]int{}
	for k, b := range p.Buckets {
		m[k] = b.Count
	}
	return m
}

func printBucketCompare(title string, a, b *paragon.ADHDPerformance) {
	ad := bucketCounts(a)
	bd := bucketCounts(b)
	keys := []string{"0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%", "100%+"}
	fmt.Println()
	fmt.Println(title)
	fmt.Println("Bucket     |        Dense |         Attn | Δ")
	fmt.Println("------------+--------------+--------------+------")
	for _, k := range keys {
		av := ad[k]
		bv := bd[k]
		fmt.Printf("%-10s | %12d | %12d | %+d\n", k, av, bv, bv-av)
	}
	fmt.Println("------------+--------------+--------------+------")
	fmt.Printf("Total      | %12d | %12d | %+d\n", a.Total, b.Total, b.Total-a.Total)
	fmt.Printf("Failures   | %12d | %12d | %+d\n", a.Failures, b.Failures, b.Failures-a.Failures)
}

/* ===== main ===== */

func main() {
	// deterministic data
	dataRng := mkFixedRng(1337)

	// ---- model shapes: 8x8 -> 4x4 -> 4x4 -> 1x4 ----
	sizes := []paragon.GridSpec{
		{Width: 8, Height: 8}, // input
		{Width: 4, Height: 4}, // hidden1
		{Width: 4, Height: 4}, // hidden2 (we'll add attn here for one model)
		{Width: 4, Height: 1}, // output 1x4
	}
	acts := []string{"relu", "relu", "relu", "softmax"}
	fully := []bool{false, true, true, true}

	// ---- build two models: dense-only vs attn-second-hidden ----
	dense := paragon.MustLoadOrCreateBundle[float32](
		"dense_2hid.bundle.json",
		"dense_2hid_v1",
		func() (*paragon.Network[float32], error) {
			return paragon.BuildGridNet[float32](paragon.BuildOpts[float32]{
				Sizes:       sizes,
				Activations: acts,
				FullyConn:   fully,
			})
		},
	)

	attn := paragon.MustLoadOrCreateBundle[float32](
		"attn_2hid.bundle.json",
		"attn_2hid_v1",
		func() (*paragon.Network[float32], error) {
			// all dense except layer 2 (index 2) where we mark alternate columns as attn
			slices := make([][]string, len(sizes))
			// hidden2 slice types: [dense, attn, dense, attn]
			slices[2] = []string{"dense", "attn", "dense", "attn"}

			attnCfg := make([]*paragon.AttnConfig[float32], len(sizes))
			attnCfg[2] = &paragon.AttnConfig[float32]{
				DK:       32, // modest head size (works well at 4x4)
				UseWo:    true,
				Share:    "layer", // share params across attn columns
				Dropout:  0.0,
				PosEnc2D: true, // tiny bias toward spatial structure
			}

			return paragon.BuildGridNet[float32](paragon.BuildOpts[float32]{
				Sizes:       sizes,
				Activations: acts,
				FullyConn:   fully,
				SliceTypes:  slices,
				Attn:        attnCfg,
			})
		},
	)

	// ---- quick pre-training peek (same random sample) ----
	inPeek := buildGlobalTwoQuery(1, dataRng)[0].X
	dense.Forward(inPeek)
	attn.Forward(inPeek)
	printFirst(dense.GetOutput(), 8, "Dense-only output (first 8 vals):")
	printFirst(attn.GetOutput(), 8, "Attn-mixed output (first 8 vals):")
	fmt.Println("Layer 2 slice types (attn model):", attn.Layers[2].SliceTypes)

	// ---- build train/test ----
	const trainN = 4096
	const testN = 1024
	train := buildGlobalTwoQuery(trainN, dataRng)
	test := buildGlobalTwoQuery(testN, dataRng)

	// ---- train both (same epochs/LR/clip) ----
	const epochs = 24
	const lr = 1e-2
	clipUp, clipLo := float32(1.0), float32(-1.0)

	fmt.Println("\n=== Training Dense ===")
	for e := 0; e < epochs; e++ {
		// simple SGD pass
		total := 0.0
		perm := rand.New(rand.NewSource(int64(2020 + e))).Perm(len(train))
		for _, i := range perm {
			dense.Forward(train[i].X)
			total += dense.ComputeLoss(train[i].Y)
			dense.Backward(train[i].Y, lr, clipUp, clipLo)
		}
		fmt.Printf("Epoch %d, Loss: %.4f\n", e, total/float64(len(train)))
	}

	fmt.Println("\n=== Training Attn ===")
	for e := 0; e < epochs; e++ {
		total := 0.0
		perm := rand.New(rand.NewSource(int64(3030 + e))).Perm(len(train))
		for _, i := range perm {
			attn.Forward(train[i].X)
			total += attn.ComputeLoss(train[i].Y)
			attn.Backward(train[i].Y, lr, clipUp, clipLo)
		}
		fmt.Printf("Epoch %d, Loss: %.4f\n", e, total/float64(len(train)))
	}

	// ---- post-training check (same peek) ----
	dense.Forward(inPeek)
	attn.Forward(inPeek)
	fmt.Println("\n=== Post-training ===")
	printFirst(dense.GetOutput(), 8, "Dense-only output (first 8 vals):")
	printFirst(attn.GetOutput(), 8, "Attn-mixed output (first 8 vals):")

	// ---- ADHD eval on test set (bucketed comparison) ----
	// We’ll use argmax class predictions vs ground truth index.
	// Your ADHD evaluates % deviation; for classification we can
	// treat “expected” as the class index and “actual” as predicted index.
	dense.Performance = paragon.NewADHDPerformance()
	attn.Performance = paragon.NewADHDPerformance()

	denseIdx := func(probs []float64) int {
		mx, mi := probs[0], 0
		for i := 1; i < len(probs); i++ {
			if probs[i] > mx {
				mx, mi = probs[i], i
			}
		}
		return mi
	}

	for _, s := range test {
		// expected class
		expIdx := 0
		for j := 0; j < 4; j++ {
			if s.Y[0][j] > 0.5 {
				expIdx = j
				break
			}
		}

		dense.Forward(s.X)
		aOut := dense.GetOutput()
		aIdx := denseIdx(aOut)
		dense.UpdateADHDPerformance(dense.EvaluatePrediction(float64(expIdx), float64(aIdx)))

		attn.Forward(s.X)
		bOut := attn.GetOutput()
		bIdx := denseIdx(bOut)
		attn.UpdateADHDPerformance(attn.EvaluatePrediction(float64(expIdx), float64(bIdx)))
	}
	dense.Performance.Score = dense.ComputeFinalScore()
	attn.Performance.Score = attn.ComputeFinalScore()

	fmt.Printf("\n=== ADHD eval (global two-query) ===\n")
	fmt.Printf("Dense: score=%.3f total=%d failures=%d\n", dense.Performance.Score, dense.Performance.Total, dense.Performance.Failures)
	fmt.Printf("Attn:  score=%.3f total=%d failures=%d\n", attn.Performance.Score, attn.Performance.Total, attn.Performance.Failures)

	printBucketCompare("\nBucket comparison (counts):", dense.Performance, attn.Performance)
}
