// engine.go
package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
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
		{Width: 8, Height: 8}, // input
		{Width: 4, Height: 4}, // hidden1
		{Width: 4, Height: 4}, // hidden2 (mix: dense/attn in columns)
		{Width: 4, Height: 1}, // logits
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
	dk          int // per-head dimension
	heads       int // number of heads
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
		Heads:       knobs.heads,
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
		net.UpdateADHDPerformance(res)

		cm[exp][pred]++
	}

	net.Performance.Score = net.ComputeFinalScore()

	return evalOut{
		name: name,
		acc:  float64(correct) / float64(len(test)),
		perf: net.Performance,
		cm:   cm,
	}
}

/* =========================
   SAVE / LOAD (bundle v3; no mid-training saves)
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

	// Dense baseline
	dense := buildDense[float32](sizes, acts, fully)

	// Variants × Heads
	type variant struct {
		base   string
		share  string // "layer"|"per-slice"
		norm   bool
		replay bool
		force  bool
	}
	var variants = []variant{
		{"attnLayer_noNorm", "layer", false, false, false},
		{"attnLayer_norm", "layer", true, false, false},
		{"attnPerSlice_noNorm", "per-slice", false, false, false},
		{"attnPerSlice_norm", "per-slice", true, false, false},
		{"attnLayer_norm_replay", "layer", true, true, true},
	}
	headsList := []int{1, 2, 3}
	// Choose per-head dk; keep total hidden dim modest: total_d = heads*dk
	perHeadDK := map[int]int{1: 64, 2: 32, 3: 24}

	type namedNet struct {
		name string
		net  *paragon.Network[float32]
	}
	var models []namedNet
	models = append(models, namedNet{"dense", dense})

	for _, h := range headsList {
		for _, v := range variants {
			name := fmt.Sprintf("%s-H%d", v.base, h)
			n := buildAttnMix[float32](sizes, acts, fully, attnKnobs[float32]{
				share:       v.share,
				useNorm:     v.norm,
				useReplay:   v.replay,
				forceReplay: v.force,
				posEncAmp:   0.02,
				normEps:     1e-6,
				dk:          perHeadDK[h],
				heads:       h,
			})
			models = append(models, namedNet{name, n})
		}
	}

	// Pre-training peek
	inPeek := test[0].X
	dense.Forward(inPeek)
	printFirst(dense.GetOutput(), 8, "Dense output (first 8):")
	// Also peek the first 3 attention models just for a sniff
	for i := 1; i <= 3 && i < len(models); i++ {
		models[i].net.Forward(inPeek)
		printFirst(models[i].net.GetOutput(), 8, models[i].name+" output (first 8):")
	}
	// Show slice types on a representative attn model (if present)
	if len(models) > 1 {
		fmt.Println("Layer 2 slice types (repr):", models[1].net.Layers[2].SliceTypes)
	}

	// Train all (parallel). No checkpointing during training.
	cfg := trainCfg{
		epochs: 28,
		lr0:    1e-2,
		lr1:    3e-3,
		clipHi: 1.0,
		clipLo: -1.0,
	}
	var wg sync.WaitGroup
	fmt.Println("\n=== Training (parallel) ===")
	for i := range models {
		seed := int64(100 + i*7)
		wg.Add(1)
		go trainNet[float32](models[i].net, train, trainCfg{
			epochs: cfg.epochs, lr0: cfg.lr0, lr1: cfg.lr1,
			clipHi: cfg.clipHi, clipLo: cfg.clipLo, seed: seed, name: models[i].name,
		}, &wg)
	}
	wg.Wait()

	// Post-training peek
	fmt.Println("\n=== Post-training ===")
	dense.Forward(inPeek)
	printFirst(dense.GetOutput(), 8, "Dense output (first 8):")
	for i := 1; i <= 3 && i < len(models); i++ {
		models[i].net.Forward(inPeek)
		printFirst(models[i].net.GetOutput(), 8, models[i].name+" output (first 8):")
	}

	// Evaluate all
	var evals []evalOut
	for _, m := range models {
		ev := evaluate(m.name, m.net, test)
		evals = append(evals, ev)
	}

	fmt.Printf("\n=== Accuracy (global two-query) ===\n")
	for _, ev := range evals {
		fmt.Printf("%-22s : %.2f%%\n", ev.name, 100*ev.acc)
	}

	// Buckets table
	names := make([]string, 0, len(evals))
	perfs := make([]*paragon.ADHDPerformance, 0, len(evals))
	for _, ev := range evals {
		names = append(names, ev.name)
		perfs = append(perfs, ev.perf)
	}
	printBucketCompare("ADHD bucket comparison", names, perfs)

	// Confusion matrices (brief: show dense + H1/H2/H3 variants of one base)
	printConfusion("dense", evals[0].cm)
	// Find and print a few representative CMs
	for _, base := range []string{"attnLayer_noNorm", "attnLayer_norm", "attnPerSlice_norm"} {
		for _, h := range headsList {
			target := fmt.Sprintf("%s-H%d", base, h)
			for _, ev := range evals {
				if ev.name == target {
					printConfusion(ev.name, ev.cm)
					break
				}
			}
		}
	}

	// Save & reload all once; re-eval to prove persistence
	outDir := "bundles_v3"
	_ = os.MkdirAll(outDir, 0o755)
	var reloaded []namedNet
	for _, m := range models {
		path := filepath.Join(outDir, m.name+".bundle.json")
		rt := saveReload(path, m.name, m.net)
		reloaded = append(reloaded, namedNet{m.name + "-RT", rt})
	}

	var ree []evalOut
	for _, m := range reloaded {
		ree = append(ree, evaluate(m.name, m.net, test))
	}

	fmt.Printf("\n=== Accuracy after reload ===\n")
	for _, ev := range ree {
		fmt.Printf("%-22s : %.2f%%\n", ev.name, 100*ev.acc)
	}

	// Buckets (reloaded)
	namesRT := make([]string, 0, len(ree))
	perfsRT := make([]*paragon.ADHDPerformance, 0, len(ree))
	for _, ev := range ree {
		namesRT = append(namesRT, ev.name)
		perfsRT = append(perfsRT, ev.perf)
	}
	printBucketCompare("ADHD bucket comparison (reloaded)", namesRT, perfsRT)

	fmt.Printf("\nScores / Failures (reloaded):\n")
	for _, ev := range ree {
		fmt.Printf("%-26s -> score=%.3f  failures=%d\n", ev.name, ev.perf.Score, ev.perf.Failures)
	}
}
