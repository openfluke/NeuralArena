package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"paragon"
	"runtime"
	"sync"
	"time"
)

// -------------------------------------------------- data helpers you already have
const (
	baseURL   = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	mnistDir  = "mnist_data"
	modelDir  = "models"
	modelFile = "mnist_model.json"
)

// -------------------------------------------------- main
func main() {
	//singleCompare()
	//benchmarkReplayVsBaseline()
	//benchmarkReplayVsBaselineN()
	//benchmarkReplayDepths()
	//benchmarkMaxReplay()
	//benchmarkReplaySweetSpot()
	benchmarkReplayBeforeAfter()
}

func singleCompare() {
	// 1) ── MNIST ──────────────────────────────────────────────────────────────
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatalf("MNIST download error: %v", err)
	}
	trainX, trainY, _ := loadMNISTData(mnistDir, true)
	testX, testY, _ := loadMNISTData(mnistDir, false)
	trainX, trainY, _, _ = paragon.SplitDataset(trainX, trainY, 0.8)

	// 2) ── common architecture  ──────────────────────────────────────────────
	layer := []struct{ Width, Height int }{{28, 28}, {16, 16}, {10, 1}}
	acts := []string{"leaky_relu", "leaky_relu", "softmax"}
	fc := []bool{true, false, true}

	// 3) ── build baseline & replay nets ──────────────────────────────────────
	baseNet := paragon.NewNetwork(layer, acts, fc)

	replayNet := paragon.NewNetwork(layer, acts, fc)
	replayNet.Layers[1].ReplayOffset = -1
	replayNet.Layers[1].ReplayPhase = "after"
	replayNet.Layers[1].MaxReplay = 1

	// 4) ── train both ────────────────────────────────────────────────────────
	fmt.Println("🧠 Training baseline …")
	baseNet.Train(trainX, trainY, 20, 0.001, true)

	fmt.Println("🧠 Training replay   …")
	replayNet.Train(trainX, trainY, 20, 0.001, true)

	// 5) ── evaluation helper ─────────────────────────────────────────────────
	type results struct {
		score   float64
		buckets map[string]int
		exp     []float64
		pred    []float64
	}
	evaluate := func(net *paragon.Network) results {
		exp, pred := []float64{}, []float64{}
		for i, in := range testX {
			net.Forward(in)
			pred = append(pred, float64(paragon.ArgMax(net.ExtractOutput())))
			exp = append(exp, float64(paragon.ArgMax(testY[i][0])))
		}
		net.EvaluateModel(exp, pred)
		b := map[string]int{}
		for k, v := range net.Performance.Buckets {
			b[k] = v.Count
		}
		return results{net.Performance.Score, b, exp, pred}
	}

	baseRes := evaluate(baseNet)
	replayRes := evaluate(replayNet)

	// 6) ── compact comparison printout ───────────────────────────────────────
	fmt.Println("\n============== ADHD COMPARISON ==============")
	fmt.Printf("Metric                     | Baseline | Replay\n")
	fmt.Printf("---------------------------+----------+---------\n")
	fmt.Printf("ADHD Score                 | %8.2f | %7.2f\n",
		baseRes.score, replayRes.score)

	fmt.Println("\nDeviation buckets (# samples):")
	keys := []string{"0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
		"50-100%", "100%+"}
	for _, k := range keys {
		fmt.Printf(" %-7s | %4d | %4d\n",
			k, baseRes.buckets[k], replayRes.buckets[k])
	}

	// 7) ── (optional) full diagnostics dumps ─────────────────────────────────
	fmt.Println("\n------ FULL DIAGNOSTICS: BASELINE ----------")
	baseNet.EvaluateFull(baseRes.exp, baseRes.pred)
	baseNet.PrintFullDiagnostics()

	fmt.Println("\n------ FULL DIAGNOSTICS: REPLAY ------------")
	replayNet.EvaluateFull(replayRes.exp, replayRes.pred)
	replayNet.PrintFullDiagnostics()
}

// -----------------------------------------------------------------------------
// BENCHMARK: 10 baseline vs 10 replay models on MNIST
// -----------------------------------------------------------------------------
func benchmarkReplayVsBaseline() {
	//---------------------------------------------------------------------------
	// 0) load data once
	//---------------------------------------------------------------------------
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatalf("MNIST download error: %v", err)
	}
	trainX, trainY, _ := loadMNISTData(mnistDir, true)
	testX, testY, _ := loadMNISTData(mnistDir, false)
	trainX, trainY, _, _ = paragon.SplitDataset(trainX, trainY, 0.8)

	//---------------------------------------------------------------------------
	// 1) prepare channels / waitgroup
	//---------------------------------------------------------------------------
	type outcome struct {
		Idx  int
		Kind string // "baseline" or "replay"
		ADHD float64
		Acc  float64
	}
	outCh := make(chan outcome, 20)
	var wg sync.WaitGroup

	//---------------------------------------------------------------------------
	// 2) helper: build‑>train‑>evaluate a single model
	//---------------------------------------------------------------------------
	run := func(idx int, kind string) {
		defer wg.Done()

		// unique seed so every model starts with different weights
		rand.Seed(time.Now().UnixNano() + int64(idx)*31)

		layer := []struct{ Width, Height int }{{28, 28}, {16, 16}, {10, 1}}
		acts := []string{"leaky_relu", "leaky_relu", "softmax"}
		fc := []bool{true, false, true}

		net := paragon.NewNetwork(layer, acts, fc)
		if kind == "replay" {
			net.Layers[1].ReplayOffset = -1
			net.Layers[1].ReplayPhase = "after"
			net.Layers[1].MaxReplay = 1
		}

		net.Train(trainX, trainY, 20, 0.001, true)

		// evaluate
		exp, pred := make([]float64, len(testX)), make([]float64, len(testX))
		correct := 0
		for i, in := range testX {
			net.Forward(in)
			predVal := float64(paragon.ArgMax(net.ExtractOutput()))
			expVal := float64(paragon.ArgMax(testY[i][0]))
			pred[i] = predVal
			exp[i] = expVal
			if predVal == expVal {
				correct++
			}
		}
		net.EvaluateModel(exp, pred)
		acc := float64(correct) / float64(len(testX)) * 100.0

		outCh <- outcome{idx, kind, net.Performance.Score, acc}
	}

	//---------------------------------------------------------------------------
	// 3) launch 20 goroutines
	//---------------------------------------------------------------------------
	for i := 0; i < 10; i++ {
		wg.Add(2)
		go run(i, "baseline")
		go run(i, "replay")
	}

	go func() {
		wg.Wait()
		close(outCh)
	}()

	//---------------------------------------------------------------------------
	// 4) collect + print
	//---------------------------------------------------------------------------
	bScores, rScores := []outcome{}, []outcome{}
	for res := range outCh {
		if res.Kind == "baseline" {
			bScores = append(bScores, res)
		} else {
			rScores = append(rScores, res)
		}
	}

	// header
	fmt.Println("\n================ 10× BENCHMARK =================")
	fmt.Printf("%-4s | %-9s | %-8s | %-8s\n", "Run", "Kind", "ADHD", "Acc%")
	fmt.Println("----------------------------------------------")

	avgBadhd, avgBacc, avgRadhd, avgRacc := 0.0, 0.0, 0.0, 0.0
	for i := 0; i < 10; i++ {
		b := bScores[i]
		r := rScores[i]
		fmt.Printf("%-4d | %-9s | %8.2f | %7.2f\n", b.Idx, "baseline", b.ADHD, b.Acc)
		fmt.Printf("%-4d | %-9s | %8.2f | %7.2f\n", r.Idx, "replay", r.ADHD, r.Acc)
		fmt.Println("----------------------------------------------")
		avgBadhd += b.ADHD
		avgBacc += b.Acc
		avgRadhd += r.ADHD
		avgRacc += r.Acc
	}
	avgBadhd /= 10
	avgBacc /= 10
	avgRadhd /= 10
	avgRacc /= 10

	fmt.Printf("AVERAGE %-9s | %8.2f | %7.2f\n", "baseline", avgBadhd, avgBacc)
	fmt.Printf("AVERAGE %-9s | %8.2f | %7.2f\n", "replay", avgRadhd, avgRacc)
	fmt.Println("==============================================")
}

// --------------------------------------------------------------
// 100‑RUN BENCHMARK  (80 % CPU utilisation)
// --------------------------------------------------------------
func benchmarkReplayVsBaselineN() {
	const (
		NModels      = 100 // pairs → 200 total nets
		Epochs       = 20
		LearningRate = 0.001
	)

	//----------------------------------------------------------------------
	// 0) LOAD DATA ONCE
	//----------------------------------------------------------------------
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatal(err)
	}
	trainX, trainY, _ := loadMNISTData(mnistDir, true)
	testX, testY, _ := loadMNISTData(mnistDir, false)
	trainX, trainY, _, _ = paragon.SplitDataset(trainX, trainY, 0.8)

	//----------------------------------------------------------------------
	// 1) CONCURRENCY GOVERNOR  (80 % of cores)
	//----------------------------------------------------------------------
	maxThreads := int(0.8 * float64(runtime.NumCPU()))
	if maxThreads < 1 {
		maxThreads = 1
	}
	sem := make(chan struct{}, maxThreads) // counting semaphore

	type score struct{ ADHD, Acc float64 }
	bScores := make([]score, NModels)
	rScores := make([]score, NModels)

	var wg sync.WaitGroup
	var mu sync.Mutex // protect slices

	buildAndRun := func(idx int, replay bool) {
		defer wg.Done()
		sem <- struct{}{}        // acquire slot
		defer func() { <-sem }() // release

		// deterministic but unique seed
		rnd := rand.New(rand.NewSource(time.Now().UnixNano() + int64(idx)*113))

		layer := []struct{ Width, Height int }{{28, 28}, {16, 16}, {10, 1}}
		acts := []string{"leaky_relu", "leaky_relu", "softmax"}
		fc := []bool{true, false, true}
		net := paragon.NewNetwork(layer, acts, fc)
		if replay {
			net.Layers[1].ReplayOffset = -1
			net.Layers[1].ReplayPhase = "after"
			net.Layers[1].MaxReplay = 1
		}

		// shuffle the training set with this model’s RNG for independence
		shuffledX := make([][][]float64, len(trainX))
		shuffledY := make([][][]float64, len(trainY))
		perm := rnd.Perm(len(trainX))
		for i, p := range perm {
			shuffledX[i] = trainX[p]
			shuffledY[i] = trainY[p]
		}
		net.Train(shuffledX, shuffledY, Epochs, LearningRate, true)

		// evaluation
		exp, pred := make([]float64, len(testX)), make([]float64, len(testX))
		correct := 0
		for i, in := range testX {
			net.Forward(in)
			p := float64(paragon.ArgMax(net.ExtractOutput()))
			t := float64(paragon.ArgMax(testY[i][0]))
			pred[i], exp[i] = p, t
			if p == t {
				correct++
			}
		}
		net.EvaluateModel(exp, pred)
		accPc := float64(correct) / 100.0 // 10 000 → percent

		mu.Lock()
		if replay {
			rScores[idx] = score{net.Performance.Score, accPc}
		} else {
			bScores[idx] = score{net.Performance.Score, accPc}
		}
		mu.Unlock()
	}

	//----------------------------------------------------------------------
	// 2) LAUNCH GOROUTINES
	//----------------------------------------------------------------------
	for i := 0; i < NModels; i++ {
		wg.Add(2)
		go buildAndRun(i, false) // baseline
		go buildAndRun(i, true)  // replay
	}
	wg.Wait()

	//----------------------------------------------------------------------
	// 3) SORTED OUTPUT
	//----------------------------------------------------------------------
	fmt.Printf("\n================ %d× BENCHMARK (%.0f%% CPUs) ================\n",
		NModels, 100*0.8)
	fmt.Printf("Run | ADHD_bas | ADHD_rep | ΔADHD  | Acc%%_bas | Acc%%_rep | ΔAcc%%\n")
	fmt.Println("------------------------------------------------------------------")

	var sumBad, sumRad, sumBac, sumRac float64
	for i := 0; i < NModels; i++ {
		b, r := bScores[i], rScores[i]
		dAd := r.ADHD - b.ADHD
		dAc := r.Acc - b.Acc
		fmt.Printf("%3d | %8.2f | %8.2f | %+6.2f | %8.2f | %8.2f | %+6.2f\n",
			i, b.ADHD, r.ADHD, dAd, b.Acc, r.Acc, dAc)
		sumBad += b.ADHD
		sumRad += r.ADHD
		sumBac += b.Acc
		sumRac += r.Acc
	}
	n := float64(NModels)
	fmt.Println("------------------------------------------------------------------")
	fmt.Printf("AVG | %8.2f | %8.2f | %+6.2f | %8.2f | %8.2f | %+6.2f\n",
		sumBad/n, sumRad/n, (sumRad-sumBad)/n,
		sumBac/n, sumRac/n, (sumRac-sumBac)/n)
	fmt.Println("===============================================================")
}

// benchmarkReplayDepths runs 10 random inits for each combo of
//   - hidden‑layer count   hCnt ∈ {2,3,4}
//   - replay depth         rDepth ∈ {0 … hCnt}
//
// It trains with 80 % of logical cores and prints mean ± sd for ADHD & accuracy.
//
// Required imports in the same file:
//
//	import (
//	    "fmt"
//	    "log"
//	    "math"
//	    "math/rand"
//	    "runtime"
//	    "sync"
//	    "time"
//	    "paragon"
//	)
func benchmarkReplayDepths() {
	// ───────── benchmark settings ──────────────────────────────────────────
	const (
		nRuns  = 10
		epochs = 15
		lr     = 0.001
		// network I/O sizes
		inputW, inputH = 28, 28
		outputW        = 10
	)

	// hideDim picks width=height so total neurons ~ original 16×16×1 (=256)
	hiddenDim := map[int]int{
		2: 12, // 2‑hidden‑layer net → 12×12 each  → 2×144 = 288 neurons
		3: 10, // 3‑hidden‑layer net → 10×10 each  → 3×100 = 300
		4: 8,  // 4‑hidden‑layer net →  8× 8 each  → 4× 64 = 256
	}

	// ───────── load MNIST once ─────────────────────────────────────────────
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatal(err)
	}
	trainX, trainY, _ := loadMNISTData(mnistDir, true)
	testX, testY, _ := loadMNISTData(mnistDir, false)
	trainX, trainY, _, _ = paragon.SplitDataset(trainX, trainY, 0.8)

	// ───────── concurrency guard (≈ 80 % CPU) ──────────────────────────────
	maxWorkers := int(0.8 * float64(runtime.NumCPU()))
	if maxWorkers < 1 {
		maxWorkers = 1
	}
	sem := make(chan struct{}, maxWorkers)
	var wg sync.WaitGroup

	// ───────── result storage ──────────────────────────────────────────────
	type key struct{ hCnt, rDepth int }
	type metric struct{ adh, acc float64 }
	results := make(map[key][]metric)
	var mu sync.Mutex

	// ───────── worker: build → train → evaluate one model ──────────────────
	runModel := func(hCnt, rDepth, runIdx int) {
		defer wg.Done()
		sem <- struct{}{}        // acquire worker slot
		defer func() { <-sem }() // release slot

		// unique RNG per run
		seed := time.Now().UnixNano() + int64(hCnt)*1e6 + int64(rDepth)*1e4 + int64(runIdx)
		rnd := rand.New(rand.NewSource(seed))

		// ----- 1. construct layer sizes ------------------------------------
		hSize := hiddenDim[hCnt]
		layer := make([]struct{ Width, Height int }, 0, hCnt+2)
		layer = append(layer, struct{ Width, Height int }{inputW, inputH})
		for i := 0; i < hCnt; i++ {
			layer = append(layer, struct{ Width, Height int }{hSize, hSize})
		}
		layer = append(layer, struct{ Width, Height int }{outputW, 1})

		// activations & connectivity
		acts := make([]string, len(layer))
		for i := range acts {
			acts[i] = "leaky_relu"
		}
		acts[len(acts)-1] = "softmax"

		fc := make([]bool, len(layer))
		fc[0], fc[len(fc)-1] = true, true // full connect input & output
		// hidden layers use local connectivity (fc[i]=false)

		net := paragon.NewNetwork(layer, acts, fc)

		// ----- 2. set replay on first rDepth hidden layers ------------------
		for l := 1; l <= rDepth && l <= hCnt; l++ {
			net.Layers[l].ReplayOffset = -1 // replay previous layer
			net.Layers[l].ReplayPhase = "after"
			net.Layers[l].MaxReplay = 1
		}

		// ----- 3. shuffle training set with this RNG ------------------------
		shX := make([][][]float64, len(trainX))
		shY := make([][][]float64, len(trainY))
		perm := rnd.Perm(len(trainX))
		for i, p := range perm {
			shX[i], shY[i] = trainX[p], trainY[p]
		}

		// ----- 4. train -----------------------------------------------------
		net.Train(shX, shY, epochs, lr, true)

		// ----- 5. evaluate --------------------------------------------------
		correct := 0
		exp, pred := make([]float64, len(testX)), make([]float64, len(testX))
		for i, in := range testX {
			net.Forward(in)
			p := float64(paragon.ArgMax(net.ExtractOutput()))
			t := float64(paragon.ArgMax(testY[i][0]))
			pred[i], exp[i] = p, t
			if p == t {
				correct++
			}
		}
		net.EvaluateModel(exp, pred)
		accPct := float64(correct) / float64(len(testX)) * 100.0

		// ----- 6. record ----------------------------------------------------
		mu.Lock()
		results[key{hCnt, rDepth}] = append(
			results[key{hCnt, rDepth}],
			metric{net.Performance.Score, accPct},
		)
		mu.Unlock()
	}

	// ───────── enqueue all runs ────────────────────────────────────────────
	for hCnt := 2; hCnt <= 4; hCnt++ {
		for rDepth := 0; rDepth <= hCnt; rDepth++ {
			for run := 0; run < nRuns; run++ {
				wg.Add(1)
				go runModel(hCnt, rDepth, run)
			}
		}
	}
	wg.Wait()

	// ───────── aggregate & print table ─────────────────────────────────────
	fmt.Println("\n========= MULTI‑HIDDEN LAYER REPLAY BENCHMARK (10 runs each) =========")
	fmt.Printf("Hidden | ReplayD | ADHD(avg) | ±sd   | Acc%%(avg) | ±sd  \n")
	fmt.Println("---------------------------------------------------------------------")
	for hCnt := 2; hCnt <= 4; hCnt++ {
		for rDepth := 0; rDepth <= hCnt; rDepth++ {
			mets := results[key{hCnt, rDepth}]
			var sumA, sumAcc float64
			for _, m := range mets {
				sumA += m.adh
				sumAcc += m.acc
			}
			n := float64(len(mets))
			meanA, meanAcc := sumA/n, sumAcc/n
			var varA, varAcc float64
			for _, m := range mets {
				varA += (m.adh - meanA) * (m.adh - meanA)
				varAcc += (m.acc - meanAcc) * (m.acc - meanAcc)
			}
			sdA := math.Sqrt(varA / n)
			sdAcc := math.Sqrt(varAcc / n)
			fmt.Printf("  %d    |   %d     | %8.2f | %5.2f |   %6.2f | %5.2f\n",
				hCnt, rDepth, meanA, sdA, meanAcc, sdAcc)
		}
		fmt.Println("---------------------------------------------------------------------")
	}
	fmt.Println("=====================================================================")
}

// benchmarkMaxReplay evaluates the replay mechanism by varying MaxReplay
func benchmarkMaxReplay() {
	const (
		nRuns        = 10    // Number of runs per MaxReplay value
		epochs       = 20    // Training epochs
		lr           = 0.001 // Learning rate
		hiddenLayers = 4     // Number of hidden layers
		hiddenSize   = 8     // 8x8 neurons per hidden layer
		maxMaxReplay = 3     // Maximum MaxReplay value to test
	)

	// Load MNIST dataset
	if err := ensureMNISTDownloads("mnist_data"); err != nil {
		log.Fatalf("MNIST download error: %v", err)
	}
	trainX, trainY, _ := loadMNISTData("mnist_data", true)
	testX, testY, _ := loadMNISTData("mnist_data", false)
	trainX, trainY, _, _ = paragon.SplitDataset(trainX, trainY, 0.8)

	// Set up concurrency using 80% of CPU cores
	maxWorkers := int(0.8 * float64(runtime.NumCPU()))
	if maxWorkers < 1 {
		maxWorkers = 1
	}
	sem := make(chan struct{}, maxWorkers)
	var wg sync.WaitGroup

	// Store results
	type key struct{ maxReplay int }
	type metric struct{ adh, acc float64 }
	results := make(map[key][]metric)
	var mu sync.Mutex

	// Worker function to train and evaluate a single model
	runModel := func(maxReplay, runIdx int) {
		defer wg.Done()
		sem <- struct{}{}        // Acquire worker slot
		defer func() { <-sem }() // Release slot

		// Unique RNG seed for each run
		seed := time.Now().UnixNano() + int64(maxReplay)*1e6 + int64(runIdx)
		rnd := rand.New(rand.NewSource(seed))

		// Define network architecture
		layers := []struct{ Width, Height int }{{28, 28}} // Input: 28x28
		for i := 0; i < hiddenLayers; i++ {
			layers = append(layers, struct{ Width, Height int }{hiddenSize, hiddenSize})
		}
		layers = append(layers, struct{ Width, Height int }{10, 1}) // Output: 10x1

		// Set activations
		acts := make([]string, len(layers))
		for i := range acts {
			acts[i] = "leaky_relu"
		}
		acts[len(acts)-1] = "softmax"

		// Set connectivity
		fc := make([]bool, len(layers))
		fc[0], fc[len(fc)-1] = true, true // Fully connected input/output

		// Initialize network
		net := paragon.NewNetwork(layers, acts, fc)

		// Configure replay for hidden layers
		for l := 1; l <= hiddenLayers; l++ {
			net.Layers[l].ReplayOffset = -1     // Replay previous layer
			net.Layers[l].ReplayPhase = "after" // Replay after processing
			net.Layers[l].MaxReplay = maxReplay // Vary replay count
		}

		// Shuffle training data
		shX := make([][][]float64, len(trainX))
		shY := make([][][]float64, len(trainY))
		perm := rnd.Perm(len(trainX))
		for i, p := range perm {
			shX[i], shY[i] = trainX[p], trainY[p]
		}

		// Train the network
		net.Train(shX, shY, epochs, lr, true)

		// Evaluate on test set
		correct := 0
		exp, pred := make([]float64, len(testX)), make([]float64, len(testX))
		for i, in := range testX {
			net.Forward(in)
			p := float64(paragon.ArgMax(net.ExtractOutput()))
			t := float64(paragon.ArgMax(testY[i][0]))
			pred[i], exp[i] = p, t
			if p == t {
				correct++
			}
		}
		net.EvaluateModel(exp, pred)
		accPct := float64(correct) / float64(len(testX)) * 100.0

		// Store results
		mu.Lock()
		k := key{maxReplay}
		results[k] = append(results[k], metric{net.Performance.Score, accPct})
		mu.Unlock()
	}

	// Launch all runs
	for maxReplay := 0; maxReplay <= maxMaxReplay; maxReplay++ {
		for run := 0; run < nRuns; run++ {
			wg.Add(1)
			go runModel(maxReplay, run)
		}
	}
	wg.Wait()

	// Display results
	fmt.Println("\n========= MAX REPLAY BENCHMARK (4 Hidden Layers, 10 runs each) =========")
	fmt.Printf("MaxReplay | ADHD(avg) | ±sd   | Acc%%(avg) | ±sd  \n")
	fmt.Println("---------------------------------------------------------------------")
	for maxReplay := 0; maxReplay <= maxMaxReplay; maxReplay++ {
		mets := results[key{maxReplay}]
		var sumA, sumAcc float64
		for _, m := range mets {
			sumA += m.adh
			sumAcc += m.acc
		}
		n := float64(len(mets))
		meanA, meanAcc := sumA/n, sumAcc/n
		var varA, varAcc float64
		for _, m := range mets {
			varA += (m.adh - meanA) * (m.adh - meanA)
			varAcc += (m.acc - meanAcc) * (m.acc - meanAcc)
		}
		sdA := math.Sqrt(varA / n)
		sdAcc := math.Sqrt(varAcc / n)
		fmt.Printf("    %d     | %8.2f | %5.2f |   %6.2f | %5.2f\n",
			maxReplay, meanA, sdA, meanAcc, sdAcc)
	}
	fmt.Println("=====================================================================")
}

// benchmarkReplaySweetSpot compares baseline vs "one‑layer replay"
// on MNIST for nets with 1 and 2 hidden layers (10 seeds each).
func benchmarkReplaySweetSpot() {
	// ---------------------- hyper‑params -----------------------------------
	const (
		nRuns         = 10
		epochs        = 25 // total epochs
		warmUpEpochs  = 5  // replay disabled during these epochs
		baseLR        = 0.001
		lrScaleReplay = 0.5 // LR multiplier after warm‑up when replay ON
	)

	// layout map: hiddenCount → hiddenWidth/Height
	hiddenDims := map[int]int{
		1: 16, // 16×16 = 256 neurons
		2: 12, // 12×12 ×2 = 288 neurons
	}

	// ---------------------- dataset ---------------------------------------
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatal(err)
	}
	trainX, trainY, _ := loadMNISTData(mnistDir, true)
	testX, testY, _ := loadMNISTData(mnistDir, false)
	trainX, trainY, _, _ = paragon.SplitDataset(trainX, trainY, 0.8)

	// ---------------------- concurrency -----------------------------------
	maxWorkers := int(0.8 * float64(runtime.NumCPU()))
	if maxWorkers < 1 {
		maxWorkers = 1
	}
	sem := make(chan struct{}, maxWorkers)
	var wg sync.WaitGroup

	type key struct {
		hCnt   int
		replay bool
	}
	type metric struct{ adh, acc float64 }
	results := make(map[key][]metric)
	var mu sync.Mutex

	// ---------------------- worker ----------------------------------------
	runModel := func(hCnt int, replay bool, runIdx int) {
		defer wg.Done()
		sem <- struct{}{}
		defer func() { <-sem }()

		seed := time.Now().UnixNano() + int64(hCnt*1e5) + int64(runIdx*137)
		rnd := rand.New(rand.NewSource(seed))

		// 1. construct layer sizes
		hSize := hiddenDims[hCnt]
		layers := []struct{ Width, Height int }{{28, 28}} // input
		for i := 0; i < hCnt; i++ {
			layers = append(layers, struct{ Width, Height int }{hSize, hSize})
		}
		layers = append(layers, struct{ Width, Height int }{10, 1}) // output

		acts := make([]string, len(layers))
		for i := range acts {
			acts[i] = "leaky_relu"
		}
		acts[len(acts)-1] = "softmax"

		fc := make([]bool, len(layers))
		fc[0], fc[len(fc)-1] = true, true // full connect I/O

		net := paragon.NewNetwork(layers, acts, fc)

		// 2. configure single‑layer replay (layer 1) if requested
		if replay {
			net.Layers[1].ReplayOffset = -1
			net.Layers[1].ReplayPhase = "after"
			net.Layers[1].MaxReplay = 1
		}

		// 3. shuffle train set for this seed
		shX := make([][][]float64, len(trainX))
		shY := make([][][]float64, len(trainY))
		perm := rnd.Perm(len(trainX))
		for i, p := range perm {
			shX[i], shY[i] = trainX[p], trainY[p]
		}

		// 4. training loop with warm‑up
		for epoch := 0; epoch < epochs; epoch++ {
			// enable/disable replay & LR scaling
			if replay && epoch >= warmUpEpochs {
				net.Layers[1].MaxReplay = 1
			} else {
				net.Layers[1].MaxReplay = 0
			}
			lr := baseLR
			if replay && epoch >= warmUpEpochs {
				lr *= lrScaleReplay
			}
			// simple one‑pass over data set each epoch
			net.Train(shX, shY, 1, lr, true)
		}

		// 5. evaluate
		correct := 0
		exp, pred := make([]float64, len(testX)), make([]float64, len(testX))
		for i, in := range testX {
			net.Forward(in)
			p := float64(paragon.ArgMax(net.ExtractOutput()))
			t := float64(paragon.ArgMax(testY[i][0]))
			pred[i], exp[i] = p, t
			if p == t {
				correct++
			}
		}
		net.EvaluateModel(exp, pred)
		accPct := float64(correct) / float64(len(testX)) * 100.0

		mu.Lock()
		results[key{hCnt, replay}] = append(results[key{hCnt, replay}],
			metric{net.Performance.Score, accPct})
		mu.Unlock()
	}

	// ---------------------- enqueue jobs ----------------------------------
	for hCnt := 1; hCnt <= 2; hCnt++ {
		for run := 0; run < nRuns; run++ {
			wg.Add(2)
			go runModel(hCnt, false, run) // baseline
			go runModel(hCnt, true, run)  // replay
		}
	}
	wg.Wait()

	// ---------------------- print summary ---------------------------------
	fmt.Println("\n========= REPLAY SWEET‑SPOT BENCHMARK (10 runs each) =========")
	fmt.Printf("Hidden | Kind    | ADHD(avg) | ±sd   | Acc%%(avg) | ±sd  \n")
	fmt.Println("-------------------------------------------------------------")
	for hCnt := 1; hCnt <= 2; hCnt++ {
		for _, replay := range []bool{false, true} {
			mets := results[key{hCnt, replay}]
			var sumA, sumAcc float64
			for _, m := range mets {
				sumA += m.adh
				sumAcc += m.acc
			}
			n := float64(len(mets))
			meanA, meanAcc := sumA/n, sumAcc/n
			var varA, varAcc float64
			for _, m := range mets {
				varA += (m.adh - meanA) * (m.adh - meanA)
				varAcc += (m.acc - meanAcc) * (m.acc - meanAcc)
			}
			sdA, sdAcc := math.Sqrt(varA/n), math.Sqrt(varAcc/n)
			kind := "baseline"
			if replay {
				kind = "replay"
			}
			fmt.Printf("  %d    | %-8s| %8.2f | %5.2f |   %6.2f | %5.2f\n",
				hCnt, kind, meanA, sdA, meanAcc, sdAcc)
		}
		fmt.Println("-------------------------------------------------------------")
	}
	fmt.Println("=============================================================")
}

// ---------------------------------------------------------------
// benchmarkReplayBeforeAfter compares baseline vs replay("before")
// vs replay("after") on a 1‑hidden‑layer MNIST network.
// ---------------------------------------------------------------
func benchmarkReplayBeforeAfter() {
	const (
		nRuns  = 10
		epochs = 20
		lr     = 0.001
	)

	// ---------- dataset ---------------------------------------------------
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatal(err)
	}
	trainX, trainY, _ := loadMNISTData(mnistDir, true)
	testX, testY, _ := loadMNISTData(mnistDir, false)
	trainX, trainY, _, _ = paragon.SplitDataset(trainX, trainY, 0.8)

	// ---------- concurrency gate -----------------------------------------
	maxW := int(0.8 * float64(runtime.NumCPU()))
	if maxW < 1 {
		maxW = 1
	}
	sem := make(chan struct{}, maxW)
	var wg sync.WaitGroup

	type variant string
	const (
		baseline  variant = "baseline"
		beforeVar variant = "replay‑before"
		afterVar  variant = "replay‑after"
	)
	allVariants := []variant{baseline, beforeVar, afterVar}

	type metric struct{ adh, acc float64 }
	results := make(map[variant][]metric)
	var mu sync.Mutex

	// ---------- worker ----------------------------------------------------
	run := func(kind variant, runIdx int) {
		defer wg.Done()
		sem <- struct{}{}
		defer func() { <-sem }()

		seed := time.Now().UnixNano() +
			int64(runIdx)*113 +
			int64(len(kind))*1e6
		rnd := rand.New(rand.NewSource(seed))

		// network shape: 28×28 → 16×16 → 10
		layers := []struct{ Width, Height int }{
			{28, 28},
			{16, 16},
			{10, 1},
		}
		acts := []string{"leaky_relu", "leaky_relu", "softmax"}
		fc := []bool{true, false, true}

		net := paragon.NewNetwork(layers, acts, fc)

		// configure replay variant
		if kind != baseline {
			net.Layers[1].ReplayOffset = -1
			net.Layers[1].ReplayPhase = map[variant]string{
				beforeVar: "before",
				afterVar:  "after",
			}[kind]
			net.Layers[1].MaxReplay = 1
		}

		// shuffle train set
		shX := make([][][]float64, len(trainX))
		shY := make([][][]float64, len(trainY))
		perm := rnd.Perm(len(trainX))
		for i, p := range perm {
			shX[i], shY[i] = trainX[p], trainY[p]
		}

		net.Train(shX, shY, epochs, lr, true)

		// evaluate
		correct := 0
		exp, pred := make([]float64, len(testX)), make([]float64, len(testX))
		for i, in := range testX {
			net.Forward(in)
			p := float64(paragon.ArgMax(net.ExtractOutput()))
			t := float64(paragon.ArgMax(testY[i][0]))
			pred[i], exp[i] = p, t
			if p == t {
				correct++
			}
		}
		net.EvaluateModel(exp, pred)
		acc := float64(correct) / float64(len(testX)) * 100.0

		mu.Lock()
		results[kind] = append(results[kind], metric{net.Performance.Score, acc})
		mu.Unlock()
	}

	// ---------- launch all jobs ------------------------------------------
	for runIdx := 0; runIdx < nRuns; runIdx++ {
		for _, v := range allVariants {
			wg.Add(1)
			go run(v, runIdx)
		}
	}
	wg.Wait()

	// ---------- summarise -------------------------------------------------
	fmt.Println("\n========= REPLAY BEFORE vs AFTER (1 hidden layer, 10 runs) =========")
	fmt.Printf("Variant        | ADHD(avg) | ±sd   | Acc%%(avg) | ±sd  \n")
	fmt.Println("-------------------------------------------------------------------")
	for _, v := range allVariants {
		mets := results[v]
		var sumA, sumAcc float64
		for _, m := range mets {
			sumA += m.adh
			sumAcc += m.acc
		}
		n := float64(len(mets))
		meanA, meanAcc := sumA/n, sumAcc/n
		var varA, varAcc float64
		for _, m := range mets {
			varA += (m.adh - meanA) * (m.adh - meanA)
			varAcc += (m.acc - meanAcc) * (m.acc - meanAcc)
		}
		sdA := math.Sqrt(varA / n)
		sdAcc := math.Sqrt(varAcc / n)
		fmt.Printf("%-13s | %8.2f | %5.2f |   %6.2f | %5.2f\n",
			string(v), meanA, sdA, meanAcc, sdAcc)
	}
	fmt.Println("===================================================================")
}
