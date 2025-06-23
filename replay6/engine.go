package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"paragon"
	"runtime"
	"sync"
	"time"
)

// -------------------------------------------------- data helpers (unchanged)
const (
	baseURL   = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	mnistDir  = "mnist_data"
	modelDir  = "models"
	modelFile = "mnist_model.json"
)

// Global file and mutex for thread-safe writing to results.txt
var (
	resultsFile *os.File
	fileMu      sync.Mutex
)

// -------------------------------------------------- main
func main() {
	// Initialize results file
	var err error
	resultsFile, err = os.OpenFile("results.txt", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatalf("Failed to open results.txt: %v", err)
	}
	defer resultsFile.Close()

	// Write timestamp header
	writeResult(fmt.Sprintf("\n=== Test Run: %s ===\n", time.Now().Format("2006-01-02 15:04:05")))

	singleCompare()
	benchmarkReplayVsBaseline()
	benchmarkReplayVsBaselineN()
	benchmarkReplayDepths()
	benchmarkMaxReplay()
	benchmarkReplaySweetSpot()
	benchmarkReplayBeforeAfter()
	benchmarkReplaySettingsMassive()
	benchmarkDeepReplaySettings()
	benchmarkDynamicReplayOptimizer()
	benchmarkAdaptiveTemporalReplay()
	benchmarkEnhancedTemporalReplay()
}

// writeResult writes formatted output to results.txt with thread safety
func writeResult(format string, args ...interface{}) {
	fileMu.Lock()
	defer fileMu.Unlock()
	_, err := fmt.Fprintf(resultsFile, format, args...)
	if err != nil {
		log.Printf("Failed to write to results.txt: %v", err)
	}
	// Ensure immediate write
	resultsFile.Sync()
}

// Helper to create a network with consistent architecture and replay settings
func createNetwork(replayType string, seed int64, hCnt int) *paragon.Network[float32] {
	// Define small architecture: 28x28 -> 3x3 (per hidden layer) -> 10x1
	layers := []struct{ Width, Height int }{{28, 28}}
	for i := 0; i < hCnt; i++ {
		layers = append(layers, struct{ Width, Height int }{3, 3})
	}
	layers = append(layers, struct{ Width, Height int }{10, 1})
	acts := make([]string, len(layers))
	for i := range acts {
		acts[i] = "leaky_relu"
	}
	acts[len(acts)-1] = "softmax"
	fc := make([]bool, len(layers))
	for i := range fc {
		fc[i] = true // Fully connected layers
	}

	net := paragon.NewNetwork[float32](layers, acts, fc, seed)

	if replayType == "static" {
		// Static replay: replay previous layer after processing
		net.Layers[1].ReplayOffset = -1
		net.Layers[1].ReplayPhase = "after"
		net.Layers[1].MaxReplay = 3
	} else if replayType == "dynamic" {
		// Dynamic replay: entropy-based gating
		net.Layers[1].ReplayEnabled = true
		net.Layers[1].ReplayBudget = 3
		net.Layers[1].ReplayGateFunc = func(input [][]float32) float64 {
			outputs := net.Layers[1].CachedOutputs
			if len(outputs) == 0 {
				return 0.5
			}
			var sum float64
			for _, v := range outputs {
				v64 := float64(v)
				if v64 > 1e-10 { // Avoid log(0)
					sum += v64 * math.Log(v64)
				}
			}
			entropy := -sum / float64(len(outputs))
			maxEntropy := math.Log(float64(len(outputs)))
			if maxEntropy == 0 {
				return 0.5
			}
			return math.Min(1.0, math.Max(0.0, entropy/maxEntropy))
		}
		net.Layers[1].ReplayGateToReps = func(score float64) int {
			if score > 0.4 { // Trigger replay for moderate entropy
				return 3
			}
			return 0
		}
	}
	return net
}

func singleCompare() {
	// 1) Load MNIST dataset
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatalf("MNIST download error: %v", err)
	}
	trainX, trainY, _ := loadMNISTData(mnistDir, true)
	testX, testY, _ := loadMNISTData(mnistDir, false)
	trainX, trainY, _, _ = paragon.SplitDataset(trainX, trainY, 0.3) // 30% training data

	// 2) Create three networks (1 hidden layer)
	baseNet := createNetwork("baseline", time.Now().UnixNano(), 1)
	staticReplayNet := createNetwork("static", time.Now().UnixNano()+1, 1)
	dynamicReplayNet := createNetwork("dynamic", time.Now().UnixNano()+2, 1)

	// 3) Train all three
	fmt.Println("🧠 Training baseline …")
	baseNet.Train(trainX, trainY, 3, 0.001, true, 5, -5)

	fmt.Println("🧠 Training static replay …")
	staticReplayNet.Train(trainX, trainY, 3, 0.001, true, 5, -5)

	fmt.Println("🧠 Training dynamic replay …")
	dynamicReplayNet.Train(trainX, trainY, 3, 0.001, true, 5, -5)

	// 4) Evaluation helper
	type results struct {
		name    string
		score   float64
		acc     float64
		buckets map[string]int
	}
	evaluate := func(net *paragon.Network[float32], name string) results {
		exp, pred := []float64{}, []float64{}
		correct := 0
		for i, in := range testX {
			net.Forward(in)
			predVal := float64(paragon.ArgMax(net.ExtractOutput()))
			expVal := float64(paragon.ArgMax(testY[i][0]))
			pred = append(pred, predVal)
			exp = append(exp, expVal)
			if predVal == expVal {
				correct++
			}
		}
		net.EvaluateModel(exp, pred)
		b := map[string]int{}
		for k, v := range net.Performance.Buckets {
			b[k] = v.Count
		}
		acc := float64(correct) / float64(len(testX)) * 100.0
		return results{name, net.Performance.Score, acc, b}
	}

	baseRes := evaluate(baseNet, "Baseline")
	staticRes := evaluate(staticReplayNet, "Static Replay")
	dynamicRes := evaluate(dynamicReplayNet, "Dynamic Replay")

	// 5) Write results to file
	writeResult("\n============== PERFORMANCE COMPARISON ==============\n")
	writeResult("Metric                     | %-12s | %-12s | %-12s\n",
		"Baseline", "Static Replay", "Dynamic Replay")
	writeResult("---------------------------+--------------+--------------+--------------\n")
	writeResult("ADHD Score                 | %12.2f | %12.2f | %12.2f\n",
		baseRes.score, staticRes.score, dynamicRes.score)
	writeResult("Accuracy %%                 | %12.2f | %12.2f | %12.2f\n",
		baseRes.acc, staticRes.acc, dynamicRes.acc)
	writeResult("\nDeviation buckets (# samples):\n")
	keys := []string{"0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%", "100%+"}
	for _, k := range keys {
		writeResult(" %-7s | %12d | %12d | %12d\n",
			k, baseRes.buckets[k], staticRes.buckets[k], dynamicRes.buckets[k])
	}
}

func benchmarkReplayVsBaseline() {
	// 1) Load data
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatalf("MNIST download error: %v", err)
	}
	trainX, trainY, _ := loadMNISTData(mnistDir, true)
	testX, testY, _ := loadMNISTData(mnistDir, false)
	trainX, trainY, _, _ = paragon.SplitDataset(trainX, trainY, 0.3)

	// 2) Setup concurrency
	type outcome struct {
		Idx  int
		Kind string
		ADHD float64
		Acc  float64
	}
	outCh := make(chan outcome, 30)
	var wg sync.WaitGroup

	// 3) Helper: build -> train -> evaluate
	run := func(idx int, kind string) {
		defer wg.Done()
		sem := make(chan struct{}, 1)
		sem <- struct{}{}
		defer func() { <-sem }()
		rnd := rand.New(rand.NewSource(time.Now().UnixNano() + int64(idx)*31))
		net := createNetwork(kind, rnd.Int63(), 1)
		fmt.Printf("🧠 Training %s for run %d …\n", kind, idx)
		net.Train(trainX, trainY, 3, 0.001, true, 5, -5)

		// Evaluate
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

	// 4) Launch 30 goroutines (10 per variant)
	for i := 0; i < 10; i++ {
		wg.Add(3)
		go run(i, "baseline")
		go run(i, "static")
		go run(i, "dynamic")
	}

	go func() {
		wg.Wait()
		close(outCh)
	}()

	// 5) Collect and write results
	bScores, sScores, dScores := []outcome{}, []outcome{}, []outcome{}
	for res := range outCh {
		switch res.Kind {
		case "baseline":
			bScores = append(bScores, res)
		case "static":
			sScores = append(sScores, res)
		case "dynamic":
			dScores = append(dScores, res)
		}
	}

	writeResult("\n================ 10× BENCHMARK ================\n")
	writeResult("%-4s | %-12s | %-8s | %-8s\n", "Run", "Kind", "ADHD", "Acc%")
	writeResult("----------------------------------------------\n")
	avgBadhd, avgSadhd, avgDadhd := 0.0, 0.0, 0.0
	avgBacc, avgSacc, avgDacc := 0.0, 0.0, 0.0
	for i := 0; i < 10; i++ {
		b, s, d := bScores[i], sScores[i], dScores[i]
		writeResult("%-4d | %-12s | %8.2f | %7.2f\n", b.Idx, "baseline", b.ADHD, b.Acc)
		writeResult("%-4d | %-12s | %8.2f | %7.2f\n", s.Idx, "static", s.ADHD, s.Acc)
		writeResult("%-4d | %-12s | %8.2f | %7.2f\n", d.Idx, "dynamic", d.ADHD, d.Acc)
		writeResult("----------------------------------------------\n")
		avgBadhd += b.ADHD
		avgSadhd += s.ADHD
		avgDadhd += d.ADHD
		avgBacc += b.Acc
		avgSacc += s.Acc
		avgDacc += d.Acc
	}
	n := float64(10)
	writeResult("AVERAGE %-12s | %8.2f | %7.2f\n", "baseline", avgBadhd/n, avgBacc/n)
	writeResult("AVERAGE %-12s | %8.2f | %7.2f\n", "static", avgSadhd/n, avgSacc/n)
	writeResult("AVERAGE %-12s | %8.2f | %7.2f\n", "dynamic", avgDadhd/n, avgDacc/n)
	writeResult("==============================================\n")
}

func benchmarkReplayVsBaselineN() {
	const (
		NModels      = 20
		Epochs       = 3
		LearningRate = 0.001
	)

	// 1) Load data
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatal(err)
	}
	trainX, trainY, _ := loadMNISTData(mnistDir, true)
	testX, testY, _ := loadMNISTData(mnistDir, false)
	trainX, trainY, _, _ = paragon.SplitDataset(trainX, trainY, 0.3)

	// 2) Concurrency governor
	maxThreads := int(0.8 * float64(runtime.NumCPU()))
	if maxThreads < 1 {
		maxThreads = 1
	}
	sem := make(chan struct{}, maxThreads)
	type score struct{ ADHD, Acc float64 }
	bScores := make([]score, NModels)
	sScores := make([]score, NModels)
	dScores := make([]score, NModels)
	var wg sync.WaitGroup
	var mu sync.Mutex

	// 3) Worker
	buildAndRun := func(idx int, kind string) {
		defer wg.Done()
		sem <- struct{}{}
		defer func() { <-sem }()
		rnd := rand.New(rand.NewSource(time.Now().UnixNano() + int64(idx)*113))
		net := createNetwork(kind, rnd.Int63(), 1)
		fmt.Printf("🧠 Training %s for run %d …\n", kind, idx)
		shuffledX := make([][][]float64, len(trainX))
		shuffledY := make([][][]float64, len(trainY))
		perm := rnd.Perm(len(trainX))
		for i, p := range perm {
			shuffledX[i] = trainX[p]
			shuffledY[i] = trainY[p]
		}
		net.Train(shuffledX, shuffledY, Epochs, LearningRate, true, 5, -5)
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
		accPc := float64(correct) / float64(len(testX)) * 100.0
		mu.Lock()
		switch kind {
		case "baseline":
			bScores[idx] = score{net.Performance.Score, accPc}
		case "static":
			sScores[idx] = score{net.Performance.Score, accPc}
		case "dynamic":
			dScores[idx] = score{net.Performance.Score, accPc}
		}
		mu.Unlock()
	}

	// 4) Launch goroutines
	for i := 0; i < NModels; i++ {
		wg.Add(3)
		go buildAndRun(i, "baseline")
		go buildAndRun(i, "static")
		go buildAndRun(i, "dynamic")
	}
	wg.Wait()

	// 5) Write results
	writeResult("\n================ %d× BENCHMARK (%.0f%% CPUs) ================\n",
		NModels, 100*0.8)
	writeResult("Run | ADHD_bas | ADHD_sta | ADHD_dyn | Acc%%_bas | Acc%%_sta | Acc%%_dyn\n")
	writeResult("------------------------------------------------------------------\n")
	var sumBad, sumSad, sumDad, sumBac, sumSac, sumDac float64
	for i := 0; i < NModels; i++ {
		b, s, d := bScores[i], sScores[i], dScores[i]
		writeResult("%3d | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f\n",
			i, b.ADHD, s.ADHD, d.ADHD, b.Acc, s.Acc, d.Acc)
		sumBad += b.ADHD
		sumSad += s.ADHD
		sumDad += d.ADHD
		sumBac += b.Acc
		sumSac += s.Acc
		sumDac += d.Acc
	}
	n := float64(NModels)
	writeResult("------------------------------------------------------------------\n")
	writeResult("AVG | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f\n",
		sumBad/n, sumSad/n, sumDad/n, sumBac/n, sumSac/n, sumDac/n)
	writeResult("===============================================================\n")
}

func benchmarkReplayDepths() {
	const (
		nRuns          = 5
		epochs         = 3
		lr             = 0.001
		inputW, inputH = 28, 28
		outputW        = 10
	)

	// 1) Load data
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatal(err)
	}
	trainX, trainY, _ := loadMNISTData(mnistDir, true)
	testX, testY, _ := loadMNISTData(mnistDir, false)
	trainX, trainY, _, _ = paragon.SplitDataset(trainX, trainY, 0.3)

	// 2) Concurrency guard
	maxWorkers := int(0.8 * float64(runtime.NumCPU()))
	if maxWorkers < 1 {
		maxWorkers = 1
	}
	sem := make(chan struct{}, maxWorkers)
	var wg sync.WaitGroup

	// 3) Result storage
	type key struct {
		hCnt  int
		rType string
	}
	type metric struct{ adh, acc float64 }
	results := make(map[key][]metric)
	var mu sync.Mutex

	// 4) Worker
	runModel := func(hCnt int, rType string, runIdx int) {
		defer wg.Done()
		sem <- struct{}{}
		defer func() { <-sem }()
		seed := time.Now().UnixNano() + int64(hCnt)*1e6 + int64(runIdx)*1e4
		rnd := rand.New(rand.NewSource(seed))
		net := createNetwork(rType, seed, hCnt)
		fmt.Printf("🧠 Training %s with %d hidden layers for run %d …\n", rType, hCnt, runIdx)
		shX := make([][][]float64, len(trainX))
		shY := make([][][]float64, len(trainY))
		perm := rnd.Perm(len(trainX))
		for i, p := range perm {
			shX[i], shY[i] = trainX[p], trainY[p]
		}
		net.Train(shX, shY, epochs, lr, true, 5, -5)
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
		results[key{hCnt, rType}] = append(results[key{hCnt, rType}], metric{net.Performance.Score, accPct})
		mu.Unlock()
	}

	// 5) Enqueue runs
	for hCnt := 1; hCnt <= 2; hCnt++ {
		for _, rType := range []string{"baseline", "static", "dynamic"} {
			for run := 0; run < nRuns; run++ {
				wg.Add(1)
				go runModel(hCnt, rType, run)
			}
		}
	}
	wg.Wait()

	// 6) Write results
	writeResult("\n========= MULTI-HIDDEN LAYER REPLAY BENCHMARK (5 runs each) =========\n")
	writeResult("Hidden | Replay Type | ADHD(avg) | ±sd   | Acc%%(avg) | ±sd  \n")
	writeResult("---------------------------------------------------------------------\n")
	for hCnt := 1; hCnt <= 2; hCnt++ {
		for _, rType := range []string{"baseline", "static", "dynamic"} {
			mets := results[key{hCnt, rType}]
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
			writeResult("  %d    | %-11s | %8.2f | %5.2f |   %6.2f | %5.2f\n",
				hCnt, rType, meanA, sdA, meanAcc, sdAcc)
		}
		writeResult("---------------------------------------------------------------------\n")
	}
	writeResult("=====================================================================\n")
}

func benchmarkMaxReplay() {
	const (
		nRuns        = 5
		epochs       = 3
		lr           = 0.001
		hiddenLayers = 1
		hiddenSize   = 3
		maxMaxReplay = 3
	)

	// 1) Load data
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatalf("MNIST download error: %v", err)
	}
	trainX, trainY, _ := loadMNISTData(mnistDir, true)
	testX, testY, _ := loadMNISTData(mnistDir, false)
	trainX, trainY, _, _ = paragon.SplitDataset(trainX, trainY, 0.3)

	// 2) Concurrency
	maxWorkers := int(0.8 * float64(runtime.NumCPU()))
	if maxWorkers < 1 {
		maxWorkers = 1
	}
	sem := make(chan struct{}, maxWorkers)
	var wg sync.WaitGroup
	type key struct{ maxReplay int }
	type metric struct{ adh, acc float64 }
	results := make(map[key][]metric)
	var mu sync.Mutex

	// 3) Worker
	runModel := func(maxReplay, runIdx int) {
		defer wg.Done()
		sem <- struct{}{}
		defer func() { <-sem }()
		seed := time.Now().UnixNano() + int64(maxReplay)*1e6 + int64(runIdx)
		rnd := rand.New(rand.NewSource(seed))
		replayType := "baseline"
		if maxReplay > 0 {
			replayType = "static"
		}
		net := createNetwork(replayType, seed, 1)
		if maxReplay > 0 {
			net.Layers[1].MaxReplay = maxReplay
		}
		fmt.Printf("🧠 Training with MaxReplay %d for run %d …\n", maxReplay, runIdx)
		shX := make([][][]float64, len(trainX))
		shY := make([][][]float64, len(trainY))
		perm := rnd.Perm(len(trainX))
		for i, p := range perm {
			shX[i], shY[i] = trainX[p], trainY[p]
		}
		net.Train(shX, shY, epochs, lr, true, 5, -5)
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
		results[key{maxReplay}] = append(results[key{maxReplay}], metric{net.Performance.Score, accPct})
		mu.Unlock()
	}

	// 4) Launch runs
	for maxReplay := 0; maxReplay <= maxMaxReplay; maxReplay++ {
		for run := 0; run < nRuns; run++ {
			wg.Add(1)
			go runModel(maxReplay, run)
		}
	}
	wg.Wait()

	// 5) Write results
	writeResult("\n========= MAX REPLAY BENCHMARK (1 Hidden Layer, 5 runs each) =========\n")
	writeResult("MaxReplay | ADHD(avg) | ±sd   | Acc%%(avg) | ±sd  \n")
	writeResult("---------------------------------------------------------------------\n")
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
		writeResult("    %d     | %8.2f | %5.2f |   %6.2f | %5.2f\n",
			maxReplay, meanA, sdA, meanAcc, sdAcc)
	}
	writeResult("=====================================================================\n")
}

func benchmarkReplaySweetSpot() {
	const (
		nRuns         = 5
		epochs        = 3
		baseLR        = 0.001
		lrScaleReplay = 0.8
	)

	// 1) Load data
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatal(err)
	}
	trainX, trainY, _ := loadMNISTData(mnistDir, true)
	testX, testY, _ := loadMNISTData(mnistDir, false)
	trainX, trainY, _, _ = paragon.SplitDataset(trainX, trainY, 0.3)

	// 2) Concurrency
	maxWorkers := int(0.8 * float64(runtime.NumCPU()))
	if maxWorkers < 1 {
		maxWorkers = 1
	}
	sem := make(chan struct{}, maxWorkers)
	var wg sync.WaitGroup
	type key struct {
		hCnt  int
		rType string
	}
	type metric struct{ adh, acc float64 }
	results := make(map[key][]metric)
	var mu sync.Mutex

	// 3) Worker
	runModel := func(hCnt int, rType string, runIdx int) {
		defer wg.Done()
		sem <- struct{}{}
		defer func() { <-sem }()
		seed := time.Now().UnixNano() + int64(hCnt*1e5) + int64(runIdx*137)
		rnd := rand.New(rand.NewSource(seed))
		net := createNetwork(rType, seed, hCnt)
		fmt.Printf("🧠 Training %s with %d hidden layers for run %d …\n", rType, hCnt, runIdx)
		shX := make([][][]float64, len(trainX))
		shY := make([][][]float64, len(trainY))
		perm := rnd.Perm(len(trainX))
		for i, p := range perm {
			shX[i], shY[i] = trainX[p], trainY[p]
		}
		lr := baseLR
		if rType != "baseline" {
			lr *= lrScaleReplay
		}
		net.Train(shX, shY, epochs, lr, true, 5, -5)
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
		results[key{hCnt, rType}] = append(results[key{hCnt, rType}], metric{net.Performance.Score, accPct})
		mu.Unlock()
	}

	// 4) Enqueue jobs
	for hCnt := 1; hCnt <= 2; hCnt++ {
		for _, rType := range []string{"baseline", "static", "dynamic"} {
			for run := 0; run < nRuns; run++ {
				wg.Add(1)
				go runModel(hCnt, rType, run)
			}
		}
	}
	wg.Wait()

	// 5) Write results
	writeResult("\n========= REPLAY SWEET-SPOT BENCHMARK (5 runs each) =========\n")
	writeResult("Hidden | Kind         | ADHD(avg) | ±sd   | Acc%%(avg) | ±sd  \n")
	writeResult("-------------------------------------------------------------\n")
	for hCnt := 1; hCnt <= 2; hCnt++ {
		for _, rType := range []string{"baseline", "static", "dynamic"} {
			mets := results[key{hCnt, rType}]
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
			writeResult("  %d    | %-12s | %8.2f | %5.2f |   %6.2f | %5.2f\n",
				hCnt, rType, meanA, sdA, meanAcc, sdAcc)
		}
		writeResult("-------------------------------------------------------------\n")
	}
	writeResult("=============================================================\n")
}

func benchmarkReplayBeforeAfter() {
	const (
		nRuns  = 5
		epochs = 3
		lr     = 0.001
	)

	// 1) Load data
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatal(err)
	}
	trainX, trainY, _ := loadMNISTData(mnistDir, true)
	testX, testY, _ := loadMNISTData(mnistDir, false)
	trainX, trainY, _, _ = paragon.SplitDataset(trainX, trainY, 0.3)

	// 2) Concurrency
	maxW := int(0.8 * float64(runtime.NumCPU()))
	if maxW < 1 {
		maxW = 1
	}
	sem := make(chan struct{}, maxW)
	var wg sync.WaitGroup
	type variant string
	const (
		baseline     variant = "baseline"
		staticBefore variant = "static-before"
		staticAfter  variant = "static-after"
		dynamic      variant = "dynamic"
	)
	allVariants := []variant{baseline, staticBefore, staticAfter, dynamic}
	type metric struct{ adh, acc float64 }
	results := make(map[variant][]metric)
	var mu sync.Mutex

	// 3) Worker
	run := func(kind variant, runIdx int) {
		defer wg.Done()
		sem <- struct{}{}
		defer func() { <-sem }()
		seed := time.Now().UnixNano() + int64(runIdx)*113 + int64(len(kind))*1e6
		rnd := rand.New(rand.NewSource(seed))
		replayType := string(kind)
		if kind == staticBefore {
			replayType = "static"
		} else if kind == staticAfter {
			replayType = "static"
		}
		net := createNetwork(replayType, seed, 1)
		if kind == staticBefore {
			net.Layers[1].ReplayPhase = "before"
		}
		fmt.Printf("🧠 Training %s for run %d …\n", kind, runIdx)
		shX := make([][][]float64, len(trainX))
		shY := make([][][]float64, len(trainY))
		perm := rnd.Perm(len(trainX))
		for i, p := range perm {
			shX[i], shY[i] = trainX[p], trainY[p]
		}
		net.Train(shX, shY, epochs, lr, true, 5, -5)
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

	// 4) Launch jobs
	for runIdx := 0; runIdx < nRuns; runIdx++ {
		for _, v := range allVariants {
			wg.Add(1)
			go run(v, runIdx)
		}
	}
	wg.Wait()

	// 5) Write results
	writeResult("\n========= REPLAY BEFORE vs AFTER vs DYNAMIC (1 hidden layer, 5 runs) =========\n")
	writeResult("Variant        | ADHD(avg) | ±sd   | Acc%%(avg) | ±sd  \n")
	writeResult("-------------------------------------------------------------------\n")
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
		writeResult("%-13s | %8.2f | %5.2f |   %6.2f | %5.2f\n",
			string(v), meanA, sdA, meanAcc, sdAcc)
	}
	writeResult("===================================================================\n")
}

func benchmarkReplaySettingsMassive() {
	const (
		nRuns          = 5
		epochs         = 2
		lr             = 0.001
		inputW, inputH = 28, 28
		outputW        = 10
	)

	// Configurations to test
	staticConfigs := []struct {
		maxReplay    int
		replayPhase  string
		replayOffset int
	}{
		{maxReplay: 0, replayPhase: "after", replayOffset: -1}, // Baseline (no replay)
		{maxReplay: 1, replayPhase: "before", replayOffset: -1},
		{maxReplay: 1, replayPhase: "after", replayOffset: -1},
		{maxReplay: 2, replayPhase: "before", replayOffset: -1},
		{maxReplay: 2, replayPhase: "after", replayOffset: -1},
		{maxReplay: 3, replayPhase: "before", replayOffset: -1},
		{maxReplay: 3, replayPhase: "after", replayOffset: -1},
		{maxReplay: 4, replayPhase: "before", replayOffset: -1},
		{maxReplay: 4, replayPhase: "after", replayOffset: -1},
		{maxReplay: 2, replayPhase: "before", replayOffset: -2},
		{maxReplay: 2, replayPhase: "after", replayOffset: -2},
	}
	dynamicConfigs := []struct {
		gateType     string
		threshold    float64
		replayBudget int
		replayPhase  string
		replayOffset int
	}{
		{gateType: "entropy", threshold: 0.3, replayBudget: 1, replayPhase: "before", replayOffset: -1},
		{gateType: "entropy", threshold: 0.3, replayBudget: 1, replayPhase: "after", replayOffset: -1},
		{gateType: "entropy", threshold: 0.5, replayBudget: 2, replayPhase: "before", replayOffset: -1},
		{gateType: "entropy", threshold: 0.5, replayBudget: 2, replayPhase: "after", replayOffset: -1},
		{gateType: "entropy", threshold: 0.7, replayBudget: 3, replayPhase: "before", replayOffset: -1},
		{gateType: "entropy", threshold: 0.7, replayBudget: 3, replayPhase: "after", replayOffset: -1},
		{gateType: "variance", threshold: 0.3, replayBudget: 2, replayPhase: "before", replayOffset: -1},
		{gateType: "variance", threshold: 0.3, replayBudget: 2, replayPhase: "after", replayOffset: -1},
		{gateType: "variance", threshold: 0.5, replayBudget: 3, replayPhase: "before", replayOffset: -1},
		{gateType: "variance", threshold: 0.5, replayBudget: 3, replayPhase: "after", replayOffset: -1},
		{gateType: "gradient", threshold: 0.3, replayBudget: 2, replayPhase: "before", replayOffset: -1},
		{gateType: "gradient", threshold: 0.3, replayBudget: 2, replayPhase: "after", replayOffset: -1},
		{gateType: "gradient", threshold: 0.5, replayBudget: 3, replayPhase: "before", replayOffset: -1},
		{gateType: "gradient", threshold: 0.5, replayBudget: 3, replayPhase: "after", replayOffset: -1},
	}

	// 1) Load data
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatal(err)
	}
	trainX, trainY, _ := loadMNISTData(mnistDir, true)
	testX, testY, _ := loadMNISTData(mnistDir, false)
	trainX, trainY, _, _ = paragon.SplitDataset(trainX, trainY, 0.25) // 25% training data

	// 2) Concurrency guard
	maxWorkers := int(0.8 * float64(runtime.NumCPU()))
	if maxWorkers < 1 {
		maxWorkers = 1
	}
	sem := make(chan struct{}, maxWorkers)
	var wg sync.WaitGroup

	// 3) Result storage
	type key struct {
		replayType string
		configDesc string
		hCnt       int
	}
	type metric struct{ adh, acc float64 }
	results := make(map[key][]metric)
	var mu sync.Mutex

	// 4) Worker
	runModel := func(hCnt int, replayType, configDesc string, maxReplay int, replayPhase string, replayOffset int, gateType string, gateThreshold float64, replayBudget int, runIdx int) {
		defer wg.Done()
		sem <- struct{}{}
		defer func() { <-sem }()
		seed := time.Now().UnixNano() + int64(hCnt*1e5) + int64(runIdx*137)
		rnd := rand.New(rand.NewSource(seed))
		net := createNetworkBIGTEST(replayType, seed, hCnt, maxReplay, replayPhase, replayOffset, gateType, gateThreshold, replayBudget)
		fmt.Printf("🧠 Training %s (%s, hCnt=%d) for run %d …\n", replayType, configDesc, hCnt, runIdx)
		shX := make([][][]float64, len(trainX))
		shY := make([][][]float64, len(trainY))
		perm := rnd.Perm(len(trainX))
		for i, p := range perm {
			shX[i], shY[i] = trainX[p], trainY[p]
		}
		net.Train(shX, shY, epochs, lr, true, 5, -5)
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
		results[key{replayType, configDesc, hCnt}] = append(results[key{replayType, configDesc, hCnt}], metric{net.Performance.Score, accPct})
		mu.Unlock()
	}

	// 5) Enqueue runs
	for hCnt := 1; hCnt <= 2; hCnt++ {
		// Static replay configurations
		for _, cfg := range staticConfigs {
			configDesc := fmt.Sprintf("MaxReplay=%d,Phase=%s,Offset=%d", cfg.maxReplay, cfg.replayPhase, cfg.replayOffset)
			replayType := "static"
			if cfg.maxReplay == 0 {
				replayType = "baseline"
				configDesc = "NoReplay"
			}
			for run := 0; run < nRuns; run++ {
				wg.Add(1)
				go runModel(hCnt, replayType, configDesc, cfg.maxReplay, cfg.replayPhase, cfg.replayOffset, "", 0.0, 0, run)
			}
		}
		// Dynamic replay configurations
		for _, cfg := range dynamicConfigs {
			configDesc := fmt.Sprintf("Gate=%s,Threshold=%.1f,Budget=%d,Phase=%s,Offset=%d",
				cfg.gateType, cfg.threshold, cfg.replayBudget, cfg.replayPhase, cfg.replayOffset)
			for run := 0; run < nRuns; run++ {
				wg.Add(1)
				go runModel(hCnt, "dynamic", configDesc, 0, cfg.replayPhase, cfg.replayOffset, cfg.gateType, cfg.threshold, cfg.replayBudget, run)
			}
		}
	}
	wg.Wait()

	// 6) Write results
	for hCnt := 1; hCnt <= 2; hCnt++ {
		writeResult("\n========= MASSIVE REPLAY SETTINGS BENCHMARK (hCnt=%d, %d runs each) =========\n", hCnt, nRuns)
		writeResult("Replay Type | Configuration                          | ADHD(avg) | ±sd   | Acc%%(avg) | ±sd  \n")
		writeResult("---------------------------------------------------------------------------\n")
		// Static configurations
		for _, cfg := range staticConfigs {
			replayType := "static"
			configDesc := fmt.Sprintf("MaxReplay=%d,Phase=%s,Offset=%d", cfg.maxReplay, cfg.replayPhase, cfg.replayOffset)
			if cfg.maxReplay == 0 {
				replayType = "baseline"
				configDesc = "NoReplay"
			}
			mets := results[key{replayType, configDesc, hCnt}]
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
			writeResult("%-11s | %-37s | %8.2f | %5.2f |   %6.2f | %5.2f\n",
				replayType, configDesc, meanA, sdA, meanAcc, sdAcc)
		}
		// Dynamic configurations
		for _, cfg := range dynamicConfigs {
			configDesc := fmt.Sprintf("Gate=%s,Threshold=%.1f,Budget=%d,Phase=%s,Offset=%d",
				cfg.gateType, cfg.threshold, cfg.replayBudget, cfg.replayPhase, cfg.replayOffset)
			mets := results[key{"dynamic", configDesc, hCnt}]
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
			writeResult("%-11s | %-37s | %8.2f | %5.2f |   %6.2f | %5.2f\n",
				"dynamic", configDesc, meanA, sdA, meanAcc, sdAcc)
		}
		writeResult("---------------------------------------------------------------------------\n")
	}
	writeResult("===========================================================================\n")
}

func createNetworkBIGTEST(replayType string, seed int64, hCnt int, maxReplay int, replayPhase string, replayOffset int, gateType string, gateThreshold float64, replayBudget int) *paragon.Network[float32] {
	layers := []struct{ Width, Height int }{{28, 28}}
	for i := 0; i < hCnt; i++ {
		layers = append(layers, struct{ Width, Height int }{3, 3})
	}
	layers = append(layers, struct{ Width, Height int }{10, 1})
	acts := make([]string, len(layers))
	for i := range acts {
		acts[i] = "leaky_relu"
	}
	acts[len(acts)-1] = "softmax"
	fc := make([]bool, len(layers))
	for i := range fc {
		fc[i] = true
	}
	net := paragon.NewNetwork[float32](layers, acts, fc, seed)
	if replayType == "static" {
		net.Layers[1].ReplayOffset = replayOffset
		net.Layers[1].ReplayPhase = replayPhase
		net.Layers[1].MaxReplay = maxReplay
	} else if replayType == "dynamic" {
		net.Layers[1].ReplayEnabled = true
		net.Layers[1].ReplayBudget = replayBudget
		net.Layers[1].ReplayPhase = replayPhase
		net.Layers[1].ReplayOffset = replayOffset
		switch gateType {
		case "entropy":
			net.Layers[1].ReplayGateFunc = func(input [][]float32) float64 {
				outputs := net.Layers[1].CachedOutputs
				if len(outputs) == 0 {
					return 0.5
				}
				var sum float64
				for _, v := range outputs {
					v64 := float64(v)
					if v64 > 1e-10 {
						sum += v64 * math.Log(v64)
					}
				}
				entropy := -sum / float64(len(outputs))
				maxEntropy := math.Log(float64(len(outputs)))
				if maxEntropy == 0 {
					return 0.5
				}
				return math.Min(1.0, math.Max(0.0, entropy/maxEntropy))
			}
		case "variance":
			net.Layers[1].ReplayGateFunc = func(input [][]float32) float64 {
				outputs := net.Layers[1].CachedOutputs
				if len(outputs) == 0 {
					return 0.5
				}
				var mean, sumSq float64
				n := float64(len(outputs))
				for _, v := range outputs {
					v64 := float64(v)
					mean += v64 / n
					sumSq += v64 * v64
				}
				variance := sumSq/n - mean*mean
				return math.Min(1.0, math.Max(0.0, variance/0.1))
			}
		case "gradient":
			net.Layers[1].ReplayGateFunc = func(input [][]float32) float64 {
				outputs := net.Layers[1].CachedOutputs
				if len(outputs) < 2 {
					return 0.5
				}
				var sumDiff float64
				for i := 1; i < len(outputs); i++ {
					diff := float64(outputs[i] - outputs[i-1])
					sumDiff += diff * diff
				}
				gradient := sumDiff / float64(len(outputs)-1)
				return math.Min(1.0, math.Max(0.0, gradient/0.1))
			}
		}
		net.Layers[1].ReplayGateToReps = func(score float64) int {
			if score > gateThreshold {
				return replayBudget
			}
			return 0
		}
	}
	return net
}

func benchmarkDeepReplaySettings() {
	const (
		nRuns          = 6
		epochs         = 1
		lr             = 0.001
		inputW, inputH = 28, 28
		outputW        = 10
	)

	// Configurations to test
	staticConfigs := []struct {
		maxReplay    int
		replayPhase  string
		replayOffset int
	}{
		{maxReplay: 0, replayPhase: "before", replayOffset: -1}, // Baseline
		{maxReplay: 1, replayPhase: "before", replayOffset: -1},
		{maxReplay: 2, replayPhase: "before", replayOffset: -1},
		{maxReplay: 3, replayPhase: "before", replayOffset: -1},
		{maxReplay: 2, replayPhase: "before", replayOffset: -2},
	}
	dynamicConfigs := []struct {
		gateType     string
		threshold    float64
		replayBudget int
		replayPhase  string
		replayOffset int
	}{
		{gateType: "variance", threshold: 0.4, replayBudget: 1, replayPhase: "before", replayOffset: -1},
		{gateType: "variance", threshold: 0.6, replayBudget: 2, replayPhase: "before", replayOffset: -1},
		{gateType: "confidence", threshold: 0.4, replayBudget: 1, replayPhase: "before", replayOffset: -1},
		{gateType: "confidence", threshold: 0.6, replayBudget: 2, replayPhase: "before", replayOffset: -1},
	}

	// 1) Load data
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatal(err)
	}
	trainX, trainY, _ := loadMNISTData(mnistDir, true)
	testX, testY, _ := loadMNISTData(mnistDir, false)
	trainX, trainY, _, _ = paragon.SplitDataset(trainX, trainY, 0.2) // 20% training data

	// 2) Concurrency guard
	maxWorkers := int(0.8 * float64(runtime.NumCPU()))
	if maxWorkers < 1 {
		maxWorkers = 1
	}
	sem := make(chan struct{}, maxWorkers)
	var wg sync.WaitGroup

	// 3) Result storage
	type key struct {
		replayType string
		configDesc string
		hCnt       int
	}
	type metric struct{ adh, acc float64 }
	results := make(map[key][]metric)
	var mu sync.Mutex

	// 4) Worker
	runModel := func(hCnt int, replayType, configDesc string, maxReplay int, replayPhase string, replayOffset int, gateType string, gateThreshold float64, replayBudget int, runIdx int) {
		defer wg.Done()
		sem <- struct{}{}
		defer func() { <-sem }()
		seed := time.Now().UnixNano() + int64(hCnt*1e5) + int64(runIdx*137)
		rnd := rand.New(rand.NewSource(seed))
		net := createNetworkBIGTEST(replayType, seed, hCnt, maxReplay, replayPhase, replayOffset, gateType, gateThreshold, replayBudget)
		fmt.Printf("🧠 Training %s (%s, hCnt=%d) for run %d …\n", replayType, configDesc, hCnt, runIdx)
		shX := make([][][]float64, len(trainX))
		shY := make([][][]float64, len(trainY))
		perm := rnd.Perm(len(trainX))
		for i, p := range perm {
			shX[i], shY[i] = trainX[p], trainY[p]
		}
		net.Train(shX, shY, epochs, lr, true, 5, -5)
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
		results[key{replayType, configDesc, hCnt}] = append(results[key{replayType, configDesc, hCnt}], metric{net.Performance.Score, accPct})
		mu.Unlock()
	}

	// 5) Enqueue runs
	for hCnt := 3; hCnt <= 5; hCnt++ {
		// Static replay configurations
		for _, cfg := range staticConfigs {
			configDesc := fmt.Sprintf("MaxReplay=%d,Phase=%s,Offset=%d", cfg.maxReplay, cfg.replayPhase, cfg.replayOffset)
			replayType := "static"
			if cfg.maxReplay == 0 {
				replayType = "baseline"
				configDesc = "NoReplay"
			}
			for run := 0; run < nRuns; run++ {
				wg.Add(1)
				go runModel(hCnt, replayType, configDesc, cfg.maxReplay, cfg.replayPhase, cfg.replayOffset, "", 0.0, 0, run)
			}
		}
		// Dynamic replay configurations
		for _, cfg := range dynamicConfigs {
			configDesc := fmt.Sprintf("Gate=%s,Threshold=%.1f,Budget=%d,Phase=%s,Offset=%d",
				cfg.gateType, cfg.threshold, cfg.replayBudget, cfg.replayPhase, cfg.replayOffset)
			for run := 0; run < nRuns; run++ {
				wg.Add(1)
				go runModel(hCnt, "dynamic", configDesc, 0, cfg.replayPhase, cfg.replayOffset, cfg.gateType, cfg.threshold, cfg.replayBudget, run)
			}
		}
	}
	wg.Wait()

	// 6) Write results
	for hCnt := 3; hCnt <= 5; hCnt++ {
		writeResult("\n========= DEEP REPLAY SETTINGS BENCHMARK (hCnt=%d, %d runs each) =========\n", hCnt, nRuns)
		writeResult("Replay Type | Configuration                          | ADHD(avg) | ±sd   | Acc%%(avg) | ±sd  \n")
		writeResult("---------------------------------------------------------------------------\n")
		// Static configurations
		for _, cfg := range staticConfigs {
			replayType := "static"
			configDesc := fmt.Sprintf("MaxReplay=%d,Phase=%s,Offset=%d", cfg.maxReplay, cfg.replayPhase, cfg.replayOffset)
			if cfg.maxReplay == 0 {
				replayType = "baseline"
				configDesc = "NoReplay"
			}
			mets := results[key{replayType, configDesc, hCnt}]
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
			writeResult("%-11s | %-37s | %8.2f | %5.2f |   %6.2f | %5.2f\n",
				replayType, configDesc, meanA, sdA, meanAcc, sdAcc)
		}
		// Dynamic configurations
		for _, cfg := range dynamicConfigs {
			configDesc := fmt.Sprintf("Gate=%s,Threshold=%.1f,Budget=%d,Phase=%s,Offset=%d",
				cfg.gateType, cfg.threshold, cfg.replayBudget, cfg.replayPhase, cfg.replayOffset)
			mets := results[key{"dynamic", configDesc, hCnt}]
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
			writeResult("%-11s | %-37s | %8.2f | %5.2f |   %6.2f | %5.2f\n",
				"dynamic", configDesc, meanA, sdA, meanAcc, sdAcc)
		}
		writeResult("---------------------------------------------------------------------------\n")
	}
	writeResult("===========================================================================\n")
}

func benchmarkDynamicReplayOptimizer() {
	const (
		nRuns          = 6
		epochs         = 1
		lr             = 0.001
		inputW, inputH = 28, 28
		outputW        = 10
		hCnt           = 3
	)

	// Configurations to test
	configs := []struct {
		gateType     string
		threshold    float64
		replayBudget int
		replayPhase  string
		replayOffset int
	}{
		// Baseline (no replay)
		{gateType: "none", threshold: 0.0, replayBudget: 0, replayPhase: "before", replayOffset: -1},
		// Variance-based
		{gateType: "variance", threshold: 0.3, replayBudget: 1, replayPhase: "before", replayOffset: -1},
		{gateType: "variance", threshold: 0.5, replayBudget: 2, replayPhase: "before", replayOffset: -1},
		{gateType: "variance", threshold: 0.7, replayBudget: 3, replayPhase: "before", replayOffset: -1},
		{gateType: "variance", threshold: 0.5, replayBudget: 2, replayPhase: "after", replayOffset: -1},
		{gateType: "variance", threshold: 0.5, replayBudget: 2, replayPhase: "before", replayOffset: -2},
		// Entropy-based
		{gateType: "entropy", threshold: 0.3, replayBudget: 1, replayPhase: "before", replayOffset: -1},
		{gateType: "entropy", threshold: 0.5, replayBudget: 2, replayPhase: "before", replayOffset: -1},
		{gateType: "entropy", threshold: 0.7, replayBudget: 3, replayPhase: "before", replayOffset: -1},
		{gateType: "entropy", threshold: 0.5, replayBudget: 2, replayPhase: "after", replayOffset: -1},
		// Confidence-based
		{gateType: "confidence", threshold: 0.3, replayBudget: 1, replayPhase: "before", replayOffset: -1},
		{gateType: "confidence", threshold: 0.5, replayBudget: 2, replayPhase: "before", replayOffset: -1},
		{gateType: "confidence", threshold: 0.7, replayBudget: 3, replayPhase: "before", replayOffset: -1},
		// Gradient-based
		{gateType: "gradient", threshold: 0.3, replayBudget: 1, replayPhase: "before", replayOffset: -1},
		{gateType: "gradient", threshold: 0.5, replayBudget: 2, replayPhase: "before", replayOffset: -1},
		{gateType: "gradient", threshold: 0.7, replayBudget: 3, replayPhase: "before", replayOffset: -1},
		// Loss-based
		{gateType: "loss", threshold: 0.3, replayBudget: 1, replayPhase: "before", replayOffset: -1},
		{gateType: "loss", threshold: 0.5, replayBudget: 2, replayPhase: "before", replayOffset: -1},
		{gateType: "loss", threshold: 0.7, replayBudget: 3, replayPhase: "before", replayOffset: -1},
		// Hybrid (Variance + Entropy)
		{gateType: "hybrid", threshold: 0.3, replayBudget: 1, replayPhase: "before", replayOffset: -1},
		{gateType: "hybrid", threshold: 0.5, replayBudget: 2, replayPhase: "before", replayOffset: -1},
		{gateType: "hybrid", threshold: 0.7, replayBudget: 3, replayPhase: "before", replayOffset: -1},
	}

	// 1) Load data
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatal(err)
	}
	trainX, trainY, _ := loadMNISTData(mnistDir, true)
	testX, testY, _ := loadMNISTData(mnistDir, false)
	trainX, trainY, _, _ = paragon.SplitDataset(trainX, trainY, 0.2) // 20% training data

	// 2) Concurrency guard
	maxWorkers := int(0.8 * float64(runtime.NumCPU()))
	if maxWorkers < 1 {
		maxWorkers = 1
	}
	sem := make(chan struct{}, maxWorkers)
	var wg sync.WaitGroup

	// 3) Result storage
	type key struct {
		replayType string
		configDesc string
	}
	type metric struct{ adh, acc float64 }
	results := make(map[key][]metric)
	var mu sync.Mutex

	// 4) Worker
	runModel := func(replayType, configDesc string, gateType string, gateThreshold float64, replayBudget int, replayPhase string, replayOffset int, runIdx int) {
		defer wg.Done()
		sem <- struct{}{}
		defer func() { <-sem }()
		seed := time.Now().UnixNano() + int64(runIdx*137)
		rnd := rand.New(rand.NewSource(seed))
		net := createNetworkBIGTEST(replayType, seed, hCnt, 0, replayPhase, replayOffset, gateType, gateThreshold, replayBudget)
		fmt.Printf("🧠 Training %s (%s) for run %d …\n", replayType, configDesc, runIdx)
		shX := make([][][]float64, len(trainX))
		shY := make([][][]float64, len(trainY))
		perm := rnd.Perm(len(trainX))
		for i, p := range perm {
			shX[i], shY[i] = trainX[p], trainY[p]
		}
		net.Train(shX, shY, epochs, lr, true, 5, -5)
		correct := 0
		exp, pred := []float64{}, []float64{}
		for i, in := range testX {
			net.Forward(in)
			p := float64(paragon.ArgMax(net.ExtractOutput()))
			t := float64(paragon.ArgMax(testY[i][0]))
			pred = append(pred, p)
			exp = append(exp, t)
			if p == t {
				correct++
			}
		}
		net.EvaluateModel(exp, pred)
		accPct := float64(correct) / float64(len(testX)) * 100.0
		mu.Lock()
		results[key{replayType, configDesc}] = append(results[key{replayType, configDesc}], metric{net.Performance.Score, accPct})
		mu.Unlock()
	}

	// 5) Enqueue runs
	for _, cfg := range configs {
		replayType := "dynamic"
		if cfg.gateType == "none" {
			replayType = "baseline"
		}
		configDesc := fmt.Sprintf("Gate=%s,Threshold=%.1f,Budget=%d,Phase=%s,Offset=%d",
			cfg.gateType, cfg.threshold, cfg.replayBudget, cfg.replayPhase, cfg.replayOffset)
		if cfg.gateType == "none" {
			configDesc = "NoReplay"
		}
		for run := 0; run < nRuns; run++ {
			wg.Add(1)
			go runModel(replayType, configDesc, cfg.gateType, cfg.threshold, cfg.replayBudget, cfg.replayPhase, cfg.replayOffset, run)
		}
	}
	wg.Wait()

	// 6) Write results
	writeResult("\n========= DYNAMIC REPLAY OPTIMIZER BENCHMARK (hCnt=%d, %d runs each) =========\n", hCnt, nRuns)
	writeResult("Replay Type | Configuration                          | ADHD(avg) | ±sd   | Acc%%(avg) | ±sd  \n")
	writeResult("---------------------------------------------------------------------------\n")
	for _, cfg := range configs {
		replayType := "dynamic"
		if cfg.gateType == "none" {
			replayType = "baseline"
		}
		configDesc := fmt.Sprintf("Gate=%s,Threshold=%.1f,Budget=%d,Phase=%s,Offset=%d",
			cfg.gateType, cfg.threshold, cfg.replayBudget, cfg.replayPhase, cfg.replayOffset)
		if cfg.gateType == "none" {
			configDesc = "NoReplay"
		}
		mets := results[key{replayType, configDesc}]
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
		writeResult("%-11s | %-37s | %8.2f | %5.2f |   %6.2f | %5.2f\n",
			replayType, configDesc, meanA, sdA, meanAcc, sdAcc)
	}
	writeResult("---------------------------------------------------------------------------\n")
	writeResult("===========================================================================\n")
}

func benchmarkAdaptiveTemporalReplay() {
	const (
		nRuns          = 6
		epochs         = 1
		lr             = 0.001
		inputW, inputH = 28, 28
		outputW        = 10
		hCnt           = 3
	)

	// Configurations to test
	configs := []struct {
		gateType     string
		threshold    float64
		replayBudget int
		replayPhase  string
		replayOffset int
	}{
		// Baseline (no replay)
		{gateType: "none", threshold: 0.0, replayBudget: 0, replayPhase: "before", replayOffset: -1},
		// Adaptive Temporal Replay
		{gateType: "temporal", threshold: 0.5, replayBudget: 2, replayPhase: "before", replayOffset: -1},
	}

	// 1) Load data
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatal(err)
	}
	trainX, trainY, _ := loadMNISTData(mnistDir, true)
	testX, testY, _ := loadMNISTData(mnistDir, false)
	trainX, trainY, _, _ = paragon.SplitDataset(trainX, trainY, 0.2) // 20% training data

	// 2) Concurrency guard
	maxWorkers := int(0.8 * float64(runtime.NumCPU()))
	if maxWorkers < 1 {
		maxWorkers = 1
	}
	sem := make(chan struct{}, maxWorkers)
	var wg sync.WaitGroup

	// 3) Result storage
	type key struct {
		replayType string
		configDesc string
	}
	type metric struct{ adh, acc float64 }
	results := make(map[key][]metric)
	var mu sync.Mutex

	// 4) Worker
	runModel := func(replayType, configDesc string, gateType string, gateThreshold float64, replayBudget int, replayPhase string, replayOffset int, runIdx int) {
		defer wg.Done()
		sem <- struct{}{}
		defer func() { <-sem }()
		seed := time.Now().UnixNano() + int64(runIdx*137)
		rnd := rand.New(rand.NewSource(seed))
		net := createNetworkTEMPORTAL(replayType, seed, hCnt, 0, replayPhase, replayOffset, gateType, gateThreshold, replayBudget)
		fmt.Printf("🧠 Training %s (%s) for run %d …\n", replayType, configDesc, runIdx)
		shX := make([][][]float64, len(trainX))
		shY := make([][][]float64, len(trainY))
		perm := rnd.Perm(len(trainX))
		for i, p := range perm {
			shX[i], shY[i] = trainX[p], trainY[p]
		}
		net.Train(shX, shY, epochs, lr, true, 5, -5)
		correct := 0
		exp, pred := []float64{}, []float64{}
		for i, in := range testX {
			net.Forward(in)
			p := float64(paragon.ArgMax(net.ExtractOutput()))
			t := float64(paragon.ArgMax(testY[i][0]))
			pred = append(pred, p)
			exp = append(exp, t)
			if p == t {
				correct++
			}
		}
		net.EvaluateModel(exp, pred)
		accPct := float64(correct) / float64(len(testX)) * 100.0
		mu.Lock()
		results[key{replayType, configDesc}] = append(results[key{replayType, configDesc}], metric{net.Performance.Score, accPct})
		mu.Unlock()
	}

	// 5) Enqueue runs
	for _, cfg := range configs {
		replayType := "dynamic"
		if cfg.gateType == "none" {
			replayType = "baseline"
		}
		configDesc := fmt.Sprintf("Gate=%s,Threshold=%.1f,Budget=%d,Phase=%s,Offset=%d",
			cfg.gateType, cfg.threshold, cfg.replayBudget, cfg.replayPhase, cfg.replayOffset)
		if cfg.gateType == "none" {
			configDesc = "NoReplay"
		}
		for run := 0; run < nRuns; run++ {
			wg.Add(1)
			go runModel(replayType, configDesc, cfg.gateType, cfg.threshold, cfg.replayBudget, cfg.replayPhase, cfg.replayOffset, run)
		}
	}
	wg.Wait()

	// 6) Write results
	writeResult("\n========= ADAPTIVE TEMPORAL REPLAY BENCHMARK (hCnt=%d, %d runs each) =========\n", hCnt, nRuns)
	writeResult("Replay Type | Configuration                          | ADHD(avg) | ±sd   | Acc%%(avg) | ±sd  \n")
	writeResult("---------------------------------------------------------------------------\n")
	for _, cfg := range configs {
		replayType := "dynamic"
		if cfg.gateType == "none" {
			replayType = "baseline"
		}
		configDesc := fmt.Sprintf("Gate=%s,Threshold=%.1f,Budget=%d,Phase=%s,Offset=%d",
			cfg.gateType, cfg.threshold, cfg.replayBudget, cfg.replayPhase, cfg.replayOffset)
		if cfg.gateType == "none" {
			configDesc = "NoReplay"
		}
		mets := results[key{replayType, configDesc}]
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
		writeResult("%-11s | %-37s | %8.2f | %5.2f |   %6.2f | %5.2f\n",
			replayType, configDesc, meanA, sdA, meanAcc, sdAcc)
	}
	writeResult("---------------------------------------------------------------------------\n")
	writeResult("===========================================================================\n")
}

func createNetworkTEMPORTAL(replayType string, seed int64, hCnt int, maxReplay int, replayPhase string, replayOffset int, gateType string, gateThreshold float64, replayBudget int) *paragon.Network[float32] {
	layers := []struct{ Width, Height int }{{28, 28}}
	for i := 0; i < hCnt; i++ {
		layers = append(layers, struct{ Width, Height int }{3, 3})
	}
	layers = append(layers, struct{ Width, Height int }{10, 1})
	acts := make([]string, len(layers))
	for i := range acts {
		acts[i] = "leaky_relu"
	}
	acts[len(acts)-1] = "softmax"
	fc := make([]bool, len(layers))
	for i := range fc {
		fc[i] = true
	}
	net := paragon.NewNetwork[float32](layers, acts, fc, seed)
	if replayType == "dynamic" {
		net.Layers[1].ReplayEnabled = true
		net.Layers[1].ReplayBudget = replayBudget
		net.Layers[1].ReplayPhase = replayPhase
		net.Layers[1].ReplayOffset = replayOffset
		switch gateType {
		case "variance":
			net.Layers[1].ReplayGateFunc = func(input [][]float32) float64 {
				outputs := net.Layers[1].CachedOutputs
				if len(outputs) == 0 {
					return 0.5
				}
				var mean, sumSq float64
				n := float64(len(outputs))
				for _, v := range outputs {
					v64 := float64(v)
					mean += v64 / n
					sumSq += v64 * v64
				}
				variance := sumSq/n - mean*mean
				score := math.Min(1.0, math.Max(0.0, variance/0.1))
				log.Printf("Variance score: %f", score)
				return score
			}
		case "temporal":
			net.Layers[1].ReplayGateFunc = func(input [][]float32) float64 {
				outputs := net.Layers[1].CachedOutputsHistory
				if len(outputs) < 5 {
					return 0.5 // Default until enough history
				}
				var avgVar float64
				for i := 0; i < len(outputs[0]); i++ {
					var mean, sumSq float64
					n := float64(len(outputs))
					for _, hist := range outputs {
						v64 := float64(hist[i])
						mean += v64 / n
						sumSq += v64 * v64
					}
					variance := sumSq/n - mean*mean
					avgVar += variance / 0.1 // Fixed divisor
				}
				score := avgVar / float64(len(outputs[0]))
				log.Printf("Temporal score: %f", score)
				return math.Min(1.0, math.Max(0.0, score))
			}
		case "hybrid":
			net.Layers[1].ReplayGateFunc = func(input [][]float32) float64 {
				outputs := net.Layers[1].CachedOutputsHistory
				currOutputs := net.Layers[1].CachedOutputs
				if len(outputs) < 5 || len(currOutputs) == 0 {
					return 0.5
				}
				// Temporal score
				var avgVar float64
				for i := 0; i < len(outputs[0]); i++ {
					var mean, sumSq float64
					n := float64(len(outputs))
					for _, hist := range outputs {
						v64 := float64(hist[i])
						mean += v64 / n
						sumSq += v64 * v64
					}
					variance := sumSq/n - mean*mean
					avgVar += variance / 0.1
				}
				temporalScore := avgVar / float64(len(outputs[0]))
				// Spatial variance score
				var mean, sumSq float64
				n := float64(len(currOutputs))
				for _, v := range currOutputs {
					v64 := float64(v)
					mean += v64 / n
					sumSq += v64 * v64
				}
				spatialVariance := sumSq/n - mean*mean
				spatialScore := math.Min(1.0, math.Max(0.0, spatialVariance/0.1))
				// Combined score
				score := 0.6*temporalScore + 0.4*spatialScore
				log.Printf("Hybrid Temporal-Spatial score: %f (Temporal: %f, Spatial: %f)", score, temporalScore, spatialScore)
				return math.Min(1.0, math.Max(0.0, score))
			}
		}
		net.Layers[1].ReplayGateToReps = func(score float64) int {
			if score > gateThreshold {
				return int(math.Min(float64(replayBudget), math.Ceil(score*float64(replayBudget))))
			}
			return 0
		}
	}
	return net
}

func benchmarkEnhancedTemporalReplay() {
	const (
		nRuns          = 6
		epochs         = 1
		lr             = 0.001
		inputW, inputH = 28, 28
		outputW        = 10
		hCnt           = 3
	)

	// Configurations to test
	configs := []struct {
		gateType     string
		threshold    float64
		replayBudget int
		replayPhase  string
		replayOffset int
	}{
		// Baseline (no replay)
		{gateType: "none", threshold: 0.0, replayBudget: 0, replayPhase: "before", replayOffset: -1},
		// Variance-based (best prior performer)
		{gateType: "variance", threshold: 0.3, replayBudget: 1, replayPhase: "before", replayOffset: -1},
		// Enhanced ATR
		{gateType: "temporal", threshold: 0.3, replayBudget: 1, replayPhase: "before", replayOffset: -1},
		{gateType: "temporal", threshold: 0.4, replayBudget: 2, replayPhase: "before", replayOffset: -1},
		{gateType: "temporal", threshold: 0.5, replayBudget: 3, replayPhase: "before", replayOffset: -1},
		// Hybrid Temporal-Spatial
		{gateType: "hybrid", threshold: 0.3, replayBudget: 1, replayPhase: "before", replayOffset: -1},
		{gateType: "hybrid", threshold: 0.4, replayBudget: 2, replayPhase: "before", replayOffset: -1},
		{gateType: "hybrid", threshold: 0.5, replayBudget: 3, replayPhase: "before", replayOffset: -1},
	}

	// 1) Load data
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatal(err)
	}
	trainX, trainY, _ := loadMNISTData(mnistDir, true)
	testX, testY, _ := loadMNISTData(mnistDir, false)
	trainX, trainY, _, _ = paragon.SplitDataset(trainX, trainY, 0.2) // 20% training data

	// 2) Concurrency guard
	maxWorkers := int(0.8 * float64(runtime.NumCPU()))
	if maxWorkers < 1 {
		maxWorkers = 1
	}
	sem := make(chan struct{}, maxWorkers)
	var wg sync.WaitGroup

	// 3) Result storage
	type key struct {
		replayType string
		configDesc string
	}
	type metric struct{ adh, acc float64 }
	results := make(map[key][]metric)
	var mu sync.Mutex

	// 4) Worker
	runModel := func(replayType, configDesc, gateType string, threshold float64, replayBudget int, replayPhase string, replayOffset int, runIdx int) {
		defer wg.Done()
		sem <- struct{}{}
		defer func() { <-sem }()
		seed := time.Now().UnixNano() + int64(runIdx*137)
		rnd := rand.New(rand.NewSource(seed))
		net := createNetworkTEMPORTAL(replayType, seed, hCnt, 0, replayPhase, replayOffset, gateType, threshold, replayBudget)
		fmt.Printf("🧠 Training %s (%s) for run %d …\n", replayType, configDesc, runIdx)
		shX := make([][][]float64, len(trainX))
		shY := make([][][]float64, len(trainY))
		perm := rnd.Perm(len(trainX))
		for i, p := range perm {
			shX[i], shY[i] = trainX[p], trainY[p]
		}
		net.Train(shX, shY, epochs, lr, true, 5, -5)
		correct := 0
		exp, pred := []float64{}, []float64{}
		for i, in := range testX {
			net.Forward(in)
			p := float64(paragon.ArgMax(net.ExtractOutput()))
			t := float64(paragon.ArgMax(testY[i][0]))
			pred = append(pred, p)
			exp = append(exp, t)
			if p == t {
				correct++
			}
		}
		net.EvaluateModel(exp, pred)
		accPct := float64(correct) / float64(len(testX)) * 100.0
		mu.Lock()
		results[key{replayType, configDesc}] = append(results[key{replayType, configDesc}], metric{net.Performance.Score, accPct})
		mu.Unlock()
	}

	// 5) Enqueue runs
	for _, cfg := range configs {
		replayType := "dynamic"
		if cfg.gateType == "none" {
			replayType = "baseline"
		}
		configDesc := fmt.Sprintf("Gate=%s,Threshold=%.1f,Budget=%d,Phase=%s,Offset=%d",
			cfg.gateType, cfg.threshold, cfg.replayBudget, cfg.replayPhase, cfg.replayOffset)
		if cfg.gateType == "none" {
			configDesc = "NoReplay"
		}
		for run := 0; run < nRuns; run++ {
			wg.Add(1)
			go runModel(replayType, configDesc, cfg.gateType, cfg.threshold, cfg.replayBudget, cfg.replayPhase, cfg.replayOffset, run)
		}
	}
	wg.Wait()

	// 6) Write results
	writeResult("\n========= ENHANCED TEMPORAL REPLAY BENCHMARK (hCnt=%d, %d runs each) =========\n", hCnt, nRuns)
	writeResult("Replay Type | Configuration                          | ADHD(avg) | ±sd   | Acc%%(avg) | ±sd  \n")
	writeResult("---------------------------------------------------------------------------\n")
	for _, cfg := range configs {
		replayType := "dynamic"
		if cfg.gateType == "none" {
			replayType = "baseline"
		}
		configDesc := fmt.Sprintf("Gate=%s,Threshold=%.1f,Budget=%d,Phase=%s,Offset=%d",
			cfg.gateType, cfg.threshold, cfg.replayBudget, cfg.replayPhase, cfg.replayOffset)
		if cfg.gateType == "none" {
			configDesc = "NoReplay"
		}
		mets := results[key{replayType, configDesc}]
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
		writeResult("%-11s | %-37s | %8.2f | %5.2f |   %6.2f | %5.2f\n",
			replayType, configDesc, meanA, sdA, meanAcc, sdAcc)
	}
	writeResult("---------------------------------------------------------------------------\n")
	writeResult("===========================================================================\n")
}
