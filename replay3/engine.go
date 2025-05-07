package main

import (
	"fmt"
	"log"
	"paragon"
	"runtime"
	"sort"
	"sync"
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
	//testReplayVariantsParallel()
	testReplayVariantsWithLowerLR()
}

func testReplayVariantsParallel() {
	// 1) Load MNIST once
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatalf("MNIST download error: %v", err)
	}
	trainX, trainY, _ := loadMNISTData(mnistDir, true)
	testX, testY, _ := loadMNISTData(mnistDir, false)
	trainX, trainY, _, _ = paragon.SplitDataset(trainX, trainY, 0.8)

	// 2) Shared architecture config
	layer := []struct{ Width, Height int }{{28, 28}, {16, 16}, {10, 1}}
	acts := []string{"leaky_relu", "leaky_relu", "softmax"}
	fc := []bool{true, false, true}

	type result struct {
		kind    string
		phase   string
		repeats int
		score   float64
		acc     float64
	}

	var results []result
	var mu sync.Mutex
	var wg sync.WaitGroup

	// 3) Limit concurrency to ~80% of available cores
	maxThreads := int(0.8 * float64(runtime.NumCPU()))
	if maxThreads < 1 {
		maxThreads = 1
	}
	sem := make(chan struct{}, maxThreads)

	// 4) Variants to test
	variants := []string{"before", "after"}
	for _, phase := range variants {
		for replay := 0; replay <= 3; replay++ {
			wg.Add(1)
			sem <- struct{}{} // acquire slot

			go func(phase string, replay int) {
				defer wg.Done()
				defer func() { <-sem }() // release slot

				// Build network
				net := paragon.NewNetwork(layer, acts, fc)
				net.Layers[1].ReplayOffset = -1
				net.Layers[1].ReplayPhase = phase
				net.Layers[1].MaxReplay = replay

				// Train
				fmt.Printf("🔁 Training Replay (%s, MaxReplay=%d)…\n", phase, replay)
				net.Train(trainX, trainY, 20, 0.001, true)

				// Evaluate
				exp, pred := []float64{}, []float64{}
				correct := 0
				for i, in := range testX {
					net.Forward(in)
					p := float64(paragon.ArgMax(net.ExtractOutput()))
					t := float64(paragon.ArgMax(testY[i][0]))
					exp = append(exp, t)
					pred = append(pred, p)
					if p == t {
						correct++
					}
				}
				net.EvaluateModel(exp, pred)

				mu.Lock()
				results = append(results, result{
					kind:    "replay",
					phase:   phase,
					repeats: replay,
					score:   net.Performance.Score,
					acc:     float64(correct) / float64(len(testX)) * 100.0,
				})
				mu.Unlock()
			}(phase, replay)
		}
	}

	// 5) Add baseline model (in parallel too)
	wg.Add(1)
	sem <- struct{}{}
	go func() {
		defer wg.Done()
		defer func() { <-sem }()

		baseNet := paragon.NewNetwork(layer, acts, fc)
		fmt.Println("🧠 Training Baseline …")
		baseNet.Train(trainX, trainY, 20, 0.001, true)

		exp, pred := []float64{}, []float64{}
		correct := 0
		for i, in := range testX {
			baseNet.Forward(in)
			p := float64(paragon.ArgMax(baseNet.ExtractOutput()))
			t := float64(paragon.ArgMax(testY[i][0]))
			exp = append(exp, t)
			pred = append(pred, p)
			if p == t {
				correct++
			}
		}
		baseNet.EvaluateModel(exp, pred)

		mu.Lock()
		results = append(results, result{
			kind:    "baseline",
			phase:   "-",
			repeats: 0,
			score:   baseNet.Performance.Score,
			acc:     float64(correct) / float64(len(testX)) * 100.0,
		})
		mu.Unlock()
	}()

	// 6) Wait for all runs to finish
	wg.Wait()

	// 7) Sort results by kind, phase, and repeats (optional)
	sort.Slice(results, func(i, j int) bool {
		if results[i].kind != results[j].kind {
			return results[i].kind < results[j].kind
		}
		if results[i].phase != results[j].phase {
			return results[i].phase < results[j].phase
		}
		return results[i].repeats < results[j].repeats
	})

	// 8) Print result table
	fmt.Println("\n================ PARALLEL REPLAY TEST ====================")
	fmt.Printf("%-10s | %-6s | Repeats | ADHD   | Acc%%\n", "Kind", "Phase")
	fmt.Println("----------------------------------------------------------")
	for _, r := range results {
		fmt.Printf("%-10s | %-6s |   %d     | %6.2f | %5.2f\n",
			r.kind, r.phase, r.repeats, r.score, r.acc)
	}
	fmt.Println("==========================================================")
}

func testReplayVariantsWithLowerLR() {
	// 1) Load data
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatalf("MNIST download error: %v", err)
	}
	trainX, trainY, _ := loadMNISTData(mnistDir, true)
	testX, testY, _ := loadMNISTData(mnistDir, false)
	trainX, trainY, _, _ = paragon.SplitDataset(trainX, trainY, 0.8)

	// 2) Shared config
	layer := []struct{ Width, Height int }{{28, 28}, {16, 16}, {10, 1}}
	acts := []string{"leaky_relu", "leaky_relu", "softmax"}
	fc := []bool{true, false, true}

	type result struct {
		kind    string
		phase   string
		repeats int
		score   float64
		acc     float64
	}

	var results []result
	var mu sync.Mutex
	var wg sync.WaitGroup

	maxThreads := int(0.8 * float64(runtime.NumCPU()))
	if maxThreads < 1 {
		maxThreads = 1
	}
	sem := make(chan struct{}, maxThreads)

	// 3) Phase and repeat configs
	phases := []string{"before", "after"}
	repeatRange := []int{0, 1, 2, 3}
	lrVariants := []struct {
		label   string
		lrScale float64
	}{
		{"replay", 1.0},        // normal LR
		{"replay_lrDown", 0.5}, // half LR for replays
	}

	for _, phase := range phases {
		for _, repeat := range repeatRange {
			for _, variant := range lrVariants {
				wg.Add(1)
				sem <- struct{}{}

				go func(phase string, repeat int, vlabel string, lrScale float64) {
					defer wg.Done()
					defer func() { <-sem }()

					net := paragon.NewNetwork(layer, acts, fc)
					net.Layers[1].ReplayOffset = -1
					net.Layers[1].ReplayPhase = phase
					net.Layers[1].MaxReplay = repeat

					lr := 0.001 * lrScale
					fmt.Printf("🔁 Training %s (Phase=%s, Replay=%d, LR=%.4f)…\n", vlabel, phase, repeat, lr)
					net.Train(trainX, trainY, 20, lr, true)

					exp, pred := []float64{}, []float64{}
					correct := 0
					for i, in := range testX {
						net.Forward(in)
						p := float64(paragon.ArgMax(net.ExtractOutput()))
						t := float64(paragon.ArgMax(testY[i][0]))
						exp = append(exp, t)
						pred = append(pred, p)
						if p == t {
							correct++
						}
					}
					net.EvaluateModel(exp, pred)

					mu.Lock()
					results = append(results, result{
						kind:    vlabel,
						phase:   phase,
						repeats: repeat,
						score:   net.Performance.Score,
						acc:     float64(correct) / float64(len(testX)) * 100.0,
					})
					mu.Unlock()
				}(phase, repeat, variant.label, variant.lrScale)
			}
		}
	}

	// 4) Baseline
	wg.Add(1)
	sem <- struct{}{}
	go func() {
		defer wg.Done()
		defer func() { <-sem }()
		baseNet := paragon.NewNetwork(layer, acts, fc)
		baseNet.Train(trainX, trainY, 20, 0.001, true)

		exp, pred := []float64{}, []float64{}
		correct := 0
		for i, in := range testX {
			baseNet.Forward(in)
			p := float64(paragon.ArgMax(baseNet.ExtractOutput()))
			t := float64(paragon.ArgMax(testY[i][0]))
			exp = append(exp, t)
			pred = append(pred, p)
			if p == t {
				correct++
			}
		}
		baseNet.EvaluateModel(exp, pred)

		mu.Lock()
		results = append(results, result{
			kind:    "baseline",
			phase:   "-",
			repeats: 0,
			score:   baseNet.Performance.Score,
			acc:     float64(correct) / float64(len(testX)) * 100.0,
		})
		mu.Unlock()
	}()

	// 5) Wait and print
	wg.Wait()

	sort.Slice(results, func(i, j int) bool {
		if results[i].kind != results[j].kind {
			return results[i].kind < results[j].kind
		}
		if results[i].phase != results[j].phase {
			return results[i].phase < results[j].phase
		}
		return results[i].repeats < results[j].repeats
	})

	fmt.Println("\n================ REPLAY + LR VARIANT TEST ====================")
	fmt.Printf("%-15s | %-6s | Repeats | ADHD   | Acc%%\n", "Kind", "Phase")
	fmt.Println("--------------------------------------------------------------")
	for _, r := range results {
		fmt.Printf("%-15s | %-6s |   %d     | %6.2f | %5.2f\n",
			r.kind, r.phase, r.repeats, r.score, r.acc)
	}
	fmt.Println("==============================================================")
}
