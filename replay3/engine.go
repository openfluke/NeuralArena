package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
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
	//testReplayVariantsWithLowerLR()
	//multiTest()
	//multiTestExtend()
	//multiTestDeepReplaySweep()

	//multiTestDeepReplaySweepLowerLR()
	//multiTestDeepReplaySweepUltraLowLR()
	multiTestHardReplaySweepLowerLR()
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

func multiTest() {
	type taskConfig struct {
		name    string
		inputW  int
		inputH  int
		outputC int
		gen     func() ([][][]float64, [][][]float64)
	}

	tasks := []taskConfig{
		{
			name:    "Fuzzy XOR Grid",
			inputW:  4,
			inputH:  4,
			outputC: 2,
			gen: func() ([][][]float64, [][][]float64) {
				var X, Y [][][]float64
				for i := 0; i < 1000; i++ {
					in := make([][]float64, 4)
					for y := 0; y < 4; y++ {
						in[y] = make([]float64, 4)
						for x := 0; x < 4; x++ {
							in[y][x] = float64(rand.Intn(2))
						}
					}
					sum := 0
					for _, row := range in {
						for _, v := range row {
							if v > 0.5 {
								sum++
							}
						}
					}
					label := sum % 2
					if rand.Float64() < 0.05 {
						label = 1 - label // 5% label noise
					}
					out := [][]float64{{0.2, 0.2}}
					out[0][label] = 0.8
					X, Y = append(X, in), append(Y, out)
				}
				return X, Y
			},
		},
		{
			name:    "Hotspot + Noise",
			inputW:  8,
			inputH:  8,
			outputC: 2,
			gen: func() ([][][]float64, [][][]float64) {
				var X, Y [][][]float64
				for i := 0; i < 1000; i++ {
					in := make([][]float64, 8)
					for y := 0; y < 8; y++ {
						in[y] = make([]float64, 8)
					}
					px := rand.Intn(8)
					py := rand.Intn(8)
					in[py][px] = 1
					for j := 0; j < 10; j++ {
						xn, yn := rand.Intn(8), rand.Intn(8)
						in[yn][xn] += rand.Float64() * 0.2 // clutter
					}
					label := 0
					if (px+py)%2 == 0 {
						label = 1
					}
					if rand.Float64() < 0.05 {
						label = 1 - label
					}
					out := [][]float64{{0.2, 0.2}}
					out[0][label] = 0.8
					X, Y = append(X, in), append(Y, out)
				}
				return X, Y
			},
		},
		{
			name:    "Global Center Mass",
			inputW:  10,
			inputH:  10,
			outputC: 2,
			gen: func() ([][][]float64, [][][]float64) {
				var X, Y [][][]float64
				for i := 0; i < 1000; i++ {
					in := make([][]float64, 10)
					for y := 0; y < 10; y++ {
						in[y] = make([]float64, 10)
					}
					points := rand.Intn(10) + 1
					cx, cy := 0.0, 0.0
					for j := 0; j < points; j++ {
						x, y := rand.Intn(10), rand.Intn(10)
						in[y][x] = 1.0
						cx += float64(x)
						cy += float64(y)
					}
					cx /= float64(points)
					cy /= float64(points)
					label := 0
					if cx+cy > 9 {
						label = 1
					}
					if rand.Float64() < 0.05 {
						label = 1 - label
					}
					out := [][]float64{{0.2, 0.2}}
					out[0][label] = 0.8
					X, Y = append(X, in), append(Y, out)
				}
				return X, Y
			},
		},
	}

	type outcome struct {
		task     string
		replay   int
		adhd     float64
		accuracy float64
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	var results []outcome

	sem := make(chan struct{}, int(0.8*float64(runtime.NumCPU())))

	for _, task := range tasks {
		for replay := 0; replay <= 3; replay++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(cfg taskConfig, replayCount int) {
				defer wg.Done()
				defer func() { <-sem }()

				X, Y := cfg.gen()
				layer := []struct{ Width, Height int }{
					{cfg.inputW, cfg.inputH},
					{12, 12},
					{cfg.outputC, 1},
				}
				acts := []string{"leaky_relu", "leaky_relu", "softmax"}
				fc := []bool{true, false, true}
				net := paragon.NewNetwork(layer, acts, fc)

				if replayCount > 0 {
					net.Layers[1].ReplayOffset = -1
					net.Layers[1].ReplayPhase = "after"
					net.Layers[1].MaxReplay = replayCount
				}

				net.Train(X, Y, 25, 0.001, true)

				exp, pred := make([]float64, len(X)), make([]float64, len(X))
				correct := 0
				for i := range X {
					net.Forward(X[i])
					p := float64(paragon.ArgMax(net.ExtractOutput()))
					t := float64(paragon.ArgMax(Y[i][0]))
					exp[i], pred[i] = t, p
					if p == t {
						correct++
					}
				}
				net.EvaluateModel(exp, pred)
				acc := float64(correct) / float64(len(X)) * 100.0

				mu.Lock()
				results = append(results, outcome{
					task: cfg.name, replay: replayCount,
					adhd: net.Performance.Score, accuracy: acc,
				})
				mu.Unlock()
			}(task, replay)
		}
	}

	wg.Wait()

	sort.Slice(results, func(i, j int) bool {
		if results[i].task != results[j].task {
			return results[i].task < results[j].task
		}
		return results[i].replay < results[j].replay
	})

	fmt.Println("\n=============== ADVERSARIAL REPLAY BENCHMARK ===============")
	fmt.Printf("%-22s | Replays | ADHD   | Acc%%\n", "Task")
	fmt.Println("------------------------------------------------------------")
	for _, r := range results {
		fmt.Printf("%-22s |    %d     | %6.2f | %5.2f\n",
			r.task, r.replay, r.adhd, r.accuracy)
	}
	fmt.Println("============================================================")
}

func multiTestExtend() {
	type taskConfig struct {
		name    string
		inputW  int
		inputH  int
		outputC int
		gen     func() ([][][]float64, [][][]float64)
	}

	tasks := []taskConfig{
		{
			name:    "Sparse Clusters",
			inputW:  10,
			inputH:  10,
			outputC: 2,
			gen: func() ([][][]float64, [][][]float64) {
				var X, Y [][][]float64
				for i := 0; i < 1000; i++ {
					in := make([][]float64, 10)
					for y := 0; y < 10; y++ {
						in[y] = make([]float64, 10)
					}
					label := rand.Intn(2)
					clusterX := rand.Intn(7)
					clusterY := rand.Intn(7)
					for j := 0; j < 5; j++ {
						dx, dy := rand.Intn(3), rand.Intn(3)
						in[clusterY+dy][clusterX+dx] = 1.0
					}
					if rand.Float64() < 0.1 {
						label = 1 - label // noise
					}
					out := [][]float64{{0.2, 0.2}}
					out[0][label] = 0.8
					X, Y = append(X, in), append(Y, out)
				}
				return X, Y
			},
		},
		{
			name:    "Noisy Ring Detection",
			inputW:  12,
			inputH:  12,
			outputC: 2,
			gen: func() ([][][]float64, [][][]float64) {
				var X, Y [][][]float64
				center := [2]int{6, 6}
				for i := 0; i < 1000; i++ {
					in := make([][]float64, 12)
					for y := 0; y < 12; y++ {
						in[y] = make([]float64, 12)
					}
					for j := 0; j < 30; j++ {
						x := rand.Intn(12)
						y := rand.Intn(12)
						dx, dy := float64(x-center[0]), float64(y-center[1])
						r := dx*dx + dy*dy
						if r > 20 && r < 30 {
							in[y][x] = 1.0
						} else {
							in[y][x] = rand.Float64() * 0.3
						}
					}
					label := 1
					if rand.Float64() < 0.1 {
						label = 0
					}
					out := [][]float64{{0.2, 0.2}}
					out[0][label] = 0.8
					X, Y = append(X, in), append(Y, out)
				}
				return X, Y
			},
		},
		{
			name:    "Scattered Bits Logic",
			inputW:  6,
			inputH:  6,
			outputC: 2,
			gen: func() ([][][]float64, [][][]float64) {
				var X, Y [][][]float64
				for i := 0; i < 1000; i++ {
					in := make([][]float64, 6)
					sum := 0.0
					for y := 0; y < 6; y++ {
						in[y] = make([]float64, 6)
						for x := 0; x < 6; x++ {
							val := float64(rand.Intn(2))
							in[y][x] = val
							sum += val
						}
					}
					label := 0
					if int(sum)%2 == 1 {
						label = 1
					}
					out := [][]float64{{0.2, 0.2}}
					out[0][label] = 0.8
					X, Y = append(X, in), append(Y, out)
				}
				return X, Y
			},
		},
	}

	type result struct {
		task     string
		replay   int
		score    float64
		accuracy float64
	}

	var results []result
	var wg sync.WaitGroup
	var mu sync.Mutex
	sem := make(chan struct{}, int(0.8*float64(runtime.NumCPU())))

	for _, task := range tasks {
		for r := 0; r <= 50; r++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(cfg taskConfig, replayCount int) {
				defer wg.Done()
				defer func() { <-sem }()

				X, Y := cfg.gen()
				layer := []struct{ Width, Height int }{
					{cfg.inputW, cfg.inputH},
					{12, 12},
					{cfg.outputC, 1},
				}
				acts := []string{"leaky_relu", "leaky_relu", "softmax"}
				fc := []bool{true, false, true}
				net := paragon.NewNetwork(layer, acts, fc)

				if replayCount > 0 {
					net.Layers[1].ReplayOffset = -1
					net.Layers[1].ReplayPhase = "after"
					net.Layers[1].MaxReplay = replayCount
				}

				net.Train(X, Y, 25, 0.001, true)

				exp, pred := make([]float64, len(X)), make([]float64, len(X))
				correct := 0
				for i := range X {
					net.Forward(X[i])
					p := float64(paragon.ArgMax(net.ExtractOutput()))
					t := float64(paragon.ArgMax(Y[i][0]))
					exp[i], pred[i] = t, p
					if p == t {
						correct++
					}
				}
				net.EvaluateModel(exp, pred)

				mu.Lock()
				results = append(results, result{
					task:     cfg.name,
					replay:   replayCount,
					score:    net.Performance.Score,
					accuracy: float64(correct) / float64(len(X)) * 100.0,
				})
				mu.Unlock()
			}(task, r)
		}
	}

	wg.Wait()

	// Optional: sort for pretty output
	sort.Slice(results, func(i, j int) bool {
		if results[i].task != results[j].task {
			return results[i].task < results[j].task
		}
		return results[i].replay < results[j].replay
	})

	// Print results
	fmt.Println("\n=========== EXTENDED MULTI-TASK REPLAY SWEEP ===========")
	fmt.Printf("%-22s | Replays | ADHD   | Acc%%\n", "Task")
	fmt.Println("----------------------------------------------------------")
	for _, r := range results {
		fmt.Printf("%-22s | %7d | %6.2f | %5.2f\n",
			r.task, r.replay, r.score, r.accuracy)
	}
	fmt.Println("==========================================================")
}

// Place this inside your main package or call it from `main()` to execute.
func multiTestDeepReplaySweep() {
	type taskConfig struct {
		name    string
		inputW  int
		inputH  int
		outputC int
		gen     func() ([][][]float64, [][][]float64)
	}

	tasks := []taskConfig{
		{
			name:    "Noisy Ring Detection",
			inputW:  12,
			inputH:  12,
			outputC: 2,
			gen: func() ([][][]float64, [][][]float64) {
				var X, Y [][][]float64
				for i := 0; i < 1000; i++ {
					in := make([][]float64, 12)
					for y := 0; y < 12; y++ {
						in[y] = make([]float64, 12)
					}
					// Draw noisy ring
					for t := 0; t < 360; t += 15 {
						angle := float64(t) * 3.14159 / 180
						x := int(6 + 4*math.Cos(angle))
						y := int(6 + 4*math.Sin(angle))
						if x >= 0 && x < 12 && y >= 0 && y < 12 {
							in[y][x] = 1.0
						}
					}
					for j := 0; j < 20; j++ {
						x := rand.Intn(12)
						y := rand.Intn(12)
						in[y][x] += rand.Float64() * 0.2
					}
					label := 1
					if rand.Float64() < 0.05 {
						label = 0
					}
					out := [][]float64{{0.2, 0.2}}
					out[0][label] = 0.8
					X, Y = append(X, in), append(Y, out)
				}
				return X, Y
			},
		},
	}

	type outcome struct {
		task     string
		layerCfg string
		replay   string
		repeats  int
		adhd     float64
		accuracy float64
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	var results []outcome
	sem := make(chan struct{}, int(0.8*float64(runtime.NumCPU())))

	layerVariants := [][]struct{ Width, Height int }{
		{{12, 12}, {32, 32}, {16, 16}, {8, 8}, {2, 1}},
		{{12, 12}, {24, 24}, {12, 12}, {6, 6}, {2, 1}},
		{{12, 12}, {48, 48}, {32, 32}, {16, 16}, {2, 1}},
	}

	phases := []string{"before", "after"}
	maxReplay := 10

	for _, task := range tasks {
		for _, layers := range layerVariants {
			for _, phase := range phases {
				for repeats := 0; repeats <= maxReplay; repeats++ {
					wg.Add(1)
					sem <- struct{}{}
					go func(cfg taskConfig, layerSet []struct{ Width, Height int }, phase string, rp int) {
						defer wg.Done()
						defer func() { <-sem }()

						X, Y := cfg.gen()
						acts := make([]string, len(layerSet))
						fc := make([]bool, len(layerSet))
						for i := range acts {
							if i == len(acts)-1 {
								acts[i] = "softmax"
								fc[i] = true
							} else {
								acts[i] = "leaky_relu"
								fc[i] = false
							}
						}
						net := paragon.NewNetwork(layerSet, acts, fc)

						for i := 1; i < len(net.Layers)-1; i++ {
							if rand.Float64() < 0.5 {
								net.Layers[i].ReplayOffset = -1
								net.Layers[i].ReplayPhase = phase
								net.Layers[i].MaxReplay = rp
							}
						}

						net.Train(X, Y, 25, 0.001, true)

						exp, pred := make([]float64, len(X)), make([]float64, len(X))
						correct := 0
						for i := range X {
							net.Forward(X[i])
							p := float64(paragon.ArgMax(net.ExtractOutput()))
							t := float64(paragon.ArgMax(Y[i][0]))
							exp[i], pred[i] = t, p
							if p == t {
								correct++
							}
						}
						net.EvaluateModel(exp, pred)
						acc := float64(correct) / float64(len(X)) * 100.0

						mu.Lock()
						results = append(results, outcome{
							task:     cfg.name,
							layerCfg: fmt.Sprintf("%d layers", len(layerSet)),
							replay:   phase,
							repeats:  rp,
							adhd:     net.Performance.Score,
							accuracy: acc,
						})
						mu.Unlock()
					}(task, layers, phase, repeats)
				}
			}
		}
	}
	wg.Wait()

	sort.Slice(results, func(i, j int) bool {
		if results[i].task != results[j].task {
			return results[i].task < results[j].task
		}
		if results[i].layerCfg != results[j].layerCfg {
			return results[i].layerCfg < results[j].layerCfg
		}
		if results[i].replay != results[j].replay {
			return results[i].replay < results[j].replay
		}
		return results[i].repeats < results[j].repeats
	})

	fmt.Println("\n=========== EXTENDED MULTI-TASK DEEP REPLAY SWEEP ===========")
	fmt.Printf("%-22s | %-10s | %-6s | Repeats | ADHD   | Acc%%\n", "Task", "Layers", "Phase")
	fmt.Println("--------------------------------------------------------------")
	for _, r := range results {
		fmt.Printf("%-22s | %-10s | %-6s |   %2d     | %6.2f | %5.2f\n",
			r.task, r.layerCfg, r.replay, r.repeats, r.adhd, r.accuracy)
	}
	fmt.Println("==============================================================")
}

func multiTestDeepReplaySweepLowerLR() {
	type taskConfig struct {
		name    string
		inputW  int
		inputH  int
		outputC int
		gen     func() ([][][]float64, [][][]float64)
	}

	tasks := []taskConfig{
		{
			name:    "Noisy Ring Detection",
			inputW:  12,
			inputH:  12,
			outputC: 2,
			gen: func() ([][][]float64, [][][]float64) {
				var X, Y [][][]float64
				for i := 0; i < 1000; i++ {
					in := make([][]float64, 12)
					for y := range in {
						in[y] = make([]float64, 12)
					}
					// Draw noisy ring
					for t := 0; t < 360; t += 15 {
						angle := float64(t) * math.Pi / 180
						x := int(6 + 4*math.Cos(angle))
						y := int(6 + 4*math.Sin(angle))
						if x >= 0 && x < 12 && y >= 0 && y < 12 {
							in[y][x] = 1.0
						}
					}
					for j := 0; j < 20; j++ {
						in[rand.Intn(12)][rand.Intn(12)] += rand.Float64() * 0.2
					}
					label := 1
					if rand.Float64() < 0.05 {
						label = 0
					}
					out := [][]float64{{0.2, 0.2}}
					out[0][label] = 0.8
					X, Y = append(X, in), append(Y, out)
				}
				return X, Y
			},
		},
	}

	type outcome struct {
		task     string
		layerCfg string
		replay   string
		repeats  int
		lr       float64
		adhd     float64
		accuracy float64
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	var results []outcome
	sem := make(chan struct{}, int(0.8*float64(runtime.NumCPU())))

	layerVariants := [][]struct{ Width, Height int }{
		{{12, 12}, {32, 32}, {16, 16}, {8, 8}, {2, 1}},
		{{12, 12}, {24, 24}, {12, 12}, {6, 6}, {2, 1}},
		{{12, 12}, {48, 48}, {32, 32}, {16, 16}, {2, 1}},
	}

	phases := []string{"before", "after"}
	learningRates := []float64{0.001, 0.0005, 0.0001}
	maxReplay := 10

	for _, task := range tasks {
		for _, layers := range layerVariants {
			for _, phase := range phases {
				for _, lr := range learningRates {
					for repeats := 0; repeats <= maxReplay; repeats++ {
						wg.Add(1)
						sem <- struct{}{}
						go func(cfg taskConfig, layerSet []struct{ Width, Height int }, phase string, lr float64, rp int) {
							defer wg.Done()
							defer func() { <-sem }()

							X, Y := cfg.gen()
							acts := make([]string, len(layerSet))
							fc := make([]bool, len(layerSet))
							for i := range acts {
								if i == len(acts)-1 {
									acts[i] = "softmax"
									fc[i] = true
								} else {
									acts[i] = "leaky_relu"
									fc[i] = false
								}
							}
							net := paragon.NewNetwork(layerSet, acts, fc)

							for i := 1; i < len(net.Layers)-1; i++ {
								if rand.Float64() < 0.5 {
									net.Layers[i].ReplayOffset = -1
									net.Layers[i].ReplayPhase = phase
									net.Layers[i].MaxReplay = rp
								}
							}

							net.Train(X, Y, 25, lr, true)

							exp, pred := make([]float64, len(X)), make([]float64, len(X))
							correct := 0
							for i := range X {
								net.Forward(X[i])
								p := float64(paragon.ArgMax(net.ExtractOutput()))
								t := float64(paragon.ArgMax(Y[i][0]))
								exp[i], pred[i] = t, p
								if p == t {
									correct++
								}
							}
							net.EvaluateModel(exp, pred)
							acc := float64(correct) / float64(len(X)) * 100.0

							mu.Lock()
							results = append(results, outcome{
								task:     cfg.name,
								layerCfg: fmt.Sprintf("%d layers", len(layerSet)),
								replay:   phase,
								repeats:  rp,
								lr:       lr,
								adhd:     net.Performance.Score,
								accuracy: acc,
							})
							mu.Unlock()
						}(task, layers, phase, lr, repeats)
					}
				}
			}
		}
	}
	wg.Wait()

	sort.Slice(results, func(i, j int) bool {
		if results[i].task != results[j].task {
			return results[i].task < results[j].task
		}
		if results[i].layerCfg != results[j].layerCfg {
			return results[i].layerCfg < results[j].layerCfg
		}
		if results[i].replay != results[j].replay {
			return results[i].replay < results[j].replay
		}
		if results[i].lr != results[j].lr {
			return results[i].lr < results[j].lr
		}
		return results[i].repeats < results[j].repeats
	})

	fmt.Println("\n=========== EXTENDED MULTI-TASK DEEP REPLAY SWEEP ===========")
	fmt.Printf("%-22s | %-10s | %-6s | LR     | Repeats | ADHD   | Acc%%\n", "Task", "Layers", "Phase")
	fmt.Println("--------------------------------------------------------------")
	for _, r := range results {
		fmt.Printf("%-22s | %-10s | %-6s | %.4f |   %2d     | %6.2f | %5.2f\n",
			r.task, r.layerCfg, r.replay, r.lr, r.repeats, r.adhd, r.accuracy)
	}
	fmt.Println("==============================================================")
}

func multiTestDeepReplaySweepUltraLowLR() {
	type taskConfig struct {
		name    string
		inputW  int
		inputH  int
		outputC int
		gen     func() ([][][]float64, [][][]float64)
	}

	tasks := []taskConfig{
		{
			name:    "Noisy Ring Detection",
			inputW:  12,
			inputH:  12,
			outputC: 2,
			gen: func() ([][][]float64, [][][]float64) {
				var X, Y [][][]float64
				for i := 0; i < 1000; i++ {
					in := make([][]float64, 12)
					for y := range in {
						in[y] = make([]float64, 12)
					}
					for t := 0; t < 360; t += 15 {
						angle := float64(t) * math.Pi / 180
						x := int(6 + 4*math.Cos(angle))
						y := int(6 + 4*math.Sin(angle))
						if x >= 0 && x < 12 && y >= 0 && y < 12 {
							in[y][x] = 1.0
						}
					}
					for j := 0; j < 20; j++ {
						in[rand.Intn(12)][rand.Intn(12)] += rand.Float64() * 0.2
					}
					label := 1
					if rand.Float64() < 0.05 {
						label = 0
					}
					out := [][]float64{{0.2, 0.2}}
					out[0][label] = 0.8
					X, Y = append(X, in), append(Y, out)
				}
				return X, Y
			},
		},
	}

	type outcome struct {
		task     string
		layerCfg string
		replay   string
		repeats  int
		lr       float64
		adhd     float64
		accuracy float64
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	var results []outcome
	sem := make(chan struct{}, int(0.8*float64(runtime.NumCPU())))

	layerVariants := [][]struct{ Width, Height int }{
		{{12, 12}, {32, 32}, {16, 16}, {8, 8}, {2, 1}},
		{{12, 12}, {48, 48}, {32, 32}, {16, 16}, {2, 1}},
	}

	phases := []string{"before", "after"}
	ultraLowLRs := []float64{0.0001, 0.00005, 0.00001, 0.000005}
	maxReplay := 10

	for _, task := range tasks {
		for _, layers := range layerVariants {
			for _, phase := range phases {
				for _, lr := range ultraLowLRs {
					for repeats := 0; repeats <= maxReplay; repeats++ {
						wg.Add(1)
						sem <- struct{}{}
						go func(cfg taskConfig, layerSet []struct{ Width, Height int }, phase string, lr float64, rp int) {
							defer wg.Done()
							defer func() { <-sem }()

							X, Y := cfg.gen()
							acts := make([]string, len(layerSet))
							fc := make([]bool, len(layerSet))
							for i := range acts {
								if i == len(acts)-1 {
									acts[i] = "softmax"
									fc[i] = true
								} else {
									acts[i] = "leaky_relu"
									fc[i] = false
								}
							}
							net := paragon.NewNetwork(layerSet, acts, fc)

							for i := 1; i < len(net.Layers)-1; i++ {
								if rand.Float64() < 0.5 {
									net.Layers[i].ReplayOffset = -1
									net.Layers[i].ReplayPhase = phase
									net.Layers[i].MaxReplay = rp
								}
							}

							net.Train(X, Y, 25, lr, true)

							exp, pred := make([]float64, len(X)), make([]float64, len(X))
							correct := 0
							for i := range X {
								net.Forward(X[i])
								p := float64(paragon.ArgMax(net.ExtractOutput()))
								t := float64(paragon.ArgMax(Y[i][0]))
								exp[i], pred[i] = t, p
								if p == t {
									correct++
								}
							}
							net.EvaluateModel(exp, pred)
							acc := float64(correct) / float64(len(X)) * 100.0

							mu.Lock()
							results = append(results, outcome{
								task:     cfg.name,
								layerCfg: fmt.Sprintf("%d layers", len(layerSet)),
								replay:   phase,
								repeats:  rp,
								lr:       lr,
								adhd:     net.Performance.Score,
								accuracy: acc,
							})
							mu.Unlock()
						}(task, layers, phase, lr, repeats)
					}
				}
			}
		}
	}
	wg.Wait()

	sort.Slice(results, func(i, j int) bool {
		if results[i].task != results[j].task {
			return results[i].task < results[j].task
		}
		if results[i].layerCfg != results[j].layerCfg {
			return results[i].layerCfg < results[j].layerCfg
		}
		if results[i].replay != results[j].replay {
			return results[i].replay < results[j].replay
		}
		if results[i].lr != results[j].lr {
			return results[i].lr < results[j].lr
		}
		return results[i].repeats < results[j].repeats
	})

	fmt.Println("\n=========== EXTENDED MULTI-TASK DEEP REPLAY SWEEP ===========")
	fmt.Printf("%-22s | %-10s | %-6s | LR       | Repeats | ADHD   | Acc%%\n", "Task", "Layers", "Phase")
	fmt.Println("--------------------------------------------------------------")
	for _, r := range results {
		fmt.Printf("%-22s | %-10s | %-6s | %.6f |   %2d     | %6.2f | %5.2f\n",
			r.task, r.layerCfg, r.replay, r.lr, r.repeats, r.adhd, r.accuracy)
	}
	fmt.Println("==============================================================")
}

func generateHardSparseXOR() ([][][]float64, [][][]float64) {
	var X, Y [][][]float64
	for i := 0; i < 1000; i++ {
		in := make([][]float64, 12)
		for y := range in {
			in[y] = make([]float64, 12)
		}

		a := rand.Intn(2)
		b := rand.Intn(2)
		c := rand.Intn(2)
		label := (a ^ b) ^ c

		// Embed sparse features
		if a == 1 {
			in[2][3] = 1.0
		}
		if b == 1 {
			in[5][6] = 1.0
		}
		if c == 1 {
			in[8][9] = 1.0
		}

		out := [][]float64{{0.2, 0.2}}
		out[0][label] = 0.8

		X = append(X, in)
		Y = append(Y, out)
	}
	return X, Y
}

func generateTemporalEcho() ([][][]float64, [][][]float64) {
	var X, Y [][][]float64
	for i := 0; i < 1000; i++ {
		in := make([][]float64, 12)
		for y := range in {
			in[y] = make([]float64, 12)
		}

		// Simulate a signal + echo
		signal := rand.Intn(2)
		delay := rand.Intn(4) + 2

		in[1][1] = float64(signal)
		if 1+delay < 12 && 1+delay < 12 {
			in[1+delay][1+delay] = float64(signal)
		}

		// Add noise
		for j := 0; j < 10; j++ {
			in[rand.Intn(12)][rand.Intn(12)] += rand.Float64() * 0.3
		}

		out := [][]float64{{0.2, 0.2}}
		out[0][signal] = 0.8
		X = append(X, in)
		Y = append(Y, out)
	}
	return X, Y
}

func multiTestHardReplaySweepLowerLR() {
	type taskConfig struct {
		name    string
		inputW  int
		inputH  int
		outputC int
		gen     func() ([][][]float64, [][][]float64)
	}

	tasks := []taskConfig{
		{
			name:    "Sparse Concept XOR",
			inputW:  12,
			inputH:  12,
			outputC: 2,
			gen:     generateHardSparseXOR,
		},
		{
			name:    "Temporal Echo Classification",
			inputW:  12,
			inputH:  12,
			outputC: 2,
			gen:     generateTemporalEcho,
		},
	}

	type outcome struct {
		task     string
		layerCfg string
		replay   string
		repeats  int
		lr       float64
		adhd     float64
		accuracy float64
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	var results []outcome
	sem := make(chan struct{}, int(0.8*float64(runtime.NumCPU())))

	layerVariants := [][]struct{ Width, Height int }{
		{{12, 12}, {32, 32}, {16, 16}, {8, 8}, {2, 1}},
	}

	phases := []string{"before", "after"}
	learningRates := []float64{0.000005, 0.00001, 0.00005, 0.0001}
	maxReplay := 10

	for _, task := range tasks {
		for _, layers := range layerVariants {
			for _, phase := range phases {
				for _, lr := range learningRates {
					for repeats := 0; repeats <= maxReplay; repeats++ {
						wg.Add(1)
						sem <- struct{}{}
						go func(cfg taskConfig, layerSet []struct{ Width, Height int }, phase string, lr float64, rp int) {
							defer wg.Done()
							defer func() { <-sem }()

							X, Y := cfg.gen()
							acts := make([]string, len(layerSet))
							fc := make([]bool, len(layerSet))
							for i := range acts {
								if i == len(acts)-1 {
									acts[i] = "softmax"
									fc[i] = true
								} else {
									acts[i] = "leaky_relu"
									fc[i] = false
								}
							}
							net := paragon.NewNetwork(layerSet, acts, fc)

							for i := 1; i < len(net.Layers)-1; i++ {
								if rand.Float64() < 0.5 {
									net.Layers[i].ReplayOffset = -1
									net.Layers[i].ReplayPhase = phase
									net.Layers[i].MaxReplay = rp
								}
							}

							net.Train(X, Y, 25, lr, true)

							exp, pred := make([]float64, len(X)), make([]float64, len(X))
							correct := 0
							for i := range X {
								net.Forward(X[i])
								p := float64(paragon.ArgMax(net.ExtractOutput()))
								t := float64(paragon.ArgMax(Y[i][0]))
								exp[i], pred[i] = t, p
								if p == t {
									correct++
								}
							}
							net.EvaluateModel(exp, pred)
							acc := float64(correct) / float64(len(X)) * 100.0

							mu.Lock()
							results = append(results, outcome{
								task:     cfg.name,
								layerCfg: fmt.Sprintf("%d layers", len(layerSet)),
								replay:   phase,
								repeats:  rp,
								lr:       lr,
								adhd:     net.Performance.Score,
								accuracy: acc,
							})
							mu.Unlock()
						}(task, layers, phase, lr, repeats)
					}
				}
			}
		}
	}
	wg.Wait()

	sort.Slice(results, func(i, j int) bool {
		if results[i].task != results[j].task {
			return results[i].task < results[j].task
		}
		if results[i].layerCfg != results[j].layerCfg {
			return results[i].layerCfg < results[j].layerCfg
		}
		if results[i].replay != results[j].replay {
			return results[i].replay < results[j].replay
		}
		if results[i].lr != results[j].lr {
			return results[i].lr < results[j].lr
		}
		return results[i].repeats < results[j].repeats
	})

	fmt.Println("\n=========== EXTENDED MULTI-TASK HARD REPLAY SWEEP ===========")
	fmt.Printf("%-28s | %-10s | %-6s | %-8s | %-7s | %-6s | %-6s\n", "Task", "Layers", "Phase", "LR", "Repeats", "ADHD", "Acc%")
	fmt.Println("--------------------------------------------------------------------------")
	for _, r := range results {
		fmt.Printf("%-28s | %-10s | %-6s | %-8.6f | %-7d | %-6.2f | %-6.2f\n",
			r.task, r.layerCfg, r.replay, r.lr, r.repeats, r.adhd, r.accuracy)
	}
	fmt.Println("==========================================================================")
}
