package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"

	"paragon"
)

const (
	modelPath    = "stock_model.json"
	numDays      = 1000
	seqLength    = 30
	epochs       = 100
	learningRate = 0.001
	fixedSeed    = 1337
)

type Config struct {
	BeforeLayer  int
	AfterLayer   int
	Replays      int
	LearningRate float64
	Epochs       int
}

type Result struct {
	Net   *paragon.Network
	Score float64
	Conf  Config
}

func main() {
	rand.Seed(fixedSeed)
	ins, tgts := generateBalancedStockData(numDays, seqLength)
	if len(ins) == 0 {
		fmt.Println("Error: insufficient training data")
		return
	}
	printLabelDistribution(tgts)

	var net *paragon.Network
	if _, err := os.Stat(modelPath); err == nil {
		fmt.Println("📥 Loading model from disk (JSON)…")
		net = &paragon.Network{}
		if err := net.LoadJSON(modelPath); err != nil {
			fmt.Println("❌ Failed to load model:", err)
			os.Remove(modelPath)
			return
		}
	} else {
		fmt.Println("🚧 Training new model…")
		net = buildModel(seqLength)
		if err := net.SaveJSON(modelPath); err != nil {
			fmt.Println("❌ Failed to save model:", err)
			return
		}
	}

	res := evaluate(net, ins, tgts)

	fmt.Println("\n============== ADHD EVALUATION ==============")
	fmt.Printf("Metric                     | Value\n")
	fmt.Printf("---------------------------+--------\n")
	fmt.Printf("ADHD Score                 | %6.2f\n", res.score)

	fmt.Println("\nDeviation buckets (# samples):")
	for _, k := range []string{"0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%", "100%+"} {
		fmt.Printf(" %-7s | %4d\n", k, res.buckets[k])
	}

	fmt.Println("\n------ FULL DIAGNOSTICS ----------")
	net.EvaluateFull(res.exp, res.pred)
	net.PrintFullDiagnostics()

	predict(net, ins[len(ins)-1])

	net, _, history := ExploreReplayVariations(
		net,
		ins, tgts,
		[]int{1, 2, 3, 4, 10, 20, 50},
		[]float64{0.0001, 0.001, 0.01},
		[]int{1, 5, 50},
		100.0,
	)

	fmt.Println("\n📈 Improvement History:")
	for i, h := range history {
		cfg := h.Conf
		fmt.Printf("Step %02d: %.2f → [before=L%d, after=L%d, reps=%d, lr=%.5f, epochs=%d]\n",
			i+1, h.Score, cfg.BeforeLayer, cfg.AfterLayer, cfg.Replays, cfg.LearningRate, cfg.Epochs)
	}
}

// ──────────────────────────────────────────────────────────────

func buildModel(seqLen int) *paragon.Network {
	layers := []struct{ Width, Height int }{
		{seqLen, 1},
		{64, 1},
		{32, 1},
		{3, 1},
	}
	acts := []string{"leaky_relu", "leaky_relu", "leaky_relu", "softmax"}
	full := []bool{true, true, true, true}
	return paragon.NewNetwork(layers, acts, full)
}

func predict(net *paragon.Network, input [][]float64) {
	net.Forward(input)
	net.ApplySoftmax()
	out := net.ExtractOutput()
	validateOutput(out)
	lbl := map[int]string{0: "down", 1: "flat", 2: "up"}[paragon.ArgMax(out)]
	fmt.Printf("\nPrediction: %s\n[down=%.2f  flat=%.2f  up=%.2f]\n",
		lbl, out[0], out[1], out[2])
}

func evaluate(net *paragon.Network, inputs, targets [][][]float64) results {
	exp, pred := []float64{}, []float64{}
	for i, in := range inputs {
		net.Forward(in)
		net.ApplySoftmax()
		out := net.ExtractOutput()
		validateOutput(out)
		pred = append(pred, float64(paragon.ArgMax(out)))
		exp = append(exp, float64(paragon.ArgMax(targets[i][0])))
	}
	net.EvaluateModel(exp, pred)
	b := make(map[string]int, len(net.Performance.Buckets))
	for k, v := range net.Performance.Buckets {
		b[k] = v.Count
	}
	return results{net.Performance.Score, b, exp, pred}
}

func validateOutput(out []float64) {
	sum := 0.0
	for _, v := range out {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			fmt.Println("❌ Invalid output value detected – deleting corrupt model")
			os.Remove(modelPath)
			os.Exit(1)
		}
		sum += v
	}
	if sum < 0.95 || sum > 1.05 {
		fmt.Printf("⚠️  Softmax sum suspicious: %.2f\n", sum)
	}
}

func printLabelDistribution(tgts [][][]float64) {
	cnt := map[int]int{}
	for _, t := range tgts {
		cnt[paragon.ArgMax(t[0])]++
	}
	fmt.Printf("\nLabel Distribution: DOWN=%d | FLAT=%d | UP=%d\n",
		cnt[0], cnt[1], cnt[2])
}

type results struct {
	score   float64
	buckets map[string]int
	exp     []float64
	pred    []float64
}

func generateBalancedStockData(numDays, seqLength int) ([][][]float64, [][][]float64) {
	rng := rand.New(rand.NewSource(fixedSeed))
	prices := make([]float64, numDays)
	prices[0] = 100.0
	for i := 1; i < numDays; i++ {
		changePct := rng.NormFloat64() * 0.5
		prices[i] = prices[i-1] * (1 + changePct/100)
		if prices[i] < 10 {
			prices[i] = 10
		}
	}

	changes := make([]int, numDays-1)
	for i := 1; i < len(prices); i++ {
		changes[i-1] = discretizePriceChange(prices[i-1], prices[i])
	}

	type sample struct{ in, out [][]float64 }
	buckets := map[int][]sample{0: {}, 1: {}, 2: {}}
	for i := 0; i < len(changes)-seqLength; i++ {
		next := changes[i+seqLength]
		in := [][]float64{make([]float64, seqLength)}
		for j := 0; j < seqLength; j++ {
			in[0][j] = float64(changes[i+j])
		}
		out := [][]float64{[]float64{0, 0, 0}}
		out[0][next] = 1.0
		buckets[next] = append(buckets[next], sample{in, out})
	}

	minCount := len(buckets[0])
	if len(buckets[1]) < minCount {
		minCount = len(buckets[1])
	}
	if len(buckets[2]) < minCount {
		minCount = len(buckets[2])
	}

	var inputs, targets [][][]float64
	for cls := 0; cls <= 2; cls++ {
		s := buckets[cls]
		rng.Shuffle(len(s), func(i, j int) { s[i], s[j] = s[j], s[i] })
		for i := 0; i < minCount; i++ {
			inputs = append(inputs, s[i].in)
			targets = append(targets, s[i].out)
		}
	}
	rng.Shuffle(len(inputs), func(i, j int) {
		inputs[i], inputs[j] = inputs[j], inputs[i]
		targets[i], targets[j] = targets[j], targets[i]
	})

	return inputs, targets
}

func discretizePriceChange(prev, curr float64) int {
	change := (curr - prev) / prev * 100
	switch {
	case change < -0.5:
		return 0
	case change > 0.5:
		return 2
	default:
		return 1
	}
}

func ExploreReplayVariations(
	parent *paragon.Network,
	inputs, targets [][][]float64,
	replayCounts []int,
	learningRates []float64,
	epochsList []int,
	earlyStopScore float64,
) (*paragon.Network, float64, []Result) {

	cloneNet := func(src *paragon.Network) *paragon.Network {
		b, _ := src.MarshalJSONModel()
		dst := &paragon.Network{}
		_ = dst.UnmarshalJSONModel(b)
		return dst
	}
	scoreNet := func(net *paragon.Network) float64 {
		return evaluate(net, inputs, targets).score
	}

	var configs []Config
	for _, r := range replayCounts {
		for _, lr := range learningRates {
			for _, ep := range epochsList {
				for l1 := 1; l1 < parent.OutputLayer; l1++ {
					for l2 := 1; l2 < parent.OutputLayer; l2++ {
						if l1 == l2 {
							continue
						}
						configs = append(configs, Config{
							BeforeLayer:  l1,
							AfterLayer:   l2,
							Replays:      r,
							LearningRate: lr,
							Epochs:       ep,
						})
					}
				}
			}
		}
	}

	bestNet := parent
	bestScore := scoreNet(parent)
	fmt.Printf("🔰 Baseline score: %.4f\n", bestScore)

	results := make(chan Result, len(configs))
	var wg sync.WaitGroup
	var stopFlag sync.Once
	var stopAll = make(chan struct{})
	var historyMu sync.Mutex
	var improvementHistory []Result

	for _, cfg := range configs {
		wg.Add(1)
		go func(cfg Config) {
			defer wg.Done()

			select {
			case <-stopAll:
				return
			default:
			}

			net := cloneNet(parent)

			layerBefore := &net.Layers[cfg.BeforeLayer]
			layerBefore.MaxReplay = cfg.Replays
			layerBefore.ReplayPhase = "before"
			layerBefore.ReplayOffset = -1

			layerAfter := &net.Layers[cfg.AfterLayer]
			layerAfter.MaxReplay = cfg.Replays
			layerAfter.ReplayPhase = "after"
			layerAfter.ReplayOffset = -1

			net.Train(inputs, targets, cfg.Epochs, cfg.LearningRate, false)
			score := scoreNet(net)

			historyMu.Lock()
			if score > bestScore {
				bestNet = net
				bestScore = score
				improvementHistory = append(improvementHistory, Result{net, score, cfg})
				fmt.Printf("✅ NEW BEST → %.2f [before=L%d, after=L%d, reps=%d, lr=%.5f, epochs=%d]\n",
					score, cfg.BeforeLayer, cfg.AfterLayer, cfg.Replays, cfg.LearningRate, cfg.Epochs)
			}
			historyMu.Unlock()

			if score >= earlyStopScore {
				stopFlag.Do(func() {
					close(stopAll)
					fmt.Printf("\n🛑 EARLY STOP: reached score %.2f [before=L%d, after=L%d, reps=%d, lr=%.5f, epochs=%d]\n",
						score, cfg.BeforeLayer, cfg.AfterLayer, cfg.Replays, cfg.LearningRate, cfg.Epochs)
				})
				return
			}
			results <- Result{net, score, cfg}
		}(cfg)
	}

	wg.Wait()
	close(results)
	return bestNet, bestScore, improvementHistory
}
