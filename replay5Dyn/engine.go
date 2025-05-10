package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"

	"paragon"
)

const (
	modelPath = "stock_model.json"
	numDays   = 1000
	seqLength = 30
	fixedSeed = 1337
)

type Config struct {
	LayerIndex   int
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

	// Load model
	base := &paragon.Network{}
	if err := base.LoadJSON(modelPath); err != nil {
		fmt.Println("❌ Failed to load model:", err)
		return
	}

	// Data
	ins, tgts := generateBalancedStockData(numDays, seqLength)
	printLabelDistribution(tgts)

	// Containers
	results := map[string][]float64{
		"standard": {},
		"manual":   {},
		"dynamic":  {},
	}

	// ─────────────── STANDARD ───────────────
	for i := 0; i < 10; i++ {
		net := Clone(base)
		fmt.Printf("\n[Standard %d] Training...\n", i+1)
		net.Train(ins, tgts, 10, 0.001, false)
		score := evaluate(net, ins, tgts).score
		fmt.Printf("→ ADHD Score: %.2f\n", score)
		results["standard"] = append(results["standard"], score)
	}

	// ─────────────── MANUAL REPLAY ───────────────
	for i := 0; i < 10; i++ {
		net := Clone(base)
		fmt.Printf("\n[Manual %d] Training with ReplayLayer1 (before -1, x5)...\n", i+1)
		net.Layers[1].ReplayOffset = -1
		net.Layers[1].ReplayPhase = "before"
		net.Layers[1].MaxReplay = 5
		net.Train(ins, tgts, 10, 0.001, false)
		score := evaluate(net, ins, tgts).score
		fmt.Printf("→ ADHD Score: %.2f\n", score)
		results["manual"] = append(results["manual"], score)
	}

	// ─────────────── DYNAMIC GATED REPLAY ───────────────
	const dynModelPath = "best_dynamic_model.json"

	var bestScore float64 = -math.MaxFloat64
	var bestNet *paragon.Network

	// ─────────────── DYNAMIC GATED REPLAY ───────────────
	if _, err := os.Stat(dynModelPath); os.IsNotExist(err) {
		fmt.Println("📁 No saved dynamic model found. Training new one...")

		for i := 0; i < 10; i++ {
			net := Clone(base)
			fmt.Printf("\n[Dynamic %d] Training with Entropy Replay Gates...\n", i+1)
			for l := 1; l < net.OutputLayer; l++ {
				layer := &net.Layers[l]
				layer.ReplayEnabled = true
				layer.ReplayBudget = 20
				layer.ReplayGateFunc = EntropyGate(layer)
				layer.ReplayGateToReps = OLDEntropyToReplay
			}

			net.Train(ins, tgts, 10, 0.001, false)
			score := evaluate(net, ins, tgts).score
			fmt.Printf("→ ADHD Score: %.2f\n", score)
			results["dynamic"] = append(results["dynamic"], score)

			if score > bestScore {
				bestScore = score
				bestNet = net
			}
		}

		// Save the best one
		if bestNet != nil {
			fmt.Printf("💾 Saving best dynamic model (Score: %.2f)...\n", bestScore)
			err := bestNet.SaveJSON(dynModelPath)
			if err != nil {
				fmt.Println("❌ Failed to save model:", err)
			}
		}

	} else {
		fmt.Println("📂 Loading existing best dynamic model...")
		net := &paragon.Network{}
		if err := net.LoadJSON(dynModelPath); err != nil {
			fmt.Println("❌ Failed to load model:", err)
		} else {
			// Restore gate functions
			for l := 1; l < net.OutputLayer; l++ {
				if net.Layers[l].ReplayEnabled {
					net.Layers[l].ReplayGateFunc = EntropyGate(&net.Layers[l])
					net.Layers[l].ReplayGateToReps = OLDEntropyToReplay
				}
			}

			// 🔍 Insert this block here to confirm replay gate behavior
			fmt.Println("\n🔬 Replay Behavior Check:")
			for l := 1; l < net.OutputLayer; l++ {
				layer := &net.Layers[l]
				if layer.ReplayEnabled {
					score := layer.ReplayGateFunc(nil)
					reps := layer.ReplayGateToReps(score)
					fmt.Printf("Layer %d → Gate Entropy Score: %.4f → Replays: %d (Budget: %d)\n",
						l, score, reps, layer.ReplayBudget)
				}
			}

			score := evaluate(net, ins, tgts).score
			fmt.Printf("→ ADHD Score from loaded dynamic model: %.2f\n", score)
			results["dynamic"] = append(results["dynamic"], score)

		}

		// 🔍 Show predictions and entropy stats
		totalCorrect := 0
		totalEntropy := 0.0
		samples := len(ins)

		for i := 0; i < samples; i++ {
			net.Forward(ins[i])
			net.ApplySoftmax()
			out := net.ExtractOutput()
			pred := paragon.ArgMax(out)
			exp := paragon.ArgMax(tgts[i][0])
			if pred == exp {
				totalCorrect++
			}

			// Calculate entropy of output
			sum := 0.0
			for _, p := range out {
				if p > 0 {
					sum -= p * math.Log2(p)
				}
			}
			entropy := sum / math.Log2(float64(len(out)))
			totalEntropy += entropy
			fmt.Printf("Sample %d: prediction=%d expected=%d entropy=%.4f\n", i, pred, exp, entropy)
		}

		accuracy := float64(totalCorrect) / float64(samples)
		avgEntropy := totalEntropy / float64(samples)

		fmt.Printf("\n✅ Dynamic Model Report:\n")
		fmt.Printf("→ Accuracy: %.2f%% (%d/%d)\n", accuracy*100, totalCorrect, samples)
		fmt.Printf("→ Average Entropy: %.4f\n", avgEntropy)
		fmt.Printf("→ Softmax Output Example (sample 0):\n")
		net.Forward(ins[0])
		net.ApplySoftmax()
		fmt.Println("→", net.ExtractOutput())

	}

	// ─────────────── SUMMARY ───────────────
	fmt.Println("\n📊 Baseline Comparison:")
	for k, scores := range results {
		avg := avg(scores)
		fmt.Printf("%-10s → avg ADHD = %.2f (n=%d)\n", k, avg, len(scores))
	}

	// ─────────────── SUMMARY ───────────────
	fmt.Println("\n📊 Baseline Comparison:")
	for mode, scores := range results {
		avg := avg(scores)
		fmt.Printf("%-10s → avg ADHD = %.2f (n=%d)\n", mode, avg, len(scores))
		fmt.Printf("  → All scores: ")
		for i, s := range scores {
			if i > 0 {
				fmt.Print(", ")
			}
			fmt.Printf("%.2f", s)
		}
		fmt.Println()
	}

	/*lambdas := []float64{0.0, 0.0001, 0.001, 0.005, 0.01, 0.02}
	sweepResults := map[float64][]float64{}

	for _, λ := range lambdas {
		for i := 0; i < 10; i++ {
			net := Clone(base)
			fmt.Printf("\n[λ=%.4f, Run %d] Training with Dynamic Replay...\n", λ, i+1)

			for l := 1; l < net.OutputLayer; l++ {
				layer := &net.Layers[l]
				layer.ReplayEnabled = true
				layer.ReplayBudget = 20
				layer.ReplayGateFunc = EntropyGate(layer)
				layer.ReplayGateToReps = EntropyToReplay
			}

			// Train with penalty
			net.TrainTestWithLambda(ins, tgts, 10, 0.001, false, λ)

			score := evaluate(net, ins, tgts).score
			fmt.Printf("→ ADHD Score: %.2f\n", score)
			sweepResults[λ] = append(sweepResults[λ], score)
		}
	}*/

}

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
	fmt.Printf("\nPrediction: %s\n[down=%.2f  flat=%.2f  up=%.2f]\n", lbl, out[0], out[1], out[2])
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
	b := make(map[string]int)
	for k, v := range net.Performance.Buckets {
		b[k] = v.Count
	}
	return results{net.Performance.Score, b, exp, pred}
}

func validateOutput(out []float64) {
	sum := 0.0
	for _, v := range out {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			fmt.Println("❌ Invalid output value – corrupt model")
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
	fmt.Printf("\nLabel Distribution: DOWN=%d | FLAT=%d | UP=%d\n", cnt[0], cnt[1], cnt[2])
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

func EntropyGate(layer *paragon.Grid) func(_ [][]float64) float64 {
	return func(_ [][]float64) float64 {
		if len(layer.CachedOutputs) == 0 {
			return 0.0
		}
		out := paragon.Softmax(layer.CachedOutputs)
		entropy := 0.0
		for _, p := range out {
			if p > 0 {
				entropy -= p * math.Log2(p)
			}
		}
		return entropy / math.Log2(float64(len(out)))
	}
}

func OLDEntropyToReplay(score float64) int {
	switch {
	case score > 0.9:
		return 10
	case score > 0.7:
		return 5
	case score > 0.5:
		return 2
	default:
		return 1
	}
}

func EntropyToReplay(score float64) int {
	return int(math.Round(score * 20)) // entropy 0.5 → 10 replays, 1.0 → 20
}

func Clone(n *paragon.Network) *paragon.Network {
	bytes, _ := n.MarshalJSONModel()
	clone := &paragon.Network{}
	_ = clone.UnmarshalJSONModel(bytes)
	return clone
}

func avg(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	sum := 0.0
	for _, x := range xs {
		sum += x
	}
	return sum / float64(len(xs))
}

//Deeper Issue: Training Cost Has No Replay Penalty
//loss += λ * float64(totalReplaysThisSample)
