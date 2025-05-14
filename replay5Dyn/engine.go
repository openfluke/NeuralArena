package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"

	"paragon"
)

const (
	modelPath    = "stock_model.json"
	dynModelPath = "best_dynamic_model.json"
	numDays      = 1000
	seqLength    = 30
	fixedSeed    = 1337
	epochs       = 10
	learnRate    = 0.001
)

type Result[T paragon.Numeric] struct {
	Net   *paragon.Network[T]
	Score float64
}

func main() {
	rand.Seed(fixedSeed)

	// Load base model
	base := &paragon.Network[float32]{}
	if err := base.LoadJSON(modelPath); err != nil {
		fmt.Println("❌ Failed to load model:", err)
		return
	}

	// Generate data
	ins, tgts := generateBalancedStockData(numDays, seqLength)
	printLabelDistribution(tgts)

	clipUpper := float32(5)
	clipLower := float32(-5)

	results := map[string][]float64{
		"standard": {},
		"manual":   {},
		"dynamic":  {},
	}

	// ─── STANDARD ───
	for i := 0; i < 10; i++ {
		net := Clone(base)
		fmt.Printf("\n[Standard %d] Training...\n", i+1)
		net.Train(ins, tgts, epochs, learnRate, false, clipUpper, clipLower)
		score := evaluate(net, ins, tgts).score
		fmt.Printf("→ ADHD Score: %.2f\n", score)
		results["standard"] = append(results["standard"], score)
	}

	// ─── MANUAL REPLAY ───
	for i := 0; i < 10; i++ {
		net := Clone(base)
		fmt.Printf("\n[Manual %d] Training with Replay Layer 1...\n", i+1)
		layer := &net.Layers[1]
		layer.ReplayEnabled = true
		layer.ReplayOffset = -1
		layer.ReplayPhase = "before"
		layer.MaxReplay = 5

		net.Train(ins, tgts, epochs, learnRate, false, clipUpper, clipLower)
		score := evaluate(net, ins, tgts).score
		fmt.Printf("→ ADHD Score: %.2f\n", score)
		results["manual"] = append(results["manual"], score)
	}

	// ─── DYNAMIC GATED REPLAY ───
	var bestScore float64 = -math.MaxFloat64
	var bestNet *paragon.Network[float32]

	if _, err := os.Stat(dynModelPath); os.IsNotExist(err) {
		fmt.Println("📁 No saved dynamic model found. Training new one...")
		for i := 0; i < 10; i++ {
			net := Clone(base)
			fmt.Printf("\n[Dynamic %d] Training with Entropy-Gated Replay...\n", i+1)
			for l := 1; l < net.OutputLayer; l++ {
				layer := &net.Layers[l]
				layer.ReplayEnabled = true
				layer.ReplayBudget = 20
				layer.ReplayGateFunc = EntropyGate[float32](layer)
				layer.ReplayGateToReps = OLDEntropyToReplay
			}
			net.Train(ins, tgts, epochs, learnRate, false, clipUpper, clipLower)
			score := evaluate(net, ins, tgts).score
			fmt.Printf("→ ADHD Score: %.2f\n", score)
			results["dynamic"] = append(results["dynamic"], score)

			if score > bestScore {
				bestScore = score
				bestNet = net
			}
		}

		if bestNet != nil {
			fmt.Printf("💾 Saving best dynamic model (Score: %.2f)...\n", bestScore)
			if err := bestNet.SaveJSON(dynModelPath); err != nil {
				fmt.Println("❌ Failed to save model:", err)
			}
		}
	} else {
		fmt.Println("📂 Loading existing best dynamic model...")
		net := &paragon.Network[float32]{}
		if err := net.LoadJSON(dynModelPath); err != nil {
			fmt.Println("❌ Failed to load model:", err)
		} else {
			for l := 1; l < net.OutputLayer; l++ {
				layer := &net.Layers[l]
				if layer.ReplayEnabled {
					layer.ReplayGateFunc = EntropyGate[float32](layer)
					layer.ReplayGateToReps = OLDEntropyToReplay
				}
			}
			score := evaluate(net, ins, tgts).score
			fmt.Printf("→ ADHD Score from loaded model: %.2f\n", score)
			results["dynamic"] = append(results["dynamic"], score)
		}
	}

	// ─── SUMMARY ───
	fmt.Println("\n📊 Summary:")
	for mode, scores := range results {
		fmt.Printf("%-10s → avg ADHD = %.2f | Scores: ", mode, avg(scores))
		for i, s := range scores {
			if i > 0 {
				fmt.Print(", ")
			}
			fmt.Printf("%.2f", s)
		}
		fmt.Println()
	}
}

// ────────── Helper Code ──────────

type evalResult struct {
	score   float64
	exp     []float64
	pred    []float64
	buckets map[string]int
}

func evaluate(net *paragon.Network[float32], inputs, targets [][][]float64) evalResult {
	exp, pred := []float64{}, []float64{}
	for i, in := range inputs {
		net.Forward(in)
		net.ApplySoftmax()
		out := net.ExtractOutput()
		pred = append(pred, float64(paragon.ArgMax(out)))
		exp = append(exp, float64(paragon.ArgMax(targets[i][0])))
	}
	net.EvaluateModel(exp, pred)
	b := map[string]int{}
	for k, v := range net.Performance.Buckets {
		b[k] = v.Count
	}
	return evalResult{net.Performance.Score, exp, pred, b}
}

func EntropyGate[T paragon.Numeric](layer *paragon.Grid[T]) func(input [][]T) float64 {
	return func(input [][]T) float64 {
		if len(input) == 0 || len(input[0]) == 0 {
			return 0.0
		}
		var values []float64
		for _, row := range input {
			for _, v := range row {
				values = append(values, float64(v))
			}
		}
		if len(values) == 0 {
			return 0.0 // <-- this guards the crash
		}
		out := paragon.Softmax(values)
		sum := 0.0
		for _, p := range out {
			if p > 0 {
				sum -= p * math.Log2(p)
			}
		}
		return sum / math.Log2(float64(len(out)))
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

func Clone[T paragon.Numeric](n *paragon.Network[T]) *paragon.Network[T] {
	bytes, _ := n.MarshalJSONModel()
	clone := &paragon.Network[T]{}
	_ = clone.UnmarshalJSONModel(bytes)
	return clone
}

func avg(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	total := 0.0
	for _, x := range xs {
		total += x
	}
	return total / float64(len(xs))
}

func printLabelDistribution(tgts [][][]float64) {
	count := map[int]int{}
	for _, t := range tgts {
		count[paragon.ArgMax(t[0])]++
	}
	fmt.Printf("\nLabel Distribution: DOWN=%d | FLAT=%d | UP=%d\n", count[0], count[1], count[2])
}

func generateBalancedStockData(numDays, seqLength int) ([][][]float64, [][][]float64) {
	rng := rand.New(rand.NewSource(fixedSeed))
	prices := make([]float64, numDays)
	prices[0] = 100.0
	for i := 1; i < numDays; i++ {
		delta := rng.NormFloat64() * 0.5
		prices[i] = math.Max(10, prices[i-1]*(1+delta/100))
	}

	changes := make([]int, numDays-1)
	for i := 1; i < len(prices); i++ {
		changes[i-1] = classifyChange(prices[i-1], prices[i])
	}

	type sample struct{ in, out [][]float64 }
	buckets := map[int][]sample{0: {}, 1: {}, 2: {}}

	for i := 0; i < len(changes)-seqLength; i++ {
		next := changes[i+seqLength]
		in := [][]float64{make([]float64, seqLength)}
		for j := 0; j < seqLength; j++ {
			in[0][j] = float64(changes[i+j])
		}
		out := [][]float64{{0, 0, 0}}
		out[0][next] = 1.0
		buckets[next] = append(buckets[next], sample{in, out})
	}

	minCount := min(len(buckets[0]), len(buckets[1]), len(buckets[2]))
	var ins, tgts [][][]float64
	for cls := 0; cls <= 2; cls++ {
		s := buckets[cls]
		rng.Shuffle(len(s), func(i, j int) { s[i], s[j] = s[j], s[i] })
		for i := 0; i < minCount; i++ {
			ins = append(ins, s[i].in)
			tgts = append(tgts, s[i].out)
		}
	}
	rng.Shuffle(len(ins), func(i, j int) {
		ins[i], ins[j] = ins[j], ins[i]
		tgts[i], tgts[j] = tgts[j], tgts[i]
	})
	return ins, tgts
}

func classifyChange(prev, curr float64) int {
	pct := (curr - prev) / prev * 100
	switch {
	case pct < -0.5:
		return 0
	case pct > 0.5:
		return 2
	default:
		return 1
	}
}

func min(a, b, c int) int {
	if a < b && a < c {
		return a
	} else if b < c {
		return b
	}
	return c
}
