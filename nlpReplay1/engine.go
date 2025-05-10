// main.go – NLP replay experiment (v3)
package main

import (
	"fmt"
	"math"
	"math/rand"

	"paragon"
)

// ─────────────── Config ───────────────
const (
	modelPath    = "nlp_model.json"
	fixedSeed    = 1337
	seqLength    = 10
	numSamples   = 1000
	numClasses   = 3
	vocabSize    = 50
	epochsStd    = 20 // standard / static
	epochsDyn    = 25 // dynamic gets a few extra sweeps
	learningRate = 0.0001
	runsPerMode  = 5
)

type results struct{ score float64 }

// ─────────────── main ───────────────
func main() {
	rand.Seed(fixedSeed)

	base := buildModel(seqLength, numClasses)
	ins, tgts := generateNLPClassificationData(numSamples, seqLength, vocabSize, numClasses)

	log := map[string][]results{
		"standard": {},
		"static":   {},
		"dynamic":  {},
	}

	// ───── STANDARD ─────
	for i := 0; i < runsPerMode; i++ {
		net := Clone(base)
		fmt.Printf("\n[Standard %d] Training…\n", i+1)
		net.Train(ins, tgts, epochsStd, learningRate, false)
		log["standard"] = append(log["standard"], evaluate(net, ins, tgts))
	}

	// ───── STATIC (fixed 5 replays per layer) ─────
	for i := 0; i < runsPerMode; i++ {
		net := Clone(base)
		fmt.Printf("\n[Static %d] Training with Replay x5…\n", i+1)
		for l := 1; l < net.OutputLayer; l++ {
			layer := &net.Layers[l]
			layer.ReplayEnabled = true // ← fixed
			layer.ReplayOffset = -1
			layer.ReplayPhase = "before"
			layer.MaxReplay = 5
		}
		net.Train(ins, tgts, epochsStd, learningRate, false)
		log["static"] = append(log["static"], evaluate(net, ins, tgts))
	}

	// ───── DYNAMIC 3.0 (adaptive) ─────
	for run := 0; run < runsPerMode; run++ {
		net := Clone(base)
		fmt.Printf("\n[Dynamic %d] Training with adaptive entropy replay…\n", run+1)

		// epoch‑varying budget: 15 → 5
		budgetForEpoch := func(epoch int) int {
			start, end := 15.0, 5.0
			p := float64(epoch) / float64(epochsDyn-1)
			return int(math.Round(start + p*(end-start)))
		}

		for epoch := 0; epoch < epochsDyn; epoch++ {
			budget := budgetForEpoch(epoch)

			for l := 1; l < net.OutputLayer; l++ {
				layer := &net.Layers[l]
				layer.ReplayEnabled = true
				layer.ReplayBudget = budget
				layer.ReplayGateFunc = EntropyGate(layer)
				layer.ReplayGateToReps = func(e float64) int {
					// reps = 1..budget, linear in entropy
					r := int(1 + e*float64(budget-1))
					if r < 1 {
						r = 1
					}
					return r
				}
			}

			// train exactly ONE epoch each pass so updated budgets apply
			net.Train(ins, tgts, 1, learningRate, false)

			if epoch%5 == 0 {
				fmt.Printf("  ↳ epoch %2d done (budget %2d)\n", epoch, budget)
			}
		}
		log["dynamic"] = append(log["dynamic"], evaluate(net, ins, tgts))
	}

	// ───── SUMMARY ─────
	fmt.Println("\n📊 NLP Replay Experiment Scores (All Models):")
	for mode, runs := range log {
		fmt.Printf("\n=== %s ===\n", mode)
		sum := 0.0
		for i, r := range runs {
			fmt.Printf("Model %d → Score: %.2f\n", i+1, r.score)
			sum += r.score
		}
		fmt.Printf("→ Average Score for %s: %.2f\n", mode, sum/float64(len(runs)))
	}
}

// ─────────────── Helpers ───────────────
func buildModel(seqLen, numClasses int) *paragon.Network {
	layers := []struct{ Width, Height int }{
		{seqLen, 1},
		{64, 1},
		{32, 1},
		{numClasses, 1},
	}
	acts := []string{"leaky_relu", "leaky_relu", "leaky_relu", "softmax"}
	full := []bool{true, true, true, true}
	return paragon.NewNetwork(layers, acts, full)
}

func generateNLPClassificationData(n, seqLen, vocabSize, numClasses int) ([][][]float64, [][][]float64) {
	rng := rand.New(rand.NewSource(fixedSeed))
	var ins, tgts [][][]float64
	for i := 0; i < n; i++ {
		input := [][]float64{make([]float64, seqLen)}
		for j := 0; j < seqLen; j++ {
			input[0][j] = float64(rng.Intn(vocabSize)) / float64(vocabSize)
		}
		label := rng.Intn(numClasses)
		target := [][]float64{{0, 0, 0}}
		target[0][label] = 1
		ins = append(ins, input)
		tgts = append(tgts, target)
	}
	return ins, tgts
}

func EntropyGate(layer *paragon.Grid) func([][]float64) float64 {
	return func(_ [][]float64) float64 {
		if len(layer.CachedOutputs) == 0 {
			return 0
		}
		out := paragon.Softmax(layer.CachedOutputs)
		h := 0.0
		for _, p := range out {
			if p > 0 {
				h -= p * math.Log2(p)
			}
		}
		return h / math.Log2(float64(len(out))) // 0‑1
	}
}

func evaluate(net *paragon.Network, inputs, targets [][][]float64) results {
	exp, pred := []float64{}, []float64{}
	for i, in := range inputs {
		net.Forward(in)
		net.ApplySoftmax()
		out := net.ExtractOutput()
		pred = append(pred, float64(paragon.ArgMax(out)))
		exp = append(exp, float64(paragon.ArgMax(targets[i][0])))
	}
	net.EvaluateModel(exp, pred)
	return results{score: net.Performance.Score}
}

func Clone(n *paragon.Network) *paragon.Network {
	b, _ := n.MarshalJSONModel()
	c := &paragon.Network{}
	_ = c.UnmarshalJSONModel(b)
	return c
}
