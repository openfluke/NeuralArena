package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"paragon"
)

const (
	filePath     = "eeg-eye-state.csv"
	inputSize    = 14
	outputSize   = 2
	epochs       = 100
	learningRate = 0.05
	valSplit     = 0.2
	fixedSeed    = 1337
)

type sample struct {
	x []float64
	y []float64
}

type result struct {
	score   float64
	buckets map[string]int
	exp     []float64
	pred    []float64
}

func main() {
	rand.Seed(fixedSeed)
	fmt.Println("📊 Loading EEG data...")

	data := loadData(filePath)
	data = balanceClasses(data)
	normalize(data)
	train, val := splitData(data, valSplit)

	// Prepare train sets for Paragon format
	X, Y := asBatch(train)
	valX, valY := asBatch(val)

	// ────────── Standard ──────────
	fmt.Println("\n🧠 Training Standard Model")
	netStd := buildModel()
	netStd.Train(X, Y, epochs, learningRate, false)
	resStd := evaluate(netStd, valX, valY)

	// ────────── Static Replay ──────────
	fmt.Println("\n🧠 Training Static Replay Model")
	netStatic := buildModel()
	for l := 1; l < netStatic.OutputLayer; l++ {
		layer := &netStatic.Layers[l]
		layer.ReplayEnabled = true
		layer.MaxReplay = 3
		layer.ReplayPhase = "before"
	}
	netStatic.Train(X, Y, epochs, learningRate, false)
	resStatic := evaluate(netStatic, valX, valY)

	// ────────── Dynamic Replay ──────────
	fmt.Println("\n🧠 Training Dynamic Replay Model")
	netDyn := buildModel()
	for l := 1; l < netDyn.OutputLayer; l++ {
		layer := &netDyn.Layers[l]
		layer.ReplayEnabled = true
		layer.ReplayBudget = 20
		layer.ReplayGateFunc = entropyGate(layer)
		layer.ReplayGateToReps = func(e float64) int {
			return 1 + int(math.Pow(e, 1.1)*float64(layer.ReplayBudget-1))
		}
	}
	netDyn.Train(X, Y, epochs, learningRate, false)
	resDyn := evaluate(netDyn, valX, valY)

	// ────────── ADHD Comparison ──────────
	fmt.Println("\n============== ADHD COMPARISON ==============")
	fmt.Printf("Metric                     | Standard | Static  | Dynamic\n")
	fmt.Printf("---------------------------+----------+---------+---------\n")
	fmt.Printf("ADHD Score                 | %8.2f | %7.2f | %7.2f\n",
		resStd.score, resStatic.score, resDyn.score)

	keys := []string{"0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%", "100%+"}
	fmt.Println("\nDeviation buckets (# samples):")
	for _, k := range keys {
		fmt.Printf(" %-7s | %4d | %4d | %4d\n",
			k, resStd.buckets[k], resStatic.buckets[k], resDyn.buckets[k])
	}

	// ────────── Diagnostics ──────────
	fmt.Println("\n------ FULL DIAGNOSTICS: STANDARD ----------")
	netStd.EvaluateFull(resStd.exp, resStd.pred)
	netStd.PrintFullDiagnostics()

	fmt.Println("\n------ FULL DIAGNOSTICS: STATIC REPLAY ----------")
	netStatic.EvaluateFull(resStatic.exp, resStatic.pred)
	netStatic.PrintFullDiagnostics()

	fmt.Println("\n------ FULL DIAGNOSTICS: DYNAMIC REPLAY ----------")
	netDyn.EvaluateFull(resDyn.exp, resDyn.pred)
	netDyn.PrintFullDiagnostics()
}

// ───── Evaluation ─────

func evaluate(net *paragon.Network, X, Y [][][]float64) result {
	exp, pred := []float64{}, []float64{}
	for i := range X {
		net.Forward(X[i])
		net.ApplySoftmax()
		p := float64(paragon.ArgMax(net.ExtractOutput()))
		t := float64(paragon.ArgMax(Y[i][0]))
		pred = append(pred, p)
		exp = append(exp, t)
	}
	net.EvaluateModel(exp, pred)
	buckets := make(map[string]int)
	for k, v := range net.Performance.Buckets {
		buckets[k] = v.Count
	}
	return result{net.Performance.Score, buckets, exp, pred}
}

func entropyGate(layer *paragon.Grid) func([][]float64) float64 {
	return func(_ [][]float64) float64 {
		if len(layer.CachedOutputs) == 0 {
			return 0.0
		}
		out := paragon.Softmax(layer.CachedOutputs)
		h := 0.0
		for _, p := range out {
			if p > 0 {
				h -= p * math.Log2(p)
			}
		}
		return h / math.Log2(float64(len(out)))
	}
}

// ───── Model Setup ─────

func buildModel() *paragon.Network {
	layers := []struct{ Width, Height int }{
		{inputSize, 1},
		{64, 1},
		{32, 1},
		{outputSize, 1},
	}
	acts := []string{"leaky_relu", "leaky_relu", "leaky_relu", "softmax"}
	full := []bool{true, true, true, true}
	return paragon.NewNetwork(layers, acts, full)
}

// ───── Data ─────

func loadData(file string) []sample {
	f, err := os.Open(file)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	r := csv.NewReader(f)
	rows, _ := r.ReadAll()
	var data []sample
	for _, row := range rows {
		if len(row) < inputSize+1 {
			continue
		}
		x := make([]float64, inputSize)
		for i := 0; i < inputSize; i++ {
			x[i], _ = strconv.ParseFloat(row[i], 64)
		}
		labelStr := strings.Trim(row[inputSize], "b'")
		yVal, _ := strconv.Atoi(labelStr)
		y := make([]float64, outputSize)
		y[yVal] = 1.0
		data = append(data, sample{x, y})
	}
	return data
}

func normalize(data []sample) {
	min := make([]float64, inputSize)
	max := make([]float64, inputSize)
	for i := range min {
		min[i] = math.MaxFloat64
	}
	for _, s := range data {
		for i := range s.x {
			if s.x[i] < min[i] {
				min[i] = s.x[i]
			}
			if s.x[i] > max[i] {
				max[i] = s.x[i]
			}
		}
	}
	for _, s := range data {
		for i := range s.x {
			s.x[i] = (s.x[i] - min[i]) / (max[i] - min[i] + 1e-6)
		}
	}
}

// balanceClasses returns an equal‑sized, class‑balanced slice
// and randomises the final order to avoid label‑ordering bias.
func balanceClasses(data []sample) []sample {
	// Separate by class
	var c0, c1 []sample
	for _, s := range data {
		if argmax(s.y) == 0 {
			c0 = append(c0, s)
		} else {
			c1 = append(c1, s)
		}
	}

	// Trim to the minority count
	n := min(len(c0), len(c1))

	// Local shuffle inside each class (optional but nice)
	rand.Shuffle(len(c0), func(i, j int) { c0[i], c0[j] = c0[j], c0[i] })
	rand.Shuffle(len(c1), func(i, j int) { c1[i], c1[j] = c1[j], c1[i] })

	// Merge and *then* shuffle to mix the labels
	balanced := append(c0[:n], c1[:n]...)
	rand.Shuffle(len(balanced), func(i, j int) { balanced[i], balanced[j] = balanced[j], balanced[i] })

	return balanced
}

func splitData(data []sample, split float64) (train, val []sample) {
	// 👇 also shuffle once more to be safe
	rand.Shuffle(len(data), func(i, j int) { data[i], data[j] = data[j], data[i] })
	n := int(float64(len(data)) * (1 - split))
	return data[:n], data[n:]
}

func asBatch(data []sample) ([][][]float64, [][][]float64) {
	X, Y := [][][]float64{}, [][][]float64{}
	for _, d := range data {
		X = append(X, [][]float64{d.x})
		Y = append(Y, [][]float64{d.y})
	}
	return X, Y
}

func argmax(v []float64) int {
	maxI := 0
	for i := range v {
		if v[i] > v[maxI] {
			maxI = i
		}
	}
	return maxI
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
