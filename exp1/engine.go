package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"paragon"
)

// Digit patterns: 2x3 binary grids flattened to 6 elements (0=off, 1=on)
var digitPatterns = map[int][]int{
	0: {1, 1, 1, 1, 0, 1}, // 110101
	1: {0, 1, 0, 0, 1, 1}, // 010011
	2: {1, 0, 1, 1, 1, 0}, // 101110
	3: {1, 0, 1, 0, 1, 1}, // 101011
	4: {0, 1, 1, 0, 1, 0}, // 011010
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// Generate training data
	trainData := generateDigitData(200)

	// Transformer configuration
	tConfig := paragon.TransformerConfig{
		DModel:      128,
		NHeads:      4,
		NLayers:     1,
		FeedForward: 256,
		VocabSize:   3, // Includes "0", "1", "[MASK]"
		MaxLength:   6,
		Activation:  "relu",
		GridRows:    2,
		GridCols:    3,
	}

	// Diffusion configuration
	dConfig := paragon.DiffusionConfig{
		NumTimesteps:      20, // More steps for gradual denoising
		MaxLength:         6,
		LearningRate:      0.001, // Increased for faster learning
		Epochs:            100,
		Temperature:       1.0,
		TopK:              2,
		MaskScheduleStart: 0.0,
		MaskScheduleEnd:   0.8, // Higher noise at later timesteps
	}

	// Initialize network and diffusion model
	network := paragon.NewTransformerEncoder(tConfig)
	// Add small random bias initialization to hidden and output layers
	for l := 1; l < len(network.Layers); l++ {
		for y := 0; y < network.Layers[l].Height; y++ {
			for x := 0; x < network.Layers[l].Width; x++ {
				network.Layers[l].Neurons[y][x].Bias = rand.NormFloat64() * 0.01
			}
		}
	}
	tokenizer := paragon.CustomTokenizer{
		Vocab:         map[string]int{"0": 0, "1": 1, "[MASK]": 2},
		ReverseVocab:  map[int]string{0: "0", 1: "1", 2: "[MASK]"},
		VocabSize:     3,
		SpecialTokens: map[int]bool{2: true},
	}
	model := paragon.NewDiffusionModel(network, dConfig, nil)
	model.Tokenizer = &tokenizer

	// Train the diffusion model
	fmt.Println("Starting diffusion training...")
	trainDiffusion(model, trainData, tConfig)

	// Generate digits and evaluate
	fmt.Println("\nGenerating digits:")
	correct := 0
	for i := 0; i < 5; i++ {
		generated := generateBetter(model, tConfig)
		closestDigit := findClosestDigit(generated)
		fmt.Printf("Generated %d: %v, Closest to digit: %d\n", i, generated, closestDigit)
		if hammingDistance(generated, digitPatterns[closestDigit]) <= 1 {
			correct++
		}
	}
	successRate := float64(correct) / 5 * 100
	fmt.Printf("Success rate: %.2f%% (%d/5)\n", successRate, correct)
}

// ### Training Function
// Computes loss only on masked positions
func trainDiffusion(model *paragon.DiffusionModel, samples [][]int, tConfig paragon.TransformerConfig) {
	data := make([][]int, len(samples))
	copy(data, samples)

	for epoch := 0; epoch < model.Config.Epochs; epoch++ {
		totalLoss := 0.0
		lr := model.Config.LearningRate

		rand.Shuffle(len(data), func(i, j int) {
			data[i], data[j] = data[j], data[i]
		})

		for _, x0 := range data {
			t := rand.Intn(model.Config.NumTimesteps)
			xt := addNoise(x0, t, model.Config.NumTimesteps, model.Config.MaskScheduleStart, model.Config.MaskScheduleEnd)

			// Build one-hot input [6][3]
			batchInput := make([][]float64, model.Config.MaxLength)
			for i, tok := range xt {
				batchInput[i] = make([]float64, tConfig.VocabSize)
				if tok == 0 {
					batchInput[i][0] = 1.0
				} else if tok == 1 {
					batchInput[i][1] = 1.0
				} else if tok == 2 { // [MASK]
					batchInput[i][2] = 1.0
				}
			}

			// Forward pass
			output2D := model.Network.ForwardTransformer(batchInput)
			preds := output2D[0]

			// Compute loss on masked positions
			loss := 0.0
			maskedCount := 0
			errorTerms := make([]float64, model.Config.MaxLength*tConfig.VocabSize)
			maskID := model.Tokenizer.Vocab["[MASK]"]
			for i, tok := range xt {
				if tok == maskID {
					maskedCount++
					start := i * tConfig.VocabSize
					end := start + tConfig.VocabSize
					probs := paragon.Softmax(preds[start:end])
					target := x0[i]
					loss -= math.Log(math.Max(probs[target], 1e-10))
					for m := 0; m < tConfig.VocabSize; m++ {
						delta := probs[m]
						if m == target {
							delta -= 1.0
						}
						if delta > 1.0 {
							delta = 1.0
						} else if delta < -1.0 {
							delta = -1.0
						}
						errorTerms[start+m] = delta
					}
				}
			}
			if maskedCount > 0 {
				totalLoss += loss / float64(maskedCount)
			}

			// Reshape error terms for backpropagation
			shaped := make([][]float64, model.Config.MaxLength)
			for i := 0; i < model.Config.MaxLength; i++ {
				start := i * tConfig.VocabSize
				shaped[i] = errorTerms[start : start+tConfig.VocabSize]
			}
			model.Network.BackwardExternal(shaped, lr)
		}

		avgLoss := totalLoss / float64(len(data))
		if epoch%10 == 0 || epoch == model.Config.Epochs-1 {
			fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, avgLoss)
		}
	}
}

// ### Noise Addition
// Adds noise based on a schedule
func addNoise(x0 []int, t, maxT int, start, end float64) []int {
	noisy := make([]int, len(x0))
	copy(noisy, x0)
	noiseLevel := start + (end-start)*float64(t)/float64(maxT-1)
	for i := range noisy {
		if rand.Float64() < noiseLevel {
			noisy[i] = 2 // [MASK]
		}
	}
	return noisy
}

// ### Generation Function
// Samples only from "0" and "1" probabilities
func generateBetter(model *paragon.DiffusionModel, tConfig paragon.TransformerConfig) []int {
	maskID := model.Tokenizer.Vocab["[MASK]"]
	xcur := make([]int, model.Config.MaxLength)
	for i := range xcur {
		xcur[i] = maskID
	}

	for t := model.Config.NumTimesteps - 1; t >= 0; t-- {
		batchInput := make([][]float64, model.Config.MaxLength)
		for i, tok := range xcur {
			batchInput[i] = make([]float64, tConfig.VocabSize)
			if tok == 0 {
				batchInput[i][0] = 1.0
			} else if tok == 1 {
				batchInput[i][1] = 1.0
			} else if tok == maskID {
				batchInput[i][2] = 1.0
			}
		}

		output2D := model.Network.ForwardTransformer(batchInput)
		preds := output2D[0]

		for i, tok := range xcur {
			if tok == maskID {
				start := i * tConfig.VocabSize
				end := start + tConfig.VocabSize
				probs := paragon.Softmax(preds[start:end]) // [p0, p1, p_mask]
				p0 := probs[0]
				p1 := probs[1]
				var p1_normalized float64
				if p0+p1 > 0 {
					p1_normalized = p1 / (p0 + p1)
				} else {
					p1_normalized = 0.5
				}
				if rand.Float64() < p1_normalized {
					xcur[i] = 1
				} else {
					xcur[i] = 0
				}
			}
		}
	}
	return xcur
}

// ### Helper Functions

// Generate training data
func generateDigitData(size int) [][]int {
	data := make([][]int, size)
	for i := 0; i < size; i++ {
		label := rand.Intn(5)
		pattern := make([]int, 6)
		copy(pattern, digitPatterns[label])
		data[i] = pattern
	}
	return data
}

// Find closest digit by Hamming distance
func findClosestDigit(generated []int) int {
	minDist := len(generated)
	closest := 0
	for digit, pattern := range digitPatterns {
		dist := hammingDistance(generated, pattern)
		if dist < minDist {
			minDist = dist
			closest = digit
		}
	}
	return closest
}

// Compute Hamming distance
func hammingDistance(a, b []int) int {
	if len(a) != len(b) {
		return -1
	}
	dist := 0
	for i := range a {
		if a[i] != b[i] {
			dist++
		}
	}
	return dist
}
