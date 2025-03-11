package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"paragon"
)

// We'll define 8×8 faces. Each face is an 8×8 grid with 0=blank, 1=eyes, 2=mouth, 3=eyebrows (or whatever).
func main() {
	rand.Seed(time.Now().UnixNano())

	// A few 8×8 faces:
	faceHappy := [][]int{
		{0, 0, 0, 3, 3, 0, 0, 0},
		{0, 1, 0, 0, 0, 0, 1, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 2, 2, 0, 0, 0},
		{0, 0, 2, 2, 2, 2, 0, 0},
		{0, 0, 0, 2, 2, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
	}
	faceSad := [][]int{
		{0, 0, 3, 3, 3, 3, 0, 0},
		{0, 1, 0, 0, 0, 0, 1, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 2, 2, 2, 2, 0, 0},
		{0, 0, 0, 2, 2, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
	}
	faceAngry := [][]int{
		{0, 3, 3, 3, 3, 3, 3, 0},
		{0, 0, 1, 0, 0, 1, 0, 0},
		{0, 3, 0, 0, 0, 0, 3, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 2, 2, 2, 2, 0, 0},
		{0, 0, 0, 2, 2, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
	}
	faceSurprised := [][]int{
		{0, 0, 3, 3, 3, 3, 0, 0},
		{0, 1, 0, 0, 0, 0, 1, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 2, 2, 2, 2, 0, 0},
		{0, 0, 2, 0, 0, 2, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
	}
	faceWacky := [][]int{
		{0, 3, 1, 3, 0, 1, 3, 0},
		{0, 1, 0, 1, 0, 1, 0, 0},
		{0, 0, 0, 0, 0, 0, 3, 0},
		{0, 0, 2, 2, 2, 2, 0, 0},
		{0, 0, 0, 2, 2, 0, 0, 0},
		{3, 0, 0, 0, 0, 0, 0, 3},
		{0, 0, 0, 1, 1, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
	}

	// Combine them all into a small dataset of 8×8 faces.
	faces := [][][]int{faceHappy, faceSad, faceAngry, faceSurprised, faceWacky}
	flatFaces := flattenFaces(faces, 8, 8)

	// Transformer config
	tConfig := paragon.TransformerConfig{
		DModel:      64,
		NHeads:      4,
		NLayers:     2,
		FeedForward: 64,
		VocabSize:   5,  // tokens {0,1,2,3,[MASK]=4}
		MaxLength:   64, // 8×8
		Activation:  "relu",
	}

	// Diffusion config with a mask schedule
	dConfig := paragon.DiffusionConfig{
		NumTimesteps:      50,
		MaxLength:         64,
		LearningRate:      0.001,
		Epochs:            200,
		Temperature:       0.8,
		TopK:              2,
		MaskScheduleStart: 0.1, // 10% masked at t=0
		MaskScheduleEnd:   0.8, // 80% masked at t=NumTimesteps-1
	}

	tokenizer := &paragon.CustomTokenizer{
		Vocab:         map[string]int{"0": 0, "1": 1, "2": 2, "3": 3, "[MASK]": 4},
		ReverseVocab:  map[int]string{0: "0", 1: "1", 2: "2", 3: "3", 4: "[MASK]"},
		VocabSize:     5,
		SpecialTokens: map[int]bool{4: true},
	}

	// Build network + improved diffusion model
	network := paragon.NewTransformerEncoder(tConfig)
	model := paragon.NewDiffusionModel(network, dConfig, []string{}) // has improved code
	model.Tokenizer = tokenizer

	fmt.Printf("Tokenizer VocabSize: %d, Vocab: %v\n", model.Tokenizer.VocabSize, model.Tokenizer.Vocab)
	fmt.Println("Starting training with Better Diffusion...")

	// Train with the improved method, printing an intermediate sample every 20 epochs
	trainBetterWithSamplesEveryN(model, flatFaces, 20)

	// Then we do a final face generation after training
	fmt.Println("\nFinal face after training, with improved sampling:")
	result := model.GenerateBetter()
	displayGridASCIIFromInts(result, 8, 8, model.Tokenizer)
}

// trainBetterWithSamplesEveryN wraps your improved method but prints a sample at intervals
func trainBetterWithSamplesEveryN(model *paragon.DiffusionModel, samples [][]int, sampleInterval int) {
	data := make([][]int, len(samples))
	copy(data, samples)

	for epoch := 0; epoch < model.Config.Epochs; epoch++ {
		totalLoss := 0.0
		// simple linear LR schedule
		lr := model.Config.LearningRate * (1.0 - float64(epoch)/float64(model.Config.Epochs))

		rand.Shuffle(len(data), func(i, j int) {
			data[i], data[j] = data[j], data[i]
		})

		for _, x0 := range data {
			// pick random step
			t := rand.Intn(model.Config.NumTimesteps)
			xt := model.BetterAddNoise(x0, t)

			// build a [MaxLength][VocabSize] input
			batchInput := make([][]float64, model.Config.MaxLength)
			for i, tok := range xt {
				row := make([]float64, model.Tokenizer.VocabSize)
				if tok >= 0 && tok < model.Tokenizer.VocabSize {
					row[tok] = 1.0
				}
				batchInput[i] = row
			}

			// forward
			output2D := model.Network.ForwardTransformer(batchInput) // shape: [1][MaxLength * VocabSize]
			preds := output2D[0]

			var loss float64
			errorTerms := make([]float64, model.Config.MaxLength*model.Tokenizer.VocabSize)

			maskID := model.Tokenizer.Vocab["[MASK]"]
			for i, tok := range xt {
				if tok == maskID {
					start := i * model.Tokenizer.VocabSize
					end := start + model.Tokenizer.VocabSize
					probs := paragon.Softmax(preds[start:end])
					target := x0[i]
					loss -= math.Log(math.Max(probs[target], 1e-10))

					for m := 0; m < model.Tokenizer.VocabSize; m++ {
						delta := probs[m]
						if m == target {
							delta -= 1.0
						}
						if delta > 5.0 {
							delta = 5.0
						} else if delta < -5.0 {
							delta = -5.0
						}
						errorTerms[start+m] = delta
					}
				}
			}

			// accumulate
			totalLoss += loss

			// reshape for backward
			shaped := make([][]float64, model.Config.MaxLength)
			for i := 0; i < model.Config.MaxLength; i++ {
				st := i * model.Tokenizer.VocabSize
				shaped[i] = errorTerms[st : st+model.Tokenizer.VocabSize]
			}
			model.Network.Backward(shaped, lr)
		}

		avgLoss := totalLoss / float64(len(data))
		if epoch%sampleInterval == 0 {
			fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, avgLoss)
			// sample an intermediate face
			sample := model.GenerateBetter()
			fmt.Println("Intermediate sample face at epoch", epoch, ":")
			displayGridASCIIFromInts(sample, 8, 8, model.Tokenizer)
		}
	}
}

// flattenFaces: convert each 8×8 face into a length-64 slice
func flattenFaces(faces [][][]int, width, height int) [][]int {
	flat := make([][]int, len(faces))
	for i, face := range faces {
		flat[i] = make([]int, width*height)
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				flat[i][y*width+x] = face[y][x]
			}
		}
	}
	return flat
}

// displayGridASCIIFromInts prints an 8×8 array of int tokens as ASCII block chars
func displayGridASCIIFromInts(tokens []int, width, height int, tokenizer *paragon.CustomTokenizer) {
	// clamp tokens
	adjusted := make([]int, width*height)
	for i := 0; i < len(adjusted); i++ {
		if i < len(tokens) && tokens[i] < 4 {
			adjusted[i] = tokens[i]
		} else {
			adjusted[i] = 0
		}
	}
	fmt.Println("Generated Face (ASCII pixels):")
	// 0->' ', 1->'█', 2->'▓', 3->'░'
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			val := adjusted[y*width+x]
			switch val {
			case 0:
				fmt.Print(" ")
			case 1:
				fmt.Print("█")
			case 2:
				fmt.Print("▓")
			case 3:
				fmt.Print("░")
			}
		}
		fmt.Println()
	}
	fmt.Println()
	// Also show the raw ints & decode
	fmt.Println("Raw token ints:", adjusted)
	fmt.Println("Decoded text:  ", tokenizer.Decode(adjusted))
}
