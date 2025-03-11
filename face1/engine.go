package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"paragon"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	faces := [][][]int{
		// Happy
		{{0, 0, 0, 0, 0},
			{0, 1, 0, 1, 0},
			{0, 0, 0, 0, 0},
			{0, 2, 2, 2, 0},
			{0, 0, 0, 0, 0}},

		// Sad
		{{0, 0, 0, 0, 0},
			{0, 1, 0, 1, 0},
			{0, 0, 0, 0, 0},
			{0, 0, 2, 0, 0},
			{0, 2, 2, 2, 0}},

		// Angry
		{{0, 0, 0, 0, 0},
			{0, 1, 3, 1, 0},
			{0, 3, 0, 3, 0},
			{0, 2, 2, 2, 0},
			{0, 0, 0, 0, 0}},

		// Surprised
		{{0, 0, 0, 0, 0},
			{0, 1, 0, 1, 0},
			{0, 3, 3, 3, 0},
			{0, 2, 0, 2, 0},
			{0, 0, 0, 0, 0}},
	}
	flatFaces := flattenFaces(faces, 5, 5)

	tConfig := paragon.TransformerConfig{
		DModel:      32,
		NHeads:      2,
		NLayers:     2,
		FeedForward: 32,
		VocabSize:   5,
		MaxLength:   25,
		Activation:  "relu",
	}

	dConfig := paragon.DiffusionConfig{
		NumTimesteps: 50,
		MaxLength:    25,
		// Feel free to adjust; 0.0005 or 0.0001 might be even more stable
		LearningRate: 0.001,
		Epochs:       200,
		// Lower temperature and topK=1 to reduce randomness:
		Temperature: 0.8,
		TopK:        1,
	}

	tokenizer := &paragon.CustomTokenizer{
		Vocab:         map[string]int{"0": 0, "1": 1, "2": 2, "3": 3, "[MASK]": 4},
		ReverseVocab:  map[int]string{0: "0", 1: "1", 2: "2", 3: "3", 4: "[MASK]"},
		VocabSize:     5,
		SpecialTokens: map[int]bool{4: true},
	}

	network := paragon.NewTransformerEncoder(tConfig)
	model := paragon.NewDiffusionModel(network, dConfig, []string{})
	model.Tokenizer = tokenizer

	fmt.Printf("Tokenizer VocabSize: %d, Vocab: %v\n", model.Tokenizer.VocabSize, model.Tokenizer.Vocab)
	fmt.Println("Starting training...")
	trainPixelDiffusion(model, flatFaces, tokenizer, dConfig, tConfig)

	fmt.Println("\nGenerating a cute face:")
	generated := model.GenerateMasked()
	displayGrid(generated, 5, 5, model.Tokenizer)
}

// flattenFaces turns each 5x5 face into a flat slice of length 25.
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

// displayGrid shows the final sequence as a grid of characters.
func displayGrid(sequence string, width, height int, tokenizer *paragon.CustomTokenizer) {
	tokens := tokenizer.Encode(sequence)
	fmt.Printf("Raw generated sequence (length %d): %v\n", len(tokens), tokens)

	// Adjust to exactly 25 tokens, clamp invalid ones
	adjusted := make([]int, width*height)
	for i := 0; i < len(adjusted); i++ {
		if i < len(tokens) {
			if tokens[i] >= 4 {
				adjusted[i] = 0
			} else {
				adjusted[i] = tokens[i]
			}
		} else {
			adjusted[i] = 0
		}
	}
	fmt.Printf("Adjusted sequence (length %d): %v\n", len(adjusted), adjusted)

	fmt.Println("Generated Face:")
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			val := adjusted[y*width+x]
			switch val {
			case 0:
				fmt.Print(" ")
			case 1:
				fmt.Print("o")
			case 2:
				fmt.Print("-")
			case 3:
				fmt.Print("^")
			case 4:
				fmt.Print("?")
			default:
				fmt.Print("X")
			}
		}
		fmt.Println()
	}
}

// trainPixelDiffusion does a masked diffusion style training loop.
// Main difference: accumulate error terms for the entire batch, then do one backward pass.
func trainPixelDiffusion(model *paragon.DiffusionModel, faces [][]int, tokenizer *paragon.CustomTokenizer,
	dConfig paragon.DiffusionConfig, tConfig paragon.TransformerConfig) {

	batchSize := 3
	numBatches := (len(faces) + batchSize - 1) / batchSize

	// Instead of the old cosine schedule, let's do a simple linear decay from initial LR to near zero
	baseLR := dConfig.LearningRate

	for epoch := 0; epoch < dConfig.Epochs; epoch++ {
		// linear decay from (baseLR) down to (0) across epochs
		progress := float64(epoch) / float64(dConfig.Epochs)
		lr := baseLR * (1.0 - progress)

		totalLoss := 0.0

		// Prepare data
		data := make([][]int, len(faces))
		for i, face := range faces {
			data[i] = make([]int, dConfig.MaxLength)
			copy(data[i], face)
		}

		// Shuffle the data each epoch
		rand.Shuffle(len(data), func(i, j int) {
			data[i], data[j] = data[j], data[i]
		})

		for i := 0; i < len(data); i += batchSize {
			end := i + batchSize
			if end > len(data) {
				end = len(data)
			}
			batch := data[i:end]

			// We'll accumulate error terms for the entire batch, then call Backward() once.
			accumulatedError := make([]float64, dConfig.MaxLength*tConfig.VocabSize)

			batchLoss := 0.0

			for _, pixels := range batch {
				// Sample a continuous t in [0,1]
				tVal := rand.Float64()
				noisyPixels := model.AddNoiseMasked(pixels, tVal)

				// Build one-hot inputs for the forward pass
				batchInput := make([][]float64, dConfig.MaxLength)
				for k := 0; k < dConfig.MaxLength; k++ {
					row := make([]float64, tConfig.VocabSize)
					tok := noisyPixels[k]
					if tok >= 0 && tok < tConfig.VocabSize {
						row[tok] = 1.0
					}
					batchInput[k] = row
				}

				// Single forward pass
				output := model.Network.ForwardTransformer(batchInput) // shape [1][MaxLength*VocabSize]

				// Compute cross-entropy only on masked tokens
				localError := make([]float64, dConfig.MaxLength*tConfig.VocabSize)
				for k := 0; k < dConfig.MaxLength; k++ {
					if noisyPixels[k] == tokenizer.Vocab["[MASK]"] {
						startIdx := k * tConfig.VocabSize
						endIdx := (k + 1) * tConfig.VocabSize
						probs := paragon.Softmax(output[0][startIdx:endIdx])
						target := pixels[k]

						// Cross-entropy contribution
						batchLoss -= math.Log(math.Max(probs[target], 1e-10))

						// Error terms for backprop
						for m := 0; m < tConfig.VocabSize; m++ {
							delta := probs[m]
							if m == target {
								delta -= 1
							}
							// gradient clipping
							if delta > 5.0 {
								delta = 5.0
							} else if delta < -5.0 {
								delta = -5.0
							}
							localError[startIdx+m] = delta
						}
					}
				}
				// Accumulate localError into accumulatedError
				for idx, val := range localError {
					accumulatedError[idx] += val
				}
			} // end batch loop

			// Average loss over samples in the batch
			batchLoss /= float64(len(batch))
			totalLoss += batchLoss

			// Now do exactly one backward pass for the entire batch
			// We must reshape accumulatedError back into shape [MaxLength][VocabSize] to feed Backward
			reshaped := make([][]float64, dConfig.MaxLength)
			for k := 0; k < dConfig.MaxLength; k++ {
				start := k * tConfig.VocabSize
				end := start + tConfig.VocabSize
				reshaped[k] = accumulatedError[start:end]
			}
			model.Network.BackwardExternal(reshaped, lr)
		}

		epochLoss := totalLoss / float64(numBatches)

		// Checkpoint sample every 20 epochs
		if epoch%20 == 0 {
			fmt.Printf("Epoch %d, LR: %.5f, Loss: %.4f\n", epoch, lr, epochLoss)
			sample := model.GenerateMasked()
			fmt.Println("Sample generation:")
			displayGrid(sample, 5, 5, model.Tokenizer)
		}
	}
}
