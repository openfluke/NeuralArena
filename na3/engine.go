package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"paragon" // Replace with actual import path
)

func main() {
	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	// Define small dataset
	sentences := []string{
		"hello world",
		"goodbye world",
		"hello there",
		"goodbye there",
	}

	// Initialize tokenizer
	tokenizer := paragon.NewCustomTokenizer(sentences)
	fmt.Printf("Vocab size: %d\n", tokenizer.VocabSize)

	// Configurations
	tConfig := paragon.TransformerConfig{
		DModel:      16,
		NHeads:      2,
		NLayers:     1,
		FeedForward: 32,
		VocabSize:   tokenizer.VocabSize,
		MaxLength:   5,
		Activation:  "relu",
	}

	dConfig := paragon.DiffusionConfig{
		NumTimesteps: 10,
		MaxLength:    5,
		LearningRate: 0.001,
		Epochs:       100,
		Temperature:  1.0,
		TopK:         3,
	}

	// Create network and model
	network := paragon.NewTransformerEncoder(tConfig)
	model := paragon.NewDiffusionModel(network, dConfig, sentences)

	// Train the model
	fmt.Println("Starting training...")
	trainMaskedDiffusion(model, sentences, tokenizer, dConfig, tConfig)

	// Generate text
	fmt.Println("\nGenerating text after training:")
	for i := 0; i < 3; i++ {
		generated := model.GenerateMasked()
		fmt.Printf("Generated %d: %s\n", i+1, generated)
	}
}

func trainMaskedDiffusion(model *paragon.DiffusionModel, sentences []string, tokenizer *paragon.CustomTokenizer,
	dConfig paragon.DiffusionConfig, tConfig paragon.TransformerConfig) {

	batchSize := 4
	numBatches := (len(sentences) + batchSize - 1) / batchSize

	for epoch := 0; epoch < dConfig.Epochs; epoch++ {
		totalLoss := 0.0
		lr := dConfig.LearningRate * (1 + math.Cos(float64(epoch)*math.Pi/float64(dConfig.Epochs))) / 2

		// Prepare data
		data := make([][]int, len(sentences))
		for i, s := range sentences {
			ids := tokenizer.Encode(s)
			if len(ids) > dConfig.MaxLength {
				data[i] = ids[:dConfig.MaxLength]
			} else {
				data[i] = make([]int, dConfig.MaxLength)
				copy(data[i], ids)
				for j := len(ids); j < dConfig.MaxLength; j++ {
					data[i][j] = tokenizer.Vocab["[PAD]"]
				}
			}
		}

		for i := 0; i < len(sentences); i += batchSize {
			end := i + batchSize
			if end > len(sentences) {
				end = len(sentences)
			}
			batch := data[i:end]

			// Prepare batch inputs and targets
			batchInputs := make([][][]float64, len(batch))
			batchTargets := make([][]int, len(batch))
			noisyBatch := make([][]int, len(batch))
			for j, tokens := range batch {
				tVal := rand.Float64()
				noisyTokens := model.AddNoiseMasked(tokens, tVal)
				noisyBatch[j] = noisyTokens
				batchInputs[j] = make([][]float64, dConfig.MaxLength)
				for k := 0; k < dConfig.MaxLength; k++ {
					batchInputs[j][k] = make([]float64, tConfig.VocabSize)
					tok := noisyTokens[k]
					if tok >= 0 && tok < tConfig.VocabSize {
						batchInputs[j][k][tok] = 1.0
					}
				}
				batchTargets[j] = tokens
			}

			// Forward pass
			batchOutputs := make([][][]float64, len(batch))
			for j, input := range batchInputs {
				batchOutputs[j] = model.Network.ForwardTransformer(input)
			}

			// Compute loss and gradients
			loss := 0.0
			errorTerms := make([][]float64, len(batch))
			for j := range batch {
				errorTerms[j] = make([]float64, dConfig.MaxLength*tConfig.VocabSize)
				for k := 0; k < dConfig.MaxLength; k++ {
					if noisyBatch[j][k] == tokenizer.Vocab["[MASK]"] {
						startIdx := k * tConfig.VocabSize
						endIdx := (k + 1) * tConfig.VocabSize
						probs := paragon.Softmax(batchOutputs[j][0][startIdx:endIdx])
						target := batchTargets[j][k]
						loss -= math.Log(math.Max(probs[target], 1e-10))
						for m := 0; m < tConfig.VocabSize; m++ {
							delta := probs[m]
							if m == target {
								delta -= 1
							}
							if delta > 5.0 {
								delta = 5.0
							} else if delta < -5.0 {
								delta = -5.0
							}
							errorTerms[j][startIdx+m] = delta
						}
					}
				}
			}
			totalLoss += loss / float64(len(batch))

			// Reshape errorTerms and perform backward pass for each sample
			for j := range batch {
				flatErrorTerms := errorTerms[j]
				reshapedErrorTerms := make([][]float64, dConfig.MaxLength)
				for k := 0; k < dConfig.MaxLength; k++ {
					start := k * tConfig.VocabSize
					end := start + tConfig.VocabSize
					reshapedErrorTerms[k] = flatErrorTerms[start:end]
				}
				model.Network.Backward(reshapedErrorTerms, lr)
			}
		}

		totalLoss /= float64(numBatches)
		if epoch%10 == 0 {
			fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, totalLoss)
			sample := model.GenerateMasked()
			fmt.Println("Sample:", sample)
		}
	}
}
