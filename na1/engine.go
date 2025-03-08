package main

import (
	"fmt"
	"math"
	"math/rand"
	"paragon"
)

func generateSentences(n int) []string {
	bases := []string{"cat", "dog", "bird", "car", "kid", "sun", "moon", "sky"}
	conditions := []string{"is tired", "barks", "is happy", "moves", "is slow", "is day", "is night", "is excited"}
	results := []string{"rests", "chases", "soars", "stops", "laughs", "shouts", "sings", "purrs"}
	var sentences []string
	for i := 0; i < n; i++ {
		base := bases[rand.Intn(len(bases))]
		cond := conditions[rand.Intn(len(conditions))]
		result := results[rand.Intn(len(results))]
		sentences = append(sentences, fmt.Sprintf("if the %s %s then it %s", base, cond, result))
	}
	return sentences
}

func main() {
	fmt.Println("V5-IMPLEMENTATION-NA1-DIFFUSION-TRANSFORMER")

	sentences := []string{
		"the cat sat on the mat",
		"a dog barked loudly",
		"birds fly in the sky",
		"the sun shines brightly",
		"a car drives fast",
		"the moon glows at night",
		"kids play in the park",
		"if birds fly then they soar",
		"the dog runs if it barks",
		"if the sun shines then it is day",
		"cats sleep if they are tired",
		"if kids play then they laugh",
		"birds sing if the sun shines",
		"if the moon glows then it is night",
		"the car stops if it is slow",
		"if cats sleep then they rest",
		"dogs chase if cats run",
		"if dogs bark then kids wake",
		"the sky darkens if it is night",
		"if birds sing then it is morning",
		"cats purr if they are happy",
		"if the car drives then it moves",
		"kids shout if they are excited",
	}
	sentences = append(sentences, generateSentences(100)...)

	tokenizer := paragon.NewCustomTokenizer(sentences)

	tConfig := paragon.TransformerConfig{
		DModel:      128,
		NHeads:      4,
		NLayers:     2,
		FeedForward: 512,
		MaxLength:   10,
		Activation:  "relu",
		VocabSize:   tokenizer.VocabSize,
	}
	nn := paragon.NewTransformerEncoder(tConfig)

	dConfig := paragon.DiffusionConfig{
		NumTimesteps: 5,
		MaxLength:    10,
		LearningRate: 0.001,
		Epochs:       6000,
		Temperature:  0.4, // Lowered for sharper sampling
		TopK:         2,
	}
	model := paragon.NewDiffusionModel(nn, dConfig, sentences)

	fmt.Println("Starting training...")
	for epoch := 0; epoch < dConfig.Epochs; epoch++ {
		lr := dConfig.LearningRate * (1 + math.Cos(float64(epoch)*math.Pi/float64(dConfig.Epochs))) / 2
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
		totalLoss := 0.0
		for _, tokens := range data {
			t := rand.Intn(dConfig.NumTimesteps)
			noisyTokens := model.AddNoise(tokens, t)
			input := make([][]float64, 1)
			input[0] = make([]float64, dConfig.MaxLength)
			for i, tok := range noisyTokens {
				input[0][i] = float64(tok)
			}
			output := nn.ForwardTransformer(input)
			loss := 0.0
			for i := 0; i < dConfig.MaxLength; i++ {
				probs := paragon.Softmax(output[i])
				target := tokens[i]
				loss -= math.Log(math.Max(probs[target], 1e-10))
			}
			totalLoss += loss / float64(dConfig.MaxLength)
			errorTerms := make([][]float64, dConfig.MaxLength)
			for i := 0; i < dConfig.MaxLength; i++ {
				probs := paragon.Softmax(output[i])
				errorTerms[i] = make([]float64, len(probs))
				for j := 0; j < len(probs); j++ {
					delta := probs[j]
					if j == tokens[i] {
						delta -= 1
					}
					if delta > 5.0 {
						delta = 5.0
					} else if delta < -5.0 {
						delta = -5.0
					}
					errorTerms[i][j] = delta
				}
			}
			nn.Backward(errorTerms, lr)
		}
		if epoch%40 == 0 {
			fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, totalLoss/float64(len(sentences)))
		}
	}

	fmt.Println("Generating text...")
	generated := model.Generate()
	fmt.Println("Final generated text:", generated)

	sampleInput := make([][]float64, 1)
	sampleInput[0] = make([]float64, tConfig.MaxLength)
	for i := range sampleInput[0] {
		sampleInput[0][i] = float64(model.Tokenizer.Vocab["if"])
	}
	output := nn.ForwardTransformer(sampleInput)
	fmt.Println("Output layer values (raw logits for first token):")
	for x := 0; x < tConfig.VocabSize; x++ {
		if x < len(output[0]) {
			fmt.Printf("%.4f ", output[0][x])
		} else {
			fmt.Printf("0.0000 ")
		}
	}
	fmt.Println()
}
