package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"paragon"
)

// We'll define 16×16 faces with 0=blank, 1=eyes, 2=mouth, 3=eyebrows.
func main() {
	rand.Seed(time.Now().UnixNano())

	// Generate more training data
	faces := generateTrainingFaces(100) // 100 synthetic faces
	flatFaces := flatten16x16(faces)

	// Adjusted configs
	tConfig := paragon.TransformerConfig{
		DModel:      32, // Reduced capacity
		NHeads:      2,
		NLayers:     2,
		FeedForward: 32,
		VocabSize:   5,
		MaxLength:   256,
		Activation:  "relu",
		GridRows:    16, // 2D encoding
		GridCols:    16,
	}
	dConfig := paragon.DiffusionConfig{
		NumTimesteps:      50,
		MaxLength:         256,
		LearningRate:      0.001,
		Epochs:            200, // More epochs with more data
		Temperature:       0.5, // Lower for sharper outputs
		TopK:              1,   // Strict sampling
		MaskScheduleStart: 0.05,
		MaskScheduleEnd:   0.9, // Broader noise range
	}

	// Tokenizer
	tokenizer := &paragon.CustomTokenizer{
		Vocab:         map[string]int{"0": 0, "1": 1, "2": 2, "3": 3, "[MASK]": 4},
		ReverseVocab:  map[int]string{0: "0", 1: "1", 2: "2", 3: "3", 4: "[MASK]"},
		VocabSize:     5,
		SpecialTokens: map[int]bool{4: true},
	}

	// Build model
	network := paragon.NewTransformerEncoder(tConfig)
	model := paragon.NewDiffusionModel(network, dConfig, []string{})
	model.Tokenizer = tokenizer

	fmt.Printf("Tokenizer VocabSize: %d, Vocab: %v\n", model.Tokenizer.VocabSize, model.Tokenizer.Vocab)
	fmt.Println("Training data samples:")
	for i := 0; i < min(3, len(faces)); i++ {
		fmt.Printf("Face %d:\n", i)
		displayGridASCII16(flatFaces[i], tokenizer)
	}

	fmt.Println("Starting 16×16 face training...")
	trainBetterWithSamples(model, flatFaces)

	// Final generation
	fmt.Println("\nFinal generation:")
	finalTokens := model.GenerateBetter()
	displayGridASCII16(finalTokens, model.Tokenizer)
}

// generateTrainingFaces creates synthetic 16x16 faces
func generateTrainingFaces(num int) [][16][16]int {
	faces := make([][16][16]int, num)
	baseFaces := [][16][16]int{buildHappyFace16(), buildSadFace16(), buildAngryFace16()}
	for i := 0; i < num; i++ {
		base := baseFaces[i%len(baseFaces)]
		faces[i] = perturbFace(base) // Add slight variations
	}
	return faces
}

// perturbFace adds random shifts to features
func perturbFace(face [16][16]int) [16][16]int {
	result := face
	shift := rand.Intn(3) - 1 // -1, 0, or 1
	for r := 0; r < 16; r++ {
		for c := 0; c < 16; c++ {
			newR := r + shift
			if newR >= 0 && newR < 16 {
				result[newR][c] = face[r][c]
			}
		}
	}
	return result
}

// buildHappyFace16: eyebrows=3, eyes=1, mouth=2, rest=0
func buildHappyFace16() [16][16]int {
	var face [16][16]int
	// eyebrows row=2..3, col=4..11 => 3
	for r := 2; r <= 3; r++ {
		for c := 4; c <= 11; c++ {
			face[r][c] = 3
		}
	}
	// eyes row=5, left col=4..5, right col=10..11 => 1
	for c := 4; c <= 5; c++ {
		face[5][c] = 1
	}
	for c := 10; c <= 11; c++ {
		face[5][c] = 1
	}
	// mouth row=10..11 => 2, col=4..11
	for r := 10; r <= 11; r++ {
		for c := 4; c <= 11; c++ {
			face[r][c] = 2
		}
	}
	return face
}

func buildSadFace16() [16][16]int {
	var face [16][16]int
	// angled eyebrows row=1..2, col=4..11 => 3
	for r := 1; r <= 2; r++ {
		for c := 4; c <= 11; c++ {
			face[r][c] = 3
		}
	}
	// eyes row=5 => left col=4..5, right col=10..11 => 1
	for c := 4; c <= 5; c++ {
		face[5][c] = 1
	}
	for c := 10; c <= 11; c++ {
		face[5][c] = 1
	}
	// frowny mouth row=10..11 => 2, but let's put a gap in the middle
	for r := 10; r <= 11; r++ {
		for c := 4; c <= 11; c++ {
			if c >= 7 && c <= 8 {
				face[r][c] = 0
			} else {
				face[r][c] = 2
			}
		}
	}
	return face
}

func buildAngryFace16() [16][16]int {
	var face [16][16]int
	// thick eyebrows row=2..3 => 3, col=3..12
	for r := 2; r <= 3; r++ {
		for c := 3; c <= 12; c++ {
			face[r][c] = 3
		}
	}
	// eyes row=6 => left col=4..5, right col=10..11 => 1
	for c := 4; c <= 5; c++ {
		face[6][c] = 1
	}
	for c := 10; c <= 11; c++ {
		face[6][c] = 1
	}
	// wide open mouth row=9..12 => 2, col=4..11
	for r := 9; r <= 12; r++ {
		for c := 4; c <= 11; c++ {
			face[r][c] = 2
		}
	}
	return face
}

// flatten16x16: from [16][16]int to length=256 slices
func flatten16x16(faces [][16][16]int) [][]int {
	out := make([][]int, len(faces))
	for i, face := range faces {
		flat := make([]int, 16*16)
		idx := 0
		for r := 0; r < 16; r++ {
			for c := 0; c < 16; c++ {
				flat[idx] = face[r][c]
				idx++
			}
		}
		out[i] = flat
	}
	return out
}

// trainBetterWithSamples => replicate the improved approach, printing an ASCII sample every 20 epochs
func trainBetterWithSamples(model *paragon.DiffusionModel, data [][]int) {
	epochs := model.Config.Epochs
	baseLR := model.Config.LearningRate

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		lr := baseLR * (1.0 - float64(epoch)/float64(epochs))

		rand.Shuffle(len(data), func(i, j int) {
			data[i], data[j] = data[j], data[i]
		})

		for _, x0 := range data {
			t := rand.Intn(model.Config.NumTimesteps)
			xt := model.BetterAddNoise(x0, t)

			batchInput := make([][]float64, model.Config.MaxLength)
			for i, tok := range xt {
				row := make([]float64, model.Tokenizer.VocabSize)
				if tok >= 0 && tok < model.Tokenizer.VocabSize {
					row[tok] = 1.0
				}
				batchInput[i] = row
			}

			output2D := model.Network.ForwardTransformer(batchInput)
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
						if delta > 1.0 { // Tighter clipping
							delta = 1.0
						} else if delta < -1.0 {
							delta = -1.0
						}
						errorTerms[start+m] = delta
					}
				}
			}
			totalLoss += loss

			shaped := make([][]float64, model.Config.MaxLength)
			for i := 0; i < model.Config.MaxLength; i++ {
				st := i * model.Tokenizer.VocabSize
				shaped[i] = errorTerms[st : st+model.Tokenizer.VocabSize]
			}
			model.Network.BackwardExternal(shaped, lr)
		}

		avgLoss := totalLoss / float64(len(data))
		if epoch%2 == 0 || epoch == epochs-1 {
			fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, avgLoss)
			sample := model.GenerateBetter()
			displayGridASCII16(sample, model.Tokenizer)
		}
	}
}

// displayGridASCII16 prints a 16×16 face from tokens, mapping 0->' ',1->'█',2->'▓',3->'░'
func displayGridASCII16(tokens []int, tokenizer *paragon.CustomTokenizer) {
	adjusted := make([]int, 16*16)
	for i := 0; i < len(adjusted); i++ {
		if i < len(tokens) && tokens[i] < 4 {
			adjusted[i] = tokens[i]
		} else {
			adjusted[i] = 0
		}
	}
	fmt.Println("Generated Face (ASCII 16×16):")
	for r := 0; r < 16; r++ {
		for c := 0; c < 16; c++ {
			val := adjusted[r*16+c]
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
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
