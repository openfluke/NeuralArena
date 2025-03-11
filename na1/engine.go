package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"paragon"
)

// Build a single one-hot row of length vocabSize, with a 1 at index=tokenID, else 0
func oneHotRow(tokenID, vocabSize int) []float64 {
	row := make([]float64, vocabSize)
	if tokenID >= 0 && tokenID < vocabSize {
		row[tokenID] = 1.0
	}
	return row
}

// Build a [MaxLength][VocabSize] array (one-hot) from a slice of tokens
func makeOneHot2D(tokens []int, maxLength, vocabSize int) [][]float64 {
	output := make([][]float64, maxLength)
	for i := 0; i < maxLength; i++ {
		output[i] = oneHotRow(tokens[i], vocabSize)
	}
	return output
}

// Just generating random sentences
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

	// 1) Prepare data
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
	// Add random generated sentences up to 500 total
	sentences = append(sentences, generateSentences(477)...)

	// 2) Tokenizer
	tokenizer := paragon.NewCustomTokenizer(sentences)

	// 3) Transformer config
	tConfig := paragon.TransformerConfig{
		DModel:      128,
		NHeads:      4,
		NLayers:     2,
		FeedForward: 512,
		MaxLength:   10, // sequence length = 10
		Activation:  "relu",
		VocabSize:   tokenizer.VocabSize, // e.g. 69
	}
	// This yields an input layer shape: Width=VocabSize (e.g. 69), Height=10

	// 4) Build network
	nn := paragon.NewTransformerEncoder(tConfig)

	// 5) Diffusion config
	dConfig := paragon.DiffusionConfig{
		NumTimesteps: 5,
		MaxLength:    10,
		LearningRate: 0.002,
		Epochs:       2000,
		Temperature:  0.4,
		TopK:         3,
	}
	// 6) Model with diffusion
	model := paragon.NewDiffusionModel(nn, dConfig, sentences)

	// 7) Training parameters
	batchSize := 10
	multithreading := true
	cpuPercent := 0.8
	numCores := runtime.NumCPU()
	numThreads := int(float64(numCores) * cpuPercent)
	if numThreads < 1 {
		numThreads = 1
	}
	numBatches := (len(sentences) + batchSize - 1) / batchSize
	fmt.Printf("Using %d threads (%d%% of %d cores), %d batches\n",
		numThreads, int(cpuPercent*100), numCores, numBatches)

	fmt.Println("Starting training...")

	// Convert the raw sentences to int[] (padded to length=10)
	data := make([][]int, len(sentences))
	for i, s := range sentences {
		ids := tokenizer.Encode(s)
		if len(ids) > dConfig.MaxLength {
			data[i] = ids[:dConfig.MaxLength]
		} else {
			padded := make([]int, dConfig.MaxLength)
			copy(padded, ids)
			for j := len(ids); j < dConfig.MaxLength; j++ {
				padded[j] = tokenizer.Vocab["[PAD]"]
			}
			data[i] = padded
		}
	}

	for epoch := 0; epoch < dConfig.Epochs; epoch++ {
		startTime := time.Now()
		// Cosine LR schedule
		lr := dConfig.LearningRate * (1 + math.Cos(float64(epoch)*math.Pi/float64(dConfig.Epochs))) / 2

		// We'll shuffle the data each epoch
		rand.Shuffle(len(data), func(i, j int) { data[i], data[j] = data[j], data[i] })

		totalLoss := 0.0
		var wg sync.WaitGroup
		lossChan := make(chan float64, numBatches)
		errorTermsChan := make(chan struct {
			terms    [][]float64
			batchIdx int
		}, numBatches)
		sem := make(chan struct{}, numThreads)

		// For storing all error terms across batches
		// shape is [nSamples][MaxLength * VocabSize]
		accumulatedErrorTerms := make([][]float64, len(data))
		for i := range accumulatedErrorTerms {
			accumulatedErrorTerms[i] = make([]float64, dConfig.MaxLength*tConfig.VocabSize)
		}

		// Launch each batch
		for i := 0; i < len(data); i += batchSize {
			end := i + batchSize
			if end > len(data) {
				end = len(data)
			}
			batchData := data[i:end]
			batchIdx := i / batchSize

			if multithreading {
				wg.Add(1)
				sem <- struct{}{}
				go func(startIdx int, batch [][]int, idx int) {
					defer wg.Done()
					defer func() { <-sem }()

					// For each sample in the batch, pick a random t, add noise
					// Then build a [10][vocabSize] one-hot input
					// Then do forward pass, cross-entropy, and store deltas
					batchErrorTerms := make([][]float64, len(batch))
					batchLoss := 0.0
					for j, tokens := range batch {
						t := rand.Intn(dConfig.NumTimesteps)
						noisy := model.AddNoise(tokens, t)
						// Build a shape [10][vocabSize] input
						oneHotInput2D := makeOneHot2D(noisy, dConfig.MaxLength, tConfig.VocabSize)
						// Forward pass
						output2D := nn.ForwardTransformer(oneHotInput2D) // shape [1][10*vocabSize]
						logits := output2D[0]                            // length 10*vocabSize

						// Cross-entropy for each position
						errorTerm := make([]float64, dConfig.MaxLength*tConfig.VocabSize)
						for k := 0; k < dConfig.MaxLength; k++ {
							startPos := k * tConfig.VocabSize
							endPos := startPos + tConfig.VocabSize
							probs := paragon.Softmax(logits[startPos:endPos])
							target := tokens[k]
							batchLoss -= math.Log(math.Max(probs[target], 1e-10))
							// Build cross-entropy deltas
							for m := 0; m < tConfig.VocabSize; m++ {
								delta := probs[m]
								if m == target {
									delta -= 1.0
								}
								// optional gradient clip
								if delta > 5.0 {
									delta = 5.0
								} else if delta < -5.0 {
									delta = -5.0
								}
								errorTerm[startPos+m] = delta
							}
						}
						batchErrorTerms[j] = errorTerm
					}

					// Average loss for this batch
					batchLoss /= float64(len(batch) * dConfig.MaxLength)
					lossChan <- batchLoss

					// Return the error terms plus the batch index
					errorTermsChan <- struct {
						terms    [][]float64
						batchIdx int
					}{batchErrorTerms, idx}

				}(i, batchData, batchIdx)
			} else {
				// Single-threaded version
				batchErrorTerms := make([][]float64, len(batchData))
				batchLoss := 0.0
				for j, tokens := range batchData {
					t := rand.Intn(dConfig.NumTimesteps)
					noisy := model.AddNoise(tokens, t)
					oneHotInput2D := makeOneHot2D(noisy, dConfig.MaxLength, tConfig.VocabSize)
					output2D := nn.ForwardTransformer(oneHotInput2D)
					logits := output2D[0]

					errorTerm := make([]float64, dConfig.MaxLength*tConfig.VocabSize)
					for k := 0; k < dConfig.MaxLength; k++ {
						startPos := k * tConfig.VocabSize
						endPos := startPos + tConfig.VocabSize
						probs := paragon.Softmax(logits[startPos:endPos])
						target := tokens[k]
						batchLoss -= math.Log(math.Max(probs[target], 1e-10))
						for m := 0; m < tConfig.VocabSize; m++ {
							delta := probs[m]
							if m == target {
								delta -= 1.0
							}
							if delta > 5.0 {
								delta = 5.0
							} else if delta < -5.0 {
								delta = -5.0
							}
							errorTerm[startPos+m] = delta
						}
					}
					batchErrorTerms[j] = errorTerm
				}
				// Average
				batchLoss /= float64(len(batchData) * dConfig.MaxLength)
				totalLoss += batchLoss

				// Immediately do backprop for this batch:
				// We have to flatten batch dimension too if we want one big call,
				// but simpler is to sum them up ourselves. Or we can do a "mini-batch" approach.
				// For demonstration, let's just do single-sample updates or sum them:
				for _, et := range batchErrorTerms {
					//shaped := [][]float64{} // shape [10][vocabSize]
					// We'll call your new "BackwardExternal" that expects final-layer deltas
					// However, to do a single combined pass for the entire batch,
					// we’d sum up the error terms. For brevity, we can do them one by one:
					nn.BackwardExternal(reshapeError(et, dConfig.MaxLength, tConfig.VocabSize), lr)
				}
			}
		}

		// If multithreading, gather up partial results and do one big backprop
		if multithreading {
			go func() {
				wg.Wait()
				close(lossChan)
				close(errorTermsChan)
			}()

			for l := range lossChan {
				totalLoss += l
			}
			// Collect error terms
			for et := range errorTermsChan {
				start := et.batchIdx * batchSize
				for j, terms := range et.terms {
					if start+j < len(accumulatedErrorTerms) {
						accumulatedErrorTerms[start+j] = terms
					}
				}
			}

			// Now do one big backward pass across all samples
			// Summation of errors is typical for mini-batch gradient descent
			// But you can also call multiple times, or average them, etc.
			bigSum := make([]float64, dConfig.MaxLength*tConfig.VocabSize)
			for _, arr := range accumulatedErrorTerms {
				for i := range arr {
					bigSum[i] += arr[i]
				}
			}
			// Optionally average by #samples
			for i := range bigSum {
				bigSum[i] /= float64(len(data))
			}

			// Now do one BackwardExternal call
			shapedError := reshapeError(bigSum, dConfig.MaxLength, tConfig.VocabSize)
			nn.BackwardExternal(shapedError, lr)
		}

		totalLoss /= float64(numBatches)

		if epoch%10 == 0 {
			fmt.Printf("%s Epoch %d, Loss: %.4f, Time: %v\n",
				time.Now().String(), epoch, totalLoss, time.Since(startTime))
		}
		// Print a sample generation every 10 epochs
		if epoch%10 == 0 && epoch > 0 {
			fmt.Println("Generating text at epoch", epoch, "...")
			g := model.Generate()
			fmt.Println("Generated text:", g)
		}
	}

	fmt.Println("Final training complete!")
	fmt.Println("Generating text...")
	finalGen := model.Generate()
	fmt.Println("Final generated text:", finalGen)

	// Test a sample input of all [CLS]:
	sampleTokens := make([]int, tConfig.MaxLength)
	clsID := tokenizer.Vocab["[CLS]"]
	for i := 0; i < tConfig.MaxLength; i++ {
		sampleTokens[i] = clsID
	}
	sampleInput2D := makeOneHot2D(sampleTokens, tConfig.MaxLength, tConfig.VocabSize)
	output2D := nn.ForwardTransformer(sampleInput2D)
	logits := output2D[0]

	fmt.Println("Output layer values (raw logits) for the first row [position=0]:")
	firstRow := logits[0:tConfig.VocabSize]
	for x := 0; x < tConfig.VocabSize; x++ {
		fmt.Printf("%.4f ", firstRow[x])
	}
	fmt.Println()
}

// reshapeError turns a 1D slice of size (MaxLength*VocabSize) into a 2D shape [MaxLength][VocabSize]
func reshapeError(flat []float64, maxLength, vocabSize int) [][]float64 {
	output := make([][]float64, maxLength)
	for i := 0; i < maxLength; i++ {
		output[i] = flat[i*vocabSize : (i+1)*vocabSize]
	}
	return output
}
