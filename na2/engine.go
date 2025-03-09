package main

import (
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"paragon"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
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

// downloadText downloads text from a URL or reads it from a local file if available.
func downloadText(url string) ([]string, error) {
	dir := "data"
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create data directory: %v", err)
	}
	hash := md5.Sum([]byte(url))
	filename := hex.EncodeToString(hash[:])
	filepath := filepath.Join(dir, filename)

	if _, err := os.Stat(filepath); err == nil {
		content, err := os.ReadFile(filepath)
		if err != nil {
			return nil, fmt.Errorf("failed to read local file: %v", err)
		}
		text := string(content)
		return splitIntoSentences(text), nil
	}

	resp, err := http.Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to download text: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %v", err)
	}

	if err := os.WriteFile(filepath, body, 0644); err != nil {
		return nil, fmt.Errorf("failed to save file: %v", err)
	}

	text := string(body)
	return splitIntoSentences(text), nil
}

// splitIntoSentences splits text into sentences and filters out empty ones.
func splitIntoSentences(text string) []string {
	sentences := strings.Split(text, ".")
	var filtered []string
	for _, s := range sentences {
		trimmed := strings.TrimSpace(s)
		if trimmed != "" {
			filtered = append(filtered, trimmed)
		}
	}
	return filtered
}

func main() {
	fmt.Println("V5-IMPLEMENTATION-NA1-DIFFUSION-TRANSFORMER-MASKED")

	url := "https://www.gutenberg.org/files/28/28-0.txt" // Pride and Prejudice as an example
	additionalSentences, err := downloadText(url)
	if err != nil {
		fmt.Printf("Error downloading text: %v\n", err)
		return
	}
	fmt.Printf("Downloaded %d additional sentences\n", len(additionalSentences))

	// Limit to 100 sentences
	if len(additionalSentences) > 100 {
		additionalSentences = additionalSentences[:100]
	}

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
	sentences = append(sentences, generateSentences(477)...) // Total 500 sentences
	sentences = append(sentences, additionalSentences...)

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
		NumTimesteps: 20, // Used in GenerateMasked for discrete steps
		MaxLength:    10,
		LearningRate: 0.002,
		Epochs:       2000,
		Temperature:  0.4,
		TopK:         3,
	}

	model := paragon.NewDiffusionModel(nn, dConfig, sentences)

	batchSize := 10
	multithreading := true
	cpuPercent := 0.8

	numCores := runtime.NumCPU()
	numThreads := int(float64(numCores) * cpuPercent)
	if numThreads < 1 {
		numThreads = 1
	}
	numBatches := (len(sentences) + batchSize - 1) / batchSize
	fmt.Printf("Using %d threads (%d%% of %d cores), %d batches\n", numThreads, int(cpuPercent*100), numCores, numBatches)

	fmt.Println("Starting training...")
	for epoch := 0; epoch < dConfig.Epochs; epoch++ {
		startTime := time.Now()
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
		var wg sync.WaitGroup
		lossChan := make(chan float64, numBatches)
		errorTermsChan := make(chan struct {
			terms    [][]float64
			batchIdx int
		}, numBatches)
		sem := make(chan struct{}, numThreads)

		accumulatedErrorTerms := make([][]float64, len(sentences))
		for i := range accumulatedErrorTerms {
			accumulatedErrorTerms[i] = make([]float64, dConfig.MaxLength*tConfig.VocabSize)
		}

		for i := 0; i < len(sentences); i += batchSize {
			end := i + batchSize
			if end > len(sentences) {
				end = len(sentences)
			}
			batchData := data[i:end]
			batchIdx := i / batchSize

			if multithreading {
				wg.Add(1)
				sem <- struct{}{}
				go func(startIdx int, batch [][]int, idx int) {
					defer wg.Done()
					defer func() { <-sem }()

					// Prepare one-hot encoded inputs for the entire batch
					batchInputs := make([][]float64, dConfig.MaxLength)
					for k := 0; k < dConfig.MaxLength; k++ {
						batchInputs[k] = make([]float64, tConfig.VocabSize)
					}
					batchTargets := make([][]int, len(batch))
					noisyBatch := make([][]int, len(batch))
					for j, tokens := range batch {
						t := rand.Float64() // Sample t ~ U[0,1]
						noisyTokens := model.AddNoiseMasked(tokens, t)
						noisyBatch[j] = noisyTokens
						for k, tok := range noisyTokens {
							if tok >= 0 && tok < tConfig.VocabSize {
								batchInputs[k][tok] = 1.0
							}
						}
						batchTargets[j] = tokens
					}
					batchOutputs := make([][][]float64, len(batch))
					for j := range batch {
						singleInput := batchInputs // Shape: [MaxLength][VocabSize]
						batchOutputs[j] = nn.ForwardTransformer(singleInput)
					}

					loss := 0.0
					batchErrorTerms := make([][]float64, len(batch))
					for j := 0; j < len(batch); j++ {
						batchErrorTerms[j] = make([]float64, dConfig.MaxLength*tConfig.VocabSize)
						numMasked := 0
						for k := 0; k < dConfig.MaxLength; k++ {
							if noisyBatch[j][k] == tokenizer.Vocab["[MASK]"] {
								numMasked++
								start := k * tConfig.VocabSize
								end := (k + 1) * tConfig.VocabSize
								probs := paragon.Softmax(batchOutputs[j][0][start:end])
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
									batchErrorTerms[j][start+m] = delta
								}
							} else {
								// Zero out error terms for non-masked positions
								for m := 0; m < tConfig.VocabSize; m++ {
									batchErrorTerms[j][k*tConfig.VocabSize+m] = 0
								}
							}
						}
						if numMasked > 0 {
							loss /= float64(numMasked)
						}
					}
					lossChan <- loss / float64(len(batch))
					errorTermsChan <- struct {
						terms    [][]float64
						batchIdx int
					}{batchErrorTerms, idx}
				}(i, batchData, batchIdx)
			} else {
				// Single-threaded version
				batchInputs := make([][]float64, dConfig.MaxLength)
				for k := 0; k < dConfig.MaxLength; k++ {
					batchInputs[k] = make([]float64, tConfig.VocabSize)
				}
				batchTargets := make([][]int, len(batchData))
				noisyBatch := make([][]int, len(batchData))
				for j, tokens := range batchData {
					t := rand.Float64()
					noisyTokens := model.AddNoiseMasked(tokens, t)
					noisyBatch[j] = noisyTokens
					for k, tok := range noisyTokens {
						if tok >= 0 && tok < tConfig.VocabSize {
							batchInputs[k][tok] = 1.0
						}
					}
					batchTargets[j] = tokens
				}
				batchOutputs := make([][][]float64, len(batchData))
				for j := range batchData {
					singleInput := batchInputs // Shape: [MaxLength][VocabSize]
					batchOutputs[j] = nn.ForwardTransformer(singleInput)
				}
				loss := 0.0
				errorTerms := make([][]float64, len(batchData))
				for j := 0; j < len(batchData); j++ {
					errorTerms[j] = make([]float64, dConfig.MaxLength*tConfig.VocabSize)
					numMasked := 0
					for k := 0; k < dConfig.MaxLength; k++ {
						if noisyBatch[j][k] == tokenizer.Vocab["[MASK]"] {
							numMasked++
							start := k * tConfig.VocabSize
							end := (k + 1) * tConfig.VocabSize
							probs := paragon.Softmax(batchOutputs[j][0][start:end])
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
								errorTerms[j][start+m] = delta
							}
						} else {
							for m := 0; m < tConfig.VocabSize; m++ {
								errorTerms[j][k*tConfig.VocabSize+m] = 0
							}
						}
					}
					if numMasked > 0 {
						loss /= float64(numMasked)
					}
				}
				totalLoss += loss / float64(len(batchData))
				nn.Backward(errorTerms, lr)
			}
		}

		if multithreading {
			go func() {
				wg.Wait()
				close(lossChan)
				close(errorTermsChan)
			}()

			for l := range lossChan {
				totalLoss += l
			}
			for et := range errorTermsChan {
				start := et.batchIdx * batchSize
				for j, terms := range et.terms {
					if start+j < len(accumulatedErrorTerms) {
						accumulatedErrorTerms[start+j] = terms
					}
				}
			}

			nn.Backward(accumulatedErrorTerms, lr)
		}

		totalLoss /= float64(numBatches)
		if epoch%10 == 0 {
			fmt.Printf("%s Epoch %d, Loss: %.4f, Time: %v\n", time.Now().String(), epoch, totalLoss, time.Since(startTime))
		}
		if epoch%10 == 0 && epoch > 0 {
			fmt.Println("Generating text at epoch", epoch, "...")
			generated := model.GenerateMasked()
			fmt.Println("Generated text:", generated)
		}
	}

	fmt.Println("Final training complete!")
	fmt.Println("Generating text...")
	generated := model.GenerateMasked()
	fmt.Println("Final generated text:", generated)

	// Sample input with one-hot encoding
	sampleInput := make([][]float64, tConfig.MaxLength)
	for k := 0; k < tConfig.MaxLength; k++ {
		sampleInput[k] = make([]float64, tConfig.VocabSize)
		tok := model.Tokenizer.Vocab["[CLS]"]
		if tok >= 0 && tok < tConfig.VocabSize {
			sampleInput[k][tok] = 1.0
		}
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
