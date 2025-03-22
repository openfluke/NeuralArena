package main

import (
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"

	"paragon"
)

// GrowDimensions61 grows all hidden layers by adding one neuron and recurses into sub-networks.
func GrowDimensions61(net *paragon.Network, currentDepth int, maxDepth int) {
	if currentDepth > maxDepth {
		return
	}
	// Add one neuron to each hidden layer (1 to OutputLayer-1)
	for layerIdx := 1; layerIdx < net.OutputLayer; layerIdx++ {
		net.AddNeuronsToLayer(layerIdx, 1)
	}
	// Recurse into sub-networks for all hidden layers
	for layerIdx := 1; layerIdx < net.OutputLayer; layerIdx++ {
		for y := 0; y < net.Layers[layerIdx].Height; y++ {
			for x := 0; x < net.Layers[layerIdx].Width; x++ {
				if net.Layers[layerIdx].Neurons[y][x].Dimension != nil {
					GrowDimensions61(net.Layers[layerIdx].Neurons[y][x].Dimension, currentDepth+1, maxDepth)
				}
			}
		}
	}
}

// Experiment61 runs an evolutionary search for NLP text generation using Forward and Backward.
func Experiment61(file *os.File) {
	fmt.Println("\n=== Experiment 61: Evolutionary NLP Text Generation with Forward/Backward ===")
	limitor := 200
	// **Step 1: Download and Prepare Text Data**
	fmt.Println("Downloading text data...")
	url := "http://www.gutenberg.org/files/1342/1342-0.txt" // Pride and Prejudice
	resp, err := http.Get(url)
	if err != nil {
		panic(fmt.Sprintf("Failed to download text: %v", err))
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		panic(fmt.Sprintf("Failed to read text: %v", err))
	}

	// Use all non-empty lines
	text := string(body)
	lines := strings.Split(text, "\n")
	var filteredLines []string
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			filteredLines = append(filteredLines, trimmed)
		}

		if i >= limitor {
			break
		}
	}

	fmt.Println(filteredLines)
	fmt.Printf("Loaded %d lines of text\n", len(filteredLines))

	// **Step 2: Tokenize the Text**
	tokenizer := paragon.NewCustomTokenizer(filteredLines)
	vocabSize := tokenizer.VocabSize
	maxLength := 10
	fmt.Printf("Vocabulary size: %d\n", vocabSize)

	// **Step 3: Set Up Feedforward Network**
	layerSizes := []struct{ Width, Height int }{
		{maxLength * vocabSize, 1}, // Input
		{128, 1},                   // Hidden 1
		{64, 1},                    // Hidden 2
		{maxLength * vocabSize, 1}, // Output
	}
	activations := []string{"linear", "relu", "relu", "softmax"}
	fullyConnected := []bool{true, true, true, true}
	initialNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)

	// **Step 4: Prepare Training Data**
	var sequences [][]int
	for _, line := range filteredLines {
		tokens := tokenizer.Encode(line)
		if len(tokens) > maxLength {
			tokens = tokens[:maxLength]
		} else {
			for len(tokens) < maxLength {
				tokens = append(tokens, tokenizer.Vocab["[PAD]"])
			}
		}
		sequences = append(sequences, tokens)

	}

	var inputs [][]float64
	var targets [][]float64
	for _, seq := range sequences {
		inputFlat := make([]float64, maxLength*vocabSize)
		targetFlat := make([]float64, maxLength*vocabSize)
		for t := 0; t < maxLength; t++ {
			if t < len(seq)-1 {
				inputFlat[t*vocabSize+seq[t]] = 1.0
				targetFlat[t*vocabSize+seq[t+1]] = 1.0
			} else {
				inputFlat[t*vocabSize+tokenizer.Vocab["[PAD]"]] = 1.0
				targetFlat[t*vocabSize+tokenizer.Vocab["[PAD]"]] = 1.0
			}
		}
		inputs = append(inputs, inputFlat)
		targets = append(targets, targetFlat)

	}

	// **Step 5: Evolution Setup**
	generations := 3
	mutationsPerCandidate := 20
	topSelection := 5
	maxDepth := 5
	epochs := 10
	learningRate := 0.0001

	// Train initial network
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		for i := range inputs {
			input2D := [][]float64{inputs[i]}
			target2D := [][]float64{targets[i]}
			initialNet.Forward(input2D)
			totalLoss += initialNet.ComputeLoss(target2D)
			initialNet.Backward(target2D, learningRate)
		}
		avgLoss := totalLoss / float64(len(inputs))
		fmt.Printf("Initial network, Epoch %d, Loss: %.4f\n", epoch, avgLoss)
	}
	baseADHDScore := ComputeADHDScore(initialNet, inputs, targets, vocabSize, maxLength)
	population := []Candidate{{net: initialNet, adhdScore: baseADHDScore}}
	fmt.Printf("Initial candidate ADHD score: %.4f\n", baseADHDScore)

	// Concurrency setup
	numThreads := int(float64(runtime.NumCPU()) * 0.8)
	if numThreads < 1 {
		numThreads = 1
	}
	sem := make(chan struct{}, numThreads)

	// **Step 6: Evolution Loop**
	for gen := 0; gen < generations; gen++ {
		fmt.Printf("Generation %d, population: %d candidates\n", gen, len(population))
		var wg sync.WaitGroup
		candidateCh := make(chan Candidate, mutationsPerCandidate*len(population))

		for _, cand := range population {
			for m := 0; m < mutationsPerCandidate; m++ {
				wg.Add(1)
				sem <- struct{}{}
				go func(parent *paragon.Network) {
					defer wg.Done()
					mutant := CloneNetwork(parent)
					epochsInside := 5
					GrowDimensions61(mutant, 1, maxDepth)

					/*for epoch := 0; epoch < epochsInside; epoch++ {
						for i := range inputs {
							input2D := [][]float64{inputs[i]}
							target2D := [][]float64{targets[i]}
							mutant.Forward(input2D)
							mutant.Backward(target2D, 0.00001)
						}
					}
					adhdScore := ComputeADHDScore(mutant, inputs, targets, vocabSize, maxLength)
					candidateCh <- Candidate{net: mutant, adhdScore: adhdScore}*/

					// Define the size of each chunk
					/*chunkSize := 25
					var adhdScores []float64

					// Training loop with chunks
					for epoch := 0; epoch < epochsInside; epoch++ {
						for start := 0; start < len(inputs); start += chunkSize {
							end := start + chunkSize
							if end > len(inputs) {
								end = len(inputs)
							}

							chunkInputs := inputs[start:end]
							chunkTargets := targets[start:end]

							// Train on this chunk
							for i := range chunkInputs {
								input2D := [][]float64{chunkInputs[i]}
								target2D := [][]float64{chunkTargets[i]}
								mutant.Forward(input2D)
								mutant.Backward(target2D, 0.00001)
							}

							// Compute ADHD score for this chunk after training
							chunkScore := ComputeADHDScore(mutant, chunkInputs, chunkTargets, vocabSize, maxLength)
							adhdScores = append(adhdScores, chunkScore)
						}
					}

					// Calculate the final ADHD score by averaging all chunk scores
					var totalScore float64
					for _, score := range adhdScores {
						totalScore += score
					}
					finalAdhdScore := totalScore / float64(len(adhdScores))
					candidateCh <- Candidate{net: mutant, adhdScore: finalAdhdScore}
					*/

					// Split data into training and validation sets
					/*trainSize := int(0.8 * float64(len(inputs)))
					trainInputs := inputs[:trainSize]
					trainTargets := targets[:trainSize]
					valInputs := inputs[trainSize:]
					valTargets := targets[trainSize:]

					// Define chunk size and declare adhdScores
					chunkSize := 25
					var adhdScores []float64

					// Training loop with chunks
					for epoch := 0; epoch < epochsInside; epoch++ {
						for start := 0; start < len(trainInputs); start += chunkSize {
							end := start + chunkSize
							if end > len(trainInputs) {
								end = len(trainInputs)
							}

							chunkInputs := trainInputs[start:end]
							chunkTargets := trainTargets[start:end]

							// Train on this chunk
							for i := range chunkInputs {
								input2D := [][]float64{chunkInputs[i]}
								target2D := [][]float64{chunkTargets[i]}
								mutant.Forward(input2D)
								mutant.Backward(target2D, 0.00001)
							}

							// Compute ADHD score for this chunk (on training data)
							chunkScore := ComputeADHDScore(mutant, chunkInputs, chunkTargets, vocabSize, maxLength)
							adhdScores = append(adhdScores, chunkScore)
							fmt.Printf("Epoch %d, Chunk %d-%d, Training ADHD Score: %.4f\n", epoch, start, end, chunkScore)
						}
					}

					// Evaluate on validation set
					adhdScore := ComputeADHDScore(mutant, valInputs, valTargets, vocabSize, maxLength)
					fmt.Printf("Final Validation ADHD Score: %.4f\n", adhdScore)
					candidateCh <- Candidate{net: mutant, adhdScore: adhdScore}
					*/

					chunkSize := 25
					var adhdScores []float64

					// Training loop with chunks
					for epoch := 0; epoch < epochsInside; epoch++ {
						for start := 0; start < len(inputs); start += chunkSize {
							end := start + chunkSize
							if end > len(inputs) {
								end = len(inputs)
							}

							chunkInputs := inputs[start:end]
							chunkTargets := targets[start:end]

							// Train on this chunk
							for i := range chunkInputs {
								input2D := [][]float64{chunkInputs[i]}
								target2D := [][]float64{chunkTargets[i]}
								mutant.Forward(input2D)
								mutant.Backward(target2D, learningRate)
							}

							// Compute ADHD score for this chunk
							chunkScore := ComputeADHDScore(mutant, chunkInputs, chunkTargets, vocabSize, maxLength)
							adhdScores = append(adhdScores, chunkScore)
						}
					}

					// Calculate total ADHD score as the sum of all chunk scores
					totalAdhdScore := 0.0
					for _, score := range adhdScores {
						totalAdhdScore += score
					}

					// Assign the total score to the candidate
					candidateCh <- Candidate{net: mutant, adhdScore: totalAdhdScore}

					<-sem
				}(cand.net)
			}
		}
		wg.Wait()
		close(candidateCh)

		var newCandidates []Candidate
		for cand := range candidateCh {
			newCandidates = append(newCandidates, cand)
		}
		allCandidates := append(population, newCandidates...)
		sort.Slice(allCandidates, func(i, j int) bool {
			return allCandidates[i].adhdScore > allCandidates[j].adhdScore // Higher ADHD score is better
		})
		if len(allCandidates) > topSelection {
			allCandidates = allCandidates[:topSelection]
		}
		population = allCandidates
		fmt.Printf("After generation %d, best ADHD score: %.4f\n", gen, population[0].adhdScore)
	}

	fmt.Printf("Initial candidate ADHD score: %.4f\n", baseADHDScore)
	best := population[0]
	fmt.Printf("Best candidate ADHD score: %.4f\n", best.adhdScore)

	// **Step 7: Generate Text**
	generatedText := generateText(best.net, tokenizer, maxLength, vocabSize)
	fmt.Println("Generated text:", generatedText)

	// Write results
	result := fmt.Sprintf("Experiment 61 Best Candidate ADHD Score: %.4f, Generated Text: %s\n", best.adhdScore, generatedText)
	if _, err := file.WriteString(result); err != nil {
		fmt.Printf("Error writing to file: %v\n", err)
	}
}

// computeAverageLoss evaluates the network over the dataset.
func computeAverageLoss(net *paragon.Network, inputs [][]float64, targets [][]float64) float64 {
	totalLoss := 0.0
	for i := range inputs {
		input2D := [][]float64{inputs[i]}
		target2D := [][]float64{targets[i]}
		net.Forward(input2D)
		totalLoss += net.ComputeLoss(target2D)
	}
	return totalLoss / float64(len(inputs))
}

// generateText generates a text sequence using the best network.
func generateText(net *paragon.Network, tokenizer *paragon.CustomTokenizer, maxLength int, vocabSize int) string {
	current := make([]int, maxLength)
	for i := range current {
		current[i] = tokenizer.Vocab["[PAD]"]
	}
	for t := 0; t < maxLength-1; t++ {
		inputFlat := make([]float64, maxLength*vocabSize)
		for i := 0; i <= t; i++ {
			inputFlat[i*vocabSize+current[i]] = 1.0
		}
		for i := t + 1; i < maxLength; i++ {
			inputFlat[i*vocabSize+tokenizer.Vocab["[PAD]"]] = 1.0
		}
		input2D := [][]float64{inputFlat}
		net.Forward(input2D)
		outputLayer := net.Layers[net.OutputLayer]
		outputNeurons := outputLayer.Neurons[0] // []*paragon.Neuron
		// Convert []*paragon.Neuron to []float64 by extracting Value fields
		outputValues := make([]float64, len(outputNeurons))
		for i, neuron := range outputNeurons {
			outputValues[i] = neuron.Value
		}
		start := t * vocabSize
		end := start + vocabSize
		// Ensure end doesn’t exceed slice bounds
		if end > len(outputValues) {
			end = len(outputValues)
		}
		probs := paragon.Softmax(outputValues[start:end])
		current[t+1] = paragon.ArgMax(probs)
	}
	return tokenizer.Decode(current)
}

func ComputeADHDScore(net *paragon.Network, inputs [][]float64, targets [][]float64, vocabSize int, maxLength int) float64 {
	// Initialize ADHD performance tracking
	net.Performance = paragon.NewADHDPerformance()

	// Iterate over each input-target pair
	for i := range inputs {
		// Forward pass with a single input
		input2D := [][]float64{inputs[i]}
		net.Forward(input2D)
		outputLayer := net.Layers[net.OutputLayer]
		outputValues := make([]float64, len(outputLayer.Neurons[0]))
		for x := 0; x < len(outputLayer.Neurons[0]); x++ {
			outputValues[x] = outputLayer.Neurons[0][x].Value
		}

		// Process each position in the sequence
		for t := 0; t < maxLength; t++ {
			start := t * vocabSize
			end := start + vocabSize
			if end > len(outputValues) {
				break // Avoid out-of-bounds access
			}

			// Get probabilities and predicted token
			probs := Softmax(outputValues[start:end])
			predictedToken := float64(ArgMax(probs))

			// Get expected token from target
			expectedToken := float64(ArgMax(targets[i][start:end]))

			// Evaluate prediction using ADHD
			result := net.EvaluatePrediction(expectedToken, predictedToken)
			net.UpdateADHDPerformance(result)
		}
	}

	// Compute and return the final ADHD score
	net.Performance.Score = net.ComputeFinalScore()
	return net.Performance.Score
}

// Assuming Softmax and ArgMax are defined elsewhere; if not, here they are:
func Softmax(values []float64) []float64 {
	maxVal := values[0]
	for _, v := range values {
		if v > maxVal {
			maxVal = v
		}
	}
	expSum := 0.0
	result := make([]float64, len(values))
	for i, v := range values {
		result[i] = math.Exp(v - maxVal)
		expSum += result[i]
	}
	for i := range result {
		result[i] /= expSum
	}
	return result
}

func ArgMax(values []float64) int {
	maxIdx := 0
	for i := 1; i < len(values); i++ {
		if values[i] > values[maxIdx] {
			maxIdx = i
		}
	}
	return maxIdx
}
