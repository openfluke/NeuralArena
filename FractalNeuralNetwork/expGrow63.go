package main

import (
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"sync"

	"paragon"
)

// Experiment63 runs an evolutionary search with extensive dimensional sub-network experimentation.
func Experiment63(file *os.File) {
	fmt.Println("\n=== Experiment 63: Evolutionary Dimensional Sub-Network Exploration ===")

	// Define the main network architecture (for Hierarchical XOR).
	layerSizes := []struct{ Width, Height int }{
		{16, 1}, // Input
		{12, 1}, // Hidden 1
		{6, 1},  // Hidden 2
		{4, 1},  // Hidden 3
		{2, 1},  // Output
	}
	activations := []string{"linear", "relu", "relu", "relu", "softmax"}
	fullyConnected := []bool{true, true, true, true, true}

	// Generate dataset for Hierarchical XOR.
	trainInputs, trainTargets := generateHierarchicalXOR(1000)
	valInputs, valTargets := generateHierarchicalXOR(200)

	// Validate input data dimensions.
	if len(trainInputs) > 0 && len(trainInputs[0]) != 1 || len(trainInputs[0][0]) != 16 {
		panic(fmt.Sprintf("trainInputs mismatch: expected [N][1][16], got [%d][%d][%d]",
			len(trainInputs), len(trainInputs[0]), len(trainInputs[0][0])))
	}

	// Evolution parameters.
	generations := 3
	mutationsPerCandidate := 3
	topSelection := 10

	// Start with a basic candidate.
	initialNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	trainer := paragon.Trainer{Network: initialNet, Config: paragon.TrainConfig{Epochs: 5, LearningRate: 0.002}}
	trainer.TrainSimple(trainInputs, trainTargets, 5)
	baseAcc := paragon.ComputeAccuracy(initialNet, valInputs, valTargets)
	population := []Candidate{{net: initialNet, acc: baseAcc}}

	// Limit concurrency to 80% of available CPU cores.
	numThreads := int(float64(runtime.NumCPU()) * 0.8)
	if numThreads < 1 {
		numThreads = 1
	}
	sem := make(chan struct{}, numThreads)

	// Evolution loop.
	for gen := 0; gen < generations; gen++ {
		fmt.Printf("Generation %d, population: %d candidates\n", gen, len(population))
		var wg sync.WaitGroup
		candidateCh := make(chan Candidate, mutationsPerCandidate*len(population))

		// Generate mutants for each candidate.
		for _, cand := range population {
			for m := 0; m < mutationsPerCandidate; m++ {
				wg.Add(1)
				sem <- struct{}{}
				go func(parent *paragon.Network) {
					defer wg.Done()
					defer func() { <-sem }()

					// Clone parent's network.
					mutant := CloneNetwork(parent) // Assuming CloneNetwork is in paragon package

					// Experiment with dimensional sub-networks across all hidden layers.
					for layerIdx := 1; layerIdx < mutant.OutputLayer; layerIdx++ {
						if rand.Float64() < 0.7 { // 70% chance to add a sub-network
							// Create a sub-network with potential nested sub-networks
							subNet := createSubNetwork(0, 3, 0.5) // Max depth 10, initial prob 0.5
							if subNet != nil {
								// Assign the sub-network to all neurons in this layer (shared)
								for y := 0; y < mutant.Layers[layerIdx].Height; y++ {
									for x := 0; x < mutant.Layers[layerIdx].Width; x++ {
										mutant.Layers[layerIdx].Neurons[y][x].Dimension = subNet
									}
								}
							}
						}
					}

					// Add neurons to a random hidden layer with 20% probability.
					if rand.Float64() < 0.2 {
						layerIdx := rand.Intn(mutant.OutputLayer-1) + 1 // Hidden layers (1, 2, 3)
						numToAdd := rand.Intn(5) + 1                    // Add 1-5 neurons
						mutant.AddNeuronsToLayer(layerIdx, numToAdd)
					}

					// Randomly tweak training parameters.
					lr := 0.00002 * (0.5 + rand.Float64()) // 0.001 to 0.003
					epochs := rand.Intn(6) + 3             // 3 to 8 epochs
					tr := paragon.Trainer{
						Network: mutant,
						Config:  paragon.TrainConfig{Epochs: epochs, LearningRate: lr},
					}
					tr.TrainSimple(trainInputs, trainTargets, epochs)

					// Evaluate with accuracy and ADHD score.
					acc := paragon.ComputeAccuracy(mutant, valInputs, valTargets)
					mutant.EvaluateModel(trainTargetsToFloat(trainTargets), trainTargetsToFloat(trainTargets))
					adhdScore := mutant.Performance.Score

					candidateCh <- Candidate{net: mutant, acc: acc, adhdScore: adhdScore}
				}(cand.net)
			}
		}
		wg.Wait()
		close(candidateCh)

		// Gather new candidates.
		var newCandidates []Candidate
		for cand := range candidateCh {
			newCandidates = append(newCandidates, cand)
		}

		// Combine and sort by a combined metric.
		allCandidates := append(population, newCandidates...)
		sort.Slice(allCandidates, func(i, j int) bool {
			scoreI := 0.7*allCandidates[i].acc + 0.3*(allCandidates[i].adhdScore/100)
			scoreJ := 0.7*allCandidates[j].acc + 0.3*(allCandidates[j].adhdScore/100)
			return scoreI > scoreJ
		})

		// Keep top candidates.
		if len(allCandidates) > topSelection {
			allCandidates = allCandidates[:topSelection]
		}
		population = allCandidates
		fmt.Printf("After generation %d, best accuracy: %.2f%%, best ADHD score: %.2f\n",
			gen, population[0].acc*100, population[0].adhdScore)
	}

	fmt.Printf("Initial candidate accuracy: %.2f%%\n", baseAcc*100)

	// Best candidate result.
	best := population[0]
	result := fmt.Sprintf("Experiment 63 Best Candidate Accuracy: %.2f%%, ADHD Score: %.2f\n",
		best.acc*100, best.adhdScore)
	fmt.Print(result)
	if _, err := file.WriteString(result); err != nil {
		fmt.Printf("Error writing to file: %v\n", err)
	}
}

// createSubNetwork generates a sub-network with potential nested sub-networks up to a specified depth.
func createSubNetwork(depth int, maxDepth int, prob float64) *paragon.Network {
	// Base case: stop recursion if maximum depth is reached.
	if depth >= maxDepth {
		return nil
	}

	// Define sub-network structure: 2 to 4 layers (input, 1-3 hidden, output).
	numLayers := rand.Intn(3) + 2 // 2 to 4 total layers
	layerSizes := make([]struct{ Width, Height int }, numLayers)
	activations := make([]string, numLayers)
	fullyConnected := make([]bool, numLayers)

	// Input layer: scalar input.
	layerSizes[0] = struct{ Width, Height int }{1, 1}
	activations[0] = "relu" // Fixed for simplicity; could be randomized.
	fullyConnected[0] = true

	// Hidden layers: width between 4 and 16.
	for i := 1; i < numLayers-1; i++ {
		width := rand.Intn(13) + 4 // 4 to 16 neurons
		layerSizes[i] = struct{ Width, Height int }{width, 1}
		activations[i] = randomActivation()
		fullyConnected[i] = rand.Float64() > 0.5 // Randomly fully connected or not.
	}

	// Output layer: scalar output.
	layerSizes[numLayers-1] = struct{ Width, Height int }{1, 1}
	activations[numLayers-1] = "linear" // Output is linear before parent neuron’s activation.
	fullyConnected[numLayers-1] = true

	// Create the sub-network.
	subNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	subNet.InitializeWeights("xavier")

	// Recursively add sub-sub-networks to hidden layers.
	for layerIdx := 1; layerIdx < subNet.OutputLayer; layerIdx++ {
		if rand.Float64() < prob {
			subSubNet := createSubNetwork(depth+1, maxDepth, prob*0.8) // Decay probability.
			if subSubNet != nil {
				// Assign the same sub-sub-network to all neurons in this layer (shared).
				for y := 0; y < subNet.Layers[layerIdx].Height; y++ {
					for x := 0; x < subNet.Layers[layerIdx].Width; x++ {
						subNet.Layers[layerIdx].Neurons[y][x].Dimension = subSubNet
					}
				}
			}
		}
	}

	return subNet
}
