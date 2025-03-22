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

// Experiment62 runs an evolutionary search with extensive dimensional sub-network experimentation.
func Experiment62(file *os.File) {
	fmt.Println("\n=== Experiment 62: Evolutionary Dimensional Sub-Network Exploration ===")

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

	// Validate input data dimensions
	if len(trainInputs) > 0 && len(trainInputs[0]) != 1 || len(trainInputs[0][0]) != 16 {
		panic(fmt.Sprintf("trainInputs mismatch: expected [N][1][16], got [%d][%d][%d]",
			len(trainInputs), len(trainInputs[0]), len(trainInputs[0][0])))
	}

	// Evolution parameters.
	generations := 3
	mutationsPerCandidate := 3
	topSelection := 10
	maxSubDepth := 4

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
					mutant := CloneNetwork(parent)

					// Experiment with dimensional sub-networks across all hidden layers.
					for layerIdx := 1; layerIdx < mutant.OutputLayer; layerIdx++ {
						if rand.Float64() < 0.7 { // 70% chance to modify this layer
							// Random sub-network configuration
							numSubLayers := rand.Intn(maxSubDepth-1) + 2 // 2 to maxSubDepth layers
							subLayerSizes := make([]struct{ Width, Height int }, numSubLayers)
							subActivations := make([]string, numSubLayers)
							subFullyConnected := make([]bool, numSubLayers)

							// Input layer: fixed at 1 to accept neuron sum
							subLayerSizes[0] = struct{ Width, Height int }{1, 1} // Fixed at 1
							subActivations[0] = "relu"
							subFullyConnected[0] = true

							// Hidden layers: variable width
							for i := 1; i < numSubLayers-1; i++ {
								width := rand.Intn(64-8+1) + 8 // 8-64 neurons
								subLayerSizes[i] = struct{ Width, Height int }{width, 1}
								subActivations[i] = randomActivation()
								subFullyConnected[i] = rand.Float64() > 0.5
							}

							// Output layer: single neuron
							subLayerSizes[numSubLayers-1] = struct{ Width, Height int }{1, 1}
							subActivations[numSubLayers-1] = "linear"
							subFullyConnected[numSubLayers-1] = true

							// Randomly choose shared or unique sub-networks
							shared := rand.Float64() < 0.3
							opts := paragon.SetLayerDimensionOptions{
								Shared:     shared,
								InitMethod: randomInitMethod(),
							}

							// Apply the sub-network
							mutant.SetLayerDimension(layerIdx, subLayerSizes, subActivations, subFullyConnected, opts)
						}
					}

					// Add neurons to a random hidden layer with 20% probability
					if rand.Float64() < 0.2 {
						layerIdx := rand.Intn(mutant.OutputLayer-1) + 1 // Hidden layers (1, 2, 3)
						numToAdd := rand.Intn(5) + 1                    // Add 1-5 neurons
						mutant.AddNeuronsToLayer(layerIdx, numToAdd)
					}

					// Randomly tweak training parameters
					lr := 0.00002 * (0.5 + rand.Float64()) // 0.001 to 0.003
					epochs := rand.Intn(6) + 3             // 3 to 8 epochs
					tr := paragon.Trainer{
						Network: mutant,
						Config:  paragon.TrainConfig{Epochs: epochs, LearningRate: lr},
					}
					tr.TrainSimple(trainInputs, trainTargets, epochs)

					// Evaluate with accuracy and ADHD score
					acc := paragon.ComputeAccuracy(mutant, valInputs, valTargets)
					mutant.EvaluateModel(trainTargetsToFloat(trainTargets), trainTargetsToFloat(trainTargets))
					adhdScore := mutant.Performance.Score

					/*correct := 0
					expectedOutputs := trainTargetsToFloat(valTargets)
					actualOutputs := make([]float64, len(valInputs))
					for i, input := range valInputs {
						mutant.Forward(input)
						outputValues := make([]float64, mutant.Layers[mutant.OutputLayer].Width)
						for x := 0; x < mutant.Layers[mutant.OutputLayer].Width; x++ {
							outputValues[x] = mutant.Layers[mutant.OutputLayer].Neurons[0][x].Value
						}
						pred := paragon.ArgMax(outputValues)
						actualOutputs[i] = float64(pred)
						if int(pred) == int(expectedOutputs[i]) {
							correct++
						}
					}
					acc := float64(correct) / float64(len(valInputs))
					mutant.EvaluateModel(expectedOutputs, actualOutputs)
					adhdScore := mutant.Performance.Score*/

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

		// Combine and sort by a combined metric
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
	result := fmt.Sprintf("Experiment 62 Best Candidate Accuracy: %.2f%%, ADHD Score: %.2f\n",
		best.acc*100, best.adhdScore)
	fmt.Print(result)
	if _, err := file.WriteString(result); err != nil {
		fmt.Printf("Error writing to file: %v\n", err)
	}
}

// randomActivation returns a randomly chosen activation function.
func randomActivation() string {
	activations := []string{"relu", "leaky_relu", "tanh", "sigmoid"}
	return activations[rand.Intn(len(activations))]
}

// randomInitMethod returns a randomly chosen weight initialization method.
func randomInitMethod() string {
	methods := []string{"xavier", "he"}
	return methods[rand.Intn(len(methods))]
}

// trainTargetsToFloat converts targets to a flat float64 slice for ADHD evaluation.
func trainTargetsToFloat(targets [][][]float64) []float64 {
	result := make([]float64, len(targets))
	for i := range targets {
		result[i] = float64(paragon.ArgMax(targets[i][0]))
	}
	return result
}
