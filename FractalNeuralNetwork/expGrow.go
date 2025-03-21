package main

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"os"
	"runtime"
	"sort"
	"sync"

	"paragon"
)

// Candidate holds a network variant along with its validation accuracy.
type Candidate struct {
	net *paragon.Network
	acc float64
}

// CloneNetwork creates a deep clone of a network by using gob encoding/decoding.
func CloneNetwork(original *paragon.Network) *paragon.Network {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	// Encode the network into the buffer.
	if err := enc.Encode(original); err != nil {
		panic(fmt.Sprintf("gob encode failed: %v", err))
	}
	dec := gob.NewDecoder(&buf)
	var clone paragon.Network
	if err := dec.Decode(&clone); err != nil {
		panic(fmt.Sprintf("gob decode failed: %v", err))
	}
	return &clone
}

// GrowDimensions recursively grows the candidate network by appending one neuron
// to layer 1 and then recursing into each neuron's sub-network (Dimension) up to maxDepth.
func OldGrowDimensions(net *paragon.Network, currentDepth int, maxDepth int) {
	if currentDepth > maxDepth {
		return
	}
	// Append one neuron to layer 1.
	net.AddNeuronsToLayer(1, 1)

	// For each neuron in layer 1, if it has a sub-network attached, grow it recursively.
	for y := 0; y < net.Layers[1].Height; y++ {
		for x := 0; x < net.Layers[1].Width; x++ {
			if net.Layers[1].Neurons[y][x].Dimension != nil {
				GrowDimensions(net.Layers[1].Neurons[y][x].Dimension, currentDepth+1, maxDepth)
			}
		}
	}
}

// GrowDimensions grows all hidden layers by adding one neuron and recurses into sub-networks.
func GrowDimensions(net *paragon.Network, currentDepth int, maxDepth int) {
	if currentDepth > maxDepth {
		return
	}
	// Add one neuron to each hidden layer (1, 2, 3)
	for layerIdx := 1; layerIdx < net.OutputLayer; layerIdx++ {
		net.AddNeuronsToLayer(layerIdx, 1)
	}
	// Recurse into sub-networks for all hidden layers
	for layerIdx := 1; layerIdx < net.OutputLayer; layerIdx++ {
		for y := 0; y < net.Layers[layerIdx].Height; y++ {
			for x := 0; x < net.Layers[layerIdx].Width; x++ {
				if net.Layers[layerIdx].Neurons[y][x].Dimension != nil {
					GrowDimensions(net.Layers[layerIdx].Neurons[y][x].Dimension, currentDepth+1, maxDepth)
				}
			}
		}
	}
}

// Experiment58 runs an evolutionary search that grows the network gradually.
func Experiment58(file *os.File) {
	fmt.Println("\n=== Experiment 58: Evolutionary Network Growth up to 10th Dimension ===")

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
	trainInputs, trainTargets := generateHierarchicalXOR(10000)
	valInputs, valTargets := generateHierarchicalXOR(2000)

	// Evolution parameters.
	generations := 5
	mutationsPerCandidate := 40 // number of mutants per candidate
	topSelection := 10          // keep top 10 candidates each generation
	maxDepth := 10
	// Start with a single basic candidate.
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

		// For each candidate in the current population, generate mutants.
		for _, cand := range population {
			for m := 0; m < mutationsPerCandidate; m++ {
				wg.Add(1)
				sem <- struct{}{}
				go func(parent *paragon.Network) {
					defer wg.Done()
					// Clone parent's network.
					mutant := CloneNetwork(parent)
					// Grow the mutant’s network by appending one neuron at layer 1 and recursively in its sub-networks.
					GrowDimensions(mutant, 1, maxDepth)
					/*if rand.Float64() < 0.5 {
						mutant.AddNeuronsToLayer(1, 1)
					} else {
						mutant.AddLayer(2, 8, 1, "relu", true)
					}*/
					// Train mutant for 5 epochs.
					tr := paragon.Trainer{Network: mutant, Config: paragon.TrainConfig{Epochs: 5, LearningRate: 0.0000001}}
					tr.TrainSimple(trainInputs, trainTargets, 5)
					acc := paragon.ComputeAccuracy(mutant, valInputs, valTargets)
					candidateCh <- Candidate{net: mutant, acc: acc}
					<-sem
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
		// Combine previous population with new candidates.
		allCandidates := append(population, newCandidates...)
		// Sort candidates by descending validation accuracy.
		sort.Slice(allCandidates, func(i, j int) bool {
			return allCandidates[i].acc > allCandidates[j].acc
		})
		// Keep only the top candidates.
		if len(allCandidates) > topSelection {
			allCandidates = allCandidates[:topSelection]
		}
		population = allCandidates
		fmt.Printf("After generation %d, best accuracy: %.2f%%\n", gen, population[0].acc*100)
	}

	fmt.Printf("Initial candidate accuracy: %.2f%%\n", baseAcc*100)

	// The best candidate is the first element.
	best := population[0]
	result := fmt.Sprintf("Experiment 58 Best Candidate Accuracy: %.2f%%\n", best.acc*100)
	fmt.Print(result)
	if _, err := file.WriteString(result); err != nil {
		fmt.Printf("Error writing to file: %v\n", err)
	}
}
