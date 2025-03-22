// experiment.go
package main

import (
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"time"

	"paragon"
)

func Experiment1(filename string) {
	rand.Seed(time.Now().UnixNano())

	numSamples := 200
	inputHeight, inputWidth := 4, 4
	numClasses := 10
	trainInputs := make([][][]float64, numSamples)
	trainTargets := make([][][]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		trainInputs[i] = make([][]float64, inputHeight)
		trainTargets[i] = make([][]float64, 1)
		trainTargets[i][0] = make([]float64, numClasses)
		label := rand.Intn(numClasses)
		trainTargets[i][0][label] = 1.0
		for y := 0; y < inputHeight; y++ {
			trainInputs[i][y] = make([]float64, inputWidth)
			for x := 0; x < inputWidth; x++ {
				trainInputs[i][y][x] = float64(label)*0.8 + rand.Float64()*0.4 - 0.2
			}
		}
	}

	// Baseline Network
	baselineSizes := []struct{ Width, Height int }{
		{inputWidth, inputHeight}, // Input: 4x4
		{8, 1},                    // Hidden: 8 neurons
		{numClasses, 1},           // Output: 10 classes
	}
	baselineActivations := []string{"linear", "relu", "softmax"}
	baselineFullyConnected := []bool{true, true, true}
	baselineNet := paragon.NewNetwork(baselineSizes, baselineActivations, baselineFullyConnected)
	baselineTrainer := paragon.Trainer{
		Network: baselineNet,
		Config: paragon.TrainConfig{
			Epochs:       150,
			LearningRate: 0.01,
			Debug:        false,
		},
	}
	fmt.Println("Training Baseline Network...")
	baselineTrainer.TrainSimple(trainInputs, trainTargets, 150)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, trainInputs, trainTargets)

	// Network with Sub-Network Dimension
	dimNet := paragon.NewNetwork(baselineSizes, baselineActivations, baselineFullyConnected)
	subSizes := []struct{ Width, Height int }{
		{8, 1},  // Input matches hidden layer size
		{16, 1}, // Hidden layer in sub-network
		{8, 1},  // Output matches hidden layer size
	}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}
	subNet := paragon.NewNetwork(subSizes, subActivations, subFullyConnected)
	dimNet.SetLayerDimension(1, subNet)
	dimTrainer := paragon.Trainer{
		Network: dimNet,
		Config: paragon.TrainConfig{
			Epochs:       150,
			LearningRate: 0.01,
			Debug:        false,
		},
	}
	fmt.Println("Training Network with Sub-Network Dimension...")
	dimTrainer.TrainSimple(trainInputs, trainTargets, 150)
	dimAcc := paragon.ComputeAccuracy(dimNet, trainInputs, trainTargets)

	// Append results to file
	f, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Printf("Failed to open file %s: %v\n", filename, err)
		return
	}
	defer f.Close()
	result := fmt.Sprintf("Experiment 1 - %s\nBaseline Accuracy: %.2f%%\nDimension Accuracy: %.2f%%\nScale: %.4f\n\n",
		time.Now().Format("2006-01-02 15:04:05"), baselineAcc*100, dimAcc*100, dimNet.Layers[1].Dimension.Scale)
	if _, err := f.WriteString(result); err != nil {
		fmt.Printf("Failed to write to file %s: %v\n", filename, err)
	} else {
		fmt.Printf("Results appended to %s\n", filename)
	}
}

// Experiment2 compares baseline, single-dimension, and multithreaded multi-dimension networks
func Experiment2(filename string) {
	// Generate dataset
	trainInputs, trainTargets := GenerateXORGridDataset("xor_grid_dataset.txt", 500)

	// Baseline Network
	baselineSizes := []struct{ Width, Height int }{
		{4, 4},  // Input: 4x4
		{8, 1},  // Hidden: 8 neurons
		{10, 1}, // Output: 10 classes
	}
	baselineActivations := []string{"linear", "relu", "softmax"}
	baselineFullyConnected := []bool{true, true, true}
	baselineNet := paragon.NewNetwork(baselineSizes, baselineActivations, baselineFullyConnected)
	baselineTrainer := paragon.Trainer{
		Network: baselineNet,
		Config: paragon.TrainConfig{
			Epochs:       150,
			LearningRate: 0.005,
			Debug:        false,
		},
	}
	fmt.Println("Training Baseline Network...")
	baselineTrainer.TrainSimple(trainInputs, trainTargets, 150)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, trainInputs, trainTargets)

	// Single Dimension Network
	dim1Net := paragon.NewNetwork(baselineSizes, baselineActivations, baselineFullyConnected)
	subSizes := []struct{ Width, Height int }{
		{8, 1},  // Input matches hidden layer
		{16, 1}, // Hidden layer
		{8, 1},  // Output matches hidden layer
	}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}
	subNet := paragon.NewNetwork(subSizes, subActivations, subFullyConnected)
	dim1Net.SetLayerDimension(1, subNet)
	dim1Trainer := paragon.Trainer{
		Network: dim1Net,
		Config: paragon.TrainConfig{
			Epochs:       150,
			LearningRate: 0.005,
			Debug:        false,
		},
	}
	fmt.Println("Training Single Dimension Network...")
	dim1Trainer.TrainSimple(trainInputs, trainTargets, 150)
	dim1Acc := paragon.ComputeAccuracy(dim1Net, trainInputs, trainTargets)

	// Multithreaded Multi-Dimension Network
	numCores := runtime.NumCPU()
	numThreads := int(float64(numCores) * 0.8) // 80% CPU usage
	fmt.Printf("Using %d threads (80%% of %d cores)\n", numThreads, numCores)

	dimensions := []int{2, 3, 4} // Trial-and-error: 2, 3, 4 sub-networks
	var wg sync.WaitGroup
	results := make(chan struct {
		dims int
		acc  float64
	}, len(dimensions))
	sem := make(chan struct{}, numThreads) // Semaphore to limit threads

	for _, numDims := range dimensions {
		wg.Add(1)
		sem <- struct{}{} // Acquire semaphore
		go func(dims int) {
			defer wg.Done()
			defer func() { <-sem }() // Release semaphore

			// Create a new network for this trial
			net := paragon.NewNetwork(baselineSizes, baselineActivations, baselineFullyConnected)
			for d := 0; d < dims; d++ {
				subNet := paragon.NewNetwork(subSizes, subActivations, subFullyConnected)
				// Overwrite the dimension each time (simulating sequential addition)
				net.SetLayerDimension(1, subNet)
			}
			trainer := paragon.Trainer{
				Network: net,
				Config: paragon.TrainConfig{
					Epochs:       150,
					LearningRate: 0.005,
					Debug:        false,
				},
			}
			fmt.Printf("Training Multi-Dimension Network with %d dimensions...\n", dims)
			trainer.TrainSimple(trainInputs, trainTargets, 150)
			acc := paragon.ComputeAccuracy(net, trainInputs, trainTargets)
			results <- struct {
				dims int
				acc  float64
			}{dims, acc}
		}(numDims)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	multiDimResults := make(map[int]float64)
	for res := range results {
		multiDimResults[res.dims] = res.acc
	}

	// Append results to file
	f, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Printf("Failed to open file %s: %v\n", filename, err)
		return
	}
	defer f.Close()

	result := fmt.Sprintf("Experiment 2 - %s\nBaseline Accuracy: %.2f%%\nSingle Dimension Accuracy: %.2f%%\nScale: %.4f\n",
		time.Now().Format("2006-01-02 15:04:05"), baselineAcc*100, dim1Acc*100, dim1Net.Layers[1].Dimension.Scale)
	for _, dims := range dimensions {
		acc := multiDimResults[dims]
		result += fmt.Sprintf("Multi-Dimension (%d dims) Accuracy: %.2f%%\n", dims, acc*100)
	}
	result += "\n"
	if _, err := f.WriteString(result); err != nil {
		fmt.Printf("Failed to write to file %s: %v\n", filename, err)
	} else {
		fmt.Printf("Results appended to %s\n", filename)
	}
}
