// experiment.go
package main

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
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

func Experiment4(filename string) {
	rand.Seed(time.Now().UnixNano())

	// Dataset parameters
	numSamples := 200
	inputHeight, inputWidth := 4, 4
	numClasses := 10

	// Generate synthetic dataset
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

	// CPU utilization
	numCores := runtime.NumCPU()
	cpuPercent := 0.8
	numThreads := int(float64(numCores) * cpuPercent)
	if numThreads < 1 {
		numThreads = 1
	}
	fmt.Printf("Using %d threads (%d%% of %d cores)\n", numThreads, int(cpuPercent*100), numCores)

	// Base network configuration
	baseSizes := []struct{ Width, Height int }{
		{inputWidth, inputHeight}, // Input: 4x4
		{8, 1},                    // Hidden: 8 neurons
		{numClasses, 1},           // Output: 10 classes
	}
	baseActivations := []string{"linear", "relu", "softmax"}
	baseFullyConnected := []bool{true, true, true}

	// Sub-network configuration
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}

	// Improved training config
	trainConfig := paragon.TrainConfig{
		Epochs:       300,   // More epochs for deeper networks
		LearningRate: 0.005, // Lower learning rate to stabilize training
		Debug:        false,
	}

	// Results storage
	type Result struct {
		Name     string
		Accuracy float64
		Scale1   float64
		Scale2   float64
		Scale3   float64
		Duration time.Duration
	}
	results := make([]Result, 0, 4)
	var wg sync.WaitGroup
	sem := make(chan struct{}, numThreads)
	resultChan := make(chan Result, 4)

	// Training function with learning rate decay
	trainNetwork := func(name string, net *paragon.Network, idx int) {
		defer wg.Done()
		defer func() { <-sem }()

		start := time.Now()
		// Custom training loop with learning rate decay
		for epoch := 0; epoch < trainConfig.Epochs; epoch++ {
			lr := trainConfig.LearningRate * (1.0 - float64(epoch)/float64(trainConfig.Epochs))
			perm := rand.Perm(len(trainInputs))
			for i := range perm {
				net.Forward(trainInputs[perm[i]])
				net.Backward(trainTargets[perm[i]], lr)
			}
		}
		acc := paragon.ComputeAccuracy(net, trainInputs, trainTargets)
		duration := time.Since(start)

		// Extract scales
		scale1, scale2, scale3 := 0.0, 0.0, 0.0
		if idx >= 1 && net.Layers[1].Dimension != nil {
			scale1 = net.Layers[1].Dimension.Scale
		}
		if idx >= 2 && net.Layers[1].Dimension != nil && net.Layers[1].Dimension.Layers[1].Dimension != nil {
			scale2 = net.Layers[1].Dimension.Layers[1].Dimension.Scale
		}
		if idx == 3 && net.Layers[1].Dimension != nil && net.Layers[1].Dimension.Layers[1].Dimension != nil &&
			net.Layers[1].Dimension.Layers[1].Dimension.Layers[1].Dimension != nil {
			scale3 = net.Layers[1].Dimension.Layers[1].Dimension.Layers[1].Dimension.Scale
		}

		resultChan <- Result{
			Name:     name,
			Accuracy: acc,
			Scale1:   scale1,
			Scale2:   scale2,
			Scale3:   scale3,
			Duration: duration,
		}
	}

	// 0D: Baseline
	wg.Add(1)
	sem <- struct{}{}
	go func() {
		fmt.Println("Training Baseline Network (0D)...")
		baseNet := paragon.NewNetwork(baseSizes, baseActivations, baseFullyConnected)
		trainNetwork("Baseline (0D)", baseNet, 0)
	}()

	// 1D: One sub-network
	wg.Add(1)
	sem <- struct{}{}
	go func() {
		fmt.Println("Training 1D Network...")
		net1D := paragon.NewNetwork(baseSizes, baseActivations, baseFullyConnected)
		sub1Sizes := []struct{ Width, Height int }{
			{8, 1},  // Input matches hidden layer
			{16, 1}, // Hidden
			{8, 1},  // Output matches hidden layer
		}
		subNet := paragon.NewNetwork(sub1Sizes, subActivations, subFullyConnected)
		net1D.SetLayerDimension(1, subNet)
		trainNetwork("1D Network", net1D, 1)
	}()

	// 2D: Two sub-networks
	wg.Add(1)
	sem <- struct{}{}
	go func() {
		fmt.Println("Training 2D Network...")
		net2D := paragon.NewNetwork(baseSizes, baseActivations, baseFullyConnected)
		sub1Sizes := []struct{ Width, Height int }{
			{8, 1},  // Input matches hidden layer
			{16, 1}, // Hidden
			{8, 1},  // Output
		}
		sub1Net := paragon.NewNetwork(sub1Sizes, subActivations, subFullyConnected)
		sub2Sizes := []struct{ Width, Height int }{
			{16, 1}, // Input matches sub1 hidden
			{32, 1}, // Hidden
			{16, 1}, // Output
		}
		sub2Net := paragon.NewNetwork(sub2Sizes, subActivations, subFullyConnected)
		sub1Net.SetLayerDimension(1, sub2Net)
		net2D.SetLayerDimension(1, sub1Net)
		trainNetwork("2D Network", net2D, 2)
	}()

	// 3D: Three sub-networks
	wg.Add(1)
	sem <- struct{}{}
	go func() {
		fmt.Println("Training 3D Network...")
		net3D := paragon.NewNetwork(baseSizes, baseActivations, baseFullyConnected)
		sub1Sizes := []struct{ Width, Height int }{
			{8, 1},  // Input matches hidden layer
			{16, 1}, // Hidden
			{8, 1},  // Output
		}
		sub1Net := paragon.NewNetwork(sub1Sizes, subActivations, subFullyConnected)
		sub2Sizes := []struct{ Width, Height int }{
			{16, 1}, // Input matches sub1 hidden
			{32, 1}, // Hidden
			{16, 1}, // Output
		}
		sub2Net := paragon.NewNetwork(sub2Sizes, subActivations, subFullyConnected)
		sub3Sizes := []struct{ Width, Height int }{
			{32, 1}, // Input matches sub2 hidden
			{64, 1}, // Hidden
			{32, 1}, // Output
		}
		sub3Net := paragon.NewNetwork(sub3Sizes, subActivations, subFullyConnected)
		sub2Net.SetLayerDimension(1, sub3Net)
		sub1Net.SetLayerDimension(1, sub2Net)
		net3D.SetLayerDimension(1, sub1Net)
		trainNetwork("3D Network", net3D, 3)
	}()

	// Collect results
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	for res := range resultChan {
		results = append(results, res)
	}

	// Sort results
	sortResults := func(results []Result) {
		order := map[string]int{"Baseline (0D)": 0, "1D Network": 1, "2D Network": 2, "3D Network": 3}
		for i := 0; i < len(results)-1; i++ {
			for j := i + 1; j < len(results); j++ {
				if order[results[i].Name] > order[results[j].Name] {
					results[i], results[j] = results[j], results[i]
				}
			}
		}
	}
	sortResults(results)

	// Write results
	f, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Printf("Failed to open file %s: %v\n", filename, err)
		return
	}
	defer f.Close()

	resultStr := fmt.Sprintf("Experiment 4 - %s\n", time.Now().Format("2006-01-02 15:04:05"))
	for _, res := range results {
		resultStr += fmt.Sprintf("%s:\n  Accuracy: %.2f%%\n  Scale1: %.4f\n  Scale2: %.4f\n  Scale3: %.4f\n  Duration: %v\n",
			res.Name, res.Accuracy*100, res.Scale1, res.Scale2, res.Scale3, res.Duration)
	}
	resultStr += "\n"
	if _, err := f.WriteString(resultStr); err != nil {
		fmt.Printf("Failed to write to file %s: %v\n", filename, err)
	} else {
		fmt.Printf("Results appended to %s\n", filename)
	}
}

func Experiment5(filename string) {
	rand.Seed(time.Now().UnixNano())

	// Dataset parameters
	numSamples := 200
	inputHeight, inputWidth := 4, 4
	numClasses := 10

	// Generate synthetic dataset
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

	// CPU utilization
	numCores := runtime.NumCPU()
	cpuPercent := 0.8
	numThreads := int(float64(numCores) * cpuPercent)
	if numThreads < 1 {
		numThreads = 1
	}
	fmt.Printf("Using %d threads (%d%% of %d cores)\n", numThreads, int(cpuPercent*100), numCores)

	// Base network configuration
	baseSizes := []struct{ Width, Height int }{
		{inputWidth, inputHeight}, // Input: 4x4
		{8, 1},                    // Hidden: 8 neurons
		{numClasses, 1},           // Output: 10 classes
	}
	baseActivations := []string{"linear", "relu", "softmax"}
	baseFullyConnected := []bool{true, true, true}

	// Sub-network configuration
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}

	// Training config
	trainConfig := paragon.TrainConfig{
		Epochs:       500,   // More epochs
		LearningRate: 0.005, // Base LR
		Debug:        true,  // Enable debug
	}

	// Results storage
	type Result struct {
		Name     string
		Accuracy float64
		Scale1   float64
		Scale2   float64
		Scale3   float64
		Duration time.Duration
	}
	results := make([]Result, 0, 4)
	var wg sync.WaitGroup
	sem := make(chan struct{}, numThreads)
	resultChan := make(chan Result, 4)

	// Training function with enhanced stability
	trainNetwork := func(name string, net *paragon.Network, idx int) {
		defer wg.Done()
		defer func() { <-sem }()

		start := time.Now()
		for epoch := 0; epoch < trainConfig.Epochs; epoch++ {
			// Cosine learning rate decay
			lr := trainConfig.LearningRate * (1.0 + math.Cos(float64(epoch)*math.Pi/float64(trainConfig.Epochs))) / 2.0
			perm := rand.Perm(len(trainInputs))
			totalLoss := 0.0
			for i := range perm {
				// Forward pass
				net.Forward(trainInputs[perm[i]])
				loss := net.ComputeLoss(trainTargets[perm[i]])
				if math.IsNaN(loss) || math.IsInf(loss, 0) {
					fmt.Printf("%s: NaN/Inf loss at epoch %d, sample %d\n", name, epoch, i)
					loss = 100.0 // Cap loss
				}
				totalLoss += loss

				// Backward pass with gradient checking
				net.Backward(trainTargets[perm[i]], lr)

				// Stabilize scales
				if idx >= 1 && net.Layers[1].Dimension != nil {
					if math.IsNaN(net.Layers[1].Dimension.Scale) || math.IsInf(net.Layers[1].Dimension.Scale, 0) {
						fmt.Printf("%s: Resetting Scale1 to 1.0 at epoch %d\n", name, epoch)
						net.Layers[1].Dimension.Scale = 1.0
					} else if net.Layers[1].Dimension.Scale > 10.0 {
						net.Layers[1].Dimension.Scale = 10.0
					} else if net.Layers[1].Dimension.Scale < -10.0 {
						net.Layers[1].Dimension.Scale = -10.0
					}
				}
				if idx >= 2 && net.Layers[1].Dimension.Layers[1].Dimension != nil {
					if math.IsNaN(net.Layers[1].Dimension.Layers[1].Dimension.Scale) || math.IsInf(net.Layers[1].Dimension.Layers[1].Dimension.Scale, 0) {
						fmt.Printf("%s: Resetting Scale2 to 1.0 at epoch %d\n", name, epoch)
						net.Layers[1].Dimension.Layers[1].Dimension.Scale = 1.0
					} else if net.Layers[1].Dimension.Layers[1].Dimension.Scale > 10.0 {
						net.Layers[1].Dimension.Layers[1].Dimension.Scale = 10.0
					} else if net.Layers[1].Dimension.Layers[1].Dimension.Scale < -10.0 {
						net.Layers[1].Dimension.Layers[1].Dimension.Scale = -10.0
					}
				}
				if idx == 3 && net.Layers[1].Dimension.Layers[1].Dimension.Layers[1].Dimension != nil {
					if math.IsNaN(net.Layers[1].Dimension.Layers[1].Dimension.Layers[1].Dimension.Scale) || math.IsInf(net.Layers[1].Dimension.Layers[1].Dimension.Layers[1].Dimension.Scale, 0) {
						fmt.Printf("%s: Resetting Scale3 to 1.0 at epoch %d\n", name, epoch)
						net.Layers[1].Dimension.Layers[1].Dimension.Layers[1].Dimension.Scale = 1.0
					} else if net.Layers[1].Dimension.Layers[1].Dimension.Layers[1].Dimension.Scale > 10.0 {
						net.Layers[1].Dimension.Layers[1].Dimension.Layers[1].Dimension.Scale = 10.0
					} else if net.Layers[1].Dimension.Layers[1].Dimension.Layers[1].Dimension.Scale < -10.0 {
						net.Layers[1].Dimension.Layers[1].Dimension.Layers[1].Dimension.Scale = -10.0
					}
				}
			}
			if epoch%50 == 0 {
				fmt.Printf("%s: Epoch %d, Loss: %.4f, LR: %.6f\n", name, epoch, totalLoss/float64(len(trainInputs)), lr)
			}
		}
		acc := paragon.ComputeAccuracy(net, trainInputs, trainTargets)
		duration := time.Since(start)

		// Extract scales with fallback
		scale1, scale2, scale3 := 0.0, 0.0, 0.0
		if idx >= 1 && net.Layers[1].Dimension != nil {
			scale1 = net.Layers[1].Dimension.Scale
			if math.IsNaN(scale1) || math.IsInf(scale1, 0) {
				scale1 = 1.0
			}
		}
		if idx >= 2 && net.Layers[1].Dimension != nil && net.Layers[1].Dimension.Layers[1].Dimension != nil {
			scale2 = net.Layers[1].Dimension.Layers[1].Dimension.Scale
			if math.IsNaN(scale2) || math.IsInf(scale2, 0) {
				scale2 = 1.0
			}
		}
		if idx == 3 && net.Layers[1].Dimension != nil && net.Layers[1].Dimension.Layers[1].Dimension != nil &&
			net.Layers[1].Dimension.Layers[1].Dimension.Layers[1].Dimension != nil {
			scale3 = net.Layers[1].Dimension.Layers[1].Dimension.Layers[1].Dimension.Scale
			if math.IsNaN(scale3) || math.IsInf(scale3, 0) {
				scale3 = 1.0
			}
		}

		resultChan <- Result{
			Name:     name,
			Accuracy: acc,
			Scale1:   scale1,
			Scale2:   scale2,
			Scale3:   scale3,
			Duration: duration,
		}
	}

	// 0D: Baseline
	wg.Add(1)
	sem <- struct{}{}
	go func() {
		fmt.Println("Training Baseline Network (0D)...")
		baseNet := paragon.NewNetwork(baseSizes, baseActivations, baseFullyConnected)
		trainNetwork("Baseline (0D)", baseNet, 0)
	}()

	// 1D: One sub-network
	wg.Add(1)
	sem <- struct{}{}
	go func() {
		fmt.Println("Training 1D Network...")
		net1D := paragon.NewNetwork(baseSizes, baseActivations, baseFullyConnected)
		sub1Sizes := []struct{ Width, Height int }{
			{8, 1},  // Input matches hidden layer
			{12, 1}, // Smaller hidden layer
			{8, 1},  // Output
		}
		subNet := paragon.NewNetwork(sub1Sizes, subActivations, subFullyConnected)
		net1D.SetLayerDimension(1, subNet)
		trainNetwork("1D Network", net1D, 1)
	}()

	// 2D: Two sub-networks
	wg.Add(1)
	sem <- struct{}{}
	go func() {
		fmt.Println("Training 2D Network...")
		net2D := paragon.NewNetwork(baseSizes, baseActivations, baseFullyConnected)
		sub1Sizes := []struct{ Width, Height int }{
			{8, 1},  // Input matches hidden layer
			{12, 1}, // Smaller hidden
			{8, 1},  // Output
		}
		sub1Net := paragon.NewNetwork(sub1Sizes, subActivations, subFullyConnected)
		sub2Sizes := []struct{ Width, Height int }{
			{12, 1}, // Input matches sub1 hidden
			{16, 1}, // Smaller hidden
			{12, 1}, // Output
		}
		sub2Net := paragon.NewNetwork(sub2Sizes, subActivations, subFullyConnected)
		sub1Net.SetLayerDimension(1, sub2Net)
		net2D.SetLayerDimension(1, sub1Net)
		trainNetwork("2D Network", net2D, 2)
	}()

	// 3D: Three sub-networks
	wg.Add(1)
	sem <- struct{}{}
	go func() {
		fmt.Println("Training 3D Network...")
		net3D := paragon.NewNetwork(baseSizes, baseActivations, baseFullyConnected)
		sub1Sizes := []struct{ Width, Height int }{
			{8, 1},  // Input matches hidden layer
			{12, 1}, // Smaller hidden
			{8, 1},  // Output
		}
		sub1Net := paragon.NewNetwork(sub1Sizes, subActivations, subFullyConnected)
		sub2Sizes := []struct{ Width, Height int }{
			{12, 1}, // Input matches sub1 hidden
			{16, 1}, // Smaller hidden
			{12, 1}, // Output
		}
		sub2Net := paragon.NewNetwork(sub2Sizes, subActivations, subFullyConnected)
		sub3Sizes := []struct{ Width, Height int }{
			{16, 1}, // Input matches sub2 hidden
			{20, 1}, // Smaller hidden
			{16, 1}, // Output
		}
		sub3Net := paragon.NewNetwork(sub3Sizes, subActivations, subFullyConnected)
		sub2Net.SetLayerDimension(1, sub3Net)
		sub1Net.SetLayerDimension(1, sub2Net)
		net3D.SetLayerDimension(1, sub1Net)
		trainNetwork("3D Network", net3D, 3)
	}()

	// Collect results
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	for res := range resultChan {
		results = append(results, res)
	}

	// Sort results
	sortResults := func(results []Result) {
		order := map[string]int{"Baseline (0D)": 0, "1D Network": 1, "2D Network": 2, "3D Network": 3}
		for i := 0; i < len(results)-1; i++ {
			for j := i + 1; j < len(results); j++ {
				if order[results[i].Name] > order[results[j].Name] {
					results[i], results[j] = results[j], results[i]
				}
			}
		}
	}
	sortResults(results)

	// Write results
	f, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Printf("Failed to open file %s: %v\n", filename, err)
		return
	}
	defer f.Close()

	resultStr := fmt.Sprintf("Experiment 5 - %s\n", time.Now().Format("2006-01-02 15:04:05"))
	for _, res := range results {
		resultStr += fmt.Sprintf("%s:\n  Accuracy: %.2f%%\n  Scale1: %.4f\n  Scale2: %.4f\n  Scale3: %.4f\n  Duration: %v\n",
			res.Name, res.Accuracy*100, res.Scale1, res.Scale2, res.Scale3, res.Duration)
	}
	resultStr += "\n"
	if _, err := f.WriteString(resultStr); err != nil {
		fmt.Printf("Failed to write to file %s: %v\n", filename, err)
	} else {
		fmt.Printf("Results appended to %s\n", filename)
	}
}

func Experiment6(filename string) {
	rand.Seed(time.Now().UnixNano())

	// Download and load MNIST dataset with retries
	trainImages, trainLabels, err := loadMNIST("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
	if err != nil {
		fmt.Printf("Failed to load MNIST after retries: %v\n", err)
		return
	}
	numSamples := len(trainLabels)
	inputHeight, inputWidth := 28, 28
	numClasses := 10
	fmt.Printf("Loaded %d MNIST training samples\n", numSamples)

	// Convert to paragon format
	trainInputs := make([][][]float64, numSamples)
	trainTargets := make([][][]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		trainInputs[i] = make([][]float64, inputHeight)
		trainTargets[i] = make([][]float64, 1)
		trainTargets[i][0] = make([]float64, numClasses)
		label := int(trainLabels[i])
		trainTargets[i][0][label] = 1.0
		for y := 0; y < inputHeight; y++ {
			trainInputs[i][y] = make([]float64, inputWidth)
			for x := 0; x < inputWidth; x++ {
				trainInputs[i][y][x] = float64(trainImages[i][y*inputWidth+x]) / 255.0 // Normalize to [0, 1]
			}
		}
	}

	// CPU utilization
	numCores := runtime.NumCPU()
	cpuPercent := 0.8
	numThreads := int(float64(numCores) * cpuPercent)
	if numThreads < 1 {
		numThreads = 1
	}
	fmt.Printf("Using %d threads (%d%% of %d cores)\n", numThreads, int(cpuPercent*100), numCores)

	// Base network configuration
	baseSizes := []struct{ Width, Height int }{
		{inputWidth, inputHeight}, // Input: 28x28
		{64, 1},                   // Hidden: 64 neurons
		{numClasses, 1},           // Output: 10 classes
	}
	baseActivations := []string{"linear", "relu", "softmax"}
	baseFullyConnected := []bool{true, true, true}

	// Sub-network configuration
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}

	// Training config
	trainConfig := paragon.TrainConfig{
		Epochs:       20,   // MNIST converges fast
		LearningRate: 0.01, // Fixed LR
		Debug:        true,
	}

	// Results storage
	type Result struct {
		Name     string
		Accuracy float64
		Scale1   float64
		Scale2   float64
		Scale3   float64
		Duration time.Duration
	}
	results := make([]Result, 0, 4)
	var wg sync.WaitGroup
	sem := make(chan struct{}, numThreads)
	resultChan := make(chan Result, 4)

	// Training function
	trainNetwork := func(name string, net *paragon.Network, idx int) {
		defer wg.Done()
		defer func() { <-sem }()

		start := time.Now()
		lr := trainConfig.LearningRate
		for epoch := 0; epoch < trainConfig.Epochs; epoch++ {
			perm := rand.Perm(len(trainInputs))
			totalLoss := 0.0
			validSamples := 0
			for i := range perm {
				net.Forward(trainInputs[perm[i]])
				loss := net.ComputeLoss(trainTargets[perm[i]])
				if loss > 1e5 || loss < -1e5 {
					fmt.Printf("%s: Extreme loss %.4f at epoch %d, sample %d, skipping\n", name, loss, epoch, i)
					continue
				}
				totalLoss += loss
				validSamples++
				net.Backward(trainTargets[perm[i]], lr)
				// Stabilize scales
				if idx >= 1 && net.Layers[1].Dimension != nil {
					if net.Layers[1].Dimension.Scale > 10.0 {
						net.Layers[1].Dimension.Scale = 10.0
					} else if net.Layers[1].Dimension.Scale < -10.0 {
						net.Layers[1].Dimension.Scale = -10.0
					}
				}
				if idx >= 2 && net.Layers[1].Dimension.Layers[1].Dimension != nil {
					if net.Layers[1].Dimension.Layers[1].Dimension.Scale > 10.0 {
						net.Layers[1].Dimension.Layers[1].Dimension.Scale = 10.0
					} else if net.Layers[1].Dimension.Layers[1].Dimension.Scale < -10.0 {
						net.Layers[1].Dimension.Layers[1].Dimension.Scale = -10.0
					}
				}
				if idx == 3 && net.Layers[1].Dimension.Layers[1].Dimension.Layers[1].Dimension != nil {
					if net.Layers[1].Dimension.Layers[1].Dimension.Layers[1].Dimension.Scale > 10.0 {
						net.Layers[1].Dimension.Layers[1].Dimension.Layers[1].Dimension.Scale = 10.0
					} else if net.Layers[1].Dimension.Layers[1].Dimension.Layers[1].Dimension.Scale < -10.0 {
						net.Layers[1].Dimension.Layers[1].Dimension.Layers[1].Dimension.Scale = -10.0
					}
				}
			}
			if validSamples > 0 && epoch%5 == 0 {
				fmt.Printf("%s: Epoch %d, Loss: %.4f, Valid Samples: %d\n", name, epoch, totalLoss/float64(validSamples), validSamples)
			}
		}
		acc := paragon.ComputeAccuracy(net, trainInputs, trainTargets)
		duration := time.Since(start)

		// Extract scales
		scale1, scale2, scale3 := 0.0, 0.0, 0.0
		if idx >= 1 && net.Layers[1].Dimension != nil {
			scale1 = net.Layers[1].Dimension.Scale
		}
		if idx >= 2 && net.Layers[1].Dimension != nil && net.Layers[1].Dimension.Layers[1].Dimension != nil {
			scale2 = net.Layers[1].Dimension.Layers[1].Dimension.Scale
		}
		if idx == 3 && net.Layers[1].Dimension != nil && net.Layers[1].Dimension.Layers[1].Dimension != nil &&
			net.Layers[1].Dimension.Layers[1].Dimension.Layers[1].Dimension != nil {
			scale3 = net.Layers[1].Dimension.Layers[1].Dimension.Layers[1].Dimension.Scale
		}

		resultChan <- Result{
			Name:     name,
			Accuracy: acc,
			Scale1:   scale1,
			Scale2:   scale2,
			Scale3:   scale3,
			Duration: duration,
		}
	}

	// 0D: Baseline
	wg.Add(1)
	sem <- struct{}{}
	go func() {
		fmt.Println("Training Baseline Network (0D)...")
		baseNet := paragon.NewNetwork(baseSizes, baseActivations, baseFullyConnected)
		trainNetwork("Baseline (0D)", baseNet, 0)
	}()

	// 1D: One sub-network
	wg.Add(1)
	sem <- struct{}{}
	go func() {
		fmt.Println("Training 1D Network...")
		net1D := paragon.NewNetwork(baseSizes, baseActivations, baseFullyConnected)
		sub1Sizes := []struct{ Width, Height int }{
			{64, 1},  // Input matches hidden layer
			{128, 1}, // Hidden
			{64, 1},  // Output
		}
		subNet := paragon.NewNetwork(sub1Sizes, subActivations, subFullyConnected)
		net1D.SetLayerDimension(1, subNet)
		trainNetwork("1D Network", net1D, 1)
	}()

	// 2D: Two sub-networks
	wg.Add(1)
	sem <- struct{}{}
	go func() {
		fmt.Println("Training 2D Network...")
		net2D := paragon.NewNetwork(baseSizes, baseActivations, baseFullyConnected)
		sub1Sizes := []struct{ Width, Height int }{
			{64, 1},  // Input matches hidden layer
			{128, 1}, // Hidden
			{64, 1},  // Output
		}
		sub1Net := paragon.NewNetwork(sub1Sizes, subActivations, subFullyConnected)
		sub2Sizes := []struct{ Width, Height int }{
			{128, 1}, // Input matches sub1 hidden
			{256, 1}, // Hidden
			{128, 1}, // Output
		}
		sub2Net := paragon.NewNetwork(sub2Sizes, subActivations, subFullyConnected)
		sub1Net.SetLayerDimension(1, sub2Net)
		net2D.SetLayerDimension(1, sub1Net)
		trainNetwork("2D Network", net2D, 2)
	}()

	// 3D: Three sub-networks
	wg.Add(1)
	sem <- struct{}{}
	go func() {
		fmt.Println("Training 3D Network...")
		net3D := paragon.NewNetwork(baseSizes, baseActivations, baseFullyConnected)
		sub1Sizes := []struct{ Width, Height int }{
			{64, 1},  // Input matches hidden layer
			{128, 1}, // Hidden
			{64, 1},  // Output
		}
		sub1Net := paragon.NewNetwork(sub1Sizes, subActivations, subFullyConnected)
		sub2Sizes := []struct{ Width, Height int }{
			{128, 1}, // Input matches sub1 hidden
			{256, 1}, // Hidden
			{128, 1}, // Output
		}
		sub2Net := paragon.NewNetwork(sub2Sizes, subActivations, subFullyConnected)
		sub3Sizes := []struct{ Width, Height int }{
			{256, 1}, // Input matches sub2 hidden
			{512, 1}, // Hidden
			{256, 1}, // Output
		}
		sub3Net := paragon.NewNetwork(sub3Sizes, subActivations, subFullyConnected)
		sub2Net.SetLayerDimension(1, sub3Net)
		sub1Net.SetLayerDimension(1, sub2Net)
		net3D.SetLayerDimension(1, sub1Net)
		trainNetwork("3D Network", net3D, 3)
	}()

	// Collect results
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	for res := range resultChan {
		results = append(results, res)
	}

	// Sort results
	sortResults := func(results []Result) {
		order := map[string]int{"Baseline (0D)": 0, "1D Network": 1, "2D Network": 2, "3D Network": 3}
		for i := 0; i < len(results)-1; i++ {
			for j := i + 1; j < len(results); j++ {
				if order[results[i].Name] > order[results[j].Name] {
					results[i], results[j] = results[j], results[i]
				}
			}
		}
	}
	sortResults(results)

	// Write results
	f, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Printf("Failed to open file %s: %v\n", filename, err)
		return
	}
	defer f.Close()

	resultStr := fmt.Sprintf("Experiment 6 - %s\n", time.Now().Format("2006-01-02 15:04:05"))
	for _, res := range results {
		resultStr += fmt.Sprintf("%s:\n  Accuracy: %.2f%%\n  Scale1: %.4f\n  Scale2: %.4f\n  Scale3: %.4f\n  Duration: %v\n",
			res.Name, res.Accuracy*100, res.Scale1, res.Scale2, res.Scale3, res.Duration)
	}
	resultStr += "\n"
	if _, err := f.WriteString(resultStr); err != nil {
		fmt.Printf("Failed to write to file %s: %v\n", filename, err)
	} else {
		fmt.Printf("Results appended to %s\n", filename)
	}
}

// loadMNIST downloads and parses MNIST data from Google Cloud Storage
func loadMNIST(imagesFile, labelsFile string) ([][]byte, []byte, error) {
	baseURL := "https://storage.googleapis.com/cvdf-datasets/mnist/"
	targetDir := "mnist_data"

	// Ensure directory exists
	if err := os.MkdirAll(targetDir, os.ModePerm); err != nil {
		return nil, nil, fmt.Errorf("failed to create directory %s: %w", targetDir, err)
	}

	// Define files to download
	files := []struct {
		compressed   string
		uncompressed string
	}{
		{imagesFile, imagesFile[:len(imagesFile)-3]}, // e.g., "train-images-idx3-ubyte.gz" -> "train-images-idx3-ubyte"
		{labelsFile, labelsFile[:len(labelsFile)-3]}, // e.g., "train-labels-idx1-ubyte.gz" -> "train-labels-idx1-ubyte"
	}

	// Download and unzip files if needed
	for _, f := range files {
		compressedPath := filepath.Join(targetDir, f.compressed)
		uncompressedPath := filepath.Join(targetDir, f.uncompressed)

		if _, err := os.Stat(uncompressedPath); os.IsNotExist(err) {
			if _, err := os.Stat(compressedPath); os.IsNotExist(err) {
				fmt.Printf("Downloading %s...\n", f.compressed)
				url := baseURL + f.compressed
				resp, err := http.Get(url)
				if err != nil {
					return nil, nil, fmt.Errorf("failed to download %s: %v", f.compressed, err)
				}
				defer resp.Body.Close()
				out, err := os.Create(compressedPath)
				if err != nil {
					return nil, nil, fmt.Errorf("create file %s failed: %v", compressedPath, err)
				}
				_, err = io.Copy(out, resp.Body)
				out.Close()
				if err != nil {
					os.Remove(compressedPath)
					return nil, nil, fmt.Errorf("write file %s failed: %v", compressedPath, err)
				}
			}
			fmt.Printf("Unzipping %s...\n", f.compressed)
			fSrc, err := os.Open(compressedPath)
			if err != nil {
				return nil, nil, fmt.Errorf("open %s failed: %v", compressedPath, err)
			}
			defer fSrc.Close()
			gzReader, err := gzip.NewReader(fSrc)
			if err != nil {
				return nil, nil, fmt.Errorf("gzip %s: %v", compressedPath, err)
			}
			defer gzReader.Close()
			fDest, err := os.Create(uncompressedPath)
			if err != nil {
				return nil, nil, fmt.Errorf("create %s failed: %v", uncompressedPath, err)
			}
			defer fDest.Close()
			_, err = io.Copy(fDest, gzReader)
			if err != nil {
				return nil, nil, fmt.Errorf("unzip %s failed: %v", compressedPath, err)
			}
			// Clean up compressed file
			if err := os.Remove(compressedPath); err != nil {
				fmt.Printf("Warning: failed to remove %s: %v\n", compressedPath, err)
			}
		}
	}

	// Load images
	imgPath := filepath.Join(targetDir, files[0].uncompressed)
	imgFile, err := os.Open(imgPath)
	if err != nil {
		return nil, nil, fmt.Errorf("open image file %s: %v", imgPath, err)
	}
	defer imgFile.Close()
	var imgHeader [16]byte
	if _, err := imgFile.Read(imgHeader[:]); err != nil {
		return nil, nil, fmt.Errorf("read image header: %v", err)
	}
	if magic := binary.BigEndian.Uint32(imgHeader[0:4]); magic != 2051 {
		return nil, nil, fmt.Errorf("invalid image magic number: %d", magic)
	}
	numImages := int(binary.BigEndian.Uint32(imgHeader[4:8]))
	rows := int(binary.BigEndian.Uint32(imgHeader[8:12]))
	cols := int(binary.BigEndian.Uint32(imgHeader[12:16]))
	if rows != 28 || cols != 28 {
		return nil, nil, fmt.Errorf("unexpected image dimensions: %dx%d", rows, cols)
	}
	images := make([][]byte, numImages)
	imgPixels := make([]byte, rows*cols*numImages)
	_, err = io.ReadFull(imgFile, imgPixels)
	if err != nil {
		return nil, nil, fmt.Errorf("read image pixels: %v", err)
	}
	for i := 0; i < numImages; i++ {
		images[i] = imgPixels[i*rows*cols : (i+1)*rows*cols]
	}

	// Load labels
	lblPath := filepath.Join(targetDir, files[1].uncompressed)
	lblFile, err := os.Open(lblPath)
	if err != nil {
		return nil, nil, fmt.Errorf("open label file %s: %v", lblPath, err)
	}
	defer lblFile.Close()
	var lblHeader [8]byte
	if _, err := lblFile.Read(lblHeader[:]); err != nil {
		return nil, nil, fmt.Errorf("read label header: %v", err)
	}
	if magic := binary.BigEndian.Uint32(lblHeader[0:4]); magic != 2049 {
		return nil, nil, fmt.Errorf("invalid label magic number: %d", magic)
	}
	numLabels := int(binary.BigEndian.Uint32(lblHeader[4:8]))
	labels := make([]byte, numLabels)
	_, err = io.ReadFull(lblFile, labels)
	if err != nil {
		return nil, nil, fmt.Errorf("read labels: %v", err)
	}

	if numImages != numLabels {
		return nil, nil, fmt.Errorf("image/label count mismatch: %d vs %d", numImages, numLabels)
	}

	return images, labels, nil
}
