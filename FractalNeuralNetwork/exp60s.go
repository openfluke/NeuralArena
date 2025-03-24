package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"paragon"
	"time"
)

func experiment64(file *os.File) {
	fmt.Println("\n=== Experiment 64: Advanced Task Suite ===")

	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	// Network configurations
	layerSizes := []struct{ Width, Height int }{{16, 1}, {32, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}
	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {4, 1}, {1, 1}}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}
	trainCfg := paragon.TrainConfig{Epochs: 2, LearningRate: 0.001}

	// List of tasks
	tasks := []string{
		"Hierarchical Pattern Recognition",
		"Recursive Function Approximation",
		"High-Dimensional Classification",
		"Time Series Prediction",
		"Simple Classification",
	}

	for _, task := range tasks {
		fmt.Printf("\n--- Task: %s ---\n", task)

		// Generate training and validation data
		trainInputs, trainTargets := generateDataForTask(task, 1000)
		valInputs, valTargets := generateDataForTask(task, 200)

		// Baseline network
		baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
		trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
		trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
		baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
		fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

		// Fractal network
		fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
		fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
		for y := 0; y < fractalNet.Layers[1].Height; y++ {
			for x := 0; x < fractalNet.Layers[1].Width; x++ {
				fractalNet.Layers[1].Neurons[y][x].Dimension.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
			}
		}
		trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}
		trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
		fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
		fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

		// Log results
		result := fmt.Sprintf("Task: %s\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
			task, baselineAcc*100, fractalAcc*100)
		file.WriteString(result)
	}
}

// generateDataForTask generates synthetic data based on the specified task
func generateDataForTask(task string, nSamples int) (inputs [][][]float64, targets [][][]float64) {
	inputs = make([][][]float64, nSamples)
	targets = make([][][]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		inputs[i] = make([][]float64, 1)
		inputs[i][0] = make([]float64, 16)
		targets[i] = make([][]float64, 1)
		targets[i][0] = make([]float64, 2)

		switch task {
		case "Hierarchical Pattern Recognition":
			// Nested conditions across 4 groups of 4 features
			for j := 0; j < 16; j++ {
				inputs[i][0][j] = rand.Float64()
			}
			count := 0
			for g := 0; g < 4; g++ {
				s1 := inputs[i][0][4*g : 4*g+2]
				s2 := inputs[i][0][4*g+2 : 4*g+4]
				if s1[0] > s1[1] && s2[0] > s2[1] {
					count++
				}
			}
			label := 0
			if count >= 3 {
				label = 1
			}
			targets[i][0][label] = 1.0

		case "Recursive Function Approximation":
			// Approximate a recursive function: y = x + 0.5 * f(x-1), f(0) = 0
			x := rand.Float64()
			f := x
			for k := 0; k < 5; k++ { // 5 iterations of recursion
				f = x + 0.5*f
			}
			for j := 0; j < 16; j++ {
				inputs[i][0][j] = x + float64(j)*0.01 // Slight variations of x
			}
			label := 0
			if f > 0.5 {
				label = 1
			}
			targets[i][0][label] = 1.0

		case "High-Dimensional Classification":
			// Complex feature interactions
			for j := 0; j < 16; j++ {
				inputs[i][0][j] = rand.Float64()
			}
			score := inputs[i][0][0]*inputs[i][0][1] + inputs[i][0][2]*inputs[i][0][3] - inputs[i][0][4]
			label := 0
			if score > 0 {
				label = 1
			}
			targets[i][0][label] = 1.0

		case "Time Series Prediction":
			// Predict next value in a synthetic series with long-term dependencies
			series := make([]float64, 16)
			for j := 0; j < 16; j++ {
				series[j] = math.Sin(float64(j)*0.1) + 0.5*math.Sin(float64(j)*0.05)
				inputs[i][0][j] = series[j]
			}
			nextVal := math.Sin(1.6*0.1) + 0.5*math.Sin(1.6*0.05)
			label := 0
			if nextVal > 0 {
				label = 1
			}
			targets[i][0][label] = 1.0

		case "Simple Classification":
			// Linearly separable data
			for j := 0; j < 16; j++ {
				inputs[i][0][j] = rand.Float64()
			}
			sum := 0.0
			for j := 0; j < 16; j++ {
				sum += inputs[i][0][j]
			}
			label := 0
			if sum > 8 {
				label = 1
			}
			targets[i][0][label] = 1.0
		}
	}
	return inputs, targets
}

// experiment65 trains baseline and fractal networks on CIFAR-10 with 100 samples and 10 epochs
func experiment65(file *os.File) {
	fmt.Println("\n=== Experiment 65: CIFAR-10 Classification ===")

	// Seed random number generator for reproducibility
	rand.Seed(time.Now().UnixNano())

	// **Network Configurations**
	// CIFAR-10 images are 32x32x3 = 3072 features when flattened
	// Output is 10 classes
	layerSizes := []struct{ Width, Height int }{
		{3072, 1}, // Input layer: flattened image
		{100, 10}, // Hidden layer 1
		{100, 10}, // Hidden layer 2
		{10, 1},   // Output layer: 10 classes
	}
	activations := []string{"linear", "relu", "relu", "softmax"}
	fullyConnected := []bool{true, true, true, true}

	// Sub-network configuration for fractal network
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}
	subLayerSizes := []struct{ Width, Height int }{
		{1, 1}, // Sub-network input
		{4, 4}, // Sub-network hidden
		{1, 1}, // Sub-network output
	}
	subActivations := []string{"relu", "relu", "relu"}
	subFullyConnected := []bool{true, true, true}

	// Training configuration
	trainCfg := paragon.TrainConfig{Epochs: 2, LearningRate: 0.01}

	// **Load CIFAR-10 Data**
	// Using 100 training samples and 20 validation samples for simplicity
	trainInputs, trainTargets, valInputs, valTargets := loadCIFAR10Subset(100, 20)

	// **Baseline Network**
	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	// **Fractal Network**
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	// Add sub-networks to hidden layer 1 (index 1)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
	// Add sub-sub-networks to the hidden layer of each sub-network
	for y := 0; y < fractalNet.Layers[1].Height; y++ {
		for x := 0; x < fractalNet.Layers[1].Width; x++ {
			subNet := fractalNet.Layers[1].Neurons[y][x].Dimension
			subNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
		}
	}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}
	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	// **Log Results**
	result := fmt.Sprintf(
		"Experiment 65: CIFAR-10 Classification\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100,
	)
	file.WriteString(result)
}

// loadCIFAR10Subset simulates loading a subset of CIFAR-10 data
func loadCIFAR10Subset(nTrain, nVal int) (trainInputs, trainTargets, valInputs, valTargets [][][]float64) {
	// CIFAR-10 has 50,000 training and 10,000 test images
	// For simplicity, generate synthetic data matching the format
	// In reality, use a library to load actual CIFAR-10 data

	// Initialize slices
	trainInputs = make([][][]float64, nTrain)
	trainTargets = make([][][]float64, nTrain)
	valInputs = make([][][]float64, nVal)
	valTargets = make([][][]float64, nVal)

	// Input: 1x3072 (height=1, width=3072)
	// Target: 1x10 (height=1, width=10, one-hot encoded)
	for i := 0; i < nTrain; i++ {
		trainInputs[i] = make([][]float64, 1)
		trainInputs[i][0] = make([]float64, 3072)
		trainTargets[i] = make([][]float64, 1)
		trainTargets[i][0] = make([]float64, 10)

		// Simulate normalized pixel values (0 to 1)
		for j := 0; j < 3072; j++ {
			trainInputs[i][0][j] = rand.Float64()
		}
		// Random one-hot encoded target (0 to 9)
		class := rand.Intn(10)
		for j := 0; j < 10; j++ {
			if j == class {
				trainTargets[i][0][j] = 1.0
			} else {
				trainTargets[i][0][j] = 0.0
			}
		}
	}

	for i := 0; i < nVal; i++ {
		valInputs[i] = make([][]float64, 1)
		valInputs[i][0] = make([]float64, 3072)
		valTargets[i] = make([][]float64, 1)
		valTargets[i][0] = make([]float64, 10)

		for j := 0; j < 3072; j++ {
			valInputs[i][0][j] = rand.Float64()
		}
		class := rand.Intn(10)
		for j := 0; j < 10; j++ {
			if j == class {
				valTargets[i][0][j] = 1.0
			} else {
				valTargets[i][0][j] = 0.0
			}
		}
	}

	return trainInputs, trainTargets, valInputs, valTargets
}
