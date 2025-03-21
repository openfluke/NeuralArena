package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"

	"paragon"
)

// generateHierarchicalXOR generates n samples for the Hierarchical XOR task.
func generateHierarchicalXORT59(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		a := rand.Intn(2)
		b := rand.Intn(2)
		c := rand.Intn(2)
		d := rand.Intn(2)
		p := a ^ b
		q := c ^ d
		r := p ^ q
		inputs[i] = [][]float64{{float64(a), float64(b), float64(c), float64(d)}}
		targets[i] = [][]float64{{float64(r)}}
	}
	return inputs, targets
}

// generateXORSubData generates synthetic data for pre-training sub-networks.
func generateXORSubDataT59(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		sumVal := rand.Float64()*4 - 2 // Range [-2, 2] to simulate weighted sums
		var target float64
		if sumVal > 0.5 && sumVal < 1.5 { // Approximate XOR behavior
			target = 1.0
		} else {
			target = 0.0
		}
		inputs[i] = [][]float64{{sumVal}}
		targets[i] = [][]float64{{target}}
	}
	return inputs, targets
}

// createBaselineNetwork creates a standard MLP network.
func createBaselineNetwork() *paragon.Network {
	layerSizes := []struct{ Width, Height int }{
		{4, 1}, // Input layer: 4 neurons
		{8, 1}, // Hidden layer 1: 8 neurons
		{8, 1}, // Hidden layer 2: 8 neurons
		{1, 1}, // Output layer: 1 neuron
	}
	activations := []string{"linear", "relu", "relu", "sigmoid"}
	fullyConnected := []bool{true, true, true, true}
	return paragon.NewNetwork(layerSizes, activations, fullyConnected)
}

// createXORSubNetwork creates a sub-network for processing sums in the fractal network.
func createXORSubNetwork() *paragon.Network {
	layerSizes := []struct{ Width, Height int }{
		{1, 1}, // Input: weighted sum (scalar)
		{2, 1}, // Hidden: 2 neurons
		{1, 1}, // Output: 1 neuron
	}
	activations := []string{"linear", "relu", "sigmoid"}
	fullyConnected := []bool{true, true, true}
	return paragon.NewNetwork(layerSizes, activations, fullyConnected)
}

// createFractalNetwork creates a fractal network with sub-networks in its neurons.
func createFractalNetwork() *paragon.Network {
	// Main network structure
	mainLayerSizes := []struct{ Width, Height int }{
		{4, 1}, // Input: 4 neurons
		{2, 1}, // Hidden layer 1: 2 neurons with sub-networks
		{1, 1}, // Output: 1 neuron with a sub-network
	}
	mainActivations := []string{"linear", "relu", "sigmoid"}
	mainFullyConnected := []bool{true, true, true}
	fractalNet := paragon.NewNetwork(mainLayerSizes, mainActivations, mainFullyConnected)

	// Sub-network structure
	subLayerSizes := []struct{ Width, Height int }{
		{1, 1}, // Input: weighted sum
		{2, 1}, // Hidden: 2 neurons
		{1, 1}, // Output: 1 neuron
	}
	subActivations := []string{"linear", "relu", "sigmoid"}
	subFullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	// Assign sub-networks to hidden layer 1
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)

	// Assign sub-network to output layer
	fractalNet.SetLayerDimension(2, subLayerSizes, subActivations, subFullyConnected, opts)

	return fractalNet
}

// computeAccuracy calculates the accuracy of a network on a dataset.
func computeAccuracy(net *paragon.Network, inputs [][][]float64, targets [][][]float64) float64 {
	correct := 0
	for i := range inputs {
		net.Forward(inputs[i])
		output := net.Layers[net.OutputLayer].Neurons[0][0].Value
		pred := 0
		if output > 0.5 {
			pred = 1
		}
		label := int(targets[i][0][0])
		if pred == label {
			correct++
		}
	}
	return float64(correct) / float64(len(inputs))
}

// mean computes the average of a float64 slice.
func mean(arr []float64) float64 {
	sum := 0.0
	for _, v := range arr {
		sum += v
	}
	return sum / float64(len(arr))
}

// stdDev computes the standard deviation of a float64 slice.
func stdDev(arr []float64) float64 {
	m := mean(arr)
	sumSq := 0.0
	for _, v := range arr {
		sumSq += (v - m) * (v - m)
	}
	return math.Sqrt(sumSq / float64(len(arr)-1))
}

// Experiment59 compares a fractal network with a baseline network on Hierarchical XOR.
func Experiment59(file *os.File) {
	fmt.Println("\n=== Experiment 59: Fractal Network vs Baseline on Hierarchical XOR ===")

	// Generate datasets
	trainInputs, trainTargets := generateHierarchicalXORT59(1000)
	valInputs, valTargets := generateHierarchicalXORT59(200)

	// Define training configuration
	trainCfg := paragon.TrainConfig{Epochs: 50, LearningRate: 0.001}

	// Generate pre-training data for sub-networks
	subTrainInputs, subTrainTargets := generateXORSubDataT59(1000)

	// Perform 10 runs for statistical reliability
	numRuns := 10
	baselineAccs := make([]float64, numRuns)
	fractalAccs := make([]float64, numRuns)

	for run := 0; run < numRuns; run++ {
		fmt.Printf("Run %d\n", run+1)

		// **Baseline Network**
		baselineNet := createBaselineNetwork()
		trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
		trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
		baselineAcc := computeAccuracy(baselineNet, valInputs, valTargets)
		baselineAccs[run] = baselineAcc

		// **Fractal Network**
		fractalNet := createFractalNetwork()

		// Pre-train a shared sub-network
		subNet := createXORSubNetwork()
		trainerSub := paragon.Trainer{Network: subNet, Config: trainCfg}
		trainerSub.TrainSimple(subTrainInputs, subTrainTargets, 20)

		// Assign pre-trained sub-network to hidden layer 1 neurons
		for y := 0; y < fractalNet.Layers[1].Height; y++ {
			for x := 0; x < fractalNet.Layers[1].Width; x++ {
				fractalNet.Layers[1].Neurons[y][x].Dimension = subNet
			}
		}

		// Assign pre-trained sub-network to output neuron
		fractalNet.Layers[2].Neurons[0][0].Dimension = subNet

		// Train the full fractal network
		trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}
		trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
		fractalAcc := computeAccuracy(fractalNet, valInputs, valTargets)
		fractalAccs[run] = fractalAcc

		// Print per-run results
		fmt.Printf("Baseline Accuracy: %.2f%%\n", baselineAcc*100)
		fmt.Printf("Fractal Accuracy: %.2f%%\n", fractalAcc*100)
	}

	// Compute and report statistics
	baselineAvg := mean(baselineAccs)
	baselineStd := stdDev(baselineAccs)
	fractalAvg := mean(fractalAccs)
	fractalStd := stdDev(fractalAccs)

	result := fmt.Sprintf("Experiment 59: Baseline Average Accuracy: %.2f%% ± %.2f\n", baselineAvg*100, baselineStd*100)
	result += fmt.Sprintf("Fractal Average Accuracy: %.2f%% ± %.2f\n\n", fractalAvg*100, fractalStd*100)
	if _, err := file.WriteString(result); err != nil {
		fmt.Printf("Error writing to file: %v\n", err)
	}
}
