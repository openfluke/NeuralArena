package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"paragon"
	"runtime"
	"sync"
	"time"
)

// MultiDimNetwork wraps a main network with multiple sub-networks
type MultiDimNetwork struct {
	MainNet *paragon.Network
	SubNets []*paragon.Network
	Scales  []float64
}

// Forward for MultiDimNetwork
func (mdn *MultiDimNetwork) Forward(inputs [][]float64) {
	// Run main network up to hidden layer
	mdn.MainNet.Forward(inputs)

	// Process hidden layer through all sub-networks in parallel
	hiddenLayer := mdn.MainNet.Layers[1]
	subInput := make([][]float64, hiddenLayer.Height)
	for y := 0; y < hiddenLayer.Height; y++ {
		subInput[y] = make([]float64, hiddenLayer.Width)
		for x := 0; x < hiddenLayer.Width; x++ {
			subInput[y][x] = hiddenLayer.Neurons[y][x].Value
		}
	}

	// Collect sub-network outputs
	subOutputs := make([][][]float64, len(mdn.SubNets))
	var wg sync.WaitGroup
	for i, subNet := range mdn.SubNets {
		wg.Add(1)
		go func(idx int, net *paragon.Network) {
			defer wg.Done()
			net.Forward(subInput)
			outLayer := net.Layers[net.OutputLayer]
			out := make([][]float64, outLayer.Height)
			for y := 0; y < outLayer.Height; y++ {
				out[y] = make([]float64, outLayer.Width)
				for x := 0; x < outLayer.Width; x++ {
					out[y][x] = outLayer.Neurons[y][x].Value
				}
			}
			subOutputs[idx] = out
		}(i, subNet)
	}
	wg.Wait()

	// Combine sub-network outputs with scales, normalize by number of dimensions
	numDims := float64(len(mdn.SubNets))
	for y := 0; y < hiddenLayer.Height; y++ {
		for x := 0; x < hiddenLayer.Width; x++ {
			combined := hiddenLayer.Neurons[y][x].Value
			for i, subOut := range subOutputs {
				combined += mdn.Scales[i] * subOut[y][x]
			}
			combined /= (1 + numDims) // Normalize to prevent scale explosion
			hiddenLayer.Neurons[y][x].Value = applyActivation(combined, hiddenLayer.Neurons[y][x].Activation)
		}
	}

	// Finish forward pass
	for l := 2; l <= mdn.MainNet.OutputLayer; l++ {
		currLayer := mdn.MainNet.Layers[l]
		for y := 0; y < currLayer.Height; y++ {
			for x := 0; x < currLayer.Width; x++ {
				neuron := currLayer.Neurons[y][x]
				sum := neuron.Bias
				for _, conn := range neuron.Inputs {
					srcNeuron := mdn.MainNet.Layers[conn.SourceLayer].Neurons[conn.SourceY][conn.SourceX]
					sum += srcNeuron.Value * conn.Weight
				}
				neuron.Value = applyActivation(sum, neuron.Activation)
			}
		}
	}

	if mdn.MainNet.Layers[mdn.MainNet.OutputLayer].Neurons[0][0].Activation == "softmax" {
		mdn.MainNet.ApplySoftmax()
	}
}

// Backward for MultiDimNetwork
func (mdn *MultiDimNetwork) Backward(targets [][]float64, learningRate float64) {
	// Compute output layer error terms
	numLayers := len(mdn.MainNet.Layers)
	errorTerms := make([][][]float64, numLayers)
	for l := range mdn.MainNet.Layers {
		errorTerms[l] = make([][]float64, mdn.MainNet.Layers[l].Height)
		for y := range errorTerms[l] {
			errorTerms[l][y] = make([]float64, mdn.MainNet.Layers[l].Width)
		}
	}

	outputLayer := mdn.MainNet.Layers[mdn.MainNet.OutputLayer]
	for y := 0; y < outputLayer.Height; y++ {
		for x := 0; x < outputLayer.Width; x++ {
			neuron := outputLayer.Neurons[y][x]
			errorTerms[mdn.MainNet.OutputLayer][y][x] = (targets[y][x] - neuron.Value) * activationDerivative(neuron.Value, neuron.Activation)
		}
	}

	// Backprop through main network from output to hidden layer
	for l := mdn.MainNet.OutputLayer; l > 1; l-- {
		currLayer := mdn.MainNet.Layers[l]
		prevLayer := mdn.MainNet.Layers[l-1]
		for y := 0; y < currLayer.Height; y++ {
			for x := 0; x < currLayer.Width; x++ {
				neuron := currLayer.Neurons[y][x]
				localErr := errorTerms[l][y][x]
				neuron.Bias += learningRate * localErr
				for i, conn := range neuron.Inputs {
					srcNeuron := prevLayer.Neurons[conn.SourceY][conn.SourceX]
					gradW := localErr * srcNeuron.Value
					if gradW > 5.0 {
						gradW = 5.0
					} else if gradW < -5.0 {
						gradW = -5.0
					}
					neuron.Inputs[i].Weight += learningRate * gradW
					errorTerms[l-1][conn.SourceY][conn.SourceX] += localErr * conn.Weight
				}
			}
		}
		for y := 0; y < prevLayer.Height; y++ {
			for x := 0; x < prevLayer.Width; x++ {
				errorTerms[l-1][y][x] *= activationDerivative(prevLayer.Neurons[y][x].Value, prevLayer.Neurons[y][x].Activation)
			}
		}
	}

	// Backprop through sub-networks
	hiddenLayer := mdn.MainNet.Layers[1]
	subInput := make([][]float64, hiddenLayer.Height)
	for y := 0; y < hiddenLayer.Height; y++ {
		subInput[y] = make([]float64, hiddenLayer.Width)
		for x := 0; x < hiddenLayer.Width; x++ {
			subInput[y][x] = hiddenLayer.Neurons[y][x].Value
		}
	}

	var wg sync.WaitGroup
	for i, subNet := range mdn.SubNets {
		wg.Add(1)
		go func(idx int, net *paragon.Network) {
			defer wg.Done()
			// Use hidden layer error terms as targets for sub-networks
			subTargets := make([][]float64, hiddenLayer.Height)
			for y := 0; y < hiddenLayer.Height; y++ {
				subTargets[y] = make([]float64, hiddenLayer.Width)
				for x := 0; x < hiddenLayer.Width; x++ {
					// Scale error terms by inverse normalization factor
					subTargets[y][x] = errorTerms[1][y][x] * (1 + float64(len(mdn.SubNets)))
				}
			}
			net.Backward(subTargets, learningRate)
			// Update scale with proper gradient
			scaleGradient := 0.0
			outLayer := net.Layers[net.OutputLayer]
			for y := 0; y < outLayer.Height; y++ {
				for x := 0; x < outLayer.Width; x++ {
					scaleGradient += subTargets[y][x] * outLayer.Neurons[y][x].Value
				}
			}
			mdn.Scales[idx] += learningRate * scaleGradient * 10 // Amplify scale learning
		}(i, subNet)
	}
	wg.Wait()

	// Backprop to input layer
	prevLayer := mdn.MainNet.Layers[0]
	for y := 0; y < hiddenLayer.Height; y++ {
		for x := 0; x < hiddenLayer.Width; x++ {
			neuron := hiddenLayer.Neurons[y][x]
			localErr := errorTerms[1][y][x]
			neuron.Bias += learningRate * localErr
			for i, conn := range neuron.Inputs {
				srcNeuron := prevLayer.Neurons[conn.SourceY][conn.SourceX]
				gradW := localErr * srcNeuron.Value
				if gradW > 5.0 {
					gradW = 5.0
				} else if gradW < -5.0 {
					gradW = -5.0
				}
				neuron.Inputs[i].Weight += learningRate * gradW
			}
		}
	}
}

// Train for MultiDimNetwork
func (mdn *MultiDimNetwork) Train(inputs [][][]float64, targets [][][]float64, epochs int, learningRate float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		perm := rand.Perm(len(inputs))
		for _, p := range perm {
			mdn.Forward(inputs[p])
			loss := mdn.MainNet.ComputeLoss(targets[p])
			if !math.IsNaN(loss) {
				totalLoss += loss
				mdn.Backward(targets[p], learningRate)
			}
		}
		fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, totalLoss/float64(len(inputs)))
	}
}

// ComputeAccuracy for MultiDimNetwork
func (mdn *MultiDimNetwork) ComputeAccuracy(inputs [][][]float64, targets [][][]float64) float64 {
	correct := 0
	for i := range inputs {
		mdn.Forward(inputs[i])
		outputLayer := mdn.MainNet.Layers[mdn.MainNet.OutputLayer]
		outputValues := make([]float64, outputLayer.Width)
		for x := 0; x < outputLayer.Width; x++ {
			outputValues[x] = outputLayer.Neurons[0][x].Value
		}
		pred := ArgMax(outputValues)
		label := ArgMax(targets[i][0])
		if pred == label {
			correct++
		}
	}
	return float64(correct) / float64(len(inputs))
}

// Experiment3 explores multi-dimensional layers with varying dimensions and scales
func Experiment3(filename string) {
	trainInputs, trainTargets := HarderGenerateXORGridDataset("xor_grid_dataset.txt", 500)

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
			Epochs:       200,
			LearningRate: 0.005,
			Debug:        false,
		},
	}
	fmt.Println("Training Baseline Network...")
	baselineTrainer.TrainSimple(trainInputs, trainTargets, 200)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, trainInputs, trainTargets)

	// Multi-Dimensional Trials
	numCores := runtime.NumCPU()
	numThreads := int(float64(numCores) * 0.8)
	fmt.Printf("Using %d threads (80%% of %d cores)\n", numThreads, numCores)

	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}

	trials := []struct {
		dims      int
		subHidden int
		scaleInit float64
	}{
		{1, 16, 0.5},
		{2, 16, 1.0},
		{3, 32, 0.5},
		{4, 16, 0.8},
		{5, 32, 1.0},
	}
	var wg sync.WaitGroup
	results := make(chan struct {
		dims   int
		acc    float64
		scales []float64
	}, len(trials))
	sem := make(chan struct{}, numThreads)

	for _, trial := range trials {
		wg.Add(1)
		sem <- struct{}{}
		go func(dims, subHidden int, scaleInit float64) {
			defer wg.Done()
			defer func() { <-sem }()

			mainNet := paragon.NewNetwork(baselineSizes, baselineActivations, baselineFullyConnected)
			mdn := &MultiDimNetwork{
				MainNet: mainNet,
				SubNets: make([]*paragon.Network, dims),
				Scales:  make([]float64, dims),
			}
			for d := 0; d < dims; d++ {
				subSizes := []struct{ Width, Height int }{
					{8, 1},         // Input matches hidden layer
					{subHidden, 1}, // Varying hidden layer size
					{8, 1},         // Output matches hidden layer
				}
				mdn.SubNets[d] = paragon.NewNetwork(subSizes, subActivations, subFullyConnected)
				mdn.Scales[d] = scaleInit + float64(d)*0.1
			}
			fmt.Printf("Training Multi-Dim Network with %d dimensions (hidden=%d, scaleInit=%.1f)...\n", dims, subHidden, scaleInit)
			mdn.Train(trainInputs, trainTargets, 200, 0.005)
			acc := mdn.ComputeAccuracy(trainInputs, trainTargets)
			results <- struct {
				dims   int
				acc    float64
				scales []float64
			}{dims, acc, mdn.Scales}
		}(trial.dims, trial.subHidden, trial.scaleInit)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	multiDimResults := make(map[int]struct {
		acc    float64
		scales []float64
	})
	for res := range results {
		multiDimResults[res.dims] = struct {
			acc    float64
			scales []float64
		}{res.acc, res.scales}
	}

	// Append results to file
	f, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Printf("Failed to open file %s: %v\n", filename, err)
		return
	}
	defer f.Close()

	result := fmt.Sprintf("Experiment 3 - %s\nBaseline Accuracy: %.2f%%\n",
		time.Now().Format("2006-01-02 15:04:05"), baselineAcc*100)
	for _, dims := range []int{1, 2, 3, 4, 5} {
		if res, ok := multiDimResults[dims]; ok {
			result += fmt.Sprintf("Multi-Dimension (%d dims) Accuracy: %.2f%%, Scales: %v\n", dims, res.acc*100, res.scales)
		}
	}
	result += "\n"
	if _, err := f.WriteString(result); err != nil {
		fmt.Printf("Failed to write to file %s: %v\n", filename, err)
	} else {
		fmt.Printf("Results appended to %s\n", filename)
	}
}

// applyActivation applies the specified activation function
func applyActivation(value float64, activation string) float64 {
	switch activation {
	case "relu":
		return math.Max(0, value)
	case "sigmoid":
		return 1 / (1 + math.Exp(-value))
	case "tanh":
		return math.Tanh(value)
	case "leaky_relu":
		if value > 0 {
			return value
		}
		return 0.01 * value
	case "elu":
		if value >= 0 {
			return value
		}
		return 1.0 * (math.Exp(value) - 1)
	case "linear":
		return value
	default:
		return value // Fallback to linear
	}
}

// activationDerivative computes the derivative of the activation function
func activationDerivative(value float64, activation string) float64 {
	switch activation {
	case "relu":
		if value > 0 {
			return 1
		}
		return 0
	case "sigmoid":
		sig := 1 / (1 + math.Exp(-value))
		return sig * (1 - sig)
	case "tanh":
		t := math.Tanh(value)
		return 1 - t*t
	case "leaky_relu":
		if value > 0 {
			return 1
		}
		return 0.01
	case "elu":
		if value >= 0 {
			return 1
		}
		return math.Exp(value)
	case "linear":
		return 1
	default:
		return 1 // Fallback to linear
	}
}

// ArgMax returns the index of the maximum value in the slice
func ArgMax(arr []float64) int {
	if len(arr) == 0 {
		return -1
	}
	maxIdx := 0
	for i := 1; i < len(arr); i++ {
		if arr[i] > arr[maxIdx] {
			maxIdx = i
		}
	}
	return maxIdx
}
