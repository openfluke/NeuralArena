package main

import (
	"fmt"
	"math/rand"
	"paragon" // <-- Make sure this import path matches your actual module import.
	"time"
)

// A tiny dataset (XOR) for demonstration
// We treat each sample as shape [height=1][width=2] for input, [height=1][width=1] for target.
var xorInputs = [][][]float64{
	{{0, 0}},
	{{0, 1}},
	{{1, 0}},
	{{1, 1}},
}
var xorTargets = [][][]float64{
	{{0}},
	{{1}},
	{{1}},
	{{0}},
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("=== 1) Partitioned Forward/Backward Demo ===")
	testPartitionTraining()

	fmt.Println("\n=== 2) Standard Gradient Training Demo ===")
	testGradientTraining()

	fmt.Println("\n=== 3) Partitioning + Dimensional Neurons Demo ===")
	testDimensionPartition()

	fmt.Println("\n=== 4) Partition + RL-Style Sub-Network Exploration ===")
	testDimensionPartitionReinforcementLearningTesting()
}

//------------------------------------------------------------
// 1) Partitioned Forward/Backward Training
//------------------------------------------------------------

func testPartitionTraining() {
	// Create a simple 3-layer network: 2 input neurons, 2 hidden, 1 output.
	layerSizes := []struct{ Width, Height int }{
		{Width: 2, Height: 1}, // input layer (2 wide, 1 high)
		{Width: 2, Height: 1}, // hidden layer
		{Width: 1, Height: 1}, // output layer
	}

	activations := []string{"linear", "relu", "sigmoid"}
	fullyConnected := []bool{false, true, true} // doesn't matter much for this minimal example

	net := paragon.NewNetwork(layerSizes, activations, fullyConnected)

	// We'll do 100 epochs, with 2 partitions in the hidden layer
	numEpochs := 100
	numTags := 2
	selectedTag := 0 // We'll start training partition #0

	for epoch := 0; epoch < numEpochs; epoch++ {
		var totalLoss float64
		for i, input := range xorInputs {
			target := xorTargets[i]

			// PARTITIONED FORWARD
			net.ForwardTagged(input, numTags, selectedTag)

			// Cross-entropy style loss from paragon
			loss := net.ComputeLoss(target)
			totalLoss += loss

			// PARTITIONED BACKWARD
			net.BackwardTagged(target, 0.1, numTags, selectedTag)
		}

		if (epoch+1)%10 == 0 {
			avgLoss := totalLoss / float64(len(xorInputs))
			fmt.Printf("Epoch %d, Tag=%d, Loss=%.4f\n", epoch+1, selectedTag, avgLoss)
		}

		// Switch partition halfway through training
		if epoch == (numEpochs / 2) {
			selectedTag = 1
		}
	}

	//------------------------------------------------------------
	// Evaluate final results
	fmt.Println("Final outputs after partitioned training:")
	for i, input := range xorInputs {
		net.ForwardTagged(input, numTags, 0) // or 1, or do both to combine
		out := net.GetOutput()
		fmt.Printf("Input=%v -> Output=%.4f (Target=%.1f)\n",
			input[0], out[0], xorTargets[i][0][0])
	}

	// Evaluate accuracy by always using ForwardTagged with, say, selectedTag=1
	acc := evaluateXORTagged(net, numTags, 1)
	fmt.Printf("Final Partitioned Training Accuracy: %.2f%%\n", acc)
}

//------------------------------------------------------------
// 2) Standard Gradient Training (no partitioning)
//------------------------------------------------------------

func testGradientTraining() {
	layerSizes := []struct{ Width, Height int }{
		{2, 1}, // input
		{2, 1}, // hidden
		{1, 1}, // output
	}
	activations := []string{"linear", "relu", "sigmoid"}
	fullyConnected := []bool{true, true, true}

	net := paragon.NewNetwork(layerSizes, activations, fullyConnected)

	numEpochs := 100
	learningRate := 0.1

	for epoch := 0; epoch < numEpochs; epoch++ {
		var totalLoss float64
		for i, input := range xorInputs {
			target := xorTargets[i]

			net.Forward(input)
			loss := net.ComputeLoss(target)
			totalLoss += loss

			net.Backward(target, learningRate)
		}
		if (epoch+1)%10 == 0 {
			fmt.Printf("Epoch %d, Loss=%.4f\n", epoch+1, totalLoss/float64(len(xorInputs)))
		}
	}

	// Evaluate final results
	fmt.Println("Final outputs after standard gradient training:")
	for i, input := range xorInputs {
		net.Forward(input)
		out := net.GetOutput()
		fmt.Printf("Input=%v -> Output=%.4f (Target=%.1f)\n",
			input[0], out[0], xorTargets[i][0][0])
	}

	// Evaluate accuracy
	acc := evaluateXORStandard(net)
	fmt.Printf("Final Standard Gradient Accuracy: %.2f%%\n", acc)
}

//------------------------------------------------------------
// 3) Partitioning + Dimensional Neurons
//------------------------------------------------------------

func testDimensionPartition() {
	// 1) Build the main "outer" network
	layerSizes := []struct{ Width, Height int }{
		{2, 1}, // input
		{2, 1}, // hidden
		{1, 1}, // output
	}
	acts := []string{"linear", "relu", "sigmoid"}
	fullyConn := []bool{true, true, true}
	mainNet := paragon.NewNetwork(layerSizes, acts, fullyConn)

	// 2) Build a small dimension sub-network & attach it to the second hidden neuron
	dimLayerSizes := []struct{ Width, Height int }{
		{1, 1}, // input
		{1, 1}, // output
	}
	dimActs := []string{"relu", "relu"}
	dimFull := []bool{true, true}
	subNet := paragon.NewNetwork(dimLayerSizes, dimActs, dimFull)

	hiddenLayer := &mainNet.Layers[1]         // layer index=1
	hiddenNeuron := hiddenLayer.Neurons[0][1] // row=0, col=1
	hiddenNeuron.Dimension = subNet

	fmt.Println("Attached a dimension sub-network to hidden neuron #1 in the mainNet.\n")

	// 3) We'll do partitioned training for 50 epochs on Tag=1
	numTags := 2
	selectedTag := 1
	epochs := 50
	for e := 0; e < epochs; e++ {
		var totalLoss float64
		for i, input := range xorInputs {
			target := xorTargets[i]

			mainNet.ForwardTagged(input, numTags, selectedTag)
			loss := mainNet.ComputeLoss(target)
			totalLoss += loss
			mainNet.BackwardTagged(target, 0.05, numTags, selectedTag)
		}
		if (e+1)%10 == 0 {
			fmt.Printf("Partition-Train Epoch %d, selectedTag=%d, avgLoss=%.4f\n",
				e+1, selectedTag, totalLoss/float64(len(xorInputs)))
		}
	}

	fmt.Println("\nOutputs after partitioning training with a dimension sub-network:")
	for i, input := range xorInputs {
		mainNet.ForwardTagged(input, numTags, selectedTag)
		out := mainNet.GetOutput()
		fmt.Printf("Input=%v -> Output=%.4f (Target=%.1f)\n",
			input[0], out[0], xorTargets[i][0][0])
	}

	// Evaluate accuracy by always using ForwardTagged with selectedTag=1
	acc := evaluateXORTagged(mainNet, numTags, selectedTag)
	fmt.Printf("Final Dimension + Partitioning Accuracy: %.2f%%\n", acc)

	// 4) Create a copy of mainNet and look for dimension neurons
	fmt.Println("\nNow let's clone the mainNet to copyNet, then do something on each dimension sub-network...")

	copyNet := paragon.NewNetwork(layerSizes, acts, fullyConn)
	err := mainNet.SaveToJSON("temp_mainnet.json")
	if err != nil {
		fmt.Println("Error saving mainNet:", err)
	}
	err = copyNet.LoadFromJSON("temp_mainnet.json")
	if err != nil {
		fmt.Println("Error loading into copyNet:", err)
	}
	fmt.Println("copyNet loaded from mainNet JSON.\n")

	// 5) Search for dimensional neurons in copyNet:
	for l := range copyNet.Layers {
		for y := range copyNet.Layers[l].Neurons {
			for x, neu := range copyNet.Layers[l].Neurons[y] {
				if neu.Dimension != nil {
					fmt.Printf("Found sub-network at layer=%d, neuron=(%d,%d). Let's 'train' or 'test' it.\n",
						l, y, x)
					// We'll just do a small forward pass for demonstration
					testInput := [][]float64{{0.5}}
					neu.Dimension.Forward(testInput)
					subOut := neu.Dimension.Layers[neu.Dimension.OutputLayer].Neurons[0][0].Value
					fmt.Printf("  Dimension sub-network out=%.4f\n", subOut)
				}
			}
		}
	}

	fmt.Println("\nFinished dimension-partitioning demonstration.")
}

//------------------------------------------------------------
// HELPER FUNCTIONS: Evaluate accuracy for XOR
//------------------------------------------------------------

// evaluateXORStandard does net.Forward(...) for each sample and
// applies a threshold of 0.5 to decide predicted class (0 or 1).
func evaluateXORStandard(net *paragon.Network) float64 {
	correct := 0
	for i, input := range xorInputs {
		net.Forward(input)
		outVal := net.GetOutput()[0]
		pred := 0.0
		if outVal >= 0.5 {
			pred = 1.0
		}
		if pred == xorTargets[i][0][0] {
			correct++
		}
	}
	return float64(correct) / float64(len(xorInputs)) * 100
}

// evaluateXORTagged does net.ForwardTagged(...) for each sample
// with the specified tag, using threshold=0.5 to decide predicted class.
func evaluateXORTagged(net *paragon.Network, numTags, selectedTag int) float64 {
	correct := 0
	for i, input := range xorInputs {
		net.ForwardTagged(input, numTags, selectedTag)
		outVal := net.GetOutput()[0]
		pred := 0.0
		if outVal >= 0.5 {
			pred = 1.0
		}
		if pred == xorTargets[i][0][0] {
			correct++
		}
	}
	return float64(correct) / float64(len(xorInputs)) * 100
}

//------------------------------------------------------------
// 4) Partition + "RL Style" Sub-Network Exploration
//------------------------------------------------------------

func testDimensionPartitionReinforcementLearningTesting() {
	//------------------------------------------------------------
	// (A) Build base main network
	//------------------------------------------------------------
	layerSizes := []struct{ Width, Height int }{
		{2, 1}, // input
		{2, 1}, // hidden
		{1, 1}, // output
	}
	acts := []string{"linear", "relu", "sigmoid"}
	fullyConn := []bool{true, true, true}
	mainNet := paragon.NewNetwork(layerSizes, acts, fullyConn)

	//------------------------------------------------------------
	// (B) We'll do a "search" over a small set of candidate sub-networks
	// to see which yields the best partitioned training accuracy if attached
	//------------------------------------------------------------
	type candidateShape struct {
		LayerSizes   []struct{ Width, Height int }
		Activations  []string
		FullyConnect []bool
	}
	candidates := []candidateShape{
		// Example 1: 1->1 sub-net (the simplest)
		{
			LayerSizes: []struct{ Width, Height int }{
				{1, 1}, // input
				{1, 1}, // output
			},
			Activations:  []string{"relu", "relu"},
			FullyConnect: []bool{true, true},
		},
		// Example 2: 1->2->1 sub-net
		{
			LayerSizes: []struct{ Width, Height int }{
				{1, 1},
				{2, 1},
				{1, 1},
			},
			Activations:  []string{"relu", "tanh", "relu"},
			FullyConnect: []bool{true, true, true},
		},
		// Example 3: 1->1->1 sub-net, but "sigmoid" to see if it's better
		{
			LayerSizes: []struct{ Width, Height int }{
				{1, 1},
				{1, 1},
				{1, 1},
			},
			Activations:  []string{"relu", "sigmoid", "relu"},
			FullyConnect: []bool{true, true, true},
		},
	}

	bestAcc := 0.0
	var bestSubNet *paragon.Network

	for i, cand := range candidates {
		fmt.Printf("Trying candidate sub-network %d with shape:\n", i+1)
		for l, sz := range cand.LayerSizes {
			fmt.Printf("  Layer %d: %dx%d, Act=%s\n", l, sz.Width, sz.Height, cand.Activations[l])
		}

		// 1) Make a fresh copy of mainNet
		tempNet := paragon.NewNetwork(layerSizes, acts, fullyConn)
		// We'll save the base mainNet (no dimension) to JSON so the copy is identical
		_ = mainNet.SaveToJSON("temp_mainnet.json")
		_ = tempNet.LoadFromJSON("temp_mainnet.json")

		// 2) Build the candidate dimension sub-network
		candSub := paragon.NewNetwork(cand.LayerSizes, cand.Activations, cand.FullyConnect)

		// 3) Attach to hidden neuron #1
		hiddenNeuron := tempNet.Layers[1].Neurons[0][1]
		hiddenNeuron.Dimension = candSub

		// 4) Partitioned train on Tag=1 for, say, 30 epochs
		numTags := 2
		selectedTag := 1
		epochs := 30
		for e := 0; e < epochs; e++ {
			for j, input := range xorInputs {
				target := xorTargets[j]
				tempNet.ForwardTagged(input, numTags, selectedTag)
				tempNet.BackwardTagged(target, 0.05, numTags, selectedTag)
			}
		}
		// Evaluate accuracy
		acc := evaluateXORTagged(tempNet, numTags, selectedTag)
		fmt.Printf("Candidate %d final accuracy=%.2f%%\n\n", i+1, acc)

		// 5) If best so far, keep track
		if acc > bestAcc {
			bestAcc = acc
			bestSubNet = candSub
		}
	}

	//------------------------------------------------------------
	// (C) Attach the best sub-network to the real mainNet, do a final training
	//------------------------------------------------------------
	if bestSubNet == nil {
		fmt.Println("No best sub-network found (shouldn't happen).")
		return
	}
	fmt.Printf("Best sub-network found with accuracy=%.2f%%\n", bestAcc)
	mainNet.Layers[1].Neurons[0][1].Dimension = bestSubNet
	fmt.Println("Attached best sub-network to mainNet at hidden neuron #1.\n")

	// Let's do a final 50 epochs partitioned training with that best sub-net attached.
	numTags := 2
	selectedTag := 1
	for e := 0; e < 50; e++ {
		for j, input := range xorInputs {
			target := xorTargets[j]
			mainNet.ForwardTagged(input, numTags, selectedTag)
			mainNet.BackwardTagged(target, 0.05, numTags, selectedTag)
		}
	}
	finalAcc := evaluateXORTagged(mainNet, numTags, selectedTag)
	fmt.Printf("Final RL-Style approach accuracy after more training: %.2f%%\n", finalAcc)
}
