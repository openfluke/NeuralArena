package main

import (
	"fmt"
	"paragon"
)

func main() {
	fmt.Println("Building partitioning nas")

	// Read and process cubes.csv
	cubes, _ := paragon.ReadCSV("cubes.csv")
	lstCubes, _ := paragon.Cleaner(cubes, []int{0}, []int{})
	lstLabelCubes, lstConvCubes, _ := paragon.Converter(lstCubes, []int{0})

	paragon.PrintTable(lstCubes)
	paragon.PrintTable(lstLabelCubes)
	paragon.PrintTable(lstConvCubes)

	// Read and process links.csv
	links, _ := paragon.ReadCSV("links.csv")
	lstLinks, _ := paragon.Cleaner(links, []int{0, 1}, []int{3, 4, 5, 6, 7})
	lstLabelLinks, lstConvLinks, _ := paragon.Converter(lstLinks, []int{0, 1, 2})

	paragon.PrintTable(lstLinks)
	paragon.PrintTable(lstLabelLinks)
	paragon.PrintTable(lstConvLinks)
	//basic(lstConvCubes)
	fullNas(lstConvCubes)
}

func basic(lstTmpData [][]string) {
	lstData, err := paragon.ConvertToFloat64(lstTmpData)
	if err != nil {
		fmt.Println("❌ Failed to convert data to float64:", err)
		return
	}

	layerSizes := []struct{ Width, Height int }{
		{96, 1}, // Input layer (8 cubes × 12 features)
		{5, 1},  // Hidden
		{12, 1}, // Output layer (reconstruct 12 features)
	}
	activations := []string{"linear", "relu", "linear"}
	fullyConnected := []bool{true, true, true}

	net := paragon.NewNetwork(layerSizes, activations, fullyConnected)

	var inputs [][][]float64
	var targets [][][]float64

	for i := 0; i < len(lstData); i++ {
		// Step 1: insert cube i's features at the right position
		prep := makePaddedVector(lstData[i], i, net)

		// Step 2: pad the rest with 0s
		padded := net.PadInputToFullSize(prep, 0.0)

		// Step 3: make sure target is 12 values
		target := lstData[i]
		if len(target) < 12 {
			tmp := make([]float64, 12)
			copy(tmp, target)
			target = tmp
		} else if len(target) > 12 {
			target = target[:12]
		}

		inputs = append(inputs, [][]float64{padded})
		targets = append(targets, [][]float64{target})
	}

	net.Train(inputs, targets, 20, 0.01, false) // Default earlyStopOnNegativeLoss=false
	fmt.Println("Training complete!")

	// ADHD Evaluation Metrics
	expected := make([]float64, len(targets)*12)
	actual := make([]float64, len(targets)*12)
	for i := range inputs {
		net.Forward(inputs[i])
		output := net.ExtractOutput()
		if len(output) == 0 || len(targets[i][0]) == 0 {
			fmt.Printf("⚠️ Skipping empty prediction at sample %d\n", i)
			continue
		}
		for j := 0; j < 12; j++ {
			expected[i*12+j] = targets[i][0][j]
			actual[i*12+j] = output[j]
		}
	}

	net.EvaluateModel(expected, actual)

	fmt.Printf("🧠 ADHD Score: %.2f\n", net.Performance.Score)
	fmt.Printf("📊 Deviation Buckets:\n")
	for bucket, stats := range net.Performance.Buckets {
		fmt.Printf(" - %s: %d samples\n", bucket, stats.Count)
	}
}

func fullNas(lstTmpData [][]string) {
	lstData, err := paragon.ConvertToFloat64(lstTmpData)
	if err != nil {
		fmt.Println("❌ Failed to convert data to float64:", err)
		return
	}

	// Define inputs and targets once, to be reused across all architectures
	var inputs [][][]float64
	var targets [][][]float64

	// Use a temporary network to determine input size (since input size is fixed at 96)
	tempLayerSizes := []struct{ Width, Height int }{
		{96, 1}, // Input
		{4, 1},  // Dummy hidden
		{12, 1}, // Output
	}
	activations := []string{"linear", "relu", "linear"}
	fullyConnected := []bool{true, true, true}
	tempNet := paragon.NewNetwork(tempLayerSizes, activations, fullyConnected)

	for i := 0; i < len(lstData); i++ {
		// Step 1: insert cube i's features at the right position
		prep := makePaddedVector(lstData[i], i, tempNet)

		// Step 2: pad the rest with 0s
		padded := tempNet.PadInputToFullSize(prep, 0.0)

		// Step 3: make sure target is 12 values
		target := lstData[i]
		if len(target) < 12 {
			tmp := make([]float64, 12)
			copy(tmp, target)
			target = tmp
		} else if len(target) > 12 {
			target = target[:12]
		}

		inputs = append(inputs, [][]float64{padded})
		targets = append(targets, [][]float64{target})
	}

	// Candidate hidden layer widths to test
	hiddenWidths := []int{4, 6, 8, 10, 12}

	var bestScore float64 = -1e9
	var bestWidth int
	var bestNetwork *paragon.Network

	for _, hiddenWidth := range hiddenWidths {
		fmt.Printf("\n🧪 Testing hidden width: %d\n", hiddenWidth)

		// Define architecture
		layerSizes := []struct{ Width, Height int }{
			{96, 1},          // Input
			{hiddenWidth, 1}, // Variable hidden
			{12, 1},          // Output (reconstruct 12 features)
		}

		net := paragon.NewNetwork(layerSizes, activations, fullyConnected)

		// Train using the precomputed inputs and targets
		net.Train(inputs, targets, 20, 0.001, true) // Default earlyStopOnNegativeLoss=false

		// Evaluate
		expected := make([]float64, len(targets)*12)
		actual := make([]float64, len(targets)*12)
		for i := range inputs {
			net.Forward(inputs[i])
			output := net.ExtractOutput()
			if len(output) == 0 || len(targets[i][0]) == 0 {
				continue
			}
			for j := 0; j < 12; j++ {
				expected[i*12+j] = targets[i][0][j]
				actual[i*12+j] = output[j]
			}
		}

		net.EvaluateModel(expected, actual)

		score := net.Performance.Score
		fmt.Printf("🔎 ADHD Score (width %d): %.2f\n", hiddenWidth, score)

		if score > bestScore {
			bestScore = score
			bestWidth = hiddenWidth
			bestNetwork = net
		}
	}

	// 🎉 Report the winner
	fmt.Printf("\n🏆 Best configuration: hidden width = %d\n", bestWidth)
	fmt.Printf("🧠 Best ADHD Score: %.2f\n", bestScore)
	fmt.Printf("📊 Final Deviation Buckets:\n")
	for bucket, stats := range bestNetwork.Performance.Buckets {
		fmt.Printf(" - %s: %d samples\n", bucket, stats.Count)
	}

	// Continue with NormalNASLayerWiseGrowing to reach ADHD score of 95 using full dataset
	fmt.Println("\n🚀 Starting NormalNASLayerWiseGrowing to reach ADHD score of 95")

	// Parameters for NAS
	minNeurons := 16 // Minimum neurons per new layer
	maxNeurons := 32 // Maximum neurons per new layer
	targetMetric := "score"
	targetValue := 95.0             // Target ADHD score
	activation := "relu"            // Consistent with previous hidden layer activation
	learningRate := 0.01            // Reduced for stability
	epochs := 20                    // Match initial training
	maxIterations := 5              // Limit to 5 iterations
	earlyStopOnNegativeLoss := true // Enable early stopping on negative loss

	// Use the best network and full dataset (inputs and targets)
	err = bestNetwork.NormalNASLayerWiseGrowing(
		inputs,  // Full padded inputs
		targets, // Full targets
		minNeurons,
		maxNeurons,
		activation,
		targetMetric,
		targetValue,
		learningRate,
		epochs,
		maxIterations,
		earlyStopOnNegativeLoss,
	)
	if err != nil {
		fmt.Printf("❌ Error in NormalNASLayerWiseGrowing: %v\n", err)
		return
	}

	// Final evaluation on the full dataset
	expected := make([]float64, len(targets)*12)
	actual := make([]float64, len(targets)*12)
	for i := range inputs {
		bestNetwork.Forward(inputs[i])
		output := bestNetwork.ExtractOutput()
		if len(output) == 0 || len(targets[i][0]) == 0 {
			fmt.Printf("⚠️ Skipping empty prediction at sample %d\n", i)
			continue
		}
		for j := 0; j < 12; j++ {
			expected[i*12+j] = targets[i][0][j]
			actual[i*12+j] = output[j]
		}
	}

	bestNetwork.EvaluateModel(expected, actual)

	// Report final results
	fmt.Printf("\n🌟 Final NAS Results:\n")
	fmt.Printf("🧠 Final ADHD Score: %.2f\n", bestNetwork.Performance.Score)
	fmt.Printf("📊 Final Deviation Buckets:\n")
	for bucket, stats := range bestNetwork.Performance.Buckets {
		fmt.Printf(" - %s: %d samples\n", bucket, stats.Count)
	}

	// Log the final network architecture
	fmt.Printf("🏛️ Final Network Architecture:\n")
	for i, layer := range bestNetwork.Layers {
		fmt.Printf("  Layer %d: %dx%d (%s)\n", i, layer.Width, layer.Height, layer.Neurons[0][0].Activation)
	}
}

func average(vec []float64) float64 {
	if len(vec) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range vec {
		sum += v
	}
	return sum / float64(len(vec))
}

// makePaddedVector inserts a cube's feature vector at the right index slot
// in the full input vector and returns the result.
func makePaddedVector(cube []float64, cubeIndex int, net *paragon.Network) []float64 {
	inputLayer := net.Layers[net.InputLayer]
	totalInputs := inputLayer.Width * inputLayer.Height
	cubeSize := len(cube)

	// Create a zeroed vector
	result := make([]float64, totalInputs)

	// Insert cube at its position
	start := cubeIndex * cubeSize
	if start+cubeSize > totalInputs {
		panic(fmt.Sprintf("cubeIndex %d with size %d exceeds input size %d", cubeIndex, cubeSize, totalInputs))
	}

	copy(result[start:start+cubeSize], cube)
	return result
}
