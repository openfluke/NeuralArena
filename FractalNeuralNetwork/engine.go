package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"paragon"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	// Open file for appending (create if doesn't exist)
	file, err := os.OpenFile("experiment_results.txt", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Printf("Error opening results file: %v\n", err)
		return
	}
	defer file.Close()

	// Write header with timestamp
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	header := fmt.Sprintf("\n=== Experiment Run: %s ===\n", timestamp)
	if _, err := file.WriteString(header); err != nil {
		fmt.Printf("Error writing to file: %v\n", err)
	}

	// Run experiments and append results
	//experimentBaselineVsFractal(file)
	//experimentFractalPretraining(file)
	//experimentMultiDimensional(file)

	// New experiments
	//experimentHierarchicalXOR(file)
	//experimentMultiScalePatterns(file)
	//experimentTemporalDependencies(file)
	//experimentSparseFeatureDetection(file)
	//experimentNonLinearComposition(file)

	// New experiments
	//experimentDeepHierarchy(file)
	//experimentPolynomialExpansion(file)
	//experimentMultiModalFusion(file)
	//experimentRecursivePatterns(file)
	//experimentContextualAmbiguity(file)

	// New experiments with deeper fractal structures
	//experimentNestedXOR(file)
	//experimentPatternHierarchy(file)
	//experimentDynamicScaling(file)
	//experimentFeatureComposition(file)
	//experimentLongRangeContext(file)
	//experimentDeepNestedXOR(file)

	// New experiments
	//experimentRecursiveFunction(file)
	//experimentMultiLevelNoise(file)
	//experimentHierarchicalRules(file)
	//experimentFractalPatternCompletion(file)

	// New complex experiments
	//experimentDeepFractalXOR(file)
	//experimentNestedPolynomialChaos(file)
	//experimentMultiScaleTemporalFractals(file)
	//experimentRecursiveContextualSwitching(file)
	//experimentFractalNoiseInterpolation(file)

	// Add the new experiment
	//experimentHierarchicalBinary(file)
	//experimentMultiScaleSignals(file)
	//experimentContextualSequences(file)

	// Add the new experiment
	experimentEnhancedDeepHierarchy(file)
	//experimentAdvancedMultiModalFusion(file)
	//experimentComplexFeatureComposition(file)
	experimentExtendedLongRangeContext(file)

	//experimentUltraComplexFeatureComposition(file)
}

// Experiment 1: Baseline vs Fractal Network
func experimentBaselineVsFractal(file *os.File) {
	fmt.Println("\n=== Experiment 1: Baseline vs Fractal Network ===")

	layerSizes := []struct{ Width, Height int }{
		{Width: 16, Height: 1},
		{Width: 32, Height: 1},
		{Width: 2, Height: 1},
	}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)

	subLayerSizes := []struct{ Width, Height int }{
		{Width: 1, Height: 1},
		{Width: 4, Height: 1},
		{Width: 1, Height: 1},
	}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: true, InitMethod: "xavier"}
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)

	trainInputs, trainTargets := generateSyntheticDataset(1000)
	valInputs, valTargets := generateSyntheticDataset(200)

	trainCfg := paragon.TrainConfig{
		Epochs:       200,
		LearningRate: 0.01,
		EarlyStopAcc: 0.90,
		Debug:        false,
	}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	// Write results to file
	result := fmt.Sprintf("Experiment 1: Baseline vs Fractal\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	if _, err := file.WriteString(result); err != nil {
		fmt.Printf("Error writing to file: %v\n", err)
	}
}

// Experiment 2: Pretraining fractal dimensions
func experimentFractalPretraining(file *os.File) {
	fmt.Println("\n=== Experiment 2: Fractal Pretraining ===")

	layerSizes := []struct{ Width, Height int }{
		{Width: 16, Height: 1},
		{Width: 32, Height: 1},
		{Width: 2, Height: 1},
	}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}

	subLayerSizes := []struct{ Width, Height int }{
		{Width: 1, Height: 1},
		{Width: 4, Height: 1},
		{Width: 1, Height: 1},
	}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: true, InitMethod: "xavier"}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)

	trainInputs, trainTargets := generateSyntheticDataset(1000)
	valInputs, valTargets := generateSyntheticDataset(200)

	trainCfg := paragon.TrainConfig{
		Epochs:       100,
		LearningRate: 0.01,
	}

	fmt.Println("Pretraining fractal sub-networks...")
	subInputs := make([][][]float64, len(trainInputs))
	subTargets := make([][][]float64, len(trainTargets))
	for i := range trainInputs {
		subInputs[i] = make([][]float64, 1)
		subInputs[i][0] = make([]float64, 1)
		subTargets[i] = make([][]float64, 1)
		subTargets[i][0] = make([]float64, 1)
		sum := 0.0
		for _, v := range trainInputs[i][0] {
			sum += v
		}
		subInputs[i][0][0] = sum / 16.0
		subTargets[i][0][0] = float64(paragon.ArgMax(trainTargets[i][0]))
	}

	subNet := fractalNet.Layers[1].Neurons[0][0].Dimension
	subTrainer := paragon.Trainer{Network: subNet, Config: trainCfg}
	subTrainer.TrainSimple(subInputs, subTargets, trainCfg.Epochs)

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training pretrained fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal (pretrained) accuracy: %.2f%%\n", fractalAcc*100)

	// Write results to file
	result := fmt.Sprintf("Experiment 2: Fractal Pretraining\nBaseline Accuracy: %.2f%%\nPretrained Fractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	if _, err := file.WriteString(result); err != nil {
		fmt.Printf("Error writing to file: %v\n", err)
	}
}

// Experiment 3: Multi-dimensional network
func experimentMultiDimensional(file *os.File) {
	fmt.Println("\n=== Experiment 3: Multi-Dimensional Network (3 Levels Deep) ===")

	layerSizes := []struct{ Width, Height int }{
		{Width: 16, Height: 1},
		{Width: 8, Height: 1},
		{Width: 2, Height: 1},
	}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes1 := []struct{ Width, Height int }{
		{Width: 1, Height: 1},
		{Width: 4, Height: 1},
		{Width: 1, Height: 1},
	}
	subActivations1 := []string{"linear", "relu", "linear"}
	subFullyConnected1 := []bool{true, true, true}

	subLayerSizes2 := []struct{ Width, Height int }{
		{Width: 1, Height: 1},
		{Width: 2, Height: 1},
		{Width: 1, Height: 1},
	}
	subActivations2 := []string{"linear", "relu", "linear"}
	subFullyConnected2 := []bool{true, true, true}

	multiNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	multiNet.SetLayerDimension(1, subLayerSizes1, subActivations1, subFullyConnected1, opts)

	for y := 0; y < multiNet.Layers[1].Height; y++ {
		for x := 0; x < multiNet.Layers[1].Width; x++ {
			subNet := multiNet.Layers[1].Neurons[y][x].Dimension
			subNet.SetLayerDimension(1, subLayerSizes2, subActivations2, subFullyConnected2, opts)
		}
	}

	trainInputs, trainTargets := generateComplexDataset(1000)
	valInputs, valTargets := generateComplexDataset(200)

	trainCfg := paragon.TrainConfig{
		Epochs:       200,
		LearningRate: 0.01,
	}

	trainer := paragon.Trainer{Network: multiNet, Config: trainCfg}
	fmt.Println("Training multi-dimensional network...")
	trainer.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	acc := paragon.ComputeAccuracy(multiNet, valInputs, valTargets)
	fmt.Printf("Multi-dimensional accuracy: %.2f%%\n", acc*100)

	// Write results to file
	result := fmt.Sprintf("Experiment 3: Multi-Dimensional (3 Levels)\nAccuracy: %.2f%%\n\n", acc*100)
	if _, err := file.WriteString(result); err != nil {
		fmt.Printf("Error writing to file: %v\n", err)
	}
}

// Experiment 4: Hierarchical XOR Problem
func experimentHierarchicalXOR(file *os.File) {
	fmt.Println("\n=== Experiment 4: Hierarchical XOR ===")

	layerSizes := []struct{ Width, Height int }{{16, 1}, {32, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {4, 1}, {1, 1}}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)

	trainInputs, trainTargets := generateHierarchicalXOR(1000)
	valInputs, valTargets := generateHierarchicalXOR(200)

	trainCfg := paragon.TrainConfig{Epochs: 300, LearningRate: 0.005}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 4: Hierarchical XOR\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

// Experiment 5: Multi-Scale Patterns
func experimentMultiScalePatterns(file *os.File) {
	fmt.Println("\n=== Experiment 5: Multi-Scale Patterns ===")

	layerSizes := []struct{ Width, Height int }{{32, 1}, {64, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {8, 1}, {1, 1}}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)

	trainInputs, trainTargets := generateMultiScalePatterns(1000)
	valInputs, valTargets := generateMultiScalePatterns(200)

	trainCfg := paragon.TrainConfig{Epochs: 300, LearningRate: 0.005}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 5: Multi-Scale Patterns\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

// Experiment 6: Temporal Dependencies
func experimentTemporalDependencies(file *os.File) {
	fmt.Println("\n=== Experiment 6: Temporal Dependencies ===")

	layerSizes := []struct{ Width, Height int }{{20, 1}, {40, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {6, 1}, {1, 1}}
	subActivations := []string{"linear", "tanh", "linear"}
	subFullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)

	trainInputs, trainTargets := generateTemporalDependencies(1000)
	valInputs, valTargets := generateTemporalDependencies(200)

	trainCfg := paragon.TrainConfig{Epochs: 300, LearningRate: 0.005}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 6: Temporal Dependencies\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

// Experiment 7: Sparse Feature Detection
func experimentSparseFeatureDetection(file *os.File) {
	fmt.Println("\n=== Experiment 7: Sparse Feature Detection ===")

	layerSizes := []struct{ Width, Height int }{{64, 1}, {128, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {10, 1}, {1, 1}}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)

	trainInputs, trainTargets := generateSparseFeatures(1000)
	valInputs, valTargets := generateSparseFeatures(200)

	trainCfg := paragon.TrainConfig{Epochs: 300, LearningRate: 0.005}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 7: Sparse Feature Detection\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

// Experiment 8: Non-Linear Composition
func experimentNonLinearComposition(file *os.File) {
	fmt.Println("\n=== Experiment 8: Non-Linear Composition ===")

	layerSizes := []struct{ Width, Height int }{{16, 1}, {32, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {6, 1}, {1, 1}}
	subActivations := []string{"linear", "tanh", "linear"}
	subFullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)

	trainInputs, trainTargets := generateNonLinearComposition(1000)
	valInputs, valTargets := generateNonLinearComposition(200)

	trainCfg := paragon.TrainConfig{Epochs: 300, LearningRate: 0.005}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 8: Non-Linear Composition\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

// Data Generation Functions

func generateHierarchicalXOR(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 16)
		for j := 0; j < 16; j++ {
			inputs[i][0][j] = float64(rand.Intn(2))
		}
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		// Hierarchical XOR: XOR of groups of 4 bits
		xor1 := int(inputs[i][0][0]) ^ int(inputs[i][0][1]) ^ int(inputs[i][0][2]) ^ int(inputs[i][0][3])
		xor2 := int(inputs[i][0][4]) ^ int(inputs[i][0][5]) ^ int(inputs[i][0][6]) ^ int(inputs[i][0][7])
		final := xor1 ^ xor2
		if final == 1 {
			targets[i][0][0] = 1
		} else {
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

func generateMultiScalePatterns(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 32)
		pattern := rand.Intn(3)
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		switch pattern {
		case 0: // Large-scale wave
			for j := 0; j < 32; j++ {
				inputs[i][0][j] = math.Sin(float64(j) / 4.0)
			}
			targets[i][0][0] = 1
		case 1: // Medium-scale wave
			for j := 0; j < 32; j++ {
				inputs[i][0][j] = math.Sin(float64(j) / 2.0)
			}
			targets[i][0][1] = 1
		case 2: // Small-scale noise
			for j := 0; j < 32; j++ {
				inputs[i][0][j] = rand.Float64() * 0.2
			}
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

func generateTemporalDependencies(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 20)
		base := rand.Float64()
		for j := 0; j < 20; j++ {
			inputs[i][0][j] = base + math.Sin(float64(j)/3.0) + rand.Float64()*0.1
		}
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		// Classify based on sum of values at specific lags
		sum := inputs[i][0][0] + inputs[i][0][5] + inputs[i][0][10]
		if sum > 1.5 {
			targets[i][0][0] = 1
		} else {
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

func generateSparseFeatures(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 64)
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		featurePos := rand.Intn(64)
		for j := 0; j < 64; j++ {
			if j == featurePos {
				inputs[i][0][j] = rand.Float64()*0.9 + 0.1
			} else {
				inputs[i][0][j] = rand.Float64() * 0.1
			}
		}
		if inputs[i][0][featurePos] > 0.5 {
			targets[i][0][0] = 1
		} else {
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

func generateNonLinearComposition(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 16)
		for j := 0; j < 16; j++ {
			inputs[i][0][j] = rand.Float64()
		}
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		// Non-linear combination: product of pairs
		prod := (inputs[i][0][0] * inputs[i][0][1]) + (inputs[i][0][2] * inputs[i][0][3])
		if prod > 0.25 {
			targets[i][0][0] = 1
		} else {
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

// Experiment 9: Deep Hierarchy
func experimentDeepHierarchy(file *os.File) {
	fmt.Println("\n=== Experiment 9: Deep Hierarchy ===")

	layerSizes := []struct{ Width, Height int }{{16, 1}, {32, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {4, 1}, {1, 1}}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
	for y := 0; y < fractalNet.Layers[1].Height; y++ {
		for x := 0; x < fractalNet.Layers[1].Width; x++ {
			fractalNet.Layers[1].Neurons[y][x].Dimension.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
		}
	}

	trainInputs, trainTargets := generateDeepHierarchy(1000)
	valInputs, valTargets := generateDeepHierarchy(200)

	trainCfg := paragon.TrainConfig{Epochs: 500, LearningRate: 0.001}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 9: Deep Hierarchy\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

// Experiment 10: Polynomial Expansion
func experimentPolynomialExpansion(file *os.File) {
	fmt.Println("\n=== Experiment 10: Polynomial Expansion ===")

	layerSizes := []struct{ Width, Height int }{{8, 1}, {16, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {4, 1}, {1, 1}}
	subActivations := []string{"linear", "tanh", "linear"}
	subFullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)

	trainInputs, trainTargets := generatePolynomialExpansion(1000)
	valInputs, valTargets := generatePolynomialExpansion(200)

	trainCfg := paragon.TrainConfig{Epochs: 500, LearningRate: 0.001}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 10: Polynomial Expansion\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

// Experiment 11: Multi-Modal Fusion
func experimentMultiModalFusion(file *os.File) {
	fmt.Println("\n=== Experiment 11: Multi-Modal Fusion ===")

	layerSizes := []struct{ Width, Height int }{{32, 1}, {64, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {6, 1}, {1, 1}}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)

	trainInputs, trainTargets := generateMultiModalFusion(1000)
	valInputs, valTargets := generateMultiModalFusion(200)

	trainCfg := paragon.TrainConfig{Epochs: 500, LearningRate: 0.001}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 11: Multi-Modal Fusion\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

// Experiment 12: Recursive Patterns
func experimentRecursivePatterns(file *os.File) {
	fmt.Println("\n=== Experiment 12: Recursive Patterns ===")

	layerSizes := []struct{ Width, Height int }{{16, 1}, {32, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {4, 1}, {1, 1}}
	subActivations := []string{"linear", "tanh", "linear"}
	subFullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)

	trainInputs, trainTargets := generateRecursivePatterns(1000)
	valInputs, valTargets := generateRecursivePatterns(200)

	trainCfg := paragon.TrainConfig{Epochs: 500, LearningRate: 0.001}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 12: Recursive Patterns\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

// Experiment 13: Contextual Ambiguity
func experimentContextualAmbiguity(file *os.File) {
	fmt.Println("\n=== Experiment 13: Contextual Ambiguity ===")

	layerSizes := []struct{ Width, Height int }{{24, 1}, {48, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {6, 1}, {1, 1}}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)

	trainInputs, trainTargets := generateContextualAmbiguity(1000)
	valInputs, valTargets := generateContextualAmbiguity(200)

	trainCfg := paragon.TrainConfig{Epochs: 500, LearningRate: 0.001}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 13: Contextual Ambiguity\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

// New Data Generation Functions

func generateDeepHierarchy(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 16)
		for j := 0; j < 16; j++ {
			inputs[i][0][j] = float64(rand.Intn(2))
		}
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		// Three-level hierarchy: XOR of XORs
		l1 := []int{int(inputs[i][0][0]) ^ int(inputs[i][0][1]), int(inputs[i][0][2]) ^ int(inputs[i][0][3])}
		l2 := l1[0] ^ l1[1]
		l3 := l2 ^ (int(inputs[i][0][4]) ^ int(inputs[i][0][5]))
		if l3 == 1 {
			targets[i][0][0] = 1
		} else {
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

func generatePolynomialExpansion(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 8)
		for j := 0; j < 8; j++ {
			inputs[i][0][j] = rand.Float64()
		}
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		// Polynomial: x0^2 + 2*x1*x2 + x3^3
		val := math.Pow(inputs[i][0][0], 2) + 2*inputs[i][0][1]*inputs[i][0][2] + math.Pow(inputs[i][0][3], 3)
		if val > 1.0 {
			targets[i][0][0] = 1
		} else {
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

func generateMultiModalFusion(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 32)
		// Two modalities: first 16 (wave), last 16 (steps)
		for j := 0; j < 16; j++ {
			inputs[i][0][j] = math.Sin(float64(j)/2.0) + rand.Float64()*0.1
			inputs[i][0][j+16] = float64(rand.Intn(3))
		}
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		waveSum := 0.0
		stepSum := 0.0
		for j := 0; j < 16; j++ {
			waveSum += inputs[i][0][j]
			stepSum += inputs[i][0][j+16]
		}
		if waveSum > 0 && stepSum > 20 {
			targets[i][0][0] = 1
		} else {
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

func generateRecursivePatterns(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 16)
		base := rand.Float64()
		for j := 0; j < 16; j++ {
			inputs[i][0][j] = base
			base = math.Sin(base) + rand.Float64()*0.1
		}
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		sum := inputs[i][0][0] + inputs[i][0][3] + inputs[i][0][6]
		if sum > 1.5 {
			targets[i][0][0] = 1
		} else {
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

func generateContextualAmbiguity(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 24)
		context := rand.Intn(2)
		for j := 0; j < 24; j++ {
			inputs[i][0][j] = rand.Float64()
		}
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		// Context changes interpretation of middle segment
		midSum := inputs[i][0][8] + inputs[i][0][9] + inputs[i][0][10]
		if context == 0 {
			if midSum > 1.5 {
				targets[i][0][0] = 1
			} else {
				targets[i][0][1] = 1
			}
		} else {
			if midSum < 1.5 {
				targets[i][0][0] = 1
			} else {
				targets[i][0][1] = 1
			}
		}
	}
	return inputs, targets
}

// Experiment 14: Nested XOR
func experimentNestedXOR(file *os.File) {
	fmt.Println("\n=== Experiment 14: Nested XOR ===")

	layerSizes := []struct{ Width, Height int }{{16, 1}, {32, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {4, 1}, {1, 1}}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
	for y := 0; y < fractalNet.Layers[1].Height; y++ {
		for x := 0; x < fractalNet.Layers[1].Width; x++ {
			fractalNet.Layers[1].Neurons[y][x].Dimension.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
		}
	}

	trainInputs, trainTargets := generateNestedXOR(1000)
	valInputs, valTargets := generateNestedXOR(200)

	trainCfg := paragon.TrainConfig{Epochs: 1000, LearningRate: 0.0005}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 14: Nested XOR\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

// Experiment 15: Pattern Hierarchy
func experimentPatternHierarchy(file *os.File) {
	fmt.Println("\n=== Experiment 15: Pattern Hierarchy ===")

	layerSizes := []struct{ Width, Height int }{{32, 1}, {64, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {6, 1}, {1, 1}}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
	for y := 0; y < fractalNet.Layers[1].Height; y++ {
		for x := 0; x < fractalNet.Layers[1].Width; x++ {
			fractalNet.Layers[1].Neurons[y][x].Dimension.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
		}
	}

	trainInputs, trainTargets := generatePatternHierarchy(1000)
	valInputs, valTargets := generatePatternHierarchy(200)

	trainCfg := paragon.TrainConfig{Epochs: 1000, LearningRate: 0.0005}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 15: Pattern Hierarchy\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

// Experiment 16: Dynamic Scaling
func experimentDynamicScaling(file *os.File) {
	fmt.Println("\n=== Experiment 16: Dynamic Scaling ===")

	layerSizes := []struct{ Width, Height int }{{20, 1}, {40, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {5, 1}, {1, 1}}
	subActivations := []string{"linear", "tanh", "linear"}
	subFullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
	for y := 0; y < fractalNet.Layers[1].Height; y++ {
		for x := 0; x < fractalNet.Layers[1].Width; x++ {
			fractalNet.Layers[1].Neurons[y][x].Dimension.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
		}
	}

	trainInputs, trainTargets := generateDynamicScaling(1000)
	valInputs, valTargets := generateDynamicScaling(200)

	trainCfg := paragon.TrainConfig{Epochs: 1000, LearningRate: 0.0005}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 16: Dynamic Scaling\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

// Experiment 17: Feature Composition
func experimentFeatureComposition(file *os.File) {
	fmt.Println("\n=== Experiment 17: Feature Composition ===")

	layerSizes := []struct{ Width, Height int }{{24, 1}, {48, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {6, 1}, {1, 1}}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
	for y := 0; y < fractalNet.Layers[1].Height; y++ {
		for x := 0; x < fractalNet.Layers[1].Width; x++ {
			fractalNet.Layers[1].Neurons[y][x].Dimension.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
		}
	}

	trainInputs, trainTargets := generateFeatureComposition(1000)
	valInputs, valTargets := generateFeatureComposition(200)

	trainCfg := paragon.TrainConfig{Epochs: 1000, LearningRate: 0.0005}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 17: Feature Composition\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

// Experiment 18: Long-Range Context
func experimentLongRangeContext(file *os.File) {
	fmt.Println("\n=== Experiment 18: Long-Range Context ===")

	layerSizes := []struct{ Width, Height int }{{32, 1}, {64, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {8, 1}, {1, 1}}
	subActivations := []string{"linear", "tanh", "linear"}
	subFullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
	for y := 0; y < fractalNet.Layers[1].Height; y++ {
		for x := 0; x < fractalNet.Layers[1].Width; x++ {
			fractalNet.Layers[1].Neurons[y][x].Dimension.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
		}
	}

	trainInputs, trainTargets := generateLongRangeContext(1000)
	valInputs, valTargets := generateLongRangeContext(200)

	trainCfg := paragon.TrainConfig{Epochs: 1000, LearningRate: 0.0005}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 18: Long-Range Context\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

// New Data Generation Functions

func generateNestedXOR(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 16)
		for j := 0; j < 16; j++ {
			inputs[i][0][j] = float64(rand.Intn(2))
		}
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		// Nested XOR: XOR of four groups, then XOR of results
		g1 := int(inputs[i][0][0]) ^ int(inputs[i][0][1]) ^ int(inputs[i][0][2]) ^ int(inputs[i][0][3])
		g2 := int(inputs[i][0][4]) ^ int(inputs[i][0][5]) ^ int(inputs[i][0][6]) ^ int(inputs[i][0][7])
		g3 := int(inputs[i][0][8]) ^ int(inputs[i][0][9]) ^ int(inputs[i][0][10]) ^ int(inputs[i][0][11])
		g4 := int(inputs[i][0][12]) ^ int(inputs[i][0][13]) ^ int(inputs[i][0][14]) ^ int(inputs[i][0][15])
		final := (g1 ^ g2) ^ (g3 ^ g4)
		if final == 1 {
			targets[i][0][0] = 1
		} else {
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

func generatePatternHierarchy(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 32)
		level := rand.Intn(3)
		base := rand.Float64()
		for j := 0; j < 32; j++ {
			switch level {
			case 0: // High-level pattern
				inputs[i][0][j] = base + math.Sin(float64(j)/8.0)
			case 1: // Mid-level pattern
				inputs[i][0][j] = base + math.Sin(float64(j)/4.0) + rand.Float64()*0.2
			case 2: // Low-level pattern
				inputs[i][0][j] = base + math.Sin(float64(j)) + rand.Float64()*0.5
			}
		}
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		sum := 0.0
		for j := 0; j < 32; j += 8 {
			sum += inputs[i][0][j]
		}
		if sum > 2.0 {
			targets[i][0][0] = 1
		} else {
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

func generateDynamicScaling(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 20)
		scale := rand.Float64() * 10
		for j := 0; j < 20; j++ {
			inputs[i][0][j] = math.Pow(float64(j)/scale, 2) + rand.Float64()*0.1
		}
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		sum := inputs[i][0][0] + inputs[i][0][10] + inputs[i][0][19]
		if sum > 1.5 {
			targets[i][0][0] = 1
		} else {
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

func generateFeatureComposition(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 24)
		for j := 0; j < 24; j++ {
			inputs[i][0][j] = rand.Float64()
		}
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		// Complex composition: (x0*x1 + x2) * (x3 + x4^2)
		f1 := inputs[i][0][0]*inputs[i][0][1] + inputs[i][0][2]
		f2 := inputs[i][0][3] + math.Pow(inputs[i][0][4], 2)
		val := f1 * f2
		if val > 0.5 {
			targets[i][0][0] = 1
		} else {
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

func generateLongRangeContext(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 32)
		context := rand.Intn(2)
		for j := 0; j < 32; j++ {
			inputs[i][0][j] = rand.Float64()
		}
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		// Context at start affects end
		contextSum := inputs[i][0][0] + inputs[i][0][1]
		endSum := inputs[i][0][30] + inputs[i][0][31]
		if context == 0 {
			if endSum > contextSum {
				targets[i][0][0] = 1
			} else {
				targets[i][0][1] = 1
			}
		} else {
			if endSum < contextSum {
				targets[i][0][0] = 1
			} else {
				targets[i][0][1] = 1
			}
		}
	}
	return inputs, targets
}

// Original Data Generation Functions (unchanged, included for completeness)
func generateSyntheticDataset(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 16)
		for j := 0; j < 16; j++ {
			inputs[i][0][j] = rand.Float64()
		}
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		if rand.Float64() > 0.5 {
			targets[i][0][0] = 1
		} else {
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

func generateComplexDataset(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 16)
		pattern := rand.Intn(4)
		sum := 0.0
		switch pattern {
		case 0:
			for j := 0; j < 16; j++ {
				inputs[i][0][j] = rand.Float64()*0.8 + 0.2
				sum += inputs[i][0][j]
			}
		case 1:
			for j := 0; j < 16; j++ {
				inputs[i][0][j] = rand.Float64() * 0.2
				sum += inputs[i][0][j]
			}
		case 2:
			for j := 0; j < 16; j++ {
				if j%2 == 0 {
					inputs[i][0][j] = rand.Float64()*0.8 + 0.2
				} else {
					inputs[i][0][j] = rand.Float64() * 0.2
				}
				sum += inputs[i][0][j]
			}
		case 3:
			for j := 0; j < 16; j++ {
				if j >= 6 && j <= 9 {
					inputs[i][0][j] = rand.Float64()*0.8 + 0.2
				} else {
					inputs[i][0][j] = rand.Float64() * 0.2
				}
				sum += inputs[i][0][j]
			}
		}
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		if sum > 8.0 {
			targets[i][0][0] = 1
		} else {
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

// Experiment 19: Deep Nested XOR (3-Level Fractal)
func experimentDeepNestedXOR(file *os.File) {
	fmt.Println("\n=== Experiment 19: Deep Nested XOR (3-Level) ===")

	layerSizes := []struct{ Width, Height int }{{16, 1}, {32, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {4, 1}, {1, 1}}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
	for y := 0; y < fractalNet.Layers[1].Height; y++ {
		for x := 0; x < fractalNet.Layers[1].Width; x++ {
			subNet := fractalNet.Layers[1].Neurons[y][x].Dimension
			subNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
			for sy := 0; sy < subNet.Layers[1].Height; sy++ {
				for sx := 0; sx < subNet.Layers[1].Width; sx++ {
					subNet.Layers[1].Neurons[sy][sx].Dimension = paragon.NewNetwork(subLayerSizes, subActivations, subFullyConnected)
				}
			}
		}
	}

	trainInputs, trainTargets := generateNestedXOR(1000) // Reuse from Experiment 14
	valInputs, valTargets := generateNestedXOR(200)

	trainCfg := paragon.TrainConfig{Epochs: 2000, LearningRate: 0.0001}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network (3-level)...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 19: Deep Nested XOR (3-Level)\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

func generateFractalSymmetry(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 32)
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		isSymmetric := rand.Intn(2) == 0

		if isSymmetric {
			// Create a symmetric pattern with nested symmetry
			for j := 0; j < 16; j++ {
				base := float64(rand.Intn(3))
				inputs[i][0][j] = base + math.Sin(float64(j)/4.0)
				inputs[i][0][31-j] = base + math.Sin(float64(31-j)/4.0)
			}
			targets[i][0][0] = 1 // Symmetric
		} else {
			// Non-symmetric with random noise
			for j := 0; j < 32; j++ {
				inputs[i][0][j] = rand.Float64()
			}
			targets[i][0][1] = 1 // Non-symmetric
		}
	}
	return inputs, targets
}

func experimentFractalSymmetry(file *os.File) {
	fmt.Println("\n=== Experiment 20: Fractal Symmetry Detection ===")

	layerSizes := []struct{ Width, Height int }{{32, 1}, {64, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {8, 1}, {1, 1}}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
	for y := 0; y < fractalNet.Layers[1].Height; y++ {
		for x := 0; x < fractalNet.Layers[1].Width; x++ {
			fractalNet.Layers[1].Neurons[y][x].Dimension.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
		}
	}

	trainInputs, trainTargets := generateFractalSymmetry(1000)
	valInputs, valTargets := generateFractalSymmetry(200)

	trainCfg := paragon.TrainConfig{Epochs: 1000, LearningRate: 0.0005}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 20: Fractal Symmetry Detection\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

func generateRecursiveFunction(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 10)
		x := float64(rand.Intn(10) + 1)
		for j := 0; j < 10; j++ {
			inputs[i][0][j] = x / 10.0 // Normalized input
		}
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		// Recursive function: x + x^2 + x^3
		y := x + math.Pow(x, 2) + math.Pow(x, 3)
		if y > 50 {
			targets[i][0][0] = 1
		} else {
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

func experimentRecursiveFunction(file *os.File) {
	fmt.Println("\n=== Experiment 21: Recursive Function Approximation ===")

	layerSizes := []struct{ Width, Height int }{{10, 1}, {20, 1}, {2, 1}}
	activations := []string{"linear", "tanh", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {5, 1}, {1, 1}}
	subActivations := []string{"linear", "tanh", "linear"}
	subFullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
	for y := 0; y < fractalNet.Layers[1].Height; y++ {
		for x := 0; x < fractalNet.Layers[1].Width; x++ {
			fractalNet.Layers[1].Neurons[y][x].Dimension.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
		}
	}

	trainInputs, trainTargets := generateRecursiveFunction(1000)
	valInputs, valTargets := generateRecursiveFunction(200)

	trainCfg := paragon.TrainConfig{Epochs: 1000, LearningRate: 0.0005}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 21: Recursive Function Approximation\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

func generateMultiLevelNoise(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 40)
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		signalType := rand.Intn(2)

		for j := 0; j < 40; j++ {
			noiseLow := rand.Float64() * 0.2
			noiseMid := math.Sin(float64(j)/5.0) * 0.3
			noiseHigh := math.Sin(float64(j)) * 0.1
			if signalType == 0 {
				inputs[i][0][j] = math.Sin(float64(j)/10.0) + noiseLow + noiseMid + noiseHigh
				targets[i][0][0] = 1
			} else {
				inputs[i][0][j] = math.Cos(float64(j)/10.0) + noiseLow + noiseMid + noiseHigh
				targets[i][0][1] = 1
			}
		}
	}
	return inputs, targets
}

func experimentMultiLevelNoise(file *os.File) {
	fmt.Println("\n=== Experiment 22: Multi-Level Noise Filtering ===")

	layerSizes := []struct{ Width, Height int }{{40, 1}, {80, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {10, 1}, {1, 1}}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
	for y := 0; y < fractalNet.Layers[1].Height; y++ {
		for x := 0; x < fractalNet.Layers[1].Width; x++ {
			fractalNet.Layers[1].Neurons[y][x].Dimension.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
		}
	}

	trainInputs, trainTargets := generateMultiLevelNoise(1000)
	valInputs, valTargets := generateMultiLevelNoise(200)

	trainCfg := paragon.TrainConfig{Epochs: 1000, LearningRate: 0.0005}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 22: Multi-Level Noise Filtering\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

func generateHierarchicalRules(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 20)
		for j := 0; j < 20; j++ {
			inputs[i][0][j] = float64(rand.Intn(2))
		}
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		// Nested rule: if (x0 AND x1) then (if (x2 OR x3) then 1 else 0) else 0
		if inputs[i][0][0] == 1 && inputs[i][0][1] == 1 {
			if inputs[i][0][2] == 1 || inputs[i][0][3] == 1 {
				targets[i][0][0] = 1
			} else {
				targets[i][0][1] = 1
			}
		} else {
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

func experimentHierarchicalRules(file *os.File) {
	fmt.Println("\n=== Experiment 23: Hierarchical Rule Learning ===")

	layerSizes := []struct{ Width, Height int }{{20, 1}, {40, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {6, 1}, {1, 1}}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
	for y := 0; y < fractalNet.Layers[1].Height; y++ {
		for x := 0; x < fractalNet.Layers[1].Width; x++ {
			fractalNet.Layers[1].Neurons[y][x].Dimension.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
		}
	}

	trainInputs, trainTargets := generateHierarchicalRules(1000)
	valInputs, valTargets := generateHierarchicalRules(200)

	trainCfg := paragon.TrainConfig{Epochs: 1000, LearningRate: 0.0005}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 23: Hierarchical Rule Learning\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

func generateFractalPatternCompletion(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 24)
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		base := float64(rand.Intn(3) + 1)
		// Fractal pattern: repeating structure with scaling
		for j := 0; j < 24; j++ {
			if j < 16 {
				inputs[i][0][j] = base * math.Pow(0.9, float64(j%4))
			} else {
				inputs[i][0][j] = 0 // Missing part to predict
			}
		}
		// Target: whether the pattern increases or decreases
		nextVal := base * math.Pow(0.9, 4)
		if nextVal > 1.5 {
			targets[i][0][0] = 1
		} else {
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

func experimentFractalPatternCompletion(file *os.File) {
	fmt.Println("\n=== Experiment 24: Fractal Pattern Completion ===")

	layerSizes := []struct{ Width, Height int }{{24, 1}, {48, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {6, 1}, {1, 1}}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
	for y := 0; y < fractalNet.Layers[1].Height; y++ {
		for x := 0; x < fractalNet.Layers[1].Width; x++ {
			fractalNet.Layers[1].Neurons[y][x].Dimension.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
		}
	}

	trainInputs, trainTargets := generateFractalPatternCompletion(1000)
	valInputs, valTargets := generateFractalPatternCompletion(200)

	trainCfg := paragon.TrainConfig{Epochs: 1000, LearningRate: 0.0005}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 24: Fractal Pattern Completion\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

func generateDeepFractalXOR(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 64)
		for j := 0; j < 64; j++ {
			inputs[i][0][j] = float64(rand.Intn(2))
		}
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		// 4-level XOR tree: 64 inputs -> 16 -> 4 -> 1
		level1 := make([]int, 16)
		for j := 0; j < 16; j++ {
			start := j * 4
			level1[j] = int(inputs[i][0][start]) ^ int(inputs[i][0][start+1]) ^
				int(inputs[i][0][start+2]) ^ int(inputs[i][0][start+3])
		}
		level2 := make([]int, 4)
		for j := 0; j < 4; j++ {
			start := j * 4
			level2[j] = level1[start] ^ level1[start+1] ^ level1[start+2] ^ level1[start+3]
		}
		level3 := level2[0] ^ level2[1] ^ level2[2] ^ level2[3]
		if level3 == 1 {
			targets[i][0][0] = 1
		} else {
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

func experimentDeepFractalXOR(file *os.File) {
	fmt.Println("\n=== Experiment 25: Deep Fractal XOR Tree ===")

	layerSizes := []struct{ Width, Height int }{{64, 1}, {128, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {8, 1}, {1, 1}}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
	for y := 0; y < fractalNet.Layers[1].Height; y++ {
		for x := 0; x < fractalNet.Layers[1].Width; x++ {
			fractalNet.Layers[1].Neurons[y][x].Dimension.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
		}
	}

	trainInputs, trainTargets := generateDeepFractalXOR(2000)
	valInputs, valTargets := generateDeepFractalXOR(400)

	trainCfg := paragon.TrainConfig{Epochs: 20, LearningRate: 0.0001}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 25: Deep Fractal XOR Tree\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

func generateNestedPolynomialChaos(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 16)
		for j := 0; j < 16; j++ {
			inputs[i][0][j] = rand.Float64()
		}
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		// Nested polynomial: ((x0*x1)^2 + x2)^3 + (x3*x4 + x5)^2
		term1 := math.Pow(inputs[i][0][0]*inputs[i][0][1], 2) + inputs[i][0][2]
		term2 := math.Pow(term1, 3)
		term3 := math.Pow(inputs[i][0][3]*inputs[i][0][4]+inputs[i][0][5], 2)
		result := term2 + term3
		if result > 10.0 {
			targets[i][0][0] = 1
		} else {
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

func experimentNestedPolynomialChaos(file *os.File) {
	fmt.Println("\n=== Experiment 26: Nested Polynomial Chaos ===")

	layerSizes := []struct{ Width, Height int }{{16, 1}, {32, 1}, {2, 1}}
	activations := []string{"linear", "tanh", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {6, 1}, {1, 1}}
	subActivations := []string{"linear", "tanh", "linear"}
	subFullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
	for y := 0; y < fractalNet.Layers[1].Height; y++ {
		for x := 0; x < fractalNet.Layers[1].Width; x++ {
			fractalNet.Layers[1].Neurons[y][x].Dimension.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
		}
	}

	trainInputs, trainTargets := generateNestedPolynomialChaos(2000)
	valInputs, valTargets := generateNestedPolynomialChaos(400)

	trainCfg := paragon.TrainConfig{Epochs: 20, LearningRate: 0.0001}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 26: Nested Polynomial Chaos\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

func generateMultiScaleTemporalFractals(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 80)
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		base := rand.Float64() * 2
		class := rand.Intn(2)
		for j := 0; j < 80; j++ {
			lowFreq := math.Sin(float64(j)/20.0) * base
			midFreq := math.Sin(float64(j)/5.0) * base * 0.5
			highFreq := math.Sin(float64(j)) * base * 0.2
			noise := rand.Float64() * 0.3
			if class == 0 {
				inputs[i][0][j] = lowFreq + midFreq + noise
			} else {
				inputs[i][0][j] = midFreq + highFreq + noise
			}
		}
		if class == 0 {
			targets[i][0][0] = 1
		} else {
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

func experimentMultiScaleTemporalFractals(file *os.File) {
	fmt.Println("\n=== Experiment 27: Multi-Scale Temporal Fractals ===")

	layerSizes := []struct{ Width, Height int }{{80, 1}, {160, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {10, 1}, {1, 1}}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
	for y := 0; y < fractalNet.Layers[1].Height; y++ {
		for x := 0; x < fractalNet.Layers[1].Width; x++ {
			fractalNet.Layers[1].Neurons[y][x].Dimension.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
		}
	}

	trainInputs, trainTargets := generateMultiScaleTemporalFractals(2000)
	valInputs, valTargets := generateMultiScaleTemporalFractals(400)

	trainCfg := paragon.TrainConfig{Epochs: 20, LearningRate: 0.0001}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 27: Multi-Scale Temporal Fractals\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

func generateRecursiveContextualSwitching(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 48)
		for j := 0; j < 48; j++ {
			inputs[i][0][j] = float64(rand.Intn(2))
		}
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		// Recursive context: first 16 bits define rules for next 16, which define last 16
		ctx1 := int(inputs[i][0][0]) ^ int(inputs[i][0][1]) ^ int(inputs[i][0][2]) ^ int(inputs[i][0][3])
		ctx2 := 0
		for j := 16; j < 20; j++ {
			if ctx1 == 1 {
				ctx2 ^= int(inputs[i][0][j])
			} else {
				ctx2 += int(inputs[i][0][j])
			}
		}
		final := 0
		for j := 32; j < 36; j++ {
			if ctx2 > 1 {
				final ^= int(inputs[i][0][j])
			} else {
				final += int(inputs[i][0][j])
			}
		}
		if final > 1 {
			targets[i][0][0] = 1
		} else {
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

func experimentRecursiveContextualSwitching(file *os.File) {
	fmt.Println("\n=== Experiment 28: Recursive Contextual Switching ===")

	layerSizes := []struct{ Width, Height int }{{48, 1}, {96, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {8, 1}, {1, 1}}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
	for y := 0; y < fractalNet.Layers[1].Height; y++ {
		for x := 0; x < fractalNet.Layers[1].Width; x++ {
			fractalNet.Layers[1].Neurons[y][x].Dimension.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
		}
	}

	trainInputs, trainTargets := generateRecursiveContextualSwitching(2000)
	valInputs, valTargets := generateRecursiveContextualSwitching(400)

	trainCfg := paragon.TrainConfig{Epochs: 20, LearningRate: 0.0001}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 28: Recursive Contextual Switching\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

func generateFractalNoiseInterpolation(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 64)
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		base := rand.Float64() * 2
		class := rand.Intn(2)
		for j := 0; j < 64; j++ {
			if j%8 < 4 { // Half the data is missing
				inputs[i][0][j] = 0
			} else {
				low := math.Sin(float64(j)/16.0) * base
				mid := math.Sin(float64(j)/4.0) * base * 0.5
				high := math.Sin(float64(j)) * base * 0.2
				if class == 0 {
					inputs[i][0][j] = low + mid
				} else {
					inputs[i][0][j] = mid + high
				}
			}
		}
		if class == 0 {
			targets[i][0][0] = 1
		} else {
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

func experimentFractalNoiseInterpolation(file *os.File) {
	fmt.Println("\n=== Experiment 29: Fractal Noise Interpolation ===")

	layerSizes := []struct{ Width, Height int }{{64, 1}, {128, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {8, 1}, {1, 1}}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
	for y := 0; y < fractalNet.Layers[1].Height; y++ {
		for x := 0; x < fractalNet.Layers[1].Width; x++ {
			fractalNet.Layers[1].Neurons[y][x].Dimension.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
		}
	}

	trainInputs, trainTargets := generateFractalNoiseInterpolation(2000)
	valInputs, valTargets := generateFractalNoiseInterpolation(400)

	trainCfg := paragon.TrainConfig{Epochs: 20, LearningRate: 0.0001}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 29: Fractal Noise Interpolation\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

func generateHierarchicalBinary(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 12)
		targets[i] = [][]float64{{0, 0}}
		if rand.Intn(2) == 0 { // Hierarchical pattern: [1, 0, 1, 0] repeated
			for j := 0; j < 12; j += 4 {
				inputs[i][0][j] = 1
				inputs[i][0][j+1] = 0
				inputs[i][0][j+2] = 1
				inputs[i][0][j+3] = 0
			}
			targets[i][0][0] = 1 // Class 1
		} else { // Random binary sequence
			for j := 0; j < 12; j++ {
				inputs[i][0][j] = float64(rand.Intn(2))
			}
			targets[i][0][1] = 1 // Class 0
		}
	}
	return inputs, targets
}

func generateMultiScaleSignals(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 10)
		targets[i] = [][]float64{{0}}
		slow := rand.Float64() * 0.1 // Slow trend factor
		fast := rand.Float64() * 0.5 // Fast oscillation factor
		for j := 0; j < 10; j++ {
			t := float64(j)
			inputs[i][0][j] = slow*t + fast*math.Sin(2*t)
		}
		t := float64(10)
		targets[i][0][0] = slow*t + fast*math.Sin(2*t) // Predict next value
	}
	return inputs, targets
}

func generateContextualSequences(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 16)
		targets[i] = [][]float64{{0, 0}}
		// First half of the sequence
		for j := 0; j < 8; j++ {
			inputs[i][0][j] = rand.Float64()
		}
		if rand.Intn(2) == 0 { // Match: Second half is first half * 2
			for j := 0; j < 8; j++ {
				inputs[i][0][j+8] = inputs[i][0][j] * 2
			}
			targets[i][0][0] = 1 // Class 1 (match)
		} else { // No match: Random second half
			for j := 0; j < 8; j++ {
				inputs[i][0][j+8] = rand.Float64()
			}
			targets[i][0][1] = 1 // Class 0 (no match)
		}
	}
	return inputs, targets
}

func experimentHierarchicalBinary(file *os.File) {
	fmt.Println("\n=== Experiment 30: Hierarchical Binary Classification ===")

	// Generate datasets
	trainInputs, trainTargets := generateHierarchicalBinary(1000)
	valInputs, valTargets := generateHierarchicalBinary(200)

	// Baseline network setup
	layerSizes := []struct{ Width, Height int }{{12, 1}, {24, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)

	// Fractal network setup
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {3, 1}, {1, 1}}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)

	// Training configuration
	trainCfg := paragon.TrainConfig{Epochs: 20, LearningRate: 0.01}

	// Train baseline network
	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)

	// Train fractal network
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}
	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)

	// Evaluate both networks
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	// Append results to file
	result := fmt.Sprintf("Experiment 30: Hierarchical Binary Classification\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	if _, err := file.WriteString(result); err != nil {
		fmt.Printf("Error writing to file: %v\n", err)
	}
}

func experimentMultiScaleSignals(file *os.File) {
	fmt.Println("\n=== Experiment 31: Multi-Scale Signal Prediction ===")

	// Generate datasets
	trainInputs, trainTargets := generateMultiScaleSignals(1000)
	valInputs, valTargets := generateMultiScaleSignals(200)

	// Baseline network setup
	layerSizes := []struct{ Width, Height int }{{10, 1}, {20, 1}, {1, 1}}
	activations := []string{"linear", "relu", "linear"}
	fullyConnected := []bool{true, true, true}
	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)

	// Fractal network setup
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {2, 1}, {1, 1}}
	subActivations := []string{"linear", "tanh", "linear"}
	subFullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)

	// Training configuration
	trainCfg := paragon.TrainConfig{Epochs: 20, LearningRate: 0.01}

	// Train baseline network
	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)

	// Train fractal network
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}
	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)

	// Compute MSE for both networks
	baselineMSE := computeMSE(baselineNet, valInputs, valTargets)
	fractalMSE := computeMSE(fractalNet, valInputs, valTargets)
	fmt.Printf("Baseline MSE: %.4f\n", baselineMSE)
	fmt.Printf("Fractal MSE: %.4f\n", fractalMSE)

	// Append results to file
	result := fmt.Sprintf("Experiment 31: Multi-Scale Signal Prediction\nBaseline MSE: %.4f\nFractal MSE: %.4f\n\n",
		baselineMSE, fractalMSE)
	if _, err := file.WriteString(result); err != nil {
		fmt.Printf("Error writing to file: %v\n", err)
	}
}

// Helper function to compute MSE
func computeMSE(net *paragon.Network, inputs [][][]float64, targets [][][]float64) float64 {
	var totalMSE float64
	for i := range inputs {
		net.Forward(inputs[i])
		output := net.Layers[net.OutputLayer].Neurons[0][0].Value
		target := targets[i][0][0]
		totalMSE += math.Pow(output-target, 2)
	}
	return totalMSE / float64(len(inputs))
}

func experimentContextualSequences(file *os.File) {
	fmt.Println("\n=== Experiment 32: Contextual Sequence Matching ===")

	// Generate datasets
	trainInputs, trainTargets := generateContextualSequences(1000)
	valInputs, valTargets := generateContextualSequences(200)

	// Baseline network setup
	layerSizes := []struct{ Width, Height int }{{16, 1}, {32, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)

	// Fractal network setup
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {4, 1}, {1, 1}}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)

	// Training configuration
	trainCfg := paragon.TrainConfig{Epochs: 20, LearningRate: 0.01}

	// Train baseline network
	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)

	// Train fractal network
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}
	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)

	// Evaluate both networks
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	// Append results to file
	result := fmt.Sprintf("Experiment 32: Contextual Sequence Matching\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	if _, err := file.WriteString(result); err != nil {
		fmt.Printf("Error writing to file: %v\n", err)
	}
}

func generateEnhancedDeepHierarchy(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 32)
		for j := 0; j < 32; j++ {
			inputs[i][0][j] = float64(rand.Intn(2))
		}
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		l1 := make([]int, 8)
		for k := 0; k < 8; k++ {
			start := k * 4
			l1[k] = int(inputs[i][0][start]) ^ int(inputs[i][0][start+1]) ^ int(inputs[i][0][start+2]) ^ int(inputs[i][0][start+3])
		}
		l2 := make([]int, 4)
		for k := 0; k < 4; k++ {
			l2[k] = l1[2*k] ^ l1[2*k+1]
		}
		l3 := l2[0] ^ l2[1]
		l4 := l3 ^ l2[2] ^ l2[3]
		if l4 == 1 {
			targets[i][0][0] = 1
		} else {
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

func experimentEnhancedDeepHierarchy(file *os.File) {
	fmt.Println("\n=== Experiment 33: Enhanced Deep Hierarchy ===")

	layerSizes := []struct{ Width, Height int }{{32, 1}, {64, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {8, 1}, {1, 1}}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
	for y := 0; y < fractalNet.Layers[1].Height; y++ {
		for x := 0; x < fractalNet.Layers[1].Width; x++ {
			fractalNet.Layers[1].Neurons[y][x].Dimension.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
		}
	}

	trainInputs, trainTargets := generateEnhancedDeepHierarchy(1000)
	valInputs, valTargets := generateEnhancedDeepHierarchy(200)

	trainCfg := paragon.TrainConfig{Epochs: 1000, LearningRate: 0.001}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 33: Enhanced Deep Hierarchy\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

func generateAdvancedMultiModalFusion(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 48)
		for j := 0; j < 16; j++ {
			inputs[i][0][j] = math.Sin(float64(j)/2.0) + rand.Float64()*0.1
			inputs[i][0][j+16] = float64(rand.Intn(3))
			inputs[i][0][j+32] = math.Cos(float64(j)/3.0) + rand.Float64()*0.2
		}
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		waveSum := 0.0
		stepSum := 0.0
		noiseSum := 0.0
		for j := 0; j < 16; j++ {
			waveSum += inputs[i][0][j]
			stepSum += inputs[i][0][j+16]
			noiseSum += inputs[i][0][j+32]
		}
		if waveSum > 0 && stepSum > 20 && noiseSum < 10 {
			targets[i][0][0] = 1
		} else {
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

func experimentAdvancedMultiModalFusion(file *os.File) {
	fmt.Println("\n=== Experiment 34: Advanced Multi-Modal Fusion ===")

	layerSizes := []struct{ Width, Height int }{{48, 1}, {96, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {8, 1}, {1, 1}}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
	for y := 0; y < fractalNet.Layers[1].Height; y++ {
		for x := 0; x < fractalNet.Layers[1].Width; x++ {
			fractalNet.Layers[1].Neurons[y][x].Dimension.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
		}
	}

	trainInputs, trainTargets := generateAdvancedMultiModalFusion(1000)
	valInputs, valTargets := generateAdvancedMultiModalFusion(200)

	trainCfg := paragon.TrainConfig{Epochs: 50, LearningRate: 0.001}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 34: Advanced Multi-Modal Fusion\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

func generateComplexFeatureComposition(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 24)
		for j := 0; j < 24; j++ {
			inputs[i][0][j] = rand.Float64()
		}
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		f1 := inputs[i][0][0]*math.Pow(inputs[i][0][1], 2) + inputs[i][0][2]*inputs[i][0][3]
		f2 := inputs[i][0][4] + math.Sin(inputs[i][0][5])
		f3 := math.Pow(inputs[i][0][6], 3)
		val := f1*f2 + f3
		if val > 1.0 {
			targets[i][0][0] = 1
		} else {
			targets[i][0][1] = 1
		}
	}
	return inputs, targets
}

func experimentComplexFeatureComposition(file *os.File) {
	fmt.Println("\n=== Experiment 36: Complex Feature Composition ===")

	layerSizes := []struct{ Width, Height int }{{24, 1}, {48, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {6, 1}, {1, 1}}
	subActivations := []string{"linear", "relu", "linear"}
	subFullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
	for y := 0; y < fractalNet.Layers[1].Height; y++ {
		for x := 0; x < fractalNet.Layers[1].Width; x++ {
			fractalNet.Layers[1].Neurons[y][x].Dimension.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
		}
	}

	trainInputs, trainTargets := generateComplexFeatureComposition(1000)
	valInputs, valTargets := generateComplexFeatureComposition(200)

	trainCfg := paragon.TrainConfig{Epochs: 50, LearningRate: 0.001}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 36: Complex Feature Composition\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

func generateExtendedLongRangeContext(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 64)
		context := rand.Intn(2)
		for j := 0; j < 64; j++ {
			inputs[i][0][j] = rand.Float64()
		}
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)
		contextSum := inputs[i][0][0] + inputs[i][0][1] + inputs[i][0][2]
		endSum := inputs[i][0][61] + inputs[i][0][62] + inputs[i][0][63]
		if context == 0 {
			if endSum > contextSum && inputs[i][0][30] > 0.5 {
				targets[i][0][0] = 1
			} else {
				targets[i][0][1] = 1
			}
		} else {
			if endSum < contextSum || inputs[i][0][30] < 0.5 {
				targets[i][0][0] = 1
			} else {
				targets[i][0][1] = 1
			}
		}
	}
	return inputs, targets
}

func experimentExtendedLongRangeContext(file *os.File) {
	fmt.Println("\n=== Experiment 37: Extended Long-Range Context ===")

	layerSizes := []struct{ Width, Height int }{{64, 1}, {128, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {8, 1}, {1, 1}}
	subActivations := []string{"linear", "tanh", "linear"}
	subFullyConnected := []bool{true, true, true}

	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
	for y := 0; y < fractalNet.Layers[1].Height; y++ {
		for x := 0; x < fractalNet.Layers[1].Width; x++ {
			fractalNet.Layers[1].Neurons[y][x].Dimension.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
		}
	}

	trainInputs, trainTargets := generateExtendedLongRangeContext(1000)
	valInputs, valTargets := generateExtendedLongRangeContext(200)

	trainCfg := paragon.TrainConfig{Epochs: 1000, LearningRate: 0.001}

	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}

	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)

	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	result := fmt.Sprintf("Experiment 37: Extended Long-Range Context\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	file.WriteString(result)
}

// Generate dataset for ultra-complex feature composition
func generateUltraComplexFeatureComposition(n int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, n)
	targets := make([][][]float64, n)
	for i := 0; i < n; i++ {
		inputs[i] = [][]float64{{}}
		inputs[i][0] = make([]float64, 48)
		for j := 0; j < 48; j++ {
			inputs[i][0][j] = rand.Float64()
		}
		targets[i] = [][]float64{{}}
		targets[i][0] = make([]float64, 2)

		// Compute intermediate features
		f1 := math.Pow(inputs[i][0][0]*math.Pow(inputs[i][0][1], 2)+inputs[i][0][2]*inputs[i][0][3], 2)
		f2 := math.Sin(inputs[i][0][4]+inputs[i][0][5]) * math.Cos(inputs[i][0][6])
		f3 := math.Log(1 + inputs[i][0][7]*inputs[i][0][8])
		f4 := math.Pow(inputs[i][0][9], 3) + math.Pow(inputs[i][0][10], 2)
		f5 := math.Exp(inputs[i][0][11]) / (1 + math.Exp(inputs[i][0][12]))

		// Final value with conditional logic
		val := f1*f2 + f3 - f4 + f5
		if (val > 0 && inputs[i][0][13] > 0.5) || (val < 0 && inputs[i][0][14] < 0.5) {
			targets[i][0][0] = 1 // Class 1
		} else {
			targets[i][0][1] = 1 // Class 0
		}
	}
	return inputs, targets
}

// Experiment 38: Ultra-Complex Feature Composition
func experimentUltraComplexFeatureComposition(file *os.File) {
	fmt.Println("\n=== Experiment 38: Ultra-Complex Feature Composition ===")

	// Network setup
	layerSizes := []struct{ Width, Height int }{{48, 1}, {96, 1}, {2, 1}}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	// Fractal sub-networks with increased depth
	subLayerSizes := []struct{ Width, Height int }{{1, 1}, {8, 1}, {4, 1}, {1, 1}}
	subActivations := []string{"linear", "relu", "relu", "linear"}
	subFullyConnected := []bool{true, true, true, true}

	// Initialize networks
	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
	for y := 0; y < fractalNet.Layers[1].Height; y++ {
		for x := 0; x < fractalNet.Layers[1].Width; x++ {
			fractalNet.Layers[1].Neurons[y][x].Dimension.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)
		}
	}

	// Generate datasets
	trainInputs, trainTargets := generateUltraComplexFeatureComposition(1000)
	valInputs, valTargets := generateUltraComplexFeatureComposition(200)

	// Training configuration
	trainCfg := paragon.TrainConfig{Epochs: 200, LearningRate: 0.001}

	// Train baseline network
	trainerBaseline := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	fmt.Println("Training baseline network...")
	trainerBaseline.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)

	// Train fractal network
	trainerFractal := paragon.Trainer{Network: fractalNet, Config: trainCfg}
	fmt.Println("Training fractal network...")
	trainerFractal.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)

	// Evaluate both networks
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baselineAcc*100)
	fmt.Printf("Fractal accuracy: %.2f%%\n", fractalAcc*100)

	// Write results to file
	result := fmt.Sprintf("Experiment 38: Ultra-Complex Feature Composition\nBaseline Accuracy: %.2f%%\nFractal Accuracy: %.2f%%\n\n",
		baselineAcc*100, fractalAcc*100)
	if _, err := file.WriteString(result); err != nil {
		fmt.Printf("Error writing to file: %v\n", err)
	}
}
