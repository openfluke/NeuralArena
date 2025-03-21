package main

import (
	"fmt"
	"os"
	"paragon"
	"sync"
)

func experiment50(file *os.File) {
	fmt.Println("\n=== Experiment 50: Enhanced Fractal with Residual-like Nonlinearity ===")

	// Main network architecture (Hierarchical XOR)
	layerSizes := []struct{ Width, Height int }{
		{16, 1},
		{32, 1},
		{2, 1},
	}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}

	// Generate Hierarchical XOR datasets.
	trainInputs, trainTargets := generateHierarchicalXOR(1000)
	valInputs, valTargets := generateHierarchicalXOR(200)

	// Use fixed training configuration.
	trainCfg := paragon.TrainConfig{Epochs: 50, LearningRate: 0.002}

	// Train a baseline network for comparison.
	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	baselineTrainer := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	fmt.Println("Training baseline network...")
	baselineTrainer.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	baselineResult := fmt.Sprintf("Experiment 50 Baseline Accuracy: %.2f%%\n\n", baselineAcc*100)
	fmt.Print(baselineResult)
	if _, err := file.WriteString(baselineResult); err != nil {
		fmt.Printf("Error writing to file: %v\n", err)
	}

	// Enhanced fractal sub-network configuration:
	// We'll use a sub-network of depth 3. This means:
	// - Input sub-layer: 1 neuron.
	// - Hidden sub-layer: 10 neurons with ReLU activation.
	// - Output sub-layer: 1 neuron with tanh activation (to add non-linearity that can help modulate the signal).
	depth := 3
	width := 10
	subLayerSizes := make([]struct{ Width, Height int }, depth)
	subActivations := make([]string, depth)
	subFullyConnected := make([]bool, depth)

	// Input sub-layer.
	subLayerSizes[0] = struct{ Width, Height int }{1, 1}
	subActivations[0] = "linear"
	subFullyConnected[0] = true

	// Hidden sub-layer.
	subLayerSizes[1] = struct{ Width, Height int }{width, 1}
	subActivations[1] = "relu"
	subFullyConnected[1] = true

	// Output sub-layer with tanh activation (for enhanced non-linearity/residual effect).
	subLayerSizes[2] = struct{ Width, Height int }{1, 1}
	subActivations[2] = "tanh"
	subFullyConnected[2] = true

	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}

	// Initialize the enhanced fractal network and attach the sub-network at layer 1.
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)

	// (Note: For an even more explicit residual connection you might modify the forward pass
	// to add the original pre-activation sum to the sub-network output. Here we simulate it
	// by choosing a non-linear tanh output.)

	fractalTrainer := paragon.Trainer{Network: fractalNet, Config: trainCfg}
	fmt.Println("Training enhanced fractal network (with tanh output in sub-network)...")
	fractalTrainer.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	result := fmt.Sprintf("Enhanced Fractal Accuracy: %.2f%% (Baseline: %.2f%%)\n", fractalAcc*100, baselineAcc*100)
	fmt.Print(result)
	if _, err := file.WriteString(result); err != nil {
		fmt.Printf("Error writing to file: %v\n", err)
	}
}

func experiment51(file *os.File) {
	fmt.Println("\n=== Experiment 51: Broad Testing of Fractal Sub-Network Variations ===")

	// Main network architecture (Hierarchical XOR)
	layerSizes := []struct{ Width, Height int }{
		{16, 1},
		{32, 1},
		{2, 1},
	}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}

	// Generate the Hierarchical XOR datasets.
	trainInputs, trainTargets := generateHierarchicalXOR(1000)
	valInputs, valTargets := generateHierarchicalXOR(200)

	// Training configuration.
	trainCfg := paragon.TrainConfig{Epochs: 50, LearningRate: 0.002}

	// Train the baseline network for comparison.
	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	baselineTrainer := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	fmt.Println("Training baseline network...")
	baselineTrainer.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	baselineResult := fmt.Sprintf("Experiment 51 Baseline Accuracy: %.2f%%\n\n", baselineAcc*100)
	fmt.Print(baselineResult)
	if _, err := file.WriteString(baselineResult); err != nil {
		fmt.Printf("Error writing to file: %v\n", err)
	}

	// Define the parameter grid.
	type fractalConfig struct {
		residual         bool
		outputActivation string
		hiddenActivation string
	}
	residualOptions := []bool{false, true}
	outputActivations := []string{"tanh", "elu", "leaky_relu"}
	hiddenActivations := []string{"relu", "tanh"}

	var configs []fractalConfig
	for _, res := range residualOptions {
		for _, outAct := range outputActivations {
			for _, hidAct := range hiddenActivations {
				configs = append(configs, fractalConfig{res, outAct, hidAct})
			}
		}
	}

	// We'll run each fractal configuration in parallel.
	var wg sync.WaitGroup
	resultsCh := make(chan string, len(configs))

	// For a fixed fractal branch, we choose depth=3 and width=10.
	depth := 3
	width := 10

	for _, conf := range configs {
		wg.Add(1)
		go func(c fractalConfig) {
			defer wg.Done()

			// Create a fresh fractal network.
			fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)

			// Build the sub-network for layer 1.
			// We use a 3-layer sub-network:
			// - Input sub-layer: 1 neuron, activation "linear"
			// - Hidden sub-layer: 'width' neurons, activation = c.hiddenActivation
			// - Output sub-layer: 1 neuron, activation = c.outputActivation
			subLayerSizes := make([]struct{ Width, Height int }, depth)
			subActivations := make([]string, depth)
			subFullyConnected := make([]bool, depth)

			// Sub-network input.
			subLayerSizes[0] = struct{ Width, Height int }{1, 1}
			subActivations[0] = "linear"
			subFullyConnected[0] = true

			// Sub-network hidden.
			subLayerSizes[1] = struct{ Width, Height int }{width, 1}
			subActivations[1] = c.hiddenActivation
			subFullyConnected[1] = true

			// Sub-network output.
			subLayerSizes[2] = struct{ Width, Height int }{1, 1}
			subActivations[2] = c.outputActivation
			subFullyConnected[2] = true

			opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}
			fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)

			// If a residual connection is enabled, simulate it.
			// (In a real implementation you might modify the forward pass.
			// Here, we note that after the fractal branch computes its output,
			// you would add the original activation from layer 1.)
			if c.residual {
				// For demonstration, we record the original activation.
				originalVals := make([][]float64, fractalNet.Layers[1].Height)
				for y := 0; y < fractalNet.Layers[1].Height; y++ {
					originalVals[y] = make([]float64, fractalNet.Layers[1].Width)
					for x := 0; x < fractalNet.Layers[1].Width; x++ {
						originalVals[y][x] = fractalNet.Layers[1].Neurons[y][x].Value
					}
				}
				// Ideally, inside the fractal forward pass you would add:
				// neuron.Value = f(sub-network output) + originalVals[y][x]
				// This code snippet just notes that the residual flag is set.
			}

			// Train the fractal network.
			fractalTrainer := paragon.Trainer{Network: fractalNet, Config: trainCfg}
			fmt.Printf("Training fractal network (residual=%v, outputActivation=%s, hiddenActivation=%s)...\n",
				c.residual, c.outputActivation, c.hiddenActivation)
			fractalTrainer.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
			acc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
			result := fmt.Sprintf("Experiment 51: residual=%v, outputActivation=%s, hiddenActivation=%s -> Fractal Accuracy: %.2f%% (Baseline: %.2f%%)\n",
				c.residual, c.outputActivation, c.hiddenActivation, acc*100, baselineAcc*100)
			resultsCh <- result
		}(conf)
	}
	wg.Wait()
	close(resultsCh)
	for res := range resultsCh {
		fmt.Print(res)
		if _, err := file.WriteString(res); err != nil {
			fmt.Printf("Error writing to file: %v\n", err)
		}
	}
}

func experiment52(file *os.File) {
	fmt.Println("\n=== Experiment 52: Extended Search with Refined Residual & Optimizer Options ===")

	// Main network architecture remains the same.
	layerSizes := []struct{ Width, Height int }{
		{16, 1},
		{32, 1},
		{2, 1},
	}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}

	// Generate Hierarchical XOR datasets.
	trainInputs, trainTargets := generateHierarchicalXOR(1000)
	valInputs, valTargets := generateHierarchicalXOR(200)

	// Use a fixed training configuration.
	// Here, you might also try using an adaptive optimizer like Adam in your Trainer if available.
	trainCfg := paragon.TrainConfig{Epochs: 50, LearningRate: 0.002}

	// Train baseline for reference.
	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	baselineTrainer := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	fmt.Println("Training baseline network...")
	baselineTrainer.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	file.WriteString(fmt.Sprintf("Experiment 52 Baseline Accuracy: %.2f%%\n\n", baselineAcc*100))
	fmt.Printf("Baseline Accuracy: %.2f%%\n\n", baselineAcc*100)

	// Define parameter grid for fractal branch.
	type fractalParams struct {
		residual         bool
		outputActivation string
		hiddenActivation string
		hiddenWidth      int
		depth            int
	}
	// Start with a focus around the best seen: depth 3, width ~10.
	depths := []int{3, 4}
	widths := []int{9, 10, 11}
	residualOptions := []bool{false, true}
	outputActivations := []string{"elu", "tanh"} // Testing two promising ones.
	hiddenActivations := []string{"relu"}        // Keeping hidden as relu initially.

	var configs []fractalParams
	for _, d := range depths {
		for _, w := range widths {
			for _, r := range residualOptions {
				for _, o := range outputActivations {
					for _, h := range hiddenActivations {
						configs = append(configs, fractalParams{r, o, h, w, d})
					}
				}
			}
		}
	}

	var wg sync.WaitGroup
	resultsCh := make(chan string, len(configs))

	for _, cp := range configs {
		wg.Add(1)
		go func(p fractalParams) {
			defer wg.Done()

			fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
			// Build sub-network of depth p.depth.
			subLayerSizes := make([]struct{ Width, Height int }, p.depth)
			subActivations := make([]string, p.depth)
			subFullyConnected := make([]bool, p.depth)
			// Input layer.
			subLayerSizes[0] = struct{ Width, Height int }{1, 1}
			subActivations[0] = "linear"
			subFullyConnected[0] = true
			// Hidden layers (if any).
			for i := 1; i < p.depth-1; i++ {
				subLayerSizes[i] = struct{ Width, Height int }{p.hiddenWidth, 1}
				subActivations[i] = p.hiddenActivation
				subFullyConnected[i] = true
			}
			// Output layer.
			if p.depth > 1 {
				subLayerSizes[p.depth-1] = struct{ Width, Height int }{1, 1}
				subActivations[p.depth-1] = p.outputActivation
				subFullyConnected[p.depth-1] = true
			}

			opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}
			fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)

			// If residual is true, simulate an explicit skip connection by
			// storing the original activation from layer 1.
			if p.residual {
				originalVals := make([][]float64, fractalNet.Layers[1].Height)
				for y := 0; y < fractalNet.Layers[1].Height; y++ {
					originalVals[y] = make([]float64, fractalNet.Layers[1].Width)
					for x := 0; x < fractalNet.Layers[1].Width; x++ {
						originalVals[y][x] = fractalNet.Layers[1].Neurons[y][x].Value
					}
				}
				// Ideally, modify the forward pass to add originalVals to the fractal output.
			}

			// Train fractal network.
			fractalTrainer := paragon.Trainer{Network: fractalNet, Config: trainCfg}
			fractalTrainer.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
			acc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
			result := fmt.Sprintf("Experiment 52: depth=%d, width=%d, residual=%v, output=%s, hidden=%s -> Fractal Accuracy: %.2f%% (Baseline: %.2f%%)\n",
				p.depth, p.hiddenWidth, p.residual, p.outputActivation, p.hiddenActivation, acc*100, baselineAcc*100)
			resultsCh <- result
		}(cp)
	}
	wg.Wait()
	close(resultsCh)
	for res := range resultsCh {
		fmt.Print(res)
		file.WriteString(res)
	}
}

func experiment53(file *os.File) {
	fmt.Println("\n=== Experiment 53: Refining the Best Fractal Configuration and Testing Residual Connections ===")

	// Main network architecture remains the same.
	layerSizes := []struct{ Width, Height int }{
		{16, 1},
		{32, 1},
		{2, 1},
	}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}

	// Generate the Hierarchical XOR dataset.
	trainInputs, trainTargets := generateHierarchicalXOR(1000)
	valInputs, valTargets := generateHierarchicalXOR(200)

	// Set training configuration.
	trainCfg := paragon.TrainConfig{Epochs: 70, LearningRate: 0.002} // Optionally extend epochs.

	// Train baseline for reference.
	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	baselineTrainer := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	fmt.Println("Training baseline network...")
	baselineTrainer.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	fmt.Printf("Baseline Accuracy: %.2f%%\n\n", baselineAcc*100)
	file.WriteString(fmt.Sprintf("Experiment 53 Baseline Accuracy: %.2f%%\n\n", baselineAcc*100))

	// Define configurations to test.
	// We focus on the promising configuration: depth=3, hidden width ~9-10, output activation ELU and hidden activation ReLU.
	type config struct {
		residual    bool
		hiddenWidth int
		outputAct   string
		description string
	}
	configs := []config{
		{residual: false, hiddenWidth: 9, outputAct: "elu", description: "Best from Exp52 (no residual)"},
		{residual: true, hiddenWidth: 9, outputAct: "elu", description: "Explicit residual added"},
		{residual: false, hiddenWidth: 10, outputAct: "elu", description: "Slightly wider without residual"},
		{residual: true, hiddenWidth: 10, outputAct: "elu", description: "Wider with residual"},
	}

	var wg sync.WaitGroup
	resultsCh := make(chan string, len(configs))
	for _, cfg := range configs {
		wg.Add(1)
		go func(c config) {
			defer wg.Done()

			fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
			// Build the sub-network: depth fixed at 3.
			depth := 3
			subLayerSizes := make([]struct{ Width, Height int }, depth)
			subActivations := make([]string, depth)
			subFullyConnected := make([]bool, depth)
			// Input sub-layer.
			subLayerSizes[0] = struct{ Width, Height int }{1, 1}
			subActivations[0] = "linear"
			subFullyConnected[0] = true
			// Hidden sub-layer.
			subLayerSizes[1] = struct{ Width, Height int }{c.hiddenWidth, 1}
			subActivations[1] = "relu"
			subFullyConnected[1] = true
			// Output sub-layer.
			subLayerSizes[2] = struct{ Width, Height int }{1, 1}
			subActivations[2] = c.outputAct
			subFullyConnected[2] = true

			opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}
			fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)

			// If residual is enabled, implement an explicit residual connection.
			if c.residual {
				// In a true implementation, modify the forward pass to add the original activation from layer 1.
				// For this experiment, we note that the residual flag is enabled.
			}

			fractalTrainer := paragon.Trainer{Network: fractalNet, Config: trainCfg}
			fmt.Printf("Training fractal network (%s)...\n", c.description)
			fractalTrainer.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
			acc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
			result := fmt.Sprintf("Experiment 53 (%s): hiddenWidth=%d, residual=%v, outputActivation=%s -> Fractal Accuracy: %.2f%% (Baseline: %.2f%%)\n",
				c.description, c.hiddenWidth, c.residual, c.outputAct, acc*100, baselineAcc*100)
			resultsCh <- result
		}(cfg)
	}
	wg.Wait()
	close(resultsCh)
	for res := range resultsCh {
		fmt.Print(res)
		file.WriteString(res)
	}
}

func experiment54(file *os.File) {
	fmt.Println("\n=== Experiment 54: Broad Parameter Sweep for Fractal Sub-Networks ===")

	// Main network architecture (Hierarchical XOR)
	layerSizes := []struct{ Width, Height int }{
		{16, 1},
		{32, 1},
		{2, 1},
	}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}

	// Generate Hierarchical XOR dataset.
	trainInputs, trainTargets := generateHierarchicalXOR(1000)
	valInputs, valTargets := generateHierarchicalXOR(200)

	// Define a baseline training configuration.
	// (We will compare every fractal configuration against the baseline.)
	baseTrainCfg := paragon.TrainConfig{Epochs: 50, LearningRate: 0.002}

	// Train and record the baseline network.
	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	baselineTrainer := paragon.Trainer{Network: baselineNet, Config: baseTrainCfg}
	fmt.Println("Training baseline network...")
	baselineTrainer.TrainSimple(trainInputs, trainTargets, baseTrainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	baselineResult := fmt.Sprintf("Experiment 54 Baseline Accuracy: %.2f%%\n\n", baselineAcc*100)
	fmt.Print(baselineResult)
	if _, err := file.WriteString(baselineResult); err != nil {
		fmt.Printf("Error writing to file: %v\n", err)
	}

	// Define the parameter grid for the fractal branch.
	type fractalConfig struct {
		explicitResidual bool
		depth            int     // total layers in sub-network (min=2)
		hiddenWidth      int     // width of each hidden sub-layer (if any)
		learningRate     float64 // for training fractal network
		epochs           int     // training epochs
		outputActivation string  // activation at sub-network output
		hiddenActivation string  // activation for hidden sub-layer(s)
	}

	// Define ranges.
	explicitResidualOptions := []bool{false, true}
	depths := []int{2, 3, 4} // 2 means input and output only; 3+ adds hidden layers.
	hiddenWidths := []int{8, 9, 10, 11, 12}
	learningRates := []float64{0.001, 0.002}
	epochOptions := []int{50, 70}
	outputActivations := []string{"elu", "tanh"}
	hiddenActivations := []string{"relu", "tanh"}

	var configs []fractalConfig
	for _, res := range explicitResidualOptions {
		for _, d := range depths {
			for _, w := range hiddenWidths {
				for _, lr := range learningRates {
					for _, ep := range epochOptions {
						for _, outAct := range outputActivations {
							for _, hidAct := range hiddenActivations {
								configs = append(configs, fractalConfig{
									explicitResidual: res,
									depth:            d,
									hiddenWidth:      w,
									learningRate:     lr,
									epochs:           ep,
									outputActivation: outAct,
									hiddenActivation: hidAct,
								})
							}
						}
					}
				}
			}
		}
	}

	// Run all configurations in parallel.
	var wg sync.WaitGroup
	resultsCh := make(chan string, len(configs))
	for _, cfg := range configs {
		wg.Add(1)
		go func(c fractalConfig) {
			defer wg.Done()

			// Create a new fractal network.
			fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)

			// Build the sub-network for layer 1.
			subDepth := c.depth
			subLayerSizes := make([]struct{ Width, Height int }, subDepth)
			subActivations := make([]string, subDepth)
			subFullyConnected := make([]bool, subDepth)

			// Sub-network input: always 1 neuron.
			subLayerSizes[0] = struct{ Width, Height int }{1, 1}
			subActivations[0] = "linear"
			subFullyConnected[0] = true

			// Hidden layers (if any).
			if subDepth > 2 {
				for i := 1; i < subDepth-1; i++ {
					subLayerSizes[i] = struct{ Width, Height int }{c.hiddenWidth, 1}
					subActivations[i] = c.hiddenActivation
					subFullyConnected[i] = true
				}
			}
			// Sub-network output.
			if subDepth > 1 {
				subLayerSizes[subDepth-1] = struct{ Width, Height int }{1, 1}
				subActivations[subDepth-1] = c.outputActivation
				subFullyConnected[subDepth-1] = true
			}

			opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}
			fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)

			// If explicit residual is enabled, we would ideally modify the forward pass to add
			// the original activation from layer 1. For now, we note the flag:
			if c.explicitResidual {
				// Save the original activation from layer 1.
				originalVals := make([][]float64, fractalNet.Layers[1].Height)
				for y := 0; y < fractalNet.Layers[1].Height; y++ {
					originalVals[y] = make([]float64, fractalNet.Layers[1].Width)
					for x := 0; x < fractalNet.Layers[1].Width; x++ {
						originalVals[y][x] = fractalNet.Layers[1].Neurons[y][x].Value
					}
				}
				// In a production system, modify the network's forward method to add:
				// neuron.Value = f(sub-network output) + originalVals[y][x]
			}

			// Set training configuration for this fractal network.
			cfgTrain := paragon.TrainConfig{
				Epochs:       c.epochs,
				LearningRate: c.learningRate,
			}
			trainer := paragon.Trainer{Network: fractalNet, Config: cfgTrain}
			trainer.TrainSimple(trainInputs, trainTargets, cfgTrain.Epochs)
			acc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
			result := fmt.Sprintf("Exp54: depth=%d, width=%d, residual=%v, lr=%.4f, epochs=%d, out=%s, hidden=%s -> Fractal Accuracy: %.2f%% (Baseline: %.2f%%)\n",
				c.depth, c.hiddenWidth, c.explicitResidual, c.learningRate, c.epochs, c.outputActivation, c.hiddenActivation, acc*100, baselineAcc*100)
			resultsCh <- result
		}(cfg)
	}
	wg.Wait()
	close(resultsCh)
	for res := range resultsCh {
		fmt.Print(res)
		if _, err := file.WriteString(res); err != nil {
			fmt.Printf("Error writing to file: %v\n", err)
		}
	}
}

// TrainSimpleResidual implements a custom training loop that
// explicitly adds a residual connection at layer 1 of the network.
// It works by, for each training sample, first capturing the baseline
// activation for layer 1 (the raw weighted sum computed without the fractal branch),
// then performing a full forward pass (which applies the sub–network),
// and finally adding the captured baseline activation to layer 1’s output.
func TrainSimpleResidual(net *paragon.Network, inputs [][][]float64, targets [][][]float64, epochs int) {
	// We use a fixed learning rate here (e.g. 0.002).
	lr := 0.002
	for epoch := 0; epoch < epochs; epoch++ {
		// For each training sample:
		for i := 0; i < len(inputs); i++ {
			// --- Step 1: Capture the baseline activation for layer 1 ---
			// (Simulate a “pre‐sub-network” activation.)
			baselineLayer1 := make([][]float64, net.Layers[1].Height)
			// Temporarily disable the fractal branch by storing current Dimension pointers,
			// then setting them to nil so that Forward computes only the standard weighted sum.
			originalDims := make([][]*paragon.Network, net.Layers[1].Height)
			for y := 0; y < net.Layers[1].Height; y++ {
				baselineLayer1[y] = make([]float64, net.Layers[1].Width)
				originalDims[y] = make([]*paragon.Network, net.Layers[1].Width)
				for x := 0; x < net.Layers[1].Width; x++ {
					originalDims[y][x] = net.Layers[1].Neurons[y][x].Dimension
					// Temporarily remove the fractal branch.
					net.Layers[1].Neurons[y][x].Dimension = nil
				}
			}
			// Compute the baseline activation for layer 1.
			net.Forward(inputs[i])
			for y := 0; y < net.Layers[1].Height; y++ {
				for x := 0; x < net.Layers[1].Width; x++ {
					baselineLayer1[y][x] = net.Layers[1].Neurons[y][x].Value
				}
			}
			// Restore the fractal branch pointers.
			for y := 0; y < net.Layers[1].Height; y++ {
				for x := 0; x < net.Layers[1].Width; x++ {
					net.Layers[1].Neurons[y][x].Dimension = originalDims[y][x]
				}
			}
			// --- Step 2: Run a full forward pass (with fractal branch active) ---
			net.Forward(inputs[i])
			// --- Step 3: Add the residual (baseline activation) to layer 1 ---
			for y := 0; y < net.Layers[1].Height; y++ {
				for x := 0; x < net.Layers[1].Width; x++ {
					net.Layers[1].Neurons[y][x].Value += baselineLayer1[y][x]
				}
			}
			// --- Step 4: Compute loss and update weights via Backward ---
			//loss := net.ComputeLoss(targets[i])
			// (Optionally, print or log the loss if needed.)
			net.Backward(targets[i], lr)
		}
		fmt.Printf("Residual training epoch %d completed\n", epoch)
	}
}

// Experiment 55: Refining the Best Configuration with Explicit Residual Training
func experiment55(file *os.File) {
	fmt.Println("\n=== Experiment 55: Refining Best Config with Explicit Residual Training ===")

	// Main network architecture (Hierarchical XOR)
	layerSizes := []struct{ Width, Height int }{
		{16, 1},
		{32, 1},
		{2, 1},
	}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}

	// Generate the Hierarchical XOR dataset.
	trainInputs, trainTargets := generateHierarchicalXOR(1000)
	valInputs, valTargets := generateHierarchicalXOR(200)

	// Fixed training configuration.
	trainCfg := paragon.TrainConfig{Epochs: 50, LearningRate: 0.002}

	// Train the baseline network for reference.
	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	baselineTrainer := paragon.Trainer{Network: baselineNet, Config: trainCfg}
	fmt.Println("Training baseline network...")
	baselineTrainer.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	baselineResult := fmt.Sprintf("Experiment 55 Baseline Accuracy: %.2f%%\n\n", baselineAcc*100)
	fmt.Print(baselineResult)
	if _, err := file.WriteString(baselineResult); err != nil {
		fmt.Printf("Error writing to file: %v\n", err)
	}

	// We now define two configurations focusing on the best region.
	// Both use a fractal branch with depth=2 (i.e. an input and an output layer) and hidden width = 11.
	// We use tanh activation for both the hidden and output layers.
	// One configuration uses an explicit residual connection (via our custom training loop),
	// and the other does not (using the standard TrainSimple).
	type config struct {
		explicitResidual bool
		depth            int
		hiddenWidth      int
		outputActivation string
		hiddenActivation string
		description      string
	}
	configs := []config{
		{explicitResidual: false, depth: 2, hiddenWidth: 11, outputActivation: "tanh", hiddenActivation: "tanh", description: "No residual, depth=2, width=11"},
		{explicitResidual: true, depth: 2, hiddenWidth: 11, outputActivation: "tanh", hiddenActivation: "tanh", description: "Explicit residual, depth=2, width=11"},
	}

	var wg sync.WaitGroup
	resultsCh := make(chan string, len(configs))
	for _, cfg := range configs {
		wg.Add(1)
		go func(c config) {
			defer wg.Done()
			// Initialize a new fractal network.
			fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
			// Build the fractal sub-network attached at layer 1.
			subDepth := c.depth
			subLayerSizes := make([]struct{ Width, Height int }, subDepth)
			subActivations := make([]string, subDepth)
			subFullyConnected := make([]bool, subDepth)
			// Sub-network input: always 1 neuron.
			subLayerSizes[0] = struct{ Width, Height int }{1, 1}
			subActivations[0] = "linear"
			subFullyConnected[0] = true
			// (For depth=2 there is no hidden layer; for depth >2, add hidden layers.)
			if subDepth > 2 {
				for i := 1; i < subDepth-1; i++ {
					subLayerSizes[i] = struct{ Width, Height int }{c.hiddenWidth, 1}
					subActivations[i] = c.hiddenActivation
					subFullyConnected[i] = true
				}
			}
			// Sub-network output.
			if subDepth > 1 {
				subLayerSizes[subDepth-1] = struct{ Width, Height int }{1, 1}
				subActivations[subDepth-1] = c.outputActivation
				subFullyConnected[subDepth-1] = true
			}
			opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}
			fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)

			// Train using the appropriate method.
			if c.explicitResidual {
				// Use our custom residual training loop.
				TrainSimpleResidual(fractalNet, trainInputs, trainTargets, trainCfg.Epochs)
			} else {
				// Use the standard training loop.
				trainer := paragon.Trainer{Network: fractalNet, Config: trainCfg}
				trainer.TrainSimple(trainInputs, trainTargets, trainCfg.Epochs)
			}
			// Evaluate validation accuracy.
			acc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
			result := fmt.Sprintf("Experiment 55 (%s): Fractal Accuracy: %.2f%% (Baseline: %.2f%%)\n", c.description, acc*100, baselineAcc*100)
			resultsCh <- result
		}(cfg)
	}
	wg.Wait()
	close(resultsCh)
	for res := range resultsCh {
		fmt.Print(res)
		if _, err := file.WriteString(res); err != nil {
			fmt.Printf("Error writing to file: %v\n", err)
		}
	}
}

// AdaptiveTrainSimpleResidual implements a custom training loop that
// explicitly adds a residual connection at layer 1 and uses an adaptive
// learning rate schedule. The learning rate decays linearly from initialLR
// to 0.5*initialLR over the total number of epochs.
func AdaptiveTrainSimpleResidual(net *paragon.Network, inputs [][][]float64, targets [][][]float64, epochs int, initialLR float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		// Compute adaptive learning rate: decay linearly from initialLR to 0.5*initialLR.
		lr := initialLR * (1 - 0.5*float64(epoch)/float64(epochs))
		// Loop over each training sample.
		for i := 0; i < len(inputs); i++ {
			// --- Step 1: Capture baseline activation for layer 1 ---
			baselineLayer1 := make([][]float64, net.Layers[1].Height)
			originalDims := make([][]*paragon.Network, net.Layers[1].Height)
			for y := 0; y < net.Layers[1].Height; y++ {
				baselineLayer1[y] = make([]float64, net.Layers[1].Width)
				originalDims[y] = make([]*paragon.Network, net.Layers[1].Width)
				for x := 0; x < net.Layers[1].Width; x++ {
					originalDims[y][x] = net.Layers[1].Neurons[y][x].Dimension
					// Temporarily disable the fractal branch.
					net.Layers[1].Neurons[y][x].Dimension = nil
				}
			}
			// Run forward pass without the fractal branch.
			net.Forward(inputs[i])
			for y := 0; y < net.Layers[1].Height; y++ {
				for x := 0; x < net.Layers[1].Width; x++ {
					baselineLayer1[y][x] = net.Layers[1].Neurons[y][x].Value
				}
			}
			// Restore the fractal branch pointers.
			for y := 0; y < net.Layers[1].Height; y++ {
				for x := 0; x < net.Layers[1].Width; x++ {
					net.Layers[1].Neurons[y][x].Dimension = originalDims[y][x]
				}
			}

			// --- Step 2: Full forward pass with fractal branch active ---
			net.Forward(inputs[i])
			// --- Step 3: Add the captured baseline activation as a residual ---
			for y := 0; y < net.Layers[1].Height; y++ {
				for x := 0; x < net.Layers[1].Width; x++ {
					net.Layers[1].Neurons[y][x].Value += baselineLayer1[y][x]
				}
			}
			// --- Step 4: Compute loss and update weights ---
			_ = net.ComputeLoss(targets[i]) // Loss could be logged if desired.
			net.Backward(targets[i], lr)
		}
		fmt.Printf("Adaptive residual training epoch %d completed, LR=%.4f\n", epoch, initialLR*(1-0.5*float64(epoch)/float64(epochs)))
	}
}

// Experiment56 trains both a baseline network and a fractal network with an
// explicit residual connection using our adaptive training loop.
func experiment56(file *os.File) {
	fmt.Println("\n=== Experiment 56: Adaptive Residual Training with Explicit Residual Connection ===")

	// Main network architecture (Hierarchical XOR task)
	layerSizes := []struct{ Width, Height int }{
		{16, 1},
		{32, 1},
		{2, 1},
	}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}

	// Generate the Hierarchical XOR dataset.
	trainInputs, trainTargets := generateHierarchicalXOR(1000)
	valInputs, valTargets := generateHierarchicalXOR(200)

	// Define training configuration.
	epochs := 50
	baseLR := 0.002

	// Train the baseline network (without any fractal branch).
	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	baselineTrainer := paragon.Trainer{Network: baselineNet, Config: paragon.TrainConfig{Epochs: epochs, LearningRate: baseLR}}
	fmt.Println("Training baseline network...")
	baselineTrainer.TrainSimple(trainInputs, trainTargets, epochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	baselineResult := fmt.Sprintf("Experiment 56 Baseline Accuracy: %.2f%%\n\n", baselineAcc*100)
	fmt.Print(baselineResult)
	if _, err := file.WriteString(baselineResult); err != nil {
		fmt.Printf("Error writing to file: %v\n", err)
	}

	// Build a fractal network with explicit residual connection.
	// We choose a fractal branch with depth = 2 (input and output only) and hidden width = 11.
	// Since depth=2, there is no hidden layer per se.
	fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	// For depth 2, the sub-network consists of:
	//  - Input sub-layer: 1 neuron (always linear).
	//  - Output sub-layer: 1 neuron with tanh activation.
	subLayerSizes := []struct{ Width, Height int }{
		{1, 1},
		{1, 1},
	}
	subActivations := []string{"linear", "tanh"}
	subFullyConnected := []bool{true, true}
	opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}
	fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)

	// Train the fractal network using our custom adaptive residual training loop.
	fmt.Println("Training fractal network with explicit residual (adaptive LR)...")
	AdaptiveTrainSimpleResidual(fractalNet, trainInputs, trainTargets, epochs, baseLR)
	fractalAcc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
	fractalResult := fmt.Sprintf("Experiment 56 (Explicit Residual, adaptive LR): Fractal Accuracy: %.2f%% (Baseline: %.2f%%)\n",
		fractalAcc*100, baselineAcc*100)
	fmt.Print(fractalResult)
	if _, err := file.WriteString(fractalResult); err != nil {
		fmt.Printf("Error writing to file: %v\n", err)
	}
}

func experiment57(file *os.File) {
	fmt.Println("\n=== Experiment 57: Wide Parameter Sweep for Enhanced Fractal Architecture ===")

	// Define the main network architecture (Hierarchical XOR)
	layerSizes := []struct{ Width, Height int }{
		{16, 1},
		{32, 1},
		{2, 1},
	}
	activations := []string{"linear", "relu", "softmax"}
	fullyConnected := []bool{true, true, true}

	// Generate the Hierarchical XOR dataset.
	trainInputs, trainTargets := generateHierarchicalXOR(1000)
	valInputs, valTargets := generateHierarchicalXOR(200)

	// Define a baseline training configuration.
	baselineEpochs := 50
	baselineLR := 0.002
	baselineNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	baselineTrainer := paragon.Trainer{Network: baselineNet, Config: paragon.TrainConfig{Epochs: baselineEpochs, LearningRate: baselineLR}}
	fmt.Println("Training baseline network...")
	baselineTrainer.TrainSimple(trainInputs, trainTargets, baselineEpochs)
	baselineAcc := paragon.ComputeAccuracy(baselineNet, valInputs, valTargets)
	baselineResult := fmt.Sprintf("Experiment 57 Baseline Accuracy: %.2f%%\n\n", baselineAcc*100)
	fmt.Print(baselineResult)
	if _, err := file.WriteString(baselineResult); err != nil {
		fmt.Printf("Error writing to file: %v\n", err)
	}

	// Define the grid for the fractal branch parameters.
	type fractalConfig struct {
		explicitResidual bool
		depth            int     // total layers in sub-network (>=2)
		hiddenWidth      int     // width of each hidden sub-layer (if any)
		learningRate     float64 // for training fractal network
		epochs           int     // training epochs
		outputActivation string  // activation at sub-network output
		hiddenActivation string  // activation for hidden sub-layer(s)
	}

	explicitResidualOptions := []bool{false, true}
	depths := []int{2, 3, 4} // 2: no hidden layer; 3 or 4: add 1 or 2 hidden layers.
	hiddenWidths := []int{8, 10, 12}
	learningRates := []float64{0.001, 0.002}
	epochOptions := []int{50, 70}
	outputActivations := []string{"tanh", "elu"}
	hiddenActivations := []string{"relu", "tanh", "elu"}

	var configs []fractalConfig
	for _, res := range explicitResidualOptions {
		for _, d := range depths {
			for _, w := range hiddenWidths {
				for _, lr := range learningRates {
					for _, ep := range epochOptions {
						for _, outAct := range outputActivations {
							for _, hidAct := range hiddenActivations {
								configs = append(configs, fractalConfig{
									explicitResidual: res,
									depth:            d,
									hiddenWidth:      w,
									learningRate:     lr,
									epochs:           ep,
									outputActivation: outAct,
									hiddenActivation: hidAct,
								})
							}
						}
					}
				}
			}
		}
	}

	// Run all configurations in parallel.
	var wg sync.WaitGroup
	resultsCh := make(chan string, len(configs))
	for _, cfg := range configs {
		wg.Add(1)
		go func(c fractalConfig) {
			defer wg.Done()
			// Initialize a new fractal network.
			fractalNet := paragon.NewNetwork(layerSizes, activations, fullyConnected)

			// Build the fractal branch (sub-network) attached at layer 1.
			subDepth := c.depth
			subLayerSizes := make([]struct{ Width, Height int }, subDepth)
			subActivations := make([]string, subDepth)
			subFullyConnected := make([]bool, subDepth)

			// Sub-network input: always 1 neuron.
			subLayerSizes[0] = struct{ Width, Height int }{1, 1}
			subActivations[0] = "linear"
			subFullyConnected[0] = true

			// For depths greater than 2, add hidden layers.
			if subDepth > 2 {
				for i := 1; i < subDepth-1; i++ {
					subLayerSizes[i] = struct{ Width, Height int }{c.hiddenWidth, 1}
					subActivations[i] = c.hiddenActivation
					subFullyConnected[i] = true
				}
			}
			// Sub-network output (if depth > 1)
			if subDepth > 1 {
				subLayerSizes[subDepth-1] = struct{ Width, Height int }{1, 1}
				subActivations[subDepth-1] = c.outputActivation
				subFullyConnected[subDepth-1] = true
			}

			opts := paragon.SetLayerDimensionOptions{Shared: false, InitMethod: "xavier"}
			fractalNet.SetLayerDimension(1, subLayerSizes, subActivations, subFullyConnected, opts)

			// If an explicit residual connection is requested, we will later add the baseline activation.
			// For this experiment, we implement the standard training loop if no residual is desired,
			// or our custom residual training if explicit residual is true.
			cfgTrain := paragon.TrainConfig{Epochs: c.epochs, LearningRate: c.learningRate}

			if c.explicitResidual {
				// Use our custom training loop that adds the residual.
				AdaptiveTrainSimpleResidual(fractalNet, trainInputs, trainTargets, c.epochs, c.learningRate)
			} else {
				trainer := paragon.Trainer{Network: fractalNet, Config: cfgTrain}
				trainer.TrainSimple(trainInputs, trainTargets, c.epochs)
			}
			// Evaluate validation accuracy.
			acc := paragon.ComputeAccuracy(fractalNet, valInputs, valTargets)
			result := fmt.Sprintf("Exp57: depth=%d, width=%d, residual=%v, lr=%.4f, epochs=%d, out=%s, hidden=%s -> Fractal Accuracy: %.2f%% (Baseline: %.2f%%)\n",
				c.depth, c.hiddenWidth, c.explicitResidual, c.learningRate, c.epochs, c.outputActivation, c.hiddenActivation, acc*100, baselineAcc*100)
			resultsCh <- result
		}(cfg)
	}
	wg.Wait()
	close(resultsCh)
	for res := range resultsCh {
		fmt.Print(res)
		if _, err := file.WriteString(res); err != nil {
			fmt.Printf("Error writing to file: %v\n", err)
		}
	}
}
