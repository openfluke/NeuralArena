package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"paragon"

	"github.com/openfluke/pilot"
	"github.com/openfluke/pilot/experiments"
)

// Constants for experiment configuration
const (
	epochs       = 5
	learningRate = 0.005
	modelsDir    = "./models"
	trainSize    = 500
	testSize     = 100
	sampleSize   = 5
	batchSize    = 4 // Added for GPU batching
)

// Timing struct to track durations of operations
type timings struct {
	init        time.Duration
	train       time.Duration
	trainEval   time.Duration
	testEval    time.Duration
	preSamples  time.Duration
	postSamples time.Duration
	total       time.Duration
}

// Results struct for experiment outcomes
type experimentResults struct {
	trainScore       float64
	testScore        float64
	preTrainOutputs  [][]float64
	postTrainOutputs [][]float64
	reloadedOutputs  [][]float64
	timings          timings
}

func main() {
	// Ensure models directory exists
	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		fmt.Printf("❌ Failed to create models directory: %v\n", err)
		return
	}

	// Load MNIST data
	data, err := loadMNISTData("./data/mnist")
	if err != nil {
		fmt.Printf("❌ Failed to load MNIST data: %v\n", err)
		return
	}

	// Validate and trim dataset
	if len(data.inputs) < trainSize+testSize {
		fmt.Printf("❌ Insufficient data: got %d samples, need %d\n", len(data.inputs), trainSize+testSize)
		return
	}
	trainInputs := data.inputs[:trainSize]
	trainTargets := data.targets[:trainSize]
	testInputs := data.inputs[trainSize : trainSize+testSize]
	testTargets := data.targets[trainSize : trainSize+testSize]
	sampleInputs := trainInputs[:sampleSize]
	sampleTargets := trainTargets[:sampleSize]
	fmt.Printf("📊 Dataset: Train=%d, Test=%d, Samples=%d\n", len(trainInputs), len(testInputs), len(sampleInputs))

	// Run experiments
	fmt.Println("\n=== CPU Experiment ===")
	cpuResults := runExperiment(false, trainInputs, trainTargets, testInputs, testTargets, sampleInputs, sampleTargets, fmt.Sprintf("%s/cpu_model.json", modelsDir))

	fmt.Println("\n=== GPU Experiment ===")
	gpuResults := runExperiment(true, trainInputs, trainTargets, testInputs, testTargets, sampleInputs, sampleTargets, fmt.Sprintf("%s/gpu_model.json", modelsDir))

	// Reload models and run samples
	fmt.Println("\n=== Reloaded Model Sample Outputs ===")
	cpuReloadedOutputs, err := runReloadedSamples(false, sampleInputs, fmt.Sprintf("%s/cpu_model.json", modelsDir))
	if err != nil {
		fmt.Printf("❌ Failed to reload CPU model: %v\n", err)
	} else {
		cpuResults.reloadedOutputs = cpuReloadedOutputs
	}

	gpuReloadedOutputs, err := runReloadedSamples(true, sampleInputs, fmt.Sprintf("%s/gpu_model.json", modelsDir))
	if err != nil {
		fmt.Printf("❌ Failed to reload GPU model: %v\n", err)
	} else {
		gpuResults.reloadedOutputs = gpuReloadedOutputs
	}

	// Compare CPU and GPU models
	fmt.Println("\n=== Model Weight Comparison ===")
	if err := compareModels(fmt.Sprintf("%s/cpu_model.json", modelsDir), fmt.Sprintf("%s/gpu_model.json", modelsDir)); err != nil {
		fmt.Printf("❌ Failed to compare models: %v\n", err)
	}

	// Display results
	displayResults(cpuResults, gpuResults, sampleInputs, sampleTargets)
}

// Experiment data structure
type experimentData struct {
	inputs  [][][]float64
	targets [][][]float64
}

// Load MNIST data (placeholder with dummy data)
func loadMNISTData(dir string) (experimentData, error) {
	mnist := experiments.NewMNISTDatasetStage(dir)
	exp := pilot.NewExperiment("MNIST", mnist)
	if err := exp.RunAll(); err != nil {
		return experimentData{}, err
	}

	// Dummy data: replace with actual MNIST loading
	inputs := make([][][]float64, trainSize+testSize)
	targets := make([][][]float64, trainSize+testSize)
	for i := range inputs {
		inputs[i] = make([][]float64, 28)
		for y := 0; y < 28; y++ {
			inputs[i][y] = make([]float64, 28)
			for x := 0; x < 28; x++ {
				inputs[i][y][x] = rand.Float64() / 255.0 // Normalize to [0, 1]
			}
		}
		targets[i] = make([][]float64, 1)
		targets[i][0] = make([]float64, 10)
		targets[i][0][rand.Intn(10)] = 1.0
	}
	return experimentData{inputs: inputs, targets: targets}, nil
}

// Run a single experiment (CPU or GPU)
func runExperiment(useGPU bool, trainInputs, trainTargets, testInputs, testTargets, sampleInputs, sampleTargets [][][]float64, modelPath string) experimentResults {
	var t timings
	startTotal := time.Now()

	// Initialize network with new model size
	start := time.Now()
	nn := paragon.NewNetwork[float32](
		[]struct{ Width, Height int }{
			{28, 28}, // Input: 28x28 (MNIST)
			{32, 32}, // Hidden 1: 32x32
			{32, 32}, // Hidden 2: 32x32
			{10, 1},  // Output: 10x1 (10 classes)
		},
		[]string{"linear", "relu", "relu", "softmax"},
		[]bool{true, true, true, true},
	)
	//nn.Debug = true // Enable debug output
	if useGPU {
		nn.WebGPUNative = true
		if err := nn.InitializeOptimizedGPU(); err != nil {
			fmt.Printf("⚠️ WebGPU initialization failed: %v\n", err)
			fmt.Println("   Falling back to CPU...")
			nn.WebGPUNative = false
		} else {
			fmt.Println("✅ WebGPU initialized")
			//	defer nn.CleanupOptimizedGPU()
		}
	}
	t.init = time.Since(start)

	// Get pre-training sample outputs
	start = time.Now()
	preTrainOutputs := make([][]float64, sampleSize)
	for i := range sampleInputs {
		nn.Forward(sampleInputs[i])
		preTrainOutputs[i] = nn.GetOutput()
	}
	t.preSamples = time.Since(start)

	// Train network
	start = time.Now()
	nn.Train(trainInputs, trainTargets, epochs, learningRate, false, float32(2), float32(-2))
	t.train = time.Since(start)

	// Save trained model
	if err := nn.SaveJSON(modelPath); err != nil {
		fmt.Printf("❌ Failed to save model to %s: %v\n", modelPath, err)
	}

	// Evaluate on training set
	start = time.Now()
	trainScore := evaluateFullNetwork(nn, trainInputs, trainTargets)
	t.trainEval = time.Since(start)

	// Evaluate on test set
	start = time.Now()
	testScore := evaluateFullNetwork(nn, testInputs, testTargets)
	t.testEval = time.Since(start)

	// Get post-training sample outputs
	start = time.Now()
	postTrainOutputs := make([][]float64, sampleSize)
	for i := range sampleInputs {
		nn.Forward(sampleInputs[i])
		postTrainOutputs[i] = nn.GetOutput()
	}
	t.postSamples = time.Since(start)

	// Total time
	t.total = time.Since(startTotal)

	return experimentResults{
		trainScore:       trainScore,
		testScore:        testScore,
		preTrainOutputs:  preTrainOutputs,
		postTrainOutputs: postTrainOutputs,
		timings:          t,
	}
}

// Run sample inputs on a reloaded model
func runReloadedSamples(useGPU bool, sampleInputs [][][]float64, modelPath string) ([][]float64, error) {
	nn := paragon.NewNetwork[float32](
		[]struct{ Width, Height int }{
			{28, 28}, {32, 32}, {32, 32}, {10, 1},
		},
		[]string{"linear", "relu", "relu", "softmax"},
		[]bool{true, true, true, true},
	)
	if useGPU {
		nn.WebGPUNative = true
		if err := nn.InitializeOptimizedGPU(); err != nil {
			return nil, fmt.Errorf("failed to initialize GPU: %w", err)
		}
	}

	if err := nn.LoadJSON(modelPath); err != nil {
		return nil, fmt.Errorf("failed to load model from %s: %w", modelPath, err)
	}

	if useGPU && nn.WebGPUNative {
		if err := nn.SyncCPUWeightsToGPU(); err != nil {
			fmt.Printf("Failed to sync CPU weights to GPU: %v\n", err)
			return nil, err
		}
	}

	outputs := make([][]float64, len(sampleInputs))
	for i := range sampleInputs {
		nn.Forward(sampleInputs[i])
		outputs[i] = nn.GetOutput()
	}

	return outputs, nil
}

// Compare weights between CPU and GPU models
func compareModels(cpuModelPath, gpuModelPath string) error {
	// Load CPU model
	cpuNetAny, err := paragon.LoadNamedNetworkFromJSONFile(cpuModelPath)
	if err != nil {
		return fmt.Errorf("failed to load CPU model: %w", err)
	}
	cpuNet, ok := cpuNetAny.(*paragon.Network[float32])
	if !ok {
		return fmt.Errorf("CPU model is not of type *Network[float32]")
	}

	// Load GPU model
	gpuNetAny, err := paragon.LoadNamedNetworkFromJSONFile(gpuModelPath)
	if err != nil {
		return fmt.Errorf("failed to load GPU model: %w", err)
	}
	gpuNet, ok := gpuNetAny.(*paragon.Network[float32])
	if !ok {
		return fmt.Errorf("GPU model is not of type *Network[float32]")
	}

	// Verify model structure
	if len(cpuNet.Layers) != len(gpuNet.Layers) {
		return fmt.Errorf("layer count mismatch: CPU=%d, GPU=%d", len(cpuNet.Layers), len(gpuNet.Layers))
	}

	// Compare weights
	totalDiff := 0.0
	totalWeights := 0
	for l := 0; l < len(cpuNet.Layers); l++ {
		cpuLayer := cpuNet.Layers[l]
		gpuLayer := gpuNet.Layers[l]
		if cpuLayer.Width != gpuLayer.Width || cpuLayer.Height != gpuLayer.Height {
			return fmt.Errorf("layer %d dimension mismatch: CPU=%dx%d, GPU=%dx%d",
				l, cpuLayer.Width, cpuLayer.Height, gpuLayer.Width, gpuLayer.Height)
		}

		for y := 0; y < cpuLayer.Height; y++ {
			for x := 0; x < cpuLayer.Width; x++ {
				cpuNeuron := cpuLayer.Neurons[y][x]
				gpuNeuron := gpuLayer.Neurons[y][x]
				if len(cpuNeuron.Inputs) != len(gpuNeuron.Inputs) {
					return fmt.Errorf("layer %d neuron (%d,%d) input count mismatch: CPU=%d, GPU=%d",
						l, x, y, len(cpuNeuron.Inputs), len(gpuNeuron.Inputs))
				}

				for i := range cpuNeuron.Inputs {
					cpuWeight := float64(cpuNeuron.Inputs[i].Weight)
					gpuWeight := float64(gpuNeuron.Inputs[i].Weight)
					totalDiff += math.Abs(cpuWeight - gpuWeight)
					totalWeights++
				}
			}
		}
	}

	if totalWeights == 0 {
		return fmt.Errorf("no weights found to compare")
	}

	meanDiff := totalDiff / float64(totalWeights)
	fmt.Printf("Mean absolute weight difference: %.6f\n", meanDiff)
	if meanDiff < 1e-6 {
		fmt.Println("Models are equivalent (weights are nearly identical)")
	} else {
		fmt.Println("Models differ significantly (weights are not equivalent)")
	}

	return nil
}

// Evaluate network performance
func evaluateFullNetwork[T paragon.Numeric](nn *paragon.Network[T], inputs, targets [][][]float64) float64 {
	expected := make([]float64, len(inputs))
	actual := make([]float64, len(inputs))
	for i := range inputs {
		nn.Forward(inputs[i])
		out := nn.ExtractOutput()
		expected[i] = float64(paragon.ArgMax(targets[i][0]))
		actual[i] = float64(paragon.ArgMax(out))
	}
	nn.EvaluateModel(expected, actual)
	return nn.Performance.Score
}

// Display experiment results and comparisons
func displayResults(cpu, gpu experimentResults, sampleInputs, sampleTargets [][][]float64) {
	// Timing breakdown
	fmt.Println("\n=== Timing Breakdown ===")
	fmt.Println("CPU Timings:")
	fmt.Printf("  Initialization:     %v\n", cpu.timings.init)
	fmt.Printf("  Pre-train samples:  %v\n", cpu.timings.preSamples)
	fmt.Printf("  Training:           %v\n", cpu.timings.train)
	fmt.Printf("  Train evaluation:   %v\n", cpu.timings.trainEval)
	fmt.Printf("  Test evaluation:    %v\n", cpu.timings.testEval)
	fmt.Printf("  Post-train samples: %v\n", cpu.timings.postSamples)
	fmt.Printf("  Total:              %v\n", cpu.timings.total)
	fmt.Println("GPU Timings:")
	fmt.Printf("  Initialization:     %v\n", gpu.timings.init)
	fmt.Printf("  Pre-train samples:  %v\n", gpu.timings.preSamples)
	fmt.Printf("  Training:           %v\n", gpu.timings.train)
	fmt.Printf("  Train evaluation:   %v\n", gpu.timings.trainEval)
	fmt.Printf("  Test evaluation:    %v\n", gpu.timings.testEval)
	fmt.Printf("  Post-train samples: %v\n", gpu.timings.postSamples)
	fmt.Printf("  Total:              %v\n", gpu.timings.total)

	// Performance scores
	fmt.Println("\n=== Performance Scores ===")
	fmt.Printf("CPU Train Score: %.4f\n", cpu.trainScore)
	fmt.Printf("CPU Test Score:  %.4f\n", cpu.testScore)
	fmt.Printf("GPU Train Score: %.4f\n", gpu.trainScore)
	fmt.Printf("GPU Test Score:  %.4f\n", gpu.testScore)

	// Sample output comparisons
	fmt.Println("\n=== Sample Output Comparisons ===")
	for i := range sampleInputs {
		fmt.Printf("\nSample %d (Target: %d):\n", i, paragon.ArgMax(sampleTargets[i][0]))
		fmt.Printf("  CPU Before: %v\n", formatOutput(cpu.preTrainOutputs[i]))
		fmt.Printf("  CPU After:  %v\n", formatOutput(cpu.postTrainOutputs[i]))
		if len(cpu.reloadedOutputs) > i {
			fmt.Printf("  CPU Reloaded: %v\n", formatOutput(cpu.reloadedOutputs[i]))
		}
		fmt.Printf("  GPU Before: %v\n", formatOutput(gpu.preTrainOutputs[i]))
		fmt.Printf("  GPU After:  %v\n", formatOutput(gpu.postTrainOutputs[i]))
		if len(gpu.reloadedOutputs) > i {
			fmt.Printf("  GPU Reloaded: %v\n", formatOutput(gpu.reloadedOutputs[i]))
		}
	}
}

// Format output for readable display
func formatOutput(output []float64) string {
	str := "["
	for i, v := range output {
		if i > 0 {
			str += ", "
		}
		str += fmt.Sprintf("%.3f", v)
	}
	return str + "]"
}
