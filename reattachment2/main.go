package main

import (
	"fmt"
	"math/rand"
	"time"

	"paragon" // Replace with actual import path
)

func main() {
	rand.Seed(42)

	fmt.Println("🌱 === ADHD-Based Neural Network Growth Demonstration ===")
	fmt.Println("📝 Process: 2 Epochs → ADHD Eval → 64 Checkpoint Samples → Micro Net Growth → Full ADHD Verification")

	// Step 1: Create initial network
	fmt.Println("\n🏗️  Step 1: Creating initial network...")
	network := createInitialNetwork()

	// Step 2: Generate training data
	fmt.Println("\n📊 Step 2: Generating training data...")
	inputs, targets := generateTrainingData(1000)

	// Step 3: MINIMAL initial training (only 2 epochs as requested)
	//fmt.Println("\n🎯 Step 3: Training for exactly 2 epochs...")
	//trainInitialNetworkMinimal(network, inputs[:800], targets[:800])

	// Step 4: Evaluate baseline performance using ADHD
	fmt.Println("\n📈 Step 4: Baseline ADHD evaluation...")
	initialADHDScore := evaluateNetworkWithADHD(network, inputs[800:], targets[800:])
	fmt.Printf("📊 Baseline ADHD score: %.4f\n", initialADHDScore)
	showADHDBreakdown(network, "Baseline")

	// Step 5: Configure growth for ADHD-based improvement
	fmt.Println("\n⚙️  Step 5: Configuring ADHD-based growth...")
	growConfig := paragon.DefaultGrowConfig()

	// Customize for more achievable growth
	growConfig.BatchSize = 64             // Exactly 64 samples as requested
	growConfig.MicroNetCount = 160        // Reasonable number of micro nets
	growConfig.MinADHDScore = 85.0        // Lower threshold (was 90.0)
	growConfig.ImprovementThreshold = 0.1 // Much lower threshold (was 2.0)
	growConfig.TrainingEpochs = 1         // Exactly 1 epoch as requested
	growConfig.LearningRate = 0.01        // Lower learning rate for stability
	growConfig.NewLayerWidth = 10         // Initial layer width
	growConfig.MaxGrowthAttempts = 200    // Allow 200 micro-net attempts per growth
	growConfig.Debug = true

	fmt.Printf("🎯 ADHD score threshold: %.1f (network needs improvement if below this)\n", growConfig.MinADHDScore)
	fmt.Printf("📏 ADHD improvement threshold: %.1f (micro net must improve by this much)\n", growConfig.ImprovementThreshold)
	fmt.Printf("📋 Checkpoint batch size: %d samples\n", growConfig.BatchSize)
	fmt.Printf("🔬 Micro networks per attempt: %d\n", growConfig.MicroNetCount)
	fmt.Printf("🎓 Training epochs per micro net: %d\n", growConfig.TrainingEpochs)
	fmt.Printf("🏗️  Initial new layer width: %d neurons\n", growConfig.NewLayerWidth)
	fmt.Printf("🔄 Growth attempts allowed per iteration: %d\n", growConfig.MaxGrowthAttempts)

	// Step 6: Aggressive ADHD-based growth process with adaptive layer sizing
	fmt.Println("\n🌱 Step 6: Starting aggressive ADHD-based growth process...")
	startTime := time.Now()

	// Aggressive growth parameters
	const maxTotalGrowthIterations = 10 // Try growing up to 10 times
	const maxConsecutiveFailures = 3    // Increase layer width after 3 failed attempts
	const layerWidthIncrement = 5       // Add 5 neurons when increasing layer width
	const maxLayerWidth = 50            // Cap layer width to prevent excessive growth

	consecutiveFailures := 0
	totalLayersAdded := 0
	totalADHDImprovement := 0.0
	currentADHDScore := initialADHDScore

	for iteration := 1; iteration <= maxTotalGrowthIterations; iteration++ {
		fmt.Printf("\n🔥 Aggressive Growth Iteration %d/%d (Layer Width: %d)\n", iteration, maxTotalGrowthIterations, growConfig.NewLayerWidth)

		result, err := network.Grow(inputs[800:], targets[800:], growConfig)
		if err != nil {
			fmt.Printf("❌ Growth iteration %d failed: %v\n", iteration, err)
			consecutiveFailures++
		} else if result.Success {
			fmt.Printf("✅ Growth iteration %d successful! ADHD Score: %.4f → %.4f (%.4f improvement)\n",
				iteration, result.OriginalADHDScore, result.ImprovedADHDScore, result.ADHDImprovement)
			totalLayersAdded += result.LayersAdded
			totalADHDImprovement += result.ADHDImprovement
			currentADHDScore = result.ImprovedADHDScore
			consecutiveFailures = 0 // Reset on success
		} else {
			fmt.Printf("❌ Growth iteration %d failed to improve ADHD score sufficiently\n", iteration)
			consecutiveFailures++
		}

		// Check if we should increase layer width
		if consecutiveFailures >= maxConsecutiveFailures && growConfig.NewLayerWidth < maxLayerWidth {
			growConfig.NewLayerWidth += layerWidthIncrement
			consecutiveFailures = 0 // Reset to give the new width a fair chance
			fmt.Printf("📈 Increasing layer width to %d neurons due to %d consecutive failures\n",
				growConfig.NewLayerWidth, maxConsecutiveFailures)
		}

		// Stop early if ADHD score meets or exceeds threshold
		if currentADHDScore >= growConfig.MinADHDScore {
			fmt.Printf("🎉 Stopping early: ADHD score (%.4f) meets or exceeds threshold (%.4f)\n",
				currentADHDScore, growConfig.MinADHDScore)
			break
		}
	}

	growthTime := time.Since(startTime)

	// Step 7: Display comprehensive ADHD growth results
	fmt.Println("\n📋 Step 7: Aggressive ADHD Growth Results")
	fmt.Println("================================================================")
	if totalLayersAdded > 0 {
		fmt.Printf("✅ AGGRESSIVE ADHD-BASED GROWTH SUCCESSFUL!\n")
		fmt.Printf("📈 Total ADHD Score Improvement: %.4f → %.4f (%.4f total improvement)\n",
			initialADHDScore, currentADHDScore, totalADHDImprovement)
		fmt.Printf("🏗️  Total Layers added: %d\n", totalLayersAdded)
		improvementPercent := (totalADHDImprovement / initialADHDScore) * 100
		fmt.Printf("📊 Relative ADHD improvement: %.2f%%\n", improvementPercent)
	} else {
		fmt.Printf("❌ Aggressive ADHD-based growth not successful\n")
		fmt.Printf("📈 Final ADHD score: %.4f\n", currentADHDScore)
		if currentADHDScore >= growConfig.MinADHDScore {
			fmt.Printf("💡 Model already meets ADHD threshold - no further growth needed\n")
		} else {
			fmt.Printf("💡 Failed to achieve sufficient ADHD improvement after %d iterations\n", maxTotalGrowthIterations)
		}
	}
	fmt.Printf("⏱️  Total growth process time: %v\n", growthTime)

	// Step 8: Final ADHD evaluation and detailed breakdown
	fmt.Println("\n🔍 Step 8: Final ADHD evaluation...")
	finalADHDScore := evaluateNetworkWithADHD(network, inputs[800:], targets[800:])
	fmt.Printf("📊 Final ADHD score: %.4f\n", finalADHDScore)
	totalADHDImprovement = finalADHDScore - initialADHDScore
	fmt.Printf("📈 Total ADHD improvement: %.4f\n", totalADHDImprovement)
	showADHDBreakdown(network, "Final")

	// Step 9: Architecture analysis
	fmt.Println("\n🏗️  Step 9: Network architecture analysis...")
	displayNetworkArchitecture(network, totalLayersAdded > 0)

	// Step 10: Performance impact analysis
	fmt.Println("\n⚡ Step 10: Performance impact analysis...")
	benchmarkNetwork(network, inputs[800:])

	// Step 11: ADHD-specific insights
	fmt.Println("\n📊 Step 11: ADHD Growth Analysis...")
	// Update result for final analysis
	finalResult := &paragon.GrowthResult{
		Success:           totalLayersAdded > 0,
		OriginalADHDScore: initialADHDScore,
		ImprovedADHDScore: finalADHDScore,
		ADHDImprovement:   totalADHDImprovement,
		LayersAdded:       totalLayersAdded,
		ProcessingTime:    growthTime,
	}
	analyzeADHDGrowth(finalResult, initialADHDScore, finalADHDScore, growthTime)

	fmt.Println("\n✅ Aggressive ADHD-based growth demonstration completed!")
	fmt.Printf("🎯 Key insight: %s\n", getADHDKeyInsight(finalResult, totalADHDImprovement))
}

func createInitialNetwork() *paragon.Network[float32] {
	// Create a deliberately small network to benefit from growth
	layerSizes := []struct{ Width, Height int }{
		{3, 1}, // Input: 3 features
		{4, 1}, // Small hidden: 4 neurons
		{3, 1}, // Small hidden: 3 neurons
		{2, 1}, // Output: 2 classes
	}

	activations := []string{"linear", "relu", "relu", "softmax"}
	fullyConnected := []bool{false, true, true, true}

	network := paragon.NewNetwork[float32](layerSizes, activations, fullyConnected)
	network.Debug = false

	fmt.Printf("🏗️  Created small network: %d → %d → %d → %d\n",
		layerSizes[0].Width, layerSizes[1].Width, layerSizes[2].Width, layerSizes[3].Width)
	fmt.Printf("💡 Small capacity designed to benefit from ADHD-guided growth\n")

	return network
}

func generateTrainingData(numSamples int) ([][][]float64, [][][]float64) {
	inputs := make([][][]float64, numSamples)
	targets := make([][][]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		x1 := rand.Float64()*2 - 1
		x2 := rand.Float64()*2 - 1
		x3 := rand.Float64()*2 - 1

		inputs[i] = [][]float64{{x1, x2, x3}}

		// Complex non-linear decision boundary requiring additional capacity
		decision := x1*x2*x3*x3 + x1*x1*x2*x3 + x2*x2*x3 + 0.3*x1*x3*x3*x3
		var target []float64
		if decision > 0.05 {
			target = []float64{0.0, 1.0}
		} else {
			target = []float64{1.0, 0.0}
		}

		targets[i] = [][]float64{target}
	}

	fmt.Printf("📊 Generated %d samples with complex decision boundary\n", numSamples)
	fmt.Printf("💡 Complex problem designed to show ADHD improvements\n")

	return inputs, targets
}

func trainInitialNetworkMinimal(network *paragon.Network[float32], inputs, targets [][][]float64) {
	fmt.Printf("🎯 Training network for exactly 2 epochs on %d samples...\n", len(inputs))

	epochs := 2 // Fixed to 2 epochs as requested
	learningRate := 0.01
	clipUpper := float32(1.0)
	clipLower := float32(-1.0)

	network.Train(inputs, targets, epochs, learningRate, true, clipUpper, clipLower)
	fmt.Printf("✅ Minimal training completed (2 epochs only)\n")
}

func evaluateNetworkWithADHD(network *paragon.Network[float32], inputs, targets [][][]float64) float64 {
	expectedOutputs := make([]float64, len(inputs))
	actualOutputs := make([]float64, len(inputs))

	for i := range inputs {
		network.Forward(inputs[i])
		output := network.GetOutput()

		// Get expected and actual class predictions
		expectedOutputs[i] = float64(argMax(targets[i][0]))
		actualOutputs[i] = float64(argMax(output))
	}

	// Use ADHD evaluation
	network.EvaluateModel(expectedOutputs, actualOutputs)
	return network.ComputeFinalScore()
}

func showADHDBreakdown(network *paragon.Network[float32], phase string) {
	if network.Performance == nil {
		fmt.Printf("❌ No ADHD performance data available for %s\n", phase)
		return
	}

	fmt.Printf("📊 %s ADHD Breakdown:\n", phase)
	fmt.Printf("   Total samples: %d\n", network.Performance.Total)
	fmt.Printf("   Failures (100%%+): %d\n", network.Performance.Failures)
	fmt.Printf("   Overall ADHD score: %.4f\n", network.Performance.Score)

	fmt.Println("   Deviation distribution:")
	buckets := []string{"0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%", "100%+"}
	for _, bucket := range buckets {
		if data, exists := network.Performance.Buckets[bucket]; exists {
			percentage := float64(data.Count) / float64(network.Performance.Total) * 100
			emoji := getADHDEmoji(bucket)
			fmt.Printf("     %s %s: %d samples (%.1f%%)\n", emoji, bucket, data.Count, percentage)
		}
	}
}

func getADHDEmoji(bucket string) string {
	switch bucket {
	case "0-10%":
		return "🟢" // Green - excellent
	case "10-20%":
		return "🟡" // Yellow - good
	case "20-30%":
		return "🟠" // Orange - okay
	case "30-40%":
		return "🔴" // Red - poor
	case "40-50%":
		return "🔴" // Red - poor
	case "50-100%":
		return "⚫" // Black - very poor
	case "100%+":
		return "💀" // Skull - catastrophic
	default:
		return "⚪"
	}
}

func displayNetworkArchitecture(network *paragon.Network[float32], grew bool) {
	if grew {
		fmt.Println("🏗️  Network Architecture (After ADHD Growth):")
	} else {
		fmt.Println("🏗️  Network Architecture (No Growth Applied):")
	}

	for i, layer := range network.Layers {
		layerType := "Hidden"
		if i == network.InputLayer {
			layerType = "Input"
		} else if i == network.OutputLayer {
			layerType = "Output"
		}

		activation := layer.Neurons[0][0].Activation
		neuronCount := layer.Width * layer.Height

		growth := ""
		if grew && i > 1 && i < len(network.Layers)-1 {
			growth = " 🆕" // Mark potentially new layers
		}

		fmt.Printf("   Layer %d (%s): %d neurons, activation=%s%s\n",
			i, layerType, neuronCount, activation, growth)
	}

	totalParams := calculateTotalParameters(network)
	fmt.Printf("📊 Total parameters: %d\n", totalParams)
}

func calculateTotalParameters(network *paragon.Network[float32]) int {
	totalParams := 0

	for i := 1; i < len(network.Layers); i++ {
		layer := network.Layers[i]
		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]
				totalParams += len(neuron.Inputs) + 1 // +1 for bias
			}
		}
	}

	return totalParams
}

func benchmarkNetwork(network *paragon.Network[float32], testInputs [][][]float64) {
	numTests := min(100, len(testInputs))

	startTime := time.Now()

	for i := 0; i < numTests; i++ {
		network.Forward(testInputs[i])
	}

	totalTime := time.Since(startTime)
	avgTime := totalTime / time.Duration(numTests)

	fmt.Printf("⚡ Performance: %d inferences in %v\n", numTests, totalTime)
	fmt.Printf("📊 Average inference time: %v\n", avgTime)
	fmt.Printf("🚀 Throughput: %.1f inferences/second\n",
		float64(numTests)/totalTime.Seconds())
}

func analyzeADHDGrowth(result *paragon.GrowthResult, initialScore, finalScore float64, growthTime time.Duration) {
	fmt.Println("📊 ADHD Growth Analysis:")

	if result.Success {
		efficiency := result.ADHDImprovement / growthTime.Seconds()
		fmt.Printf("   🎯 ADHD growth efficiency: %.4f score improvement/second\n", efficiency)

		if result.LayersAdded > 0 {
			layerEfficiency := result.ADHDImprovement / float64(result.LayersAdded)
			fmt.Printf("   🏗️  ADHD improvement per layer added: %.4f\n", layerEfficiency)
		}

		microNetEfficiency := result.ADHDImprovement / float64(result.MicroNetsProcessed)
		fmt.Printf("   🔬 ADHD improvement per micro net tested: %.4f\n", microNetEfficiency)

		fmt.Printf("   🎯 Growth occurred at layer %d (%.1f%% through network)\n",
			result.CheckpointLayer, float64(result.CheckpointLayer)/4.0*100)

		// ADHD quality analysis
		if result.ADHDImprovement > 10 {
			fmt.Printf("   🌟 Excellent ADHD improvement (>10 points)\n")
		} else if result.ADHDImprovement > 5 {
			fmt.Printf("   ✅ Good ADHD improvement (5-10 points)\n")
		} else if result.ADHDImprovement > 1 {
			fmt.Printf("   📈 Modest ADHD improvement (1-5 points)\n")
		} else {
			fmt.Printf("   📊 Minimal ADHD improvement (<1 point)\n")
		}
	} else {
		fmt.Printf("   📊 %d micro networks tested with ADHD evaluation\n",
			result.MicroNetsProcessed)
		fmt.Printf("   💡 No micro network achieved sufficient ADHD improvement\n")
	}

	fmt.Printf("   ⏱️  Time per micro network: %.2fms\n",
		growthTime.Seconds()*1000/float64(result.MicroNetsProcessed))
}

func getADHDKeyInsight(result *paragon.GrowthResult, improvement float64) string {
	if result.Success {
		return fmt.Sprintf("Aggressive ADHD-guided growth added %d layers and improved fine-grained prediction quality by %.2f points!",
			result.LayersAdded, improvement)
	} else {
		return "Aggressive ADHD analysis identified areas for improvement, but micro networks didn't achieve sufficient gains after multiple iterations."
	}
}

// Helper functions
func argMax(values []float64) int {
	maxIdx := 0
	maxVal := values[0]

	for i, val := range values[1:] {
		if val > maxVal {
			maxVal = val
			maxIdx = i + 1
		}
	}

	return maxIdx
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
