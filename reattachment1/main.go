package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"paragon" // Replace with actual import path
)

func main() {
	rand.Seed(42)

	fmt.Println("=== Complete Neural Network Surgery Verification with Timing ===")

	// File paths
	networkFile := "original_network.json"
	microNetworkFile := "micro_network.json"

	// Step 1: Create or load original network
	fmt.Println("\n🏗️  Step 1: Setting up original network...")
	network, isLoaded := setupNetwork(networkFile)

	// Step 2: Set test parameters
	testInput := [][]float64{{0.1, 0.5, 0.9}}
	checkpointLayer := 2
	tolerance := 1e-10

	fmt.Printf("📊 Test input: %v\n", testInput[0])
	fmt.Printf("🎯 Checkpoint layer: %d\n", checkpointLayer)

	// Step 3: Extract or load micro network
	fmt.Println("\n🔬 Step 2: Setting up micro network...")
	microNet, _ := setupMicroNetwork(network, checkpointLayer, microNetworkFile, isLoaded)

	// Step 4: THE 3-WAY VERIFICATION with timing
	fmt.Println("\n🧪 Step 3: Running 3-way verification...")
	startTime := time.Now()

	isEquivalent, outputs := microNet.VerifyThreeWayEquivalence(network, testInput, tolerance)

	verificationTime := time.Since(startTime)
	fmt.Printf("⏱️  Verification completed in: %v\n", verificationTime)

	// Display results
	fmt.Printf("🔍 Verification 1 - Main network full forward: [%.6f, %.6f]\n",
		outputs[0][0], outputs[0][1])
	fmt.Printf("🔍 Verification 2 - Main network from checkpoint: [%.6f, %.6f]\n",
		outputs[1][0], outputs[1][1])
	fmt.Printf("🔍 Verification 3 - Micro network from checkpoint: [%.6f, %.6f]\n",
		outputs[2][0], outputs[2][1])

	// Show differences
	fmt.Println("\n📋 Verification Results:")
	fmt.Printf("   Full vs Main-Checkpoint: %s (diff: [%.10f, %.10f])\n",
		getCheckMark(abs(outputs[0][0]-outputs[1][0]) < tolerance),
		abs(outputs[0][0]-outputs[1][0]), abs(outputs[0][1]-outputs[1][1]))

	fmt.Printf("   Main-Checkpoint vs Micro-Checkpoint: %s (diff: [%.10f, %.10f])\n",
		getCheckMark(abs(outputs[1][0]-outputs[2][0]) < tolerance),
		abs(outputs[1][0]-outputs[2][0]), abs(outputs[1][1]-outputs[2][1]))

	fmt.Printf("   Full vs Micro-Checkpoint: %s (diff: [%.10f, %.10f])\n",
		getCheckMark(abs(outputs[0][0]-outputs[2][0]) < tolerance),
		abs(outputs[0][0]-outputs[2][0]), abs(outputs[0][1]-outputs[2][1]))

	if isEquivalent {
		fmt.Println("\n🎉 ALL THREE OUTPUTS MATCH PERFECTLY!")
		fmt.Println("✅ Micro network is functionally equivalent to main network")
	} else {
		fmt.Println("\n⚠️  OUTPUTS DON'T MATCH - Investigation needed")
	}

	// Step 5: Test micro network normal vs checkpoint difference with timing
	fmt.Println("\n🔬 Step 4: Testing micro network normal vs checkpoint...")
	startTime = time.Now()

	checkpointState := network.GetLayerState(checkpointLayer)
	isDifferent, normalOutput, checkpointOutput := microNet.VerifyMicroNormalDiffers(testInput, checkpointState, 1e-6)

	differenceTestTime := time.Since(startTime)
	fmt.Printf("⏱️  Difference test completed in: %v\n", differenceTestTime)

	fmt.Printf("🔍 Micro normal forward: [%.6f, %.6f]\n", normalOutput[0], normalOutput[1])
	fmt.Printf("🔍 Micro checkpoint forward: [%.6f, %.6f]\n", checkpointOutput[0], checkpointOutput[1])
	fmt.Printf("📊 Normal vs Checkpoint different: %s (this should be TRUE)\n", getCheckMark(isDifferent))

	if isDifferent {
		fmt.Println("✅ Micro network normal path correctly differs from checkpoint path")
	} else {
		fmt.Println("⚠️  Micro network paths are identical (unexpected)")
	}

	// Step 6: Demonstrate complete surgery with timing
	fmt.Println("\n🚀 Step 5: Demonstrating complete surgery...")
	_ = demonstrateCompleteSurgery(network, testInput, checkpointLayer)

	// Step 7: Always save networks after surgery (they may have been modified)
	fmt.Println("\n💾 Step 6: Saving networks after surgery...")

	// Extract fresh micro network from potentially modified original network
	freshMicroNet := network.ExtractMicroNetwork(checkpointLayer)
	saveNetworks(network, freshMicroNet, networkFile, microNetworkFile)

	// Step 8: Performance summary
	fmt.Println("\n📊 Performance Summary:")
	fmt.Printf("   3-Way Verification: %v\n", verificationTime)
	fmt.Printf("   Difference Testing: %v\n", differenceTestTime)

	fmt.Println("\n✅ Complete verification test finished!")
}

func setupNetwork(networkFile string) (*paragon.Network[float32], bool) {
	if fileExists(networkFile) {
		fmt.Printf("📁 Loading existing network from %s...\n", networkFile)
		startTime := time.Now()

		networkAny, err := paragon.LoadNamedNetworkFromJSONFile(networkFile)
		if err != nil {
			log.Printf("Failed to load network: %v", err)
			fmt.Println("🏗️  Creating new network instead...")
			return createNewNetwork(), false
		}

		network, ok := networkAny.(*paragon.Network[float32])
		if !ok {
			log.Printf("Unexpected network type: %T", networkAny)
			fmt.Println("🏗️  Creating new network instead...")
			return createNewNetwork(), false
		}

		loadTime := time.Since(startTime)
		fmt.Printf("✅ Network loaded successfully in %v\n", loadTime)
		return network, true
	} else {
		fmt.Println("🏗️  Creating new network...")
		return createNewNetwork(), false
	}
}

func setupMicroNetwork(originalNet *paragon.Network[float32], checkpointLayer int, microFile string, originalLoaded bool) (*paragon.MicroNetwork[float32], bool) {
	if originalLoaded && fileExists(microFile) {
		fmt.Printf("📁 Loading existing micro network from %s...\n", microFile)
		startTime := time.Now()

		// Load micro network
		microNetworkAny, err := paragon.LoadNamedNetworkFromJSONFile(microFile)
		if err != nil {
			log.Printf("Failed to load micro network: %v", err)
			fmt.Println("🔬 Extracting new micro network instead...")
			return extractNewMicroNetwork(originalNet, checkpointLayer), false
		}

		microNetwork, ok := microNetworkAny.(*paragon.Network[float32])
		if !ok {
			log.Printf("Unexpected micro network type: %T", microNetworkAny)
			fmt.Println("🔬 Extracting new micro network instead...")
			return extractNewMicroNetwork(originalNet, checkpointLayer), false
		}

		loadTime := time.Since(startTime)
		fmt.Printf("✅ Micro network loaded successfully in %v\n", loadTime)

		// Reconstruct MicroNetwork wrapper
		microNet := &paragon.MicroNetwork[float32]{
			Network:       microNetwork,
			SourceLayers:  []int{0, checkpointLayer, originalNet.OutputLayer},
			CheckpointIdx: 1,
		}

		return microNet, true
	} else {
		fmt.Println("🔬 Extracting new micro network...")
		return extractNewMicroNetwork(originalNet, checkpointLayer), false
	}
}

func createNewNetwork() *paragon.Network[float32] {
	startTime := time.Now()

	layerSizes := []struct{ Width, Height int }{
		{3, 1}, {8, 1}, {6, 1}, {2, 1},
	}
	activations := []string{"linear", "relu", "relu", "softmax"}
	fullyConnected := []bool{false, true, true, true}

	network := paragon.NewNetwork[float32](layerSizes, activations, fullyConnected)
	network.Debug = false

	creationTime := time.Since(startTime)
	fmt.Printf("✅ Created network: %d → %d → %d → %d in %v\n",
		layerSizes[0].Width, layerSizes[1].Width, layerSizes[2].Width, layerSizes[3].Width, creationTime)

	return network
}

func extractNewMicroNetwork(originalNet *paragon.Network[float32], checkpointLayer int) *paragon.MicroNetwork[float32] {
	startTime := time.Now()

	microNet := originalNet.ExtractMicroNetwork(checkpointLayer)

	extractionTime := time.Since(startTime)
	fmt.Printf("✅ Micro network extracted: %d layers in %v\n", len(microNet.Network.Layers), extractionTime)

	return microNet
}

func demonstrateCompleteSurgery(network *paragon.Network[float32], testInput [][]float64, checkpointLayer int) *paragon.MicroNetwork[float32] {
	testInputs := [][][]float64{
		testInput,
		{{0.3, 0.7, 0.2}},
		{{0.8, 0.1, 0.6}},
	}

	// Get original performance
	startTime := time.Now()
	network.Forward(testInput)
	originalOutput := network.GetOutput()
	forwardTime := time.Since(startTime)

	fmt.Printf("📊 Original network output: [%.6f, %.6f] (computed in %v)\n",
		originalOutput[0], originalOutput[1], forwardTime)

	// Perform complete surgery with timing
	fmt.Println("🏥 Performing complete network surgery...")
	startTime = time.Now()

	tolerance := 1e-6
	microNet, err := network.NetworkSurgery(checkpointLayer, testInputs, tolerance)
	if err != nil {
		log.Printf("Surgery failed: %v", err)

	}

	surgeryTime := time.Since(startTime)
	fmt.Printf("⏱️  Surgery completed in: %v\n", surgeryTime)

	// Test final result
	startTime = time.Now()
	network.Forward(testInput)
	finalOutput := network.GetOutput()
	finalForwardTime := time.Since(startTime)

	fmt.Printf("📊 Post-surgery output: [%.6f, %.6f] (computed in %v)\n",
		finalOutput[0], finalOutput[1], finalForwardTime)
	fmt.Printf("🏆 Surgery complete! Micro network has %d layers\n", len(microNet.Network.Layers))

	// Show the difference
	outputDiff := abs(originalOutput[0]-finalOutput[0]) + abs(originalOutput[1]-finalOutput[1])
	if outputDiff < 1e-6 {
		fmt.Println("✅ Surgery preserved original functionality perfectly")
	} else {
		fmt.Printf("🔧 Surgery modified network (total output change: %.8f)\n", outputDiff)
	}

	// Performance comparison
	fmt.Printf("⏱️  Performance: Original forward (%v) vs Post-surgery forward (%v)\n",
		forwardTime, finalForwardTime)

	return microNet
}

func saveNetworks(network *paragon.Network[float32], microNet *paragon.MicroNetwork[float32], networkFile, microFile string) {
	startTime := time.Now()

	// Save original network
	if err := network.SaveJSON(networkFile); err != nil {
		log.Printf("Failed to save network: %v", err)
	} else {
		fmt.Printf("💾 Original network saved to %s\n", networkFile)
	}

	// Save micro network
	if err := microNet.Network.SaveJSON(microFile); err != nil {
		log.Printf("Failed to save micro network: %v", err)
	} else {
		fmt.Printf("💾 Micro network saved to %s\n", microFile)
	}

	saveTime := time.Since(startTime)
	fmt.Printf("⏱️  Networks saved in: %v\n", saveTime)
}

func fileExists(filename string) bool {
	_, err := os.Stat(filename)
	return !os.IsNotExist(err)
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func getCheckMark(condition bool) string {
	if condition {
		return "✅"
	}
	return "❌"
}
