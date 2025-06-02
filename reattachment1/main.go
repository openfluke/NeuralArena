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
	modifiedNetworkFile := "modified_network.json"
	microNetworkFile := "micro_network.json"

	// Step 1: Create or load networks
	fmt.Println("\n🏗️  Step 1: Setting up networks...")
	network, originalNet, isLoaded := setupNetworks(networkFile, modifiedNetworkFile)

	// Step 2: Set test parameters
	testInput := [][]float64{{0.1, 0.5, 0.9}}
	checkpointLayer := 2
	tolerance := 1e-10

	fmt.Printf("📊 Test input: %v\n", testInput[0])
	fmt.Printf("🎯 Checkpoint layer: %d\n", checkpointLayer)

	// Step 3: Extract or load micro network from ORIGINAL network
	fmt.Println("\n🔬 Step 2: Setting up micro network...")
	var microNet *paragon.MicroNetwork[float32]
	if isLoaded && originalNet != nil {
		fmt.Println("🔬 Extracting micro network from preserved original network...")
		microNet = extractNewMicroNetwork(originalNet, checkpointLayer)
	} else {
		fmt.Println("🔬 Extracting micro network from current network...")
		microNet = extractNewMicroNetwork(network, checkpointLayer)
	}

	// Step 4: THE 3-WAY VERIFICATION with timing
	fmt.Println("\n🧪 Step 3: Running 3-way verification...")

	if isLoaded && originalNet != nil {
		fmt.Println("🔍 Testing compatibility between original structure and modified network...")
	}

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
		if isLoaded && originalNet != nil {
			fmt.Println("✅ Modified network maintains compatibility with original structure")
		} else {
			fmt.Println("✅ Micro network is functionally equivalent to main network")
		}
	} else {
		fmt.Println("\n⚠️  OUTPUTS DON'T MATCH - Investigation needed")
		if isLoaded && originalNet != nil {
			fmt.Println("⚠️  Modified network is incompatible with original structure!")
			fmt.Println("💡 This could indicate surgery corruption or incompatible changes")
		}
		return // Skip surgery if verification fails
	}

	// Step 5: Test micro network normal vs checkpoint difference
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

	// Step 6: Demonstrate complete surgery (only on unmodified networks)
	if !isLoaded {
		fmt.Println("\n🚀 Step 5: Demonstrating complete surgery...")
		resultMicroNet := demonstrateCompleteSurgery(network, testInput, checkpointLayer)

		// Step 7: Save all networks after surgery
		fmt.Println("\n💾 Step 6: Saving networks after surgery...")
		if resultMicroNet != nil {
			saveAllNetworks(network, network, resultMicroNet, networkFile, modifiedNetworkFile, microNetworkFile)
		}
	} else {
		fmt.Println("\n🚀 Step 5: Surgery skipped on loaded modified network")
		fmt.Println("💡 Surgery is only performed on fresh networks to maintain verification integrity")
		fmt.Println("💡 Delete saved files to start fresh and perform new surgery")
	}

	// Step 8: Performance summary
	fmt.Println("\n📊 Performance Summary:")
	fmt.Printf("   3-Way Verification: %v\n", verificationTime)
	fmt.Printf("   Difference Testing: %v\n", differenceTestTime)

	fmt.Println("\n✅ Complete verification test finished!")
}

func setupNetworks(networkFile, modifiedNetworkFile string) (*paragon.Network[float32], *paragon.Network[float32], bool) {
	if fileExists(modifiedNetworkFile) {
		// Load modified network as primary, original as backup
		fmt.Printf("📁 Loading modified network from %s...\n", modifiedNetworkFile)
		modifiedNet := loadNetworkFromFile(modifiedNetworkFile)

		var originalNet *paragon.Network[float32]
		if fileExists(networkFile) {
			fmt.Printf("📁 Loading original network from %s...\n", networkFile)
			originalNet = loadNetworkFromFile(networkFile)
		}

		return modifiedNet, originalNet, true
	} else if fileExists(networkFile) {
		// Load single network (could be original or modified)
		fmt.Printf("📁 Loading network from %s...\n", networkFile)
		network := loadNetworkFromFile(networkFile)
		return network, nil, true
	} else {
		// Create fresh network
		fmt.Println("🏗️  Creating new network...")
		network := createNewNetwork()
		return network, nil, false
	}
}

func loadNetworkFromFile(filename string) *paragon.Network[float32] {
	startTime := time.Now()

	networkAny, err := paragon.LoadNamedNetworkFromJSONFile(filename)
	if err != nil {
		log.Printf("Failed to load network from %s: %v", filename, err)
		fmt.Println("🏗️  Creating new network instead...")
		return createNewNetwork()
	}

	network, ok := networkAny.(*paragon.Network[float32])
	if !ok {
		log.Printf("Unexpected network type from %s: %T", filename, networkAny)
		fmt.Println("🏗️  Creating new network instead...")
		return createNewNetwork()
	}

	loadTime := time.Since(startTime)
	fmt.Printf("✅ Network loaded successfully in %v\n", loadTime)
	return network
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

	startTime := time.Now()
	network.Forward(testInput)
	originalOutput := network.GetOutput()
	forwardTime := time.Since(startTime)

	fmt.Printf("📊 Original network output: [%.6f, %.6f] (computed in %v)\n",
		originalOutput[0], originalOutput[1], forwardTime)

	fmt.Println("🏥 Performing complete network surgery...")
	startTime = time.Now()

	tolerance := 1e-6
	microNet, err := network.NetworkSurgery(checkpointLayer, testInputs, tolerance)

	surgeryTime := time.Since(startTime)
	fmt.Printf("⏱️  Surgery completed in: %v\n", surgeryTime)

	if err != nil {
		log.Printf("Surgery failed: %v", err)
		fmt.Println("⚠️  Surgery failed - skipping post-surgery analysis")

		startTime = time.Now()
		network.Forward(testInput)
		finalOutput := network.GetOutput()
		finalForwardTime := time.Since(startTime)

		fmt.Printf("📊 Post-surgery output: [%.6f, %.6f] (computed in %v)\n",
			finalOutput[0], finalOutput[1], finalForwardTime)

		return nil
	}

	startTime = time.Now()
	network.Forward(testInput)
	finalOutput := network.GetOutput()
	finalForwardTime := time.Since(startTime)

	fmt.Printf("📊 Post-surgery output: [%.6f, %.6f] (computed in %v)\n",
		finalOutput[0], finalOutput[1], finalForwardTime)

	if microNet != nil && microNet.Network != nil {
		fmt.Printf("🏆 Surgery complete! Micro network has %d layers\n", len(microNet.Network.Layers))
	}

	outputDiff := abs(originalOutput[0]-finalOutput[0]) + abs(originalOutput[1]-finalOutput[1])
	if outputDiff < 1e-6 {
		fmt.Println("✅ Surgery preserved original functionality perfectly")
	} else {
		fmt.Printf("🔧 Surgery modified network (total output change: %.8f)\n", outputDiff)
	}

	fmt.Printf("⏱️  Performance: Original forward (%v) vs Post-surgery forward (%v)\n",
		forwardTime, finalForwardTime)

	return microNet
}

func saveAllNetworks(originalNet, modifiedNet *paragon.Network[float32], microNet *paragon.MicroNetwork[float32],
	originalFile, modifiedFile, microFile string) {
	startTime := time.Now()

	// Save original network (pre-surgery)
	if err := originalNet.SaveJSON(originalFile); err != nil {
		log.Printf("Failed to save original network: %v", err)
	} else {
		fmt.Printf("💾 Original network saved to %s\n", originalFile)
	}

	// Save modified network (post-surgery)
	if err := modifiedNet.SaveJSON(modifiedFile); err != nil {
		log.Printf("Failed to save modified network: %v", err)
	} else {
		fmt.Printf("💾 Modified network saved to %s\n", modifiedFile)
	}

	// Save micro network
	if microNet != nil && microNet.Network != nil {
		if err := microNet.Network.SaveJSON(microFile); err != nil {
			log.Printf("Failed to save micro network: %v", err)
		} else {
			fmt.Printf("💾 Micro network saved to %s\n", microFile)
		}
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
