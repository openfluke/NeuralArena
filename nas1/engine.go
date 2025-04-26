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
	basic(lstConvCubes)
	//fullNas(lstConvCubes)
}

func basic(lstTmpData [][]string) {
	lstData, err := paragon.ConvertToFloat64(lstTmpData)
	if err != nil {
		fmt.Println("❌ Failed to convert data to float64:", err)
		return
	}

	lstDataRemove := paragon.RemoveColumns(lstData, []int{0})
	lstDataExpectedOutputPadding := paragon.Padding(lstDataRemove, 12, 0.0)

	fmt.Println("Original row:", lstData[0])
	fmt.Println("Expected output Padded row:", lstDataExpectedOutputPadding[0])

	lstDataSelected := paragon.SelectColumns(lstData, []int{0})
	lstDataInputPadding := paragon.Padding(lstDataSelected, 99, 0.0)

	fmt.Println("input without padding row:", lstDataSelected[0])
	fmt.Println("Expected intput Padded row:", lstDataInputPadding[0])

	// Convert to 3D slices
	fmt.Println("Convert to 3D slices inputs3D")
	inputs3D := make([][][]float64, len(lstDataInputPadding))
	for i, row := range lstDataInputPadding {
		inputs3D[i] = [][]float64{row} // [1][99]
		fmt.Println(inputs3D[i])
	}

	fmt.Println("Convert to 3D slices targets3D")
	targets3D := make([][][]float64, len(lstDataExpectedOutputPadding))
	for i, row := range lstDataExpectedOutputPadding {
		targets3D[i] = [][]float64{row} // [1][12]
		fmt.Println(targets3D[i])
	}

	layerSizes := []struct{ Width, Height int }{
		{99, 1}, // Input layer (8 cubes × 12 features)
		{5, 5},  // Hidden
		{5, 5},  // Hidden
		{12, 1}, // Output layer (reconstruct 12 features)
	}
	activations := []string{"linear", "relu", "relu", "relu"}
	fullyConnected := []bool{true, true, true, true}

	net := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	net.Train(inputs3D, targets3D, 20, 0.1, true)

	/*fmt.Println(lstDataInputPadding)
	fmt.Println(lstDataExpectedOutputPadding)

	fmt.Println(inputs3D)
	fmt.Println(targets3D)*/

	// Assuming inputs3D and targets3D are your input and target datasets
	expected := make([]float64, 0)
	actual := make([]float64, 0)
	for i := range inputs3D {
		net.Forward(inputs3D[i])               // Run the network on the input
		output := net.ExtractOutput()          // Get the predicted output (e.g., []float64 of length 12)
		target := targets3D[i][0]              // Get the true target (e.g., []float64 of length 12)
		expected = append(expected, target...) // Flatten target into expected
		actual = append(actual, output...)     // Flatten prediction into actual
	}

	fmt.Println("---------SimplePRINT----------")
	net.EvaluateModel(expected, actual) // Evaluate using the ADHD metric
	fmt.Printf("🧠 ADHD Score: %.2f\n", net.Performance.Score)
	fmt.Printf("📊 Deviation Buckets:\n")
	for bucket, stats := range net.Performance.Buckets {
		fmt.Printf(" - %s: %d samples\n", bucket, stats.Count)
	}

	fmt.Println("---------PrintFullDiagnostics----------")
	net.EvaluateFull(expected, actual)
	net.PrintFullDiagnostics()

	fmt.Println("---------PrintSAMPLEDiagnostics----------")

	// Prepare per-sample slices for sample-level comparison
	expectedVectors := [][]float64{}
	actualVectors := [][]float64{}

	for i := range inputs3D {
		net.Forward(inputs3D[i])
		actualVectors = append(actualVectors, net.ExtractOutput())
		expectedVectors = append(expectedVectors, targets3D[i][0])
	}

	// Compute and print sample-level diagnostics
	perf := paragon.ComputePerSamplePerformance(expectedVectors, actualVectors, 0.01, net)
	paragon.PrintSampleDiagnostics(perf, 0.01)

	// --- Neural Architecture Search ---------------------------------
	/*fmt.Println("---------Neural Architecture Search----------")
	numClones := 100                // how many mutated copies to evaluate
	nasEpochs := 5                  // epochs of training per clone
	baseLR := 0.01                  // starting learning-rate (decays inside NAS)
	weightMutationRate := 0.05      // std-dev of Gaussian noise added to weights
	earlyStopOnNegativeLoss := true // keep the training loop stable

	bestNet, bestScore, improved, err := net.NormalNASLayerWiseGrowingEnhanced(
		numClones,
		nasEpochs,
		baseLR,
		weightMutationRate,
		inputs3D,
		targets3D,
		earlyStopOnNegativeLoss,
	)
	if err != nil {
		fmt.Printf("❌ NAS failed: %v\n", err)
		return
	}

	// --- Evaluate best candidate ------------------------------------
	expectedNAS, actualNAS := make([]float64, 0), make([]float64, 0)
	for i := range inputs3D {
		bestNet.Forward(inputs3D[i])
		actualNAS = append(actualNAS, bestNet.ExtractOutput()...)
		expectedNAS = append(expectedNAS, targets3D[i][0]...)
	}
	bestNet.EvaluateModel(expectedNAS, actualNAS)

	fmt.Printf("🧠 Original Network ADHD Score: %.2f\n", net.Performance.Score)
	fmt.Printf("🧠 Best NAS Network ADHD Score: %.2f\n", bestScore)
	fmt.Printf("📊 Best NAS Deviation Buckets:\n")
	for bucket, stats := range bestNet.Performance.Buckets {
		fmt.Printf(" - %s: %d samples\n", bucket, stats.Count)
	}
	if improved {
		fmt.Println("✅ NAS improved the network performance!")
	} else {
		fmt.Println("⚠️ NAS did not improve the network performance.")
	}*/

	// ---------- Iterative Neural Architecture Search -----------------
	fmt.Println("---------Iterative Neural Architecture Search----------")

	// --- NAS hyper-params you can tweak once here --------------------
	/*numClones := 100 // per round
	nasEpochs := 5
	baseLR := 0.01
	weightMutationRate := 0.05
	earlyStop := true
	targetADHD := 95.0 // stop when we reach this
	maxAttempts := 10  // safety cap
	NormalNASLayerWiseGrowingEnhanced := true
	// -----------------------------------------------------------------

	parentNet := net // start from the network you already trained
	parentScore := net.Performance.Score
	initialScore := parentScore
	fmt.Printf("🔰 Starting ADHD Score: %.2f\n", parentScore)

	for attempt := 1; attempt <= maxAttempts && parentScore < targetADHD; attempt++ {
		fmt.Printf("\n🔄 NAS Attempt %d / %d (current ADHD %.2f)\n",
			attempt, maxAttempts, parentScore)

		candNet, candScore, improved, err := parentNet.NormalNASLayerWiseGrowingEnhanced(
			numClones,
			nasEpochs,
			baseLR,
			weightMutationRate,
			inputs3D,
			targets3D,
			earlyStop,
			NormalNASLayerWiseGrowingEnhanced,
		)
		if err != nil {
			fmt.Printf("❌ NAS attempt %d failed: %v – skipping\n", attempt, err)
			continue
		}

		// Evaluate candidate on full set (already done inside NAS, but good to keep)
		// candScore is the ADHD score computed inside the NAS function.
		if improved && candScore > parentScore {
			fmt.Printf("✅ Improved: %.2f  →  %.2f\n", parentScore, candScore)
			parentNet = candNet
			parentScore = candScore
		} else {
			fmt.Printf("⚠️ No improvement this round (best %.2f)\n", parentScore)
			// Optionally raise mutation strength to escape local minima
			weightMutationRate *= 1.2
		}
	}

	// ------------------- Final report --------------------------------
	fmt.Println("\n====================== Final NAS Report ======================")
	fmt.Printf("🧠 Original ADHD Score: %.2f\n", initialScore)
	fmt.Printf("🧠 Best   ADHD Score: %.2f\n", parentScore)
	if parentScore >= targetADHD {
		fmt.Printf("🎉 Target of %.1f reached!\n", targetADHD)
	} else {
		fmt.Printf("🔎 Target not reached after %d attempts (best %.2f)\n",
			maxAttempts, parentScore)
	}
	fmt.Println("==============================================================")*/

	// capture the original score
	initialScore := net.Performance.Score

	// run the iterative NAS
	bestNet, bestScore := net.IterativeInitNAS(
		100,       // numClones
		5,         // nasEpochs
		0.01,      // baseLR
		0.05,      // weightMutationRate
		true,      // earlyStopOnNegativeLoss
		true,      // enableActMutation
		95.0,      // targetADHD
		10,        // maxAttempts
		inputs3D,  // inputs
		targets3D, // targets
	)

	// final report
	fmt.Println("\n====================== Final NAS Report ======================")
	fmt.Printf("🧠 Original ADHD Score: %.2f\n", initialScore)
	fmt.Printf("🧠 Best   ADHD Score: %.2f\n", bestScore)
	if bestScore >= 95.0 {
		fmt.Printf("🎉 Target of %.1f reached!\n", 95.0)
	} else {
		fmt.Printf("🔎 Target not reached after %d attempts (best %.2f)\n", 10, bestScore)
	}
	fmt.Println("==============================================================")

	_ = bestNet
}
