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
	lstDataInputPadding := paragon.Padding(lstDataSelected, 96, 0.0)

	fmt.Println("input without padding row:", lstDataSelected[0])
	fmt.Println("Expected intput Padded row:", lstDataInputPadding[0])

	// Convert to 3D slices
	fmt.Println("Convert to 3D slices inputs3D")
	inputs3D := make([][][]float64, len(lstDataInputPadding))
	for i, row := range lstDataInputPadding {
		inputs3D[i] = [][]float64{row} // [1][96]
		fmt.Println(inputs3D[i])
	}

	fmt.Println("Convert to 3D slices targets3D")
	targets3D := make([][][]float64, len(lstDataExpectedOutputPadding))
	for i, row := range lstDataExpectedOutputPadding {
		targets3D[i] = [][]float64{row} // [1][12]
		fmt.Println(targets3D[i])
	}

	layerSizes := []struct{ Width, Height int }{
		{96, 1}, // Input layer (8 cubes × 12 features)
		{10, 1}, // Hidden
		{12, 1}, // Output layer (reconstruct 12 features)
	}
	activations := []string{"linear", "relu", "relu"}
	fullyConnected := []bool{true, true, true}

	net := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	net.Train(inputs3D, targets3D, 20, 0.01, true)

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

}
