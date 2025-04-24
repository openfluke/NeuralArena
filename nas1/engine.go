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

	net.Train(inputs, targets, 20, 0.01)
	fmt.Println("Training complete!")

	// ADHD Evaluation Metrics
	expected := make([]float64, len(targets))
	actual := make([]float64, len(targets))

	for i := range inputs {
		net.Forward(inputs[i])

		output := net.ExtractOutput() // returns []float64 from final layer
		if len(output) == 0 || len(targets[i][0]) == 0 {
			fmt.Printf("⚠️ Skipping empty prediction at sample %d\n", i)
			continue
		}

		// Use average prediction (or a specific value if desired)
		expected[i] = average(targets[i][0])
		actual[i] = average(output)
	}

	net.EvaluateModel(expected, actual)

	fmt.Printf("🧠 ADHD Score: %.2f\n", net.Performance.Score)
	fmt.Printf("📊 Deviation Buckets:\n")
	for bucket, stats := range net.Performance.Buckets {
		fmt.Printf(" - %s: %d samples\n", bucket, stats.Count)
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
