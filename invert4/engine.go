package main

import (
	"fmt"
	"log"

	"paragon"
)

const (
	baseURL   = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	mnistDir  = "mnist_data"
	modelDir  = "models"
	modelFile = "mnist_model.json"
)

func main() {
	// --- Load MNIST ---
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatalf("MNIST download error: %v", err)
	}
	trainInputs, trainTargets, err := loadMNISTData(mnistDir, true)
	if err != nil {
		log.Fatalf("Training load failed: %v", err)
	}
	testInputs, testTargets, err := loadMNISTData(mnistDir, false)
	if err != nil {
		log.Fatalf("Test load failed: %v", err)
	}
	trainSetInputs, trainSetTargets, _, _ := paragon.SplitDataset(trainInputs, trainTargets, 0.8)

	// --- Build Model ---
	layerSizes := []struct{ Width, Height int }{{28, 28}, {16, 16}, {10, 1}}
	activations := []string{"leaky_relu", "leaky_relu", "softmax"}
	fullyConnected := []bool{true, false, true}
	//modelPath := filepath.Join(modelDir, modelFile)

	var nn *paragon.Network
	fmt.Println("🧠 No pre-trained model found. Starting training...")
	nn = paragon.NewNetwork(layerSizes, activations, fullyConnected)
	nn.Train(trainSetInputs, trainSetTargets, 10, 0.01, true)
	fmt.Println("✅ Training complete.")

	// --- ADHD Evaluation ---
	var expected, predicted []float64
	for i, input := range testInputs {
		nn.Forward(input)
		out := extractOutput(nn)
		pred := paragon.ArgMax(out)
		trueLabel := paragon.ArgMax(testTargets[i][0])
		expected = append(expected, float64(trueLabel))
		predicted = append(predicted, float64(pred))
	}
	nn.EvaluateModel(expected, predicted)

	// --- Unified ADHD Diagnostics ---
	fmt.Println("\n---------SimplePRINT----------")
	fmt.Printf("🧠 ADHD Score: %.2f\n", nn.Performance.Score)
	fmt.Println("📊 Deviation Buckets:")
	for bucket, stats := range nn.Performance.Buckets {
		fmt.Printf(" - %-7s → %d samples\n", bucket, stats.Count)
	}

	fmt.Println("\n---------PrintFullDiagnostics----------")
	nn.EvaluateFull(expected, predicted)
	nn.PrintFullDiagnostics()

	fmt.Println("\n---------PrintSAMPLEDiagnostics----------")
	expectedVectors := make([][]float64, len(testInputs))
	actualVectors := make([][]float64, len(testInputs))
	for i := range testInputs {
		nn.Forward(testInputs[i])
		actualVectors[i] = nn.ExtractOutput()
		expectedVectors[i] = testTargets[i][0]
	}

	perSample := paragon.ComputePerSamplePerformance(expectedVectors, actualVectors, 0.01, nn)
	paragon.PrintSampleDiagnostics(perSample, 0.01)

	output := trainTargets[0]
	reconstructed := nn.ReverseInferFromOutput(output)
	_ = SaveFloatImage(reconstructed, "reconstructed_trained.png")

	// --- Reverse Inference: Generate Images from One-Hot Softmax Vectors ---
	for digit := 0; digit <= 9; digit++ {
		output := make([][]float64, 1)  // batch of 1
		output[0] = make([]float64, 10) // 10-class output
		output[0][digit] = 1.0          // One-hot at index `digit`

		reconstructed := nn.ReverseInferFromOutput(output)
		filename := fmt.Sprintf("reconstructed_%d.png", digit)
		err := SaveFloatImage(reconstructed, filename)
		if err != nil {
			fmt.Printf("❌ Failed to save image %s: %v\n", filename, err)
		} else {
			fmt.Printf("✅ Saved: %s\n", filename)
		}
	}

	// --- Reverse Inference from First Sample ---
	//RunReverseTest(trainInputs, trainTargets, nil, "before_proxy")

	//bestStudent := TuneStudentWithProxy(trainInputs, trainTargets)

	//RunReverseAttributionTuning(trainInputs, trainTargets, bestStudent)

	//bestStudent.SetBiasWeightFromReverseAttributionPercent(trainInputs[0], trainTargets[0], 0.010)

	//RunReverseTest(trainInputs, trainTargets, bestStudent, "after_proxy")

	//RunReverseSetPercentTuning(trainInputs, trainTargets, bestStudent, (0.000016667)*2)
	//RunReverseSetPercentTuning(trainInputs, trainTargets, bestStudent, 1.0)
	//RunBidirectionalConstraintTraining(trainInputs, trainTargets, bestStudent, 0.01, 0.9)
	//RunSandwichTraining(trainInputs, trainTargets, bestStudent, 0.01, 0.9)

}

func createStudentNet() *paragon.Network {

	layerSizes := []struct{ Width, Height int }{{28, 28}, {16, 16}, {10, 1}}
	activations := []string{"leaky_relu", "leaky_relu", "softmax"}
	fullyConnected := []bool{true, false, true}

	nn := paragon.NewNetwork(layerSizes, activations, fullyConnected)

	return nn
}

func RunReverseTest(dataset [][][]float64, targets [][][]float64, student *paragon.Network, label string) {
	if student == nil {
		student = createStudentNet()
	}

	// Save reverse image of first sample for visualization
	output := targets[0]
	reconstructed := student.ReverseInferFromOutput(output)
	_ = SaveFloatImage(dataset[0], "original_input.png")
	_ = SaveFloatImage(reconstructed, fmt.Sprintf("reconstructed_%s.png", label))

	// ADHD evaluation on full batch
	var expected, predicted []float64
	for i := range dataset {
		student.Forward(dataset[i])
		expectedLabel := float64(paragon.ArgMax(targets[i][0]))
		predictedLabel := float64(paragon.ArgMax(student.ExtractOutput()))
		expected = append(expected, expectedLabel)
		predicted = append(predicted, predictedLabel)
	}

	student.EvaluateModel(expected, predicted)
	fmt.Printf("🧠 ADHD Score [%s]: %.2f\n", label, student.Performance.Score)

}

func TuneStudentWithProxy(dataset [][][]float64, targets [][][]float64) *paragon.Network {
	type Config struct {
		MaxUpdate  float64
		Damping    float64
		ProxyDecay float64
	}

	configs := []Config{
		{0.5, 0.1, 0.9},
		{0.5, 0.3, 0.9},
		{0.5, 0.7, 0.9},
		{1.0, 0.3, 0.9},
		{5.0, 0.3, 0.9},
	}

	var bestStudent *paragon.Network
	bestScore := -1.0

	for i, cfg := range configs {
		student := createStudentNet()

		// Apply PropagateProxyError for each sample
		for j := range dataset {
			errorSignal := 1.0 // Could be refined with actual deviation
			student.PropagateProxyError(dataset[j], errorSignal,
				0.01, cfg.MaxUpdate, cfg.Damping, cfg.ProxyDecay)
		}

		// Full batch evaluation
		var expected, predicted []float64
		for j := range dataset {
			student.Forward(dataset[j])
			expectedLabel := float64(paragon.ArgMax(targets[j][0]))
			predictedLabel := float64(paragon.ArgMax(student.ExtractOutput()))
			expected = append(expected, expectedLabel)
			predicted = append(predicted, predictedLabel)
		}

		student.EvaluateModel(expected, predicted)
		score := student.Performance.Score

		fmt.Printf("🔁 Config %d → Max=%.2f, Damp=%.2f → ADHD Score: %.2f\n",
			i, cfg.MaxUpdate, cfg.Damping, score)

		if score > bestScore {
			bestScore = score
			bestStudent = student
		}
	}

	return bestStudent
}

func RunReverseAttributionTuning(dataset [][][]float64, targets [][][]float64, student *paragon.Network) {
	fmt.Println("\n---------Reverse Attribution Tuning----------")

	for i := range dataset {
		fmt.Println(i)
		target := targets[i]
		student.TuneWithReverseAttribution(dataset[i], target, 0.1) // Step size can be tuned
		break
	}

}

func RunReverseSetPercentTuning(dataset [][][]float64, targets [][][]float64, student *paragon.Network, percent float64) {
	fmt.Printf("\n---------Reverse Attribution Direct Set (%.2f%% per sample) ----------\n", percent*100)

	total := len(dataset)
	checkpoint := total / 20 // every 5%

	if checkpoint == 0 {
		checkpoint = 1 // handle very small datasets
	}

	for i := range dataset {
		student.SetBiasWeightFromReverseAttributionPercent(dataset[i], targets[i], percent)

		// Evaluate every 5% of the dataset
		if (i+1)%checkpoint == 0 || i == total-1 {
			var expected, predicted []float64
			for j := range dataset {
				student.Forward(dataset[j])
				expectedLabel := float64(paragon.ArgMax(targets[j][0]))
				predictedLabel := float64(paragon.ArgMax(student.ExtractOutput()))
				expected = append(expected, expectedLabel)
				predicted = append(predicted, predictedLabel)
			}

			student.EvaluateModel(expected, predicted)
			fmt.Printf("🧠 ADHD Score after %3d/%d samples (%.0f%%): %.2f\n",
				i+1, total, float64(i+1)*100.0/float64(total), student.Performance.Score)
		}
	}
}

func RunBidirectionalConstraintTraining(dataset [][][]float64, targets [][][]float64, student *paragon.Network, percent, decay float64) {
	fmt.Printf("\n--------- Bidirectional Constraint Training (%.2f%% strength, decay=%.2f) ----------\n", percent*100, decay)

	total := len(dataset)
	checkpoint := total / 20 // every 5%

	if checkpoint == 0 {
		checkpoint = 1
	}

	for i := range dataset {
		student.PropagateBidirectionalConstraint(dataset[i], targets[i], percent, decay)

		// Score every 5%
		if (i+1)%checkpoint == 0 || i == total-1 {
			var expected, predicted []float64
			for j := range dataset {
				student.Forward(dataset[j])
				expectedLabel := float64(paragon.ArgMax(targets[j][0]))
				predictedLabel := float64(paragon.ArgMax(student.ExtractOutput()))
				expected = append(expected, expectedLabel)
				predicted = append(predicted, predictedLabel)
			}
			student.EvaluateModel(expected, predicted)
			fmt.Printf("🧠 ADHD Score after %3d/%d samples (%.0f%%): %.2f\n",
				i+1, total, float64(i+1)*100.0/float64(total), student.Performance.Score)
		}
	}
}

func RunSandwichTraining(dataset [][][]float64, targets [][][]float64, student *paragon.Network, lr, decay float64) {
	fmt.Printf("\n--------- Sandwich Constraint Training (lr=%.4f, decay=%.2f) ----------\n", lr, decay)

	total := len(dataset)
	checkpoint := total / 20 // 5% progress reporting
	if checkpoint == 0 {
		checkpoint = 1
	}

	for i := range dataset {
		student.PropagateSandwichConstraint(dataset[i], targets[i], lr, decay)

		// Score ADHD every 5%
		if (i+1)%checkpoint == 0 || i == total-1 {
			var expected, predicted []float64
			for j := range dataset {
				student.Forward(dataset[j])
				expectedLabel := float64(paragon.ArgMax(targets[j][0]))
				predictedLabel := float64(paragon.ArgMax(student.ExtractOutput()))
				expected = append(expected, expectedLabel)
				predicted = append(predicted, predictedLabel)
			}

			student.EvaluateModel(expected, predicted)
			fmt.Printf("🧠 ADHD Score after %3d/%d (%.0f%%): %.2f\n",
				i+1, total, float64(i+1)*100.0/float64(total), student.Performance.Score)
		}
	}
}
